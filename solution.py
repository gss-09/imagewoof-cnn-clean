"""From-scratch sequential CNN, training CLI, and self-contained grading interface."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


CLASS_NAMES = (
    "shih_tzu", "rhodesian_ridgeback", "beagle", "english_foxhound",
    "border_terrier", "australian_terrier", "golden_retriever",
    "old_english_sheepdog", "samoyed", "dingo",
)
# Fixed scaling to [-1, 1]; no external dataset statistics are used.
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
ARCHITECTURES = {
    "DogCNN-v1": (32, 64, 128),
    "DogCNN-v2": (32, 64, 128, 192),
}


class DogCNN(nn.Module):
    """Pairs of convolutions followed by pooling, with a ten-class classifier."""

    def __init__(self, architecture="DogCNN-v1"):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
        self.architecture = architecture
        block_channels = ARCHITECTURES[architecture]
        layers = []
        in_channels = 3
        for channels in block_channels:
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ])
                in_channels = channels
            layers.append(nn.MaxPool2d(2))
        layers.extend([
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(block_channels[-1], len(CLASS_NAMES)),
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        return self.network(images)


def load_model(path="model.pth"):
    """Build the model, load weights on CPU, and return it in eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != list(CLASS_NAMES):
        raise ValueError("Checkpoint class mapping does not match the official IDs")
    if checkpoint["mean"] != list(MEAN) or checkpoint["std"] != list(STD):
        raise ValueError("Checkpoint normalization does not match this implementation")
    model = DogCNN(checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def predict(model, images):
    """Raw RGB uint8 NHWC images -> NumPy int64 IDs in classes.txt order."""
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8:
        raise TypeError("images must be a uint8 NumPy array")
    if images.ndim != 4 or images.shape[1:] != (128, 128, 3):
        raise ValueError("images must have shape (N, 128, 128, 3)")
    model.eval()
    device = next(model.parameters()).device
    results = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + 32]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            batch = (batch / 255.0 - 0.5) / 0.5
            results.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(results).astype(np.int64) if results else np.empty(0, dtype=np.int64)


def read_classes(path):
    entries = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if entries != [[str(i), name] for i, name in enumerate(CLASS_NAMES)]:
        raise ValueError(f"Unexpected official class mapping in {path}")
    return list(CLASS_NAMES)


class DogDataset(torch.utils.data.Dataset):
    """Enumerate folders in official numeric order, never alphabetical order."""

    def __init__(self, root, transform, per_class=None, seed=42):
        self.transform = transform
        self.samples = []
        rng = random.Random(seed)
        root = Path(root)
        actual = {p.name for p in root.iterdir() if p.is_dir()}
        if actual != set(CLASS_NAMES):
            raise ValueError(f"Unexpected breed folders in {root}: {actual}")
        for label, name in enumerate(CLASS_NAMES):
            files = sorted((root / name).glob("*.png"))
            if not files:
                raise ValueError(f"No images found for {name}")
            if per_class is not None:
                files = rng.sample(files, min(per_class, len(files)))
            self.samples.extend((path, label) for path in files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (128, 128):
                raise ValueError(f"Expected 128 x 128 image: {path}")
            return self.transform(image), label


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def choose_device(requested):
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable to this Python process. Run train_mps.command "
            "from macOS Terminal, outside the restricted agent shell. "
            "No CPU training has been started."
        )
    if device.type != "cpu":
        # Availability alone is insufficient: confirm a real accelerator operation.
        probe = torch.ones((2, 2), device=device)
        result = (probe @ probe).cpu()
        if not torch.equal(result, torch.full((2, 2), 2.0)):
            raise RuntimeError(f"Accelerator operation failed on {device}")
    return device


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = correct = total = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            predicted = logits.detach().argmax(1)
            total_loss += loss.item() * len(labels)
            correct += (predicted == labels).sum().item()
            total += len(labels)
            confusion += torch.bincount(
                (labels * 10 + predicted).cpu(), minlength=100,
            ).reshape(10, 10)
    return {"loss": total_loss / total, "accuracy": correct / total,
            "confusion_matrix": confusion.tolist()}


def check_public_interface(model, dataset, expected_accuracy):
    """Use raw validation pixels to check the exact grader-facing path."""
    from PIL import Image
    correct = 0
    for start in range(0, len(dataset), 32):
        samples = dataset.samples[start:start + 32]
        raw = []
        for path, _ in samples:
            with Image.open(path) as image:
                raw.append(np.array(image.convert("RGB")))
        predictions = predict(model, np.stack(raw))
        labels = np.array([label for _, label in samples])
        correct += int((predictions == labels).sum())
    accuracy = correct / len(dataset)
    if abs(accuracy - expected_accuracy) > 1e-9:
        raise RuntimeError(f"Public accuracy {accuracy} differs from validation {expected_accuracy}")
    return accuracy


def train(args):
    # Training-only dependency; importing for grading needs no torchvision or data.
    from torchvision import transforms

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.workers < 0:
        raise ValueError("epochs, batch-size, and patience must be positive; workers nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if device.type == "cpu" and args.device == "auto" and not args.smoke:
        raise RuntimeError(
            "No accelerator is accessible. Refusing to silently start full CPU training. "
            "Run train_mps.command from macOS Terminal for MPS, or explicitly "
            "select --device cpu if CPU training is intended."
        )
    read_classes(args.data_root / "imagewoof_train/classes.txt")
    read_classes(args.data_root / "imagewoof_val/classes.txt")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12),
                                 ratio=(0.5, 2.0), value=0),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    train_data = DogDataset(args.data_root / "imagewoof_train/train", train_transform,
                            16 if args.smoke else None, args.seed)
    val_data = DogDataset(args.data_root / "imagewoof_val/val", val_transform,
                          8 if args.smoke else None, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,
                                               generator=generator, **loader_options)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **loader_options)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(device=str(device), torch_version=str(torch.__version__),
                  train_images=len(train_data), validation_images=len(val_data),
                  architecture=args.architecture, class_names=list(CLASS_NAMES),
                  mean=list(MEAN), std=list(STD), split="provided split",
                  augmentation="random resized crop scale 0.85–1.0, horizontal flip, color jitter 0.15, random erasing p=0.25 scale=0.02–0.12 ratio=0.5–2.0 value=0 after normalization",
                  optimizer="AdamW", scheduler="cosine", label_smoothing=0.1,
                  determinism="fixed seeds; deterministic algorithms warn on unsupported operations")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "split.json").write_text(json.dumps({
        "train": [str(p.relative_to(args.data_root)) for p, _ in train_data.samples],
        "validation": [str(p.relative_to(args.data_root)) for p, _ in val_data.samples],
    }, indent=2) + "\n")

    model = (load_model(args.init_checkpoint) if args.init_checkpoint is not None
             else DogCNN(args.architecture))
    if model.architecture != args.architecture:
        raise ValueError("--architecture must match the initialization checkpoint")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = min(args.epochs, 2) if args.smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy, best_epoch = -1.0, 0
    best_validation = None
    def save_best(epoch):
        torch.save({"architecture": args.architecture, "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }, "class_names": list(CLASS_NAMES), "mean": list(MEAN), "std": list(STD),
            "seed": args.seed, "epoch": epoch, "validation_accuracy": best_accuracy,
            "smoke_run": args.smoke,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        }, output / "model.pth")

    if args.init_checkpoint is not None:
        best_validation = run_epoch(model, val_loader, criterion, device)
        best_accuracy = best_validation["accuracy"]
        save_best(0)
        (output / "initial_validation.json").write_text(
            json.dumps(best_validation, indent=2) + "\n")
        print(f"Initialized from {args.init_checkpoint}; validation: {best_accuracy:.2%}; "
              "fresh AdamW optimizer and cosine schedule", flush=True)
    history = []
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"train: {len(train_data)}; validation: {len(val_data)}; smoke: {args.smoke}", flush=True)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training = run_epoch(model, train_loader, criterion, device, optimizer)
        validation = run_epoch(model, val_loader, criterion, device)
        row = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train=training,
                   validation=validation, seconds=time.perf_counter() - started)
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"Epoch {epoch}/{epochs}: loss={training['loss']:.4f}, "
              f"train={training['accuracy']:.2%}, val={validation['accuracy']:.2%}, "
              f"{row['seconds']:.1f}s", flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            best_validation = validation
            save_best(epoch)
        scheduler.step()
        if epoch - best_epoch >= args.patience:
            break

    restored = load_model(output / "model.pth").to(device)
    public_accuracy = check_public_interface(restored, val_data, best_accuracy)
    best_confusion = np.array(best_validation["confusion_matrix"])
    report = dict(smoke_run=args.smoke, best_epoch=best_epoch,
                  validation_images=len(val_data), validation_accuracy=best_accuracy,
                  public_predict_accuracy=public_accuracy,
                  per_class_accuracy={name: float(best_confusion[i, i] / best_confusion[i].sum())
                                      for i, name in enumerate(CLASS_NAMES)})
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Checkpoint reloaded; public predict accuracy: {public_accuracy:.2%}. "
          f"Artifacts: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True, help="New run directory; must not exist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), default="DogCNN-v1",
                        help="v1: six convolutions; v2: eight convolutions with a fourth pooling stage")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Continue from saved weights with a fresh optimizer and schedule; "
                             "retain initial validation as the score to beat")
    parser.add_argument("--smoke", action="store_true", help="At most 2 epochs; 160 train / 80 val images")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
