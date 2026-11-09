"""COSC 4368 HW2: a from-scratch sequential CNN for the ten Imagewoof dog breeds.

Grading interface (needs only torch and numpy):
    model = load_model("model.pth")
    labels = predict(model, images)   # images: uint8 array of shape (N, 128, 128, 3)

Training (reproduces the submitted model; needs torchvision, Pillow and the supplied data):
    python solution.py --output runs/my_run

Recipe of the submitted model: supplied train + validation images pooled and re-split
99/1 per breed with seed 42 (10,255 train / 105 validation), 150 epochs, SGD with
Nesterov momentum, warm-up then cosine learning rate, label smoothing, and an
exponential moving average (EMA) of the weights. The saved checkpoint is the EMA
weights from the epoch with the highest validation accuracy (epoch 113, 94/105).
"""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


# Official label order from classes.txt: output ID i means CLASS_NAMES[i].
CLASS_NAMES = (
    "shih_tzu", "rhodesian_ridgeback", "beagle", "english_foxhound",
    "border_terrier", "australian_terrier", "golden_retriever",
    "old_english_sheepdog", "samoyed", "dingo",
)
# Pixels are scaled from [0, 255] to [-1, 1]; no dataset statistics are used.
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
ARCHITECTURE = "DogCNN-v4"
BLOCK_CHANNELS = (64, 128, 256, 512, 512)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class DogCNN(nn.Module):
    """Plain sequential CNN: five blocks of (conv-BN-ReLU) x 2 + max-pool, then classifier.

    Input 3 x 128 x 128 -> after five pools 512 x 4 x 4 -> global average pool -> 10 scores.
    10 convolutions, 9,413,066 parameters. No skip connections.
    """

    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for channels in BLOCK_CHANNELS:
            for _ in range(2):
                layers += [
                    nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ]
                in_channels = channels
            layers.append(nn.MaxPool2d(2))  # halves height and width
        layers += [
            nn.AdaptiveAvgPool2d(1),  # global average pooling
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(BLOCK_CHANNELS[-1], len(CLASS_NAMES)),
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        return self.network(images)


# ----------------------------------------------------------------------------
# Grading interface
# ----------------------------------------------------------------------------

def load_model(path="model.pth"):
    """Build the network, load the trained weights on CPU, and return it in eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError(f"Checkpoint is {checkpoint['architecture']}, expected {ARCHITECTURE}")
    if checkpoint["class_names"] != list(CLASS_NAMES):
        raise ValueError("Checkpoint class order does not match classes.txt")
    model = DogCNN()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def predict(model, images):
    """Classify raw RGB images.

    images: NumPy uint8 array of shape (N, 128, 128, 3) with values 0-255.
    Returns a NumPy int64 array of shape (N,) with class IDs 0-9 (classes.txt order).
    """
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8:
        raise TypeError("images must be a uint8 NumPy array")
    if images.ndim != 4 or images.shape[1:] != (128, 128, 3):
        raise ValueError("images must have shape (N, 128, 128, 3)")
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):  # small batches keep memory bounded
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + 32]))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)  # NHWC -> NCHW
            batch = (batch / 255.0 - 0.5) / 0.5  # same scaling as training
            predictions.append(model(batch).argmax(1).cpu().numpy())
    if not predictions:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(predictions).astype(np.int64)


# ----------------------------------------------------------------------------
# Training (only runs from the command line)
# ----------------------------------------------------------------------------

class DogDataset(torch.utils.data.Dataset):
    """(path, label) pairs read in official class order, never alphabetical folder order."""

    def __init__(self, root, transform):
        self.transform = transform
        self.samples = []
        for label, name in enumerate(CLASS_NAMES):
            files = sorted((Path(root) / name).glob("*.png"))
            if not files:
                raise ValueError(f"No images found in {Path(root) / name}")
            self.samples += [(path, label) for path in files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def resplit(samples, fraction, seed):
    """Pool images and hold out `fraction` of each breed (seeded) as the new validation set."""
    rng = random.Random(seed)
    train, validation = [], []
    for label in range(len(CLASS_NAMES)):
        items = sorted(sample for sample in samples if sample[1] == label)
        rng.shuffle(items)
        held_out = round(len(items) * fraction)
        validation += items[:held_out]
        train += items[held_out:]
    return sorted(train), sorted(validation)


def seed_worker(worker_id):
    """Give each data-loading worker its own reproducible random state."""
    del worker_id  # required by DataLoader, not used
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class WeightAverage:
    """Exponential moving average (EMA) of the weights: a smoothed copy of the same CNN."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """average = decay * average + (1 - decay) * current, after every optimizer step."""
        for average, current in zip(self.model.state_dict().values(), model.state_dict().values()):
            if average.dtype.is_floating_point:
                average.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                average.copy_(current)  # integer counters in BatchNorm


def run_epoch(model, loader, criterion, device, optimizer=None, after_step=None):
    """One pass over `loader`. Trains when an optimizer is given; otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)  # rows: true breed, columns: predicted
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
                after_step()
            predicted = logits.detach().argmax(1)
            total_loss += loss.item() * len(labels)
            correct += (predicted == labels).sum().item()
            total += len(labels)
            confusion += torch.bincount((labels * 10 + predicted).cpu(), minlength=100).reshape(10, 10)
    return {"loss": total_loss / total, "accuracy": correct / total,
            "confusion_matrix": confusion.tolist()}


def public_accuracy(model, dataset):
    """Accuracy of the saved model through predict() on raw validation pixels."""
    from PIL import Image
    images = []
    for path, _ in dataset.samples:
        with Image.open(path) as image:
            images.append(np.array(image.convert("RGB")))
    labels = np.array([label for _, label in dataset.samples])
    return float((predict(model, np.stack(images)) == labels).mean())


def train(args):
    from torchvision import transforms

    # Fixed seeds for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    # Augmentation for training images only; validation images are just scaled.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.35, 1.0)),  # crop 35-100% of the area, resize to 128
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    # Pool the supplied train and validation folders, then re-split per breed.
    train_data = DogDataset(args.data_root / "imagewoof_train/train", train_transform)
    val_data = DogDataset(args.data_root / "imagewoof_val/val", val_transform)
    train_data.samples, val_data.samples = resplit(
        train_data.samples + val_data.samples, args.val_fraction, args.seed)

    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_options)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **loader_options)
    (output / "config.json").write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        | {"device": str(device), "train_images": len(train_data), "validation_images": len(val_data)},
        indent=2) + "\n")
    (output / "split.json").write_text(json.dumps({
        "train": [str(p.relative_to(args.data_root)) for p, _ in train_data.samples],
        "validation": [str(p.relative_to(args.data_root)) for p, _ in val_data.samples],
    }, indent=2) + "\n")

    model = DogCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.weight_decay)
    # Learning rate: linear warm-up for the first epochs, then cosine decay to zero.
    warmup, epochs = args.warmup_epochs, args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (
        (e + 1) / warmup if e < warmup else
        0.5 * (1 + math.cos(math.pi * (e - warmup) / (epochs - warmup)))))
    average = WeightAverage(model, args.ema_decay)

    best_accuracy, best_epoch, best_validation, history = -1.0, 0, None, []
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"train: {len(train_data)}; validation: {len(val_data)}", flush=True)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training = run_epoch(model, train_loader, criterion, device, optimizer,
                             lambda: average.update(model))
        raw = run_epoch(model, val_loader, criterion, device)                 # current weights
        validation = run_epoch(average.model, val_loader, criterion, device)  # EMA weights (selected)
        seconds = time.perf_counter() - started
        history.append(dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train=training,
                            validation=validation, validation_raw=raw, seconds=seconds))
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"Epoch {epoch}/{epochs}: loss={training['loss']:.4f}, train={training['accuracy']:.2%}, "
              f"val={validation['accuracy']:.2%}, raw={raw['accuracy']:.2%}, "
              f"{seconds:.1f}s", flush=True)
        # Keep the EMA weights from the epoch with the highest validation accuracy.
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch, best_validation = validation["accuracy"], epoch, validation
            torch.save({"architecture": ARCHITECTURE,
                        "state_dict": {k: v.detach().cpu() for k, v in average.model.state_dict().items()},
                        "class_names": list(CLASS_NAMES), "mean": list(MEAN), "std": list(STD),
                        "seed": args.seed, "epoch": epoch, "validation_accuracy": best_accuracy},
                       output / "model.pth")
        scheduler.step()

    # Reload the saved checkpoint and confirm predict() reproduces its validation accuracy.
    reloaded = public_accuracy(load_model(output / "model.pth").to(device), val_data)
    confusion = np.array(best_validation["confusion_matrix"])
    summary = dict(best_epoch=best_epoch, validation_images=len(val_data),
                   validation_accuracy=best_accuracy, public_predict_accuracy=reloaded,
                   per_class_accuracy={name: float(confusion[i, i] / confusion[i].sum())
                                       for i, name in enumerate(CLASS_NAMES)})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Best epoch {best_epoch}: {best_accuracy:.2%}; reloaded predict(): {reloaded:.2%}. "
          f"Saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train the DogCNN-v4 submission model.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder containing imagewoof_train/ and imagewoof_val/")
    parser.add_argument("--output", type=Path, required=True, help="New run folder (must not exist)")
    parser.add_argument("--val-fraction", type=float, default=0.01, help="Held-out share of each breed")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6, help="Data-loading processes")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
