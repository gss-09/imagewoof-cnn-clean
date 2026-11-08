"""Check class labels, preprocessing, and portable checkpoint inference."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from solution import CLASS_NAMES, MEAN, STD, DogDataset, load_model, predict, read_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    read_classes(root / "imagewoof_train/classes.txt")
    read_classes(root / "imagewoof_val/classes.txt")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = DogDataset(root / "imagewoof_train/train", transform)
    val = DogDataset(root / "imagewoof_val/val", transform)
    assert len(train) == 9065 and len(val) == 1295
    assert not ({p for p, _ in train.samples} & {p for p, _ in val.samples})
    for dataset in (train, val):
        for path, label in dataset.samples:
            assert CLASS_NAMES[label] == path.parent.name

    # Include every breed and cross the public function's internal batch boundary.
    chosen = []
    for label in range(10):
        chosen.extend([(p, y) for p, y in val.samples if y == label][:4])
    raw = []
    for path, _ in chosen:
        with Image.open(path) as image:
            raw.append(np.array(image.convert("RGB")))
    images = np.stack(raw)
    model = load_model(args.checkpoint)
    assert not model.training
    actual = predict(model, images)
    assert actual.shape == (40,) and actual.dtype == np.int64
    assert np.all((actual >= 0) & (actual < 10))
    assert np.array_equal(actual, predict(model, images))
    assert np.array_equal(actual, np.concatenate([predict(model, x[None]) for x in images]))
    assert predict(model, images[:0]).shape == (0,)
    with torch.inference_mode():
        training_path = model(torch.stack([transform(Image.fromarray(x)) for x in images]))
    assert np.array_equal(actual, training_path.argmax(1).numpy())
    for bad in (images.astype(np.float32), images.transpose(0, 3, 1, 2)):
        try:
            predict(model, bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Malformed input was accepted")

    # A new process gets only the submission code, weights, and raw inputs.
    with tempfile.TemporaryDirectory() as directory:
        stage = Path(directory)
        shutil.copy2(root / "solution.py", stage / "solution.py")
        shutil.copy2(args.checkpoint, stage / "model.pth")
        np.save(stage / "inputs.npy", images)
        subprocess.run([
            sys.executable, "-c",
            "import numpy as np, torch; torch.set_num_threads(4); "
            "from solution import load_model, predict; "
            "np.save('outputs.npy', predict(load_model(), np.load('inputs.npy')))",
        ], cwd=stage, check=True, timeout=120)
        assert np.array_equal(actual, np.load(stage / "outputs.npy"))
    print(json.dumps({
        "status": "passed", "training_images": len(train), "validation_images": len(val),
        "checked": ["official labels for all files", "raw-pixel preprocessing parity",
                    "repeatability", "single and multiple inference batches", "empty batch",
                    "input validation", "isolated fresh-process CPU checkpoint loading"],
        "device": "cpu",
    }, indent=2))


if __name__ == "__main__":
    main()
