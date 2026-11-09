"""Final grader-style checks, run from a folder containing only the unzipped submission. Args: project_root"""
import sys, json, time
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, ".")
ok = lambda name, cond, extra="": print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
t = time.time(); import solution; t = time.time() - t
ok("import is fast and starts no training", t < 10, f"{t:.1f}s")
ok("import needs no PIL/torchvision", not any(m in sys.modules for m in ("PIL", "torchvision")))
import numpy as np, torch
from torch import nn
from PIL import Image
m = solution.load_model()  # PDF default path "model.pth"
ok("load_model() with default path, CPU", next(m.parameters()).device.type == "cpu")
ok("returned in eval mode", not m.training and all(not x.training for x in m.modules()))
allowed = (nn.Sequential, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Dropout, nn.Linear, solution.DogCNN)
ok("only allowed layer types", all(isinstance(x, allowed) for x in m.modules()), sorted({type(x).__name__ for x in m.modules()}))
ok("10 conv layers, 9,413,066 params", sum(isinstance(x, nn.Conv2d) for x in m.modules()) == 10 and sum(p.numel() for p in m.parameters()) == 9413066)
ckpt = torch.load("model.pth", map_location="cpu", weights_only=True)
fresh = solution.DogCNN(); missing = fresh.load_state_dict(ckpt["state_dict"], strict=True)
ok("architecture matches weights exactly (strict load)", not missing.missing_keys and not missing.unexpected_keys)
ok("checkpoint records seed 42, epoch 113", ckpt["seed"] == 42 and ckpt["epoch"] == 113, f"val {ckpt['validation_accuracy']:.4f}")
for f in ("imagewoof_train/classes.txt", "imagewoof_val/classes.txt"):
    ok(f"class order matches {f}", [l.split() for l in open(root / f) if l.strip()] == [[str(i), n] for i, n in enumerate(solution.CLASS_NAMES)])
files = json.load(open(sys.argv[2] if len(sys.argv) > 2 else root / "saved_models/v4_resplit9901_8952/split.json"))["validation"]
x = np.stack([np.array(Image.open(root / f).convert("RGB")) for f in files])
y = np.array([solution.CLASS_NAMES.index(Path(f).parent.name) for f in files])
ok("validation input is (N,128,128,3) uint8 RGB", x.shape == (105, 128, 128, 3) and x.dtype == np.uint8)
p = solution.predict(m, x)
ok("output is int numpy array, shape (N,), ids 0-9", isinstance(p, np.ndarray) and p.shape == (105,) and np.issubdtype(p.dtype, np.integer) and p.min() >= 0 and p.max() <= 9, str(p.dtype))
ok("reproduces validation accuracy 94/105 = 89.52%", int((p == y).sum()) == 94, f"{int((p == y).sum())}/105")
ok("deterministic on repeat", bool((solution.predict(m, x) == p).all()))
ok("batch size does not change predictions", bool((np.concatenate([solution.predict(m, x[i:i + 7]) for i in range(0, 105, 7)]) == p).all()))
ok("single image works", solution.predict(m, x[:1]).shape == (1,))
ok("empty batch works", solution.predict(m, x[:0]).shape == (0,))
ok("input array not modified", bool((np.stack([np.array(Image.open(root / f).convert("RGB")) for f in files[:5]]) == x[:5]).all()))
big = np.stack([np.array(Image.open(f).convert("RGB")) for f in sorted((root / "imagewoof_val/val").glob("*/*.png"))])
t = time.time(); q = solution.predict(m, big); t = time.time() - t
ok("runs on 1,295 images on CPU", q.shape == (1295,), f"{t:.1f}s; ~{t * 2591 / 1295:.0f}s for the 2,591 hidden images")
if torch.cuda.is_available():
    g = solution.predict(m.cuda(), x)
    ok("accelerator: same predictions after model.cuda()", bool((g == p).all()), torch.cuda.get_device_name(0))
    g2 = solution.predict(solution.load_model("model.pth").cuda(), x)
    ok("accelerator: fresh load_model + cuda gives 94/105", int((g2 == y).sum()) == 94)
else:
    print("INFO no CUDA on this machine (accelerator checked separately)")
