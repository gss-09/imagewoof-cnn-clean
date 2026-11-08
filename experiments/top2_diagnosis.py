import sys, collections, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, "saved_models/v4_sgd_wd001_8672")
import solution as S
model = S.load_model("saved_models/v4_sgd_wd001_8672/model.pth")
imgs, labels = [], []
for lab, name in enumerate(S.CLASS_NAMES):
    for p in sorted(Path("imagewoof_val/val", name).glob("*.png")):
        imgs.append(np.array(Image.open(p).convert("RGB"))); labels.append(lab)
x = np.stack(imgs); y = np.array(labels)
logits = []
with torch.inference_mode():
    for i in range(0, len(x), 64):
        b = torch.from_numpy(x[i:i+64]).permute(0,3,1,2).float(); b = (b/255-0.5)/0.5
        logits.append(model(b))
logits = torch.cat(logits); top = logits.topk(2, 1).indices.numpy()
assert (top[:,0] == S.predict(model, x)).all()
t1 = top[:,0] == y; t2 = t1 | (top[:,1] == y)
print(f"top-1 {t1.mean()*100:.2f}% ({t1.sum()}/{len(y)})  top-2 {t2.mean()*100:.2f}% ({t2.sum()}/{len(y)})")
print(f"of {(~t1).sum()} wrong, {(t2 & ~t1).sum()} had the right breed as 2nd guess")
for c, n in enumerate(S.CLASS_NAMES):
    m = y == c; print(f"  {n:22s} top1 {t1[m].mean()*100:5.1f}%  top2 {t2[m].mean()*100:5.1f}%")
pairs = collections.Counter(tuple(sorted((S.CLASS_NAMES[a], S.CLASS_NAMES[b]))) for a, b in zip(y[~t1], top[~t1,0]))
print("most confused pairs (both directions):")
for (a, b), k in pairs.most_common(6): print(f"  {a} <-> {b}: {k}")
