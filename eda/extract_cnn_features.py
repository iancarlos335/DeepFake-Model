"""Frozen ImageNet ResNet-18 embeddings (512-d, global-average-pooled) for every face crop
listed in derived/manifest.csv. Output: derived/resnet18_feats.npz with `feats` aligned to
manifest rows (NaN for rows without a crop)."""
import os, time
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PI = "/Users/iancarlos/Repositories/Barao/8Semestre/PI"
D = f"{PI}/derived"
torch.manual_seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(8)

man = pd.read_csv(f"{D}/manifest.csv", low_memory=False)
idx = np.where(man.crop_file.notna())[0]
tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Crops(Dataset):
    def __len__(self):
        return len(idx)

    def __getitem__(self, k):
        return tf(Image.open(f"{D}/{man.crop_file.iloc[idx[k]]}").convert("RGB"))


m = torchvision.models.resnet18()
m.load_state_dict(torch.load(f"{D}/weights/resnet18-f37072fd.pth", map_location="cpu"))
m.fc = torch.nn.Identity()
m.eval().to(dev)
out = np.full((len(man), 512), np.nan, np.float32)
dl = DataLoader(Crops(), batch_size=128, num_workers=0)
t0, pos = time.time(), 0
with torch.no_grad():
    for b, x in enumerate(dl):
        f = m(x.to(dev)).cpu().numpy()
        out[idx[pos:pos + len(f)]] = f
        pos += len(f)
        if b % 40 == 0:
            print(f"{pos}/{len(idx)} {time.time() - t0:.0f}s dev={dev}", flush=True)
np.savez_compressed(f"{D}/resnet18_feats.npz", feats=out)
print("done", pos, round(time.time() - t0), flush=True)
