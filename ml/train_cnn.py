"""A1/A2 - Ajuste fino de CNN pre-treinada (ResNet18; EfficientNet-B0 opcional) como classificador de pixels (real x FaceSwap, FF++ c23).

Protocolo (igual ao baseline do Marco 1): classe real = originais do FF++, fake = FaceSwap; splits oficiais
por video (coluna `split` do manifest). CelebA NAO entra no treino. Variaveis da mascara nao sao usadas.

Uso:
  python ml/train_cnn.py --smoke                 # 2 batches, valida o pipeline
  python ml/train_cnn.py --bench                 # mede img/s de treino
  python ml/train_cnn.py --epochs 6 --fpv 5      # treino completo (retoma sozinho do ultimo checkpoint)
Saidas em derived/cnn/<run>/: last.pt, best.pt (melhor AUC de validacao), log.csv, config.json
"""
import argparse, io, json, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

PI = os.environ.get("PI_DIR", "/Users/iancarlos/Repositories/Barao/8Semestre/PI")
D = f"{PI}/derived"
torch.multiprocessing.set_sharing_strategy("file_system")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_manifest(groups=("ffpp_real", "ffpp_faceswap")):
    m = pd.read_csv(f"{D}/manifest.csv", low_memory=False)
    m = m[m.group.isin(groups) & m.crop_file.notna() & (m.detected == 1)].reset_index(drop=True)
    m["video_key"] = m.group + "/" + m.video_id.astype(str)
    return m


class Crops(Dataset):
    def __init__(self, df, size=224, train=False):
        self.df, self.size, self.train = df.reset_index(drop=True), size, train
        self.norm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(MEAN, STD)])

    def aug(self, im):
        # reduz dependencia de atalhos de compressao: JPEG aleatorio, desfoque, down/up-scaling, flip
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            s = random.choice([0.5, 0.65, 0.8])
            w = im.width
            im = im.resize((int(w * s),) * 2, Image.BILINEAR).resize((w, w), Image.BILINEAR)
        if random.random() < 0.3:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
        if random.random() < 0.5:
            b = io.BytesIO()
            im.save(b, "JPEG", quality=random.randint(50, 95))
            b.seek(0)
            im = Image.open(b).convert("RGB")
        return im

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(f"{D}/{r.crop_file}").convert("RGB")
        if self.train:
            im = self.aug(im)
        im = im.resize((self.size, self.size), Image.BILINEAR)
        return self.norm(im), torch.tensor(float(r.label))


def build_model(arch="resnet18", pretrained=True):
    """resnet18 e o padrao: em CPU (sem GPU/MPS) o EfficientNet-B0 treina a ~2 img/s (convolucao depthwise),
    contra ~17 img/s da ResNet18, que alem disso e a mesma rede do linear probe do Marco 1 (comparacao direta)."""
    if arch == "resnet18":
        m = torchvision.models.resnet18()
        if pretrained:
            m.load_state_dict(torch.load(f"{D}/weights/resnet18-f37072fd.pth", map_location="cpu"))
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))
    elif arch == "effb0":
        m = torchvision.models.efficientnet_b0()
        if pretrained:
            m.load_state_dict(torch.load(f"{D}/weights/efficientnet_b0_rwightman-7f5810bc.pth", map_location="cpu"))
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 1))
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def predict(model, df, size=224, bs=64, workers=0):
    model.eval()
    dl = DataLoader(Crops(df, size, False), batch_size=bs, num_workers=workers)
    out = []
    for x, _ in dl:
        out.append(model(x).squeeze(1).float().numpy())
    return np.concatenate(out)


def sample_epoch(tr, fpv, rng):
    if fpv <= 0:
        return tr
    return tr.groupby("video_key", group_keys=False).apply(
        lambda g: g.sample(min(fpv, len(g)), random_state=int(rng.integers(1 << 31)))).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "effb0"])
    ap.add_argument("--run", default="resnet18_ft_v1")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--fpv", type=int, default=5, help="frames por video por epoca (0 = todos os 10)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0, help="0: workers com shm falham neste ambiente")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    m = load_manifest()
    tr, va = m[m.split == "train"], m[m.split == "val"]
    # os splits oficiais nao compartilham videos
    assert not set(tr.video_key) & set(va.video_key) and not set(tr.video_key) & set(m[m.split == "test"].video_key)
    model = build_model(a.arch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(a.seed)

    if a.smoke or a.bench:
        n = 2 if a.smoke else 6
        dl = DataLoader(Crops(tr.sample(a.bs * n, random_state=0), a.size, True), batch_size=a.bs, shuffle=True,
                        num_workers=a.workers)
        model.train(); t0 = time.time(); seen = 0
        for b, (x, y) in enumerate(dl):
            loss = lossf(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
            seen += len(x)
            print(f"batch {b} loss={loss.item():.4f} {seen / (time.time() - t0):.1f} img/s", flush=True)
        return

    out = f"{D}/cnn/{a.run}"; os.makedirs(out, exist_ok=True)
    json.dump(vars(a) | {"n_train_frames": len(tr), "n_val_frames": len(va)}, open(f"{out}/config.json", "w"), indent=1)
    steps_per_epoch = int(np.ceil(len(sample_epoch(tr, a.fpv, np.random.default_rng(0))) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch + 5,
                                                pct_start=0.15)
    start_ep, best, logrows = 0, -1.0, []
    if os.path.exists(f"{out}/last.pt"):  # retoma
        ck = torch.load(f"{out}/last.pt", map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, best, logrows = ck["epoch"] + 1, ck["best"], ck["log"]
        print(f"retomando da epoca {start_ep}", flush=True)
    for ep in range(start_ep, a.epochs):
        rng = np.random.default_rng(a.seed + ep)
        ep_df = sample_epoch(tr, a.fpv, rng)
        dl = DataLoader(Crops(ep_df, a.size, True), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                        persistent_workers=False)
        model.train(); t0 = time.time(); run_loss, nb = 0.0, 0
        for x, y in dl:
            loss = lossf(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run_loss += loss.item(); nb += 1
            if nb % 25 == 0:
                print(f"ep{ep} {nb}/{len(dl)} loss={run_loss / nb:.4f} {nb * a.bs / (time.time() - t0):.1f} img/s", flush=True)
        t_tr = time.time() - t0
        logit = predict(model, va, a.size, workers=a.workers)
        y = va.label.values
        auc = roc_auc_score(y, logit)
        vloss = float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(logit), torch.tensor(y, dtype=torch.float32)))
        acc = float(((logit > 0) == y).mean())
        logrows.append(dict(epoch=ep, train_loss=run_loss / max(nb, 1), val_loss=vloss, val_auc=auc, val_acc=acc,
                            train_sec=t_tr, total_sec=time.time() - t0))
        print(logrows[-1], flush=True)
        if auc > best:
            best = auc; torch.save({"model": model.state_dict(), "arch": a.arch, "epoch": ep, "val_auc": auc}, f"{out}/best.pt")
        torch.save({"model": model.state_dict(), "arch": a.arch, "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "log": logrows}, f"{out}/last.pt")
        pd.DataFrame(logrows).to_csv(f"{out}/log.csv", index=False)
    print("fim; melhor val_auc =", best, flush=True)


if __name__ == "__main__":
    main()
