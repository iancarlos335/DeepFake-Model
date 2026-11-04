"""A3 - Temperature scaling na validacao + exportacao de derived/agent/preds_cnn.parquet (contrato 5.1).

  python ml/calibrate_export.py --run resnet18_ft_v1 [--ckpt best]

Saidas:
  derived/agent/preds_cnn.parquet   crop_file, video_id, split, label, p_cnn_raw, p_cnn_cal, logit, model_id
  derived/cnn/<run>/calibration.json  temperatura T e ECE/Brier/NLL antes e depois (validacao e teste)
T e ajustado SO na validacao (1 parametro, minimiza NLL); o teste nao e tocado.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import train_cnn as T

D = T.D


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="resnet18_ft_v1")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ck = torch.load(f"{D}/cnn/{a.run}/{a.ckpt}.pt", map_location="cpu")
    arch = ck.get("arch", "resnet18")
    model = T.build_model(arch, pretrained=False)
    model.load_state_dict(ck["model"])

    m = pd.read_csv(f"{D}/manifest.csv", low_memory=False, dtype={"video_id": str, "target_id": str, "source_id": str})
    m = m[m.group.isin(["ffpp_real", "ffpp_faceswap"]) & m.crop_file.notna() & (m.detected == 1)].reset_index(drop=True)
    t0 = time.time()
    m["logit"] = T.predict(model, m, 224, bs=64, workers=0)
    print(f"inferencia {len(m)} imagens em {time.time() - t0:.0f}s", flush=True)

    va, te = m[m.split == "val"], m[m.split == "test"]
    nll = lambda t: log_loss(va.label, sigmoid(va.logit.values / t), labels=[0, 1])
    Topt = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    m["p_cnn_raw"] = sigmoid(m.logit.values)
    m["p_cnn_cal"] = sigmoid(m.logit.values / Topt)
    m["model_id"] = f"{a.run}_{a.ckpt}"
    cal = {"run": a.run, "ckpt": a.ckpt, "epoch_ckpt": int(ck["epoch"]), "temperature": Topt}
    for name, d in (("val", m[m.split == "val"]), ("test", m[m.split == "test"]), ("train", m[m.split == "train"])):
        y = d.label.values
        for col in ("p_cnn_raw", "p_cnn_cal"):
            cal[f"{name}_{col}"] = dict(nll=float(log_loss(y, d[col], labels=[0, 1])), brier=float(brier_score_loss(y, d[col])),
                                        ece=ece(y, d[col].values), auc=float(roc_auc_score(y, d[col])),
                                        acc=float(((d[col] >= .5) == y).mean()))
    os.makedirs(f"{D}/agent", exist_ok=True)
    out = m[["crop_file", "video_id", "split", "label", "p_cnn_raw", "p_cnn_cal", "logit", "model_id"]]
    out.to_parquet(f"{D}/agent/preds_cnn.parquet", index=False)
    json.dump(cal, open(f"{D}/cnn/{a.run}/calibration.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in cal.items() if k in ("temperature", "epoch_ckpt")}))
    for k, v in cal.items():
        if isinstance(v, dict):
            print(k, {x: round(y, 4) for x, y in v.items()})


if __name__ == "__main__":
    main()
