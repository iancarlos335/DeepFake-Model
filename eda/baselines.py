"""Baseline classifiers for Marco 1 (section 4).

Model: L2-regularised logistic regression (scikit-learn) on standardised features.
Protocol: official FaceForensics++ video splits (720/140/140); C chosen on validation AUC;
threshold fixed at 0.5 (classes balanced); test metrics at frame level and at video level
(mean probability over the video's 10 frames); 95% CIs by cluster bootstrap over target
videos (real A and fake A_B share a cluster), 2,000 resamples, seed 0.
Mask-derived variables are excluded: masks are not available at inference time.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss, confusion_matrix)

PI = "/Users/iancarlos/Repositories/Barao/8Semestre/PI"
D = f"{PI}/derived"
HAND = ["sharpness", "hf_ratio", "blockiness", "brightness", "saturation", "det_prob", "face_px", "eye_dist_rel"]
LOGF = {"sharpness", "hf_ratio", "face_px"}
CS = [1e-3, 1e-2, 1e-1, 1, 10, 100]


def load():
    man = pd.read_csv(f"{D}/manifest.csv", low_memory=False)
    spec = np.load(f"{D}/manifest_spectra.npz")["spectra"]
    man["cluster"] = np.where(man.source == "CelebA", man.video_id, man.target_id)
    return man, spec


def hand_matrix(df):
    X = df[HAND].copy()
    for c in LOGF:
        X[c] = np.log10(X[c])
    return X.values


def spec_matrix(spec, rows):
    return np.log10(spec[rows] + 1e-6)


def fit(Xtr, ytr, Xva, yva):
    best = None
    for C in CS:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        m.fit(Xtr, ytr)
        a = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or a > best[0]:
            best = (a, C, m)
    return best  # (val_auc, C, model)


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, yh), prec=precision_score(y, yh, zero_division=0),
                rec=recall_score(y, yh), f1=f1_score(y, yh), auc=roc_auc_score(y, p),
                brier=brier_score_loss(y, p), tn=tn, fp=fp, fn=fn, tp=tp)


def video_level(df, p):
    v = df.assign(p=p).groupby(["group", "video_id"]).agg(y=("label", "first"), p=("p", "mean"), cluster=("cluster", "first")).reset_index()
    return v


def cluster_ci(df, p, level="frame", B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = df.assign(p=p)
    if level == "video":
        d = video_level(df, p)
        d = d.rename(columns={"y": "label"})
    groups = {c: g for c, g in d.groupby("cluster")}
    keys = np.array(list(groups))
    acc, auc, f1 = [], [], []
    for _ in range(B):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
        if s.label.nunique() < 2:
            continue
        yh = (s.p >= 0.5).astype(int)
        acc.append(accuracy_score(s.label, yh)); auc.append(roc_auc_score(s.label, s.p)); f1.append(f1_score(s.label, yh))
    q = lambda a: tuple(np.round(np.percentile(a, [2.5, 97.5]), 3))
    return dict(acc_ci=q(acc), auc_ci=q(auc), f1_ci=q(f1))
