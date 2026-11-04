"""A4 - Tabela comparativa de modelos no teste do FF++ c23 (mesmo protocolo do baseline do Marco 1).

Reajusta as 3 regressoes logisticas do baseline (mesmas funcoes de eda/baselines.py, C escolhido pela AUC de
validacao) e le as predicoes da CNN ajustada de derived/agent/preds_cnn.parquet. Metricas por frame e por
video (media das probabilidades dos frames do video), IC95% por bootstrap agrupado por video alvo (2000, seed 0).

  python ml/compare_models.py
Saidas: derived/agent/tabela_modelos.csv, derived/agent/tabela_modelos.json, derived/agent/preds_baselines.parquet
"""
import json, os, sys
import numpy as np, pandas as pd

PI = os.environ.get("PI_DIR", "/Users/iancarlos/Repositories/Barao/8Semestre/PI")
D = f"{PI}/derived"
sys.path.insert(0, f"{PI}/eda")
import baselines as B
from sklearn.metrics import brier_score_loss


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


man, spec = B.load()
ok = man[man.detected == 1].copy(); ok["row"] = ok.index
ff = ok[ok.source == "FaceForensics++"]
tr, va, te = (ff[ff.split == s] for s in ["train", "val", "test"])
R = np.load(f"{D}/resnet18_feats.npz")["feats"]
REPS = {"LR + 8 atributos manuais": B.hand_matrix,
        "LR + espectro radial (baseline)": lambda d: B.spec_matrix(spec, d.row.values),
        "LR + ResNet-18 congelada": lambda d: R[d.row.values]}
preds, cfg = {}, {}
for name, mk in REPS.items():
    val_auc, C, m = B.fit(mk(tr), tr.label.values, mk(va), va.label.values)
    preds[name] = m.predict_proba(mk(te))[:, 1]
    cfg[name] = dict(val_auc=val_auc, C=C)

cnn = pd.read_parquet(f"{D}/agent/preds_cnn.parquet")
te_c = te.merge(cnn[["crop_file", "p_cnn_raw", "p_cnn_cal"]], on="crop_file", how="left")
assert te_c.p_cnn_cal.notna().all() and len(te_c) == len(te)
vsplit = cnn[cnn.split == "val"]
from sklearn.metrics import roc_auc_score
preds["ResNet-18 ajustada (CNN), bruta"] = te_c.p_cnn_raw.values
preds["ResNet-18 ajustada (CNN), calibrada (T)"] = te_c.p_cnn_cal.values
cfg["ResNet-18 ajustada (CNN), bruta"] = dict(val_auc=float(roc_auc_score(vsplit.label, vsplit.p_cnn_raw)))
cfg["ResNet-18 ajustada (CNN), calibrada (T)"] = dict(val_auc=float(roc_auc_score(vsplit.label, vsplit.p_cnn_cal)))

COST = {"LR + 8 atributos manuais": "segundos em CPU; coeficientes legiveis",
        "LR + espectro radial (baseline)": "segundos em CPU; 128 coeficientes, direcao interpretavel",
        "LR + ResNet-18 congelada": "~10 min de extracao em CPU; caixa-preta nas features",
        "ResNet-18 ajustada (CNN), bruta": "~46 min de treino em CPU; ~46 img/s na inferencia; caixa-preta",
        "ResNet-18 ajustada (CNN), calibrada (T)": "igual, + 1 parametro (T) ajustado na validacao"}
rows, ytest = [], te.label.values
for name, p in preds.items():
    fr = B.metrics(ytest, p); v = B.video_level(te, p); vd = B.metrics(v.y.values, v.p.values)
    cf, cv = B.cluster_ci(te, p), B.cluster_ci(te, p, "video")
    rows.append({"modelo": name, "AUC_val": cfg[name]["val_auc"], "C": cfg[name].get("C"),
                 "acc_frame": fr["acc"], "prec_frame": fr["prec"], "rec_frame": fr["rec"], "f1_frame": fr["f1"], "auc_frame": fr["auc"],
                 "auc_frame_ic95": f"[{cf['auc_ci'][0]:.3f}, {cf['auc_ci'][1]:.3f}]", "acc_frame_ic95": f"[{cf['acc_ci'][0]:.3f}, {cf['acc_ci'][1]:.3f}]",
                 "brier_frame": fr["brier"], "ece_frame": ece(ytest, p),
                 "acc_video": vd["acc"], "f1_video": vd["f1"], "auc_video": vd["auc"], "auc_video_ic95": f"[{cv['auc_ci'][0]:.3f}, {cv['auc_ci'][1]:.3f}]",
                 "acc_video_ic95": f"[{cv['acc_ci'][0]:.3f}, {cv['acc_ci'][1]:.3f}]", "brier_video": vd["brier"],
                 "tn_video": int(vd["tn"]), "fp_video": int(vd["fp"]), "fn_video": int(vd["fn"]), "tp_video": int(vd["tp"]),
                 "custo_interpretabilidade": COST[name]})
tab = pd.DataFrame(rows)
os.makedirs(f"{D}/agent", exist_ok=True)
tab.to_csv(f"{D}/agent/tabela_modelos.csv", index=False)
pd.DataFrame({"crop_file": te.crop_file.values, **{k: v for k, v in preds.items() if k.startswith("LR")}}).to_parquet(f"{D}/agent/preds_baselines.parquet", index=False)
n = dict(frames=len(te), videos=int(te.groupby(["group", "video_id"]).ngroups), frac_fake=float(ytest.mean()))
json.dump({"n_test": n, "rows": rows}, open(f"{D}/agent/tabela_modelos.json", "w"), indent=1, default=float)
print(n)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(tab[["modelo", "AUC_val", "acc_frame", "f1_frame", "auc_frame", "auc_frame_ic95", "brier_frame", "ece_frame", "acc_video", "f1_video", "auc_video", "auc_video_ic95"]].round(3).to_string())
# conferencia com o baseline reportado no Marco 1
old = json.load(open(f"{D}/baseline_results.json"))
print("Marco1 spec: acc_frame %.3f auc_frame %.3f acc_video %.3f" % (old["spec"]["frame"]["acc"], old["spec"]["frame"]["auc"], old["spec"]["video"]["acc"]))
