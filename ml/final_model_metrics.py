"""A5 - modelo final de ML (classificador de pixels): metricas exatas no teste + figura da secao 3.
Escolha: ResNet-18 ajustada, saida calibrada por temperature scaling (T ajustado so na validacao).
Saidas: derived/agent/metricas_modelo_final.json, derived/agent/fig_modelo_final.png"""
import json, os, sys
import numpy as np, pandas as pd
PI = os.environ.get("PI_DIR", "/Users/iancarlos/Repositories/Barao/8Semestre/PI"); D = f"{PI}/derived"
sys.path.insert(0, f"{PI}/eda"); import baselines as B
from sklearn.metrics import roc_curve, roc_auc_score
cnn = pd.read_parquet(f"{D}/agent/preds_cnn.parquet")
man, _ = B.load(); ok = man[man.detected == 1]
te = ok[(ok.source == "FaceForensics++") & (ok.split == "test")].merge(cnn[["crop_file", "p_cnn_cal"]], on="crop_file")
assert len(te) == 2800
p, y = te.p_cnn_cal.values, te.label.values
fr = B.metrics(y, p); v = B.video_level(te, p); vd = B.metrics(v.y.values, v.p.values)
cf, cv = B.cluster_ci(te, p), B.cluster_ci(te, p, "video")
cal = json.load(open(f"{D}/cnn/resnet18_ft_v1/calibration.json")); cfg = json.load(open(f"{D}/cnn/resnet18_ft_v1/config.json"))
tab = pd.read_csv(f"{D}/agent/tabela_modelos.csv").set_index("modelo")
ece = float(tab.loc["ResNet-18 ajustada (CNN), calibrada (T)", "ece_frame"])
out = dict(modelo="ResNet-18 ajustada (ImageNet -> FF++ c23), saida calibrada por temperature scaling",
           model_id=cnn.model_id.iloc[0], temperatura=cal["temperature"], epoca_checkpoint=cal["epoch_ckpt"], limiar=0.5,
           teste=dict(frames=len(te), videos=int(len(v)), frac_fake=float(y.mean())),
           frame={k: float(x) for k, x in fr.items()} | dict(ece=ece, auc_ic95=cf["auc_ci"], acc_ic95=cf["acc_ci"], f1_ic95=cf["f1_ci"]),
           video={k: float(x) for k, x in vd.items()} | dict(auc_ic95=cv["auc_ci"], acc_ic95=cv["acc_ci"], f1_ic95=cv["f1_ci"]),
           controle_celeba_real_teste=json.load(open(f"{D}/cnn/resnet18_ft_v1/celeba_real_check.json")),
           treino=dict(arch="resnet18", epocas=cfg["epochs"], frames_por_video_por_epoca=cfg["fpv"], lr=cfg["lr"], bs=cfg["bs"], seed=cfg["seed"]))
json.dump(out, open(f"{D}/agent/metricas_modelo_final.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("frame", "video")}, indent=1))


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
bl = pd.read_parquet(f"{D}/agent/preds_baselines.parquet").set_index("crop_file").loc[te.crop_file]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.35))
ax = axs[0]; ax.plot([0, 1], [0, 1], color="0.8", lw=0.8)
for name, pp, c, lw in [("LR + espectro radial (baseline)", bl["LR + espectro radial (baseline)"].values, "#E69F00", 1.4),
                        ("LR + ResNet-18 congelada", bl["LR + ResNet-18 congelada"].values, "#999999", 1.2),
                        ("ResNet-18 ajustada (calibrada)", p, "#0072B2", 2.2)]:
    f_, t_, _ = roc_curve(y, pp); ax.plot(f_, t_, color=c, lw=lw, label=f"{name} (AUC {roc_auc_score(y, pp):.3f})")
ax.set_xlabel("taxa de falsos positivos"); ax.set_ylabel("taxa de verdadeiros positivos"); ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.set_title("ROC por frame no teste do FF++ c23", loc="left")
ax = axs[1]; lg = pd.read_csv(f"{D}/cnn/resnet18_ft_v1/log.csv")
ax.plot(lg.epoch + 1, lg.train_loss, color="#0072B2", marker="o", ms=3, label="treino")
ax.plot(lg.epoch + 1, lg.val_loss, color="#D55E00", marker="o", ms=3, label="validação")
ax.set_xlabel("época"); ax.set_ylabel("perda (entropia cruzada)"); ax.legend(frameon=False); ax.set_title("Perda de validação estabiliza após a época 3", loc="left")
ax = axs[2]; Mc = np.array([[vd["tn"], vd["fp"]], [vd["fn"], vd["tp"]]]).astype(int)
ax.imshow(Mc, cmap="Blues", vmin=0, vmax=140)
for i in range(2):
    for j in range(2): ax.text(j, i, f"{Mc[i, j]}\n({Mc[i, j] / 140 * 100:.0f}%)", ha="center", va="center", color="white" if Mc[i, j] > 80 else "black")
ax.set_xticks([0, 1], ["real", "fake"]); ax.set_yticks([0, 1], ["real", "fake"]); ax.set_xlabel("predito"); ax.set_ylabel("verdadeiro")
ax.set_title(f"Por vídeo: {vd['acc'] * 100:.1f}% de acurácia", loc="left")
fig.savefig(f"{D}/agent/fig_modelo_final.png", dpi=200, bbox_inches="tight")
