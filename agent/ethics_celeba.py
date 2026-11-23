"""Análise ética: taxa de falsos positivos (rosto REAL classificado como fake) por grupo do CelebA.

  python agent/ethics_celeba.py [--n_random 4000] [--limit N]

Todas as imagens do CelebA são rostos reais; logo, "acusar de fake" é sempre um falso positivo. Duas condições:
  bruta      recortes já extraídos no Marco 1 (JPEG q≈75, rosto de ≈98 px).
  igualada   imagem original redimensionada para que o rosto tenha ≈160 px (mediana do FF++) e recomprimida em
             H.264 CRF 23 (um quadro), depois processada pelo pipeline completo. É uma aproximação: não reproduz
             o vídeo do FF++, mas aproxima tamanho do rosto e tipo de compressão.
Grupos (atributos do CelebA): Male, Young, Pale_Skin (único proxy de tom de pele disponível), Eyeglasses, Wearing_Hat.
O CelebA não foi usado em nenhuma etapa de treino, ajuste ou escolha de limiares.
"""
import argparse, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, imageio_ffmpeg

sys.path.insert(0, os.path.dirname(__file__))
import core, evaluate as ev

PI = core.PI; D = f"{PI}/derived"; OUT = f"{D}/agent"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe(); TARGET_FACE = 160.0
ATTRS = {"Male": "sexo (Male = 1: masculino)", "Young": "idade (Young = 1: jovem)", "Pale_Skin": "pele pálida",
         "Eyeglasses": "óculos", "Wearing_Hat": "chapéu"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def equalize(src, face_px, tmp, W=178, H=218):
    s = TARGET_FACE / face_px
    w, h = int(round(W * s / 2) * 2), int(round(H * s / 2) * 2)
    dst = os.path.join(tmp, os.path.basename(src) + ".mp4")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, "-vf", f"scale={w}:{h}:flags=bicubic", "-frames:v", "1",
                    "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p", dst], check=True)
    cap = cv2.VideoCapture(dst); ok, fr = cap.read(); cap.release(); os.remove(dst)
    return fr if ok else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_random", type=int, default=4000); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    m = pd.read_csv(f"{D}/manifest.csv", low_memory=False)
    c = m[(m.group == "celeba_real") & (m.detected == 1) & m.crop_file.notna()].copy()
    c["image_id"] = c.video_id.astype(str)
    at = pd.read_csv(f"{PI}/Celeba/list_attr_celeba.csv")[["image_id"] + list(ATTRS)]
    c = c.merge(at, on="image_id", how="left")
    ag = core.Agent(threads=4)
    # ---- condição bruta: todos os recortes já extraídos
    t0 = time.time(); logit = []
    for r in c.itertuples():
        im = cv2.imread(f"{D}/{r.crop_file}"); logit.append(ag.cnn_prob(im)[0])
    c["p_cnn"] = 1 / (1 + np.exp(-np.array(logit) / ag.T))
    pf = []
    for r in c.itertuples():
        feats = dict(hf_ratio=r.hf_ratio, blockiness=r.blockiness, det_prob=r.det_prob, eye_dist_rel=r.eye_dist_rel)
        pf.append(ag.infer(r.p_cnn, feats)["p_fake"])
    c["p_fake"] = pf
    print(f"condição bruta: {len(c)} imagens em {time.time() - t0:.0f}s", flush=True)
    # ---- condição igualada: Pale_Skin completo + amostra aleatória
    rng = np.random.default_rng(0)
    sub = pd.concat([c[c.Pale_Skin == 1], c.iloc[rng.choice(len(c), min(a.n_random, len(c)), replace=False)]]).drop_duplicates("image_id")
    if a.limit:
        sub = sub.iloc[: a.limit]
    tmp = tempfile.mkdtemp(prefix="eq_"); res = {}
    src = lambda iid: f"{PI}/Celeba/img_align_celeba/img_align_celeba/{iid}"
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        futs = {r.image_id: ex.submit(equalize, src(r.image_id), r.face_px, tmp) for r in sub.itertuples()}
        for i, (iid, f) in enumerate(futs.items(), 1):
            fr = f.result()
            if fr is None:
                res[iid] = dict(p_cnn_eq=np.nan, p_fake_eq=np.nan, ok_eq=0); continue
            R = ag.analyze(fr)
            res[iid] = dict(p_cnn_eq=R.get("p_cal", np.nan), p_fake_eq=R.get("p_fake", np.nan), ok_eq=int(R.get("ok", False)))
            if i % 500 == 0:
                print(f"  igualada {i}/{len(sub)} {time.time() - t0:.0f}s", flush=True)
    eq = pd.DataFrame.from_dict(res, orient="index").rename_axis("image_id").reset_index()
    c = c.merge(eq, on="image_id", how="left")
    c.drop(columns=[x for x in c.columns if x.startswith(("le_", "re_", "no_", "lm_", "rm_"))]).to_parquet(f"{OUT}/celeba_ethics_frames.parquet", index=False)
    # ---- FPR por grupo
    rows = []
    def add(cond, grp_name, mask, pcol, pcnn_col):
        d = c[mask & c[pcol].notna()]; n = len(d)
        for sys_name, p in [("Rede completa", d[pcol].values), ("CNN + regra", d[pcnn_col].values)]:
            dec = ev.rule(p); k = int((dec == 1).sum()); inc = int((dec == -1).sum()); lo, hi = wilson(k, n)
            rows.append(dict(condicao=cond, atributo=grp_name[0], grupo=grp_name[1], sistema=sys_name, n=n, acusados_fake=k,
                             fpr=k / n if n else np.nan, fpr_lo=lo, fpr_hi=hi, inconclusivos=inc, taxa_inconclusivo=inc / n if n else np.nan))
    for cond, pcol, pc in [("bruta", "p_fake", "p_cnn"), ("igualada", "p_fake_eq", "p_cnn_eq")]:
        add(cond, ("(todos)", "todos"), c.index == c.index, pcol, pc)
        for at_, desc in ATTRS.items():
            add(cond, (at_, f"{at_}=1"), c[at_] == 1, pcol, pc); add(cond, (at_, f"{at_}=0"), c[at_] == -1, pcol, pc)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/fpr_por_grupo.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[df.sistema == "Rede completa"][["condicao", "grupo", "n", "acusados_fake", "fpr", "fpr_lo", "fpr_hi", "taxa_inconclusivo"]].round(4).to_string())
    print("p_fake mediana bruta/igualada:", c.p_fake.median().round(4), c.p_fake_eq.median().round(4))


if __name__ == "__main__":
    main()
