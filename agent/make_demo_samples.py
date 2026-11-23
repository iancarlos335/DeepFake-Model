"""Seleciona exemplos do TESTE do FF++ para a demonstração e grava em agent/demo_samples (não versionado).
Nomes neutros (amostra_XX); o gabarito fica em demo_samples/gabarito.csv. Inclui casos certos, inconclusivos e erros."""
import json, os, shutil, sys
import cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import core, build_model as bm
PI = core.PI; D = f"{PI}/derived"; OUT = os.path.join(os.path.dirname(__file__), "demo_samples")
os.makedirs(OUT, exist_ok=True)
P = json.load(open(core.PARAMS)); bayes = core.Bayes(P); cuts = {k: np.array(v) for k, v in P["evidence_cuts"].items()}
f = bm.load_frames(); te = f[f.split == "test"].copy()
X = bm.discretize_df(te, cuts)
ps = []
for i in range(len(te)):
    ps.append(bayes.posterior({n: int(X.iloc[i][n]) for n in P["nodes"]})[0])
te["p_fake"] = ps; te["dec"] = [core.decide(p)[0] for p in ps]
rng = np.random.default_rng(7)
def pick(mask, n): 
    d = te[mask]; return d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)]
fk, rl = te.label == 1, te.label == 0
sel = [("fake correto", pick(fk & (te.dec == "fake"), 3)), ("fake inconclusivo", pick(fk & (te.dec == "inconclusivo"), 1)), ("fake erro (real)", pick(fk & (te.dec == "real"), 1)),
       ("real correto", pick(rl & (te.dec == "real"), 3)), ("real inconclusivo", pick(rl & (te.dec == "inconclusivo"), 1)), ("real erro (fake)", pick(rl & (te.dec == "fake"), 1))]
rows = []
for kind, d in sel:
    for r in d.itertuples():
        rows.append((kind, r))
order = rng.permutation(len(rows))
gab = []
for n, j in enumerate(order, 1):
    kind, r = rows[j]
    V = f"{PI}/FaceForensics/" + ("original_sequences/youtube/c23/videos" if r.group == "ffpp_real" else "manipulated_sequences/FaceSwap/c23/videos")
    _, frs = core.read_video_frames(f"{V}/{r.video_id}.mp4", 10); fr = dict(frs).get(int(r.frame_idx))
    if fr is None: continue
    name = f"amostra_{n:02d}.png"; cv2.imwrite(os.path.join(OUT, name), fr)
    gab.append(dict(arquivo=name, verdade="fake" if r.label else "real", video=r.video_id, frame=int(r.frame_idx), caso=kind, p_fake_esperada=round(r.p_fake, 4), decisao_esperada=r.dec))
# CelebA (fora do domínio de treino)
cel = pd.read_csv(f"{D}/manifest.csv", low_memory=False)
cel = cel[(cel.group == "celeba_real") & (cel.split == "test") & (cel.detected == 1)].iloc[:2]
for k, r in enumerate(cel.itertuples(), 1):
    src = f"{PI}/Celeba/img_align_celeba/img_align_celeba/{r.video_id}"; name = f"celeba_{k:02d}.jpg"; shutil.copy(src, os.path.join(OUT, name))
    gab.append(dict(arquivo=name, verdade="real", video=r.video_id, frame=-1, caso="CelebA (fora do domínio)", p_fake_esperada=np.nan, decisao_esperada=""))
# vídeos: um real e um fake com decisão correta por vídeo
g = te.groupby(["group", "video_id"]).agg(p=("p_fake", "mean")).reset_index()
for grp, tag, ok in [("ffpp_real", "real", lambda p: p < 0.3), ("ffpp_faceswap", "fake", lambda p: p > 0.9)]:
    c = g[(g.group == grp) & g.p.map(ok)].sample(1, random_state=3).iloc[0]
    V = f"{PI}/FaceForensics/" + ("original_sequences/youtube/c23/videos" if grp == "ffpp_real" else "manipulated_sequences/FaceSwap/c23/videos")
    gab.append(dict(arquivo="", verdade=tag, video=c.video_id, frame=-1, caso="vídeo", p_fake_esperada=round(c.p, 4), decisao_esperada=tag, _src=f"{V}/{c.video_id}.mp4"))
vids = [x for x in gab if x.get("_src")]
for n, x in enumerate(rng.permutation(len(vids)), 1):
    v = vids[x]; name = f"video_{n:02d}.mp4"; shutil.copy(v["_src"], os.path.join(OUT, name)); v["arquivo"] = name; v.pop("_src")
pd.DataFrame(gab).to_csv(os.path.join(OUT, "gabarito.csv"), index=False)
print(pd.DataFrame(gab)[["arquivo", "verdade", "caso", "p_fake_esperada", "decisao_esperada"]].to_string())
