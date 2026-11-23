"""Proxy de FF++ c40: reencoda os videos c23 de validacao e teste com H.264 CRF 40 (libx264) e roda o agente
nos mesmos indices de frame do c23. NAO e o c40 oficial (que parte do video bruto); e uma recompressao c23 -> CRF 40.

  python agent/make_c40_proxy.py [--limit N] [--splits val test]
Saida: derived/agent/c40proxy_frames.parquet  (uma linha por frame; evidencias, logit da CNN, p_cnn_cal)
"""
import argparse, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, imageio_ffmpeg

sys.path.insert(0, os.path.dirname(__file__))
import core

PI = core.PI; FF = f"{PI}/FaceForensics"
V_O = f"{FF}/original_sequences/youtube/c23/videos"; V_F = f"{FF}/manipulated_sequences/FaceSwap/c23/videos"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def encode(src, dst, crf=40):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, "-an", "-c:v", "libx264", "-crf", str(crf),
                    "-preset", "veryfast", "-pix_fmt", "yuv420p", "-threads", "2", dst], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--crf", type=int, default=40); ap.add_argument("--enc_workers", type=int, default=4)
    ap.add_argument("--out", default="c40proxy_frames.parquet")
    a = ap.parse_args()
    units = []
    for s in a.splits:
        for x, y in json.load(open(f"{FF}/splits/{s}.json")):
            units += [(f"{x}_{y}", s), (f"{y}_{x}", s)]
    if a.limit:
        units = units[: a.limit]
    ag = core.Agent(threads=4)
    tmp = tempfile.mkdtemp(prefix="c40proxy_")
    rows, t0 = [], time.time()

    def job(u):
        pair, split = u; tgt = pair.split("_")[0]
        n_fake = int(cv2.VideoCapture(f"{V_F}/{pair}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sorted({int((j + 0.5) / 10 * n_fake) for j in range(10)})
        fk = encode(f"{V_F}/{pair}.mp4", f"{tmp}/f_{pair}.mp4", a.crf)
        og = encode(f"{V_O}/{tgt}.mp4", f"{tmp}/o_{tgt}.mp4", a.crf)
        return u, idxs, fk, og

    with ThreadPoolExecutor(a.enc_workers) as ex:
        futs = [ex.submit(job, u) for u in units]
        for i, f in enumerate(futs, 1):
            (pair, split), idxs, fk, og = f.result()
            tgt = pair.split("_")[0]
            for grp, vid, path in [("ffpp_faceswap", pair, fk), ("ffpp_real", tgt, og)]:
                cap = cv2.VideoCapture(path); want = set(idxs); k = 0; frames = {}
                while k <= max(idxs):
                    if not cap.grab():
                        break
                    if k in want:
                        ok, fr = cap.retrieve()
                        if ok:
                            frames[k] = fr
                    k += 1
                cap.release()
                for ix in idxs:
                    base = dict(group=grp, video_id=vid, target_id=tgt, split=split, frame_idx=ix, label=int(grp != "ffpp_real"), compression=f"crf{a.crf}_proxy")
                    if ix not in frames:
                        rows.append(base | dict(detected=0)); continue
                    b, p, l = ag.detect(frames[ix])
                    if b is None or l is None:
                        rows.append(base | dict(detected=0)); continue
                    c = ag.crop_from(frames[ix], b, p, l)
                    logit, _ = ag.cnn_prob(c["crop"])
                    rows.append(base | dict(detected=1, logit=logit, p_cnn_raw=float(1 / (1 + np.exp(-logit))),
                                            p_cnn_cal=float(1 / (1 + np.exp(-logit / ag.T))), **c["feats"]))
            for p_ in (fk, og):
                os.remove(p_)
            if i % 10 == 0 or i == len(units):
                print(f"{i}/{len(units)} pares, {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(f"{PI}/derived/agent/{a.out}", index=False)
    pd.DataFrame(rows).to_parquet(f"{PI}/derived/agent/{a.out}", index=False)
    print("fim", len(rows), "linhas", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
