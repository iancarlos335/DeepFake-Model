"""EDA feature extraction for FaceForensics++ (c23, original + FaceSwap) and CelebA.

For every FaceSwap video A_B (target A, source face B) we sample K frames uniformly
within the fake's length and read the SAME frame indices from the original A, the
fake A_B and the FaceSwap mask A_B. For CelebA we draw a seeded, partition-stratified
random sample. Each image goes through the same MTCNN detector and the same crop /
feature code, so differences between groups are not caused by preprocessing.

Outputs (PI/derived/):
  manifest.csv            one row per image (metadata + features)
  spectra.npz             radial power spectrum (128 bins) per row, keyed by row order
  crops/<group>/*.png     256x256 face crops (lossless) for later modelling
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool

PI = "/Users/iancarlos/Repositories/Barao/8Semestre/PI"
FF = f"{PI}/FaceForensics"
V_O = f"{FF}/original_sequences/youtube/c23/videos"
V_F = f"{FF}/manipulated_sequences/FaceSwap/c23/videos"
V_M = f"{FF}/manipulated_sequences/FaceSwap/masks/videos"
CEL = f"{PI}/Celeba"
OUT = f"{PI}/derived"
CROP = 256
MARGIN = 1.3
NBINS = 128
STD_LUMA = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99], float)

_det = None


def _init():
    global _det
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    from facenet_pytorch import MTCNN
    _det = MTCNN(keep_all=True, device="cpu", post_process=False)


# ---------------------------------------------------------------- features
def radial_spectrum(gray):
    g = gray.astype(np.float64); g -= g.mean()
    w = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * w))) ** 2
    h, wd = P.shape; yy, xx = np.indices(P.shape)
    r = np.hypot((yy - h / 2) / h, (xx - wd / 2) / wd)  # cycles/pixel, Nyquist = 0.5
    bins = np.minimum((r / 0.5 * NBINS).astype(int), NBINS)
    spec = np.bincount(bins.ravel(), P.ravel(), NBINS + 1)[:NBINS] / np.maximum(np.bincount(bins.ravel(), None, NBINS + 1)[:NBINS], 1)
    hf = P[(r > 0.25) & (r <= 0.5)].sum() / P[(r > 0.02) & (r <= 0.5)].sum()
    return spec.astype(np.float32), float(hf)


def blockiness(gray, x0, y0, x1, y1, B=8):
    """Mean |diff| across absolute 8-px block boundaries / mean |diff| elsewhere (rows+cols)."""
    g = gray[y0:y1, x0:x1].astype(np.float32)
    if g.shape[0] < 3 * B or g.shape[1] < 3 * B:
        return np.nan
    dx = np.abs(np.diff(g, axis=1)); dy = np.abs(np.diff(g, axis=0))
    cx = (np.arange(dx.shape[1]) + x0) % B == B - 1  # diff between col c and c+1
    cy = (np.arange(dy.shape[0]) + y0) % B == B - 1
    bnd = np.concatenate([dx[:, cx].ravel(), dy[cy, :].ravel()]).mean()
    ins = np.concatenate([dx[:, ~cx].ravel(), dy[~cy, :].ravel()]).mean()
    return float(bnd / max(ins, 1e-6))


def square_crop(box, W, H):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = min(MARGIN * max(x1 - x0, y1 - y0), W, H)
    cx = min(max(cx, s / 2), W - s / 2); cy = min(max(cy, s / 2), H - s / 2)
    a, b = int(round(cx - s / 2)), int(round(cy - s / 2))
    s = int(round(s))
    return max(a, 0), max(b, 0), min(a + s, W), min(b + s, H)


def image_features(bgr, crop_path=None):
    H, W = bgr.shape[:2]
    row = dict(width=W, height=H)
    boxes, probs, lms = _det.detect(np.ascontiguousarray(bgr[..., ::-1]), landmarks=True)
    if boxes is None:
        row.update(detected=0, n_faces=0)
        return row, None, None
    k = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))  # largest face
    bx = boxes[k]; lm = lms[k]
    x0, y0, x1, y1 = square_crop(bx, W, H)
    crop = cv2.resize(bgr[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    spec, hf = radial_spectrum(gray)
    fw, fh = bx[2] - bx[0], bx[3] - bx[1]
    eye_d = float(np.hypot(*(lm[1] - lm[0])))
    row.update(detected=1, n_faces=len(boxes), det_prob=float(probs[k]),
               face_px=float(np.sqrt(max(fw, 1) * max(fh, 1))), face_frac=float(fw * fh / (W * H)),
               bx0=float(bx[0]), by0=float(bx[1]), bx1=float(bx[2]), by1=float(bx[3]),
               eye_dist_rel=eye_d / max(fw, 1),
               sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
               hf_ratio=hf,
               blockiness=blockiness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), x0, y0, x1, y1),
               brightness=float(hsv[..., 2].mean()), saturation=float(hsv[..., 1].mean()))
    for j, nm in enumerate(["le", "re", "no", "lm", "rm"]):
        row[f"{nm}_x"], row[f"{nm}_y"] = float(lm[j][0]), float(lm[j][1])
    if crop_path:
        cv2.imwrite(crop_path, crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return row, spec, (x0, y0, x1, y1)


def mask_texture(bgr, mask_bin, face_px):
    """log ratio of mean |Laplacian| inside the swap mask vs a ring just outside it."""
    if mask_bin.sum() < 50:
        return np.nan, np.nan
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    r = max(3, int(0.15 * face_px))
    dil = cv2.dilate(mask_bin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    ring = (dil > 0) & ~mask_bin
    ero = cv2.erode(mask_bin.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # avoid the seam itself
    if ring.sum() < 50 or ero.sum() < 50:
        return np.nan, np.nan
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dcol = float(np.linalg.norm(lab[ero].mean(0) - lab[ring].mean(0)))
    return float(np.log(lap[ero].mean() / max(lap[ring].mean(), 1e-6))), dcol


# ---------------------------------------------------------------- readers
def read_frames(path, idxs):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); want = set(idxs); out = {}; i = 0
    while i <= max(idxs):
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok:
                out[i] = fr
        i += 1
    cap.release()
    return n, out


def ff_unit(args):
    pair, split, K = args
    tgt = pair.split("_")[0]
    rows, specs = [], []
    try:
        n_fake = int(cv2.VideoCapture(f"{V_F}/{pair}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sorted({int((j + 0.5) / K * n_fake) for j in range(K)})
        _, F = read_frames(f"{V_F}/{pair}.mp4", idxs)
        n_orig, O = read_frames(f"{V_O}/{tgt}.mp4", idxs)
        _, M = read_frames(f"{V_M}/{pair}.mp4", idxs)
    except Exception as e:  # noqa: BLE001
        return [dict(group="ffpp_faceswap", video_id=pair, error=repr(e))], [None]
    for grp, vid, frames, nfr in [("ffpp_real", tgt, O, n_orig), ("ffpp_faceswap", pair, F, n_fake)]:
        for i in idxs:
            base = dict(group=grp, source="FaceForensics++", label=int(grp != "ffpp_real"), video_id=vid,
                        target_id=tgt, source_id=pair.split("_")[1], split=split, frame_idx=i,
                        video_frames=nfr, compression="c23")
            if i not in frames:
                base.update(decode_ok=0, detected=0); rows.append(base); specs.append(None); continue
            base["decode_ok"] = 1
            cp = f"{OUT}/crops/{grp}/{vid}_{i:04d}.png"
            r, spec, box = image_features(frames[i], cp)
            base.update(r); base["crop_file"] = os.path.relpath(cp, OUT) if spec is not None else None
            if i in M and M[i].shape[:2] == frames[i].shape[:2]:
                mb = M[i].max(axis=2) > 10
                base["mask_frac"] = float(mb.mean())
                fpx = base.get("face_px", np.sqrt(mb.sum()))
                base["tex_logratio"], base["color_dist"] = mask_texture(frames[i], mb, fpx)
            rows.append(base); specs.append(spec)
    return rows, specs


def jpeg_quality(path):
    from PIL import Image
    try:
        q = Image.open(path).quantization
        t = np.array(q[0], float)
        s = np.mean(t / STD_LUMA) * 100
        return float((200 - s) / 2 if s <= 100 else 5000 / s)
    except Exception:  # noqa: BLE001
        return np.nan


def celeba_unit(args):
    img_id, part = args
    p = f"{CEL}/img_align_celeba/img_align_celeba/{img_id}"
    base = dict(group="celeba_real", source="CelebA", label=0, video_id=img_id, split=["train", "val", "test"][part],
                compression="jpeg", jpeg_q=jpeg_quality(p))
    bgr = cv2.imread(p)
    if bgr is None:
        base.update(decode_ok=0, detected=0); return [base], [None]
    base["decode_ok"] = 1
    cp = f"{OUT}/crops/celeba_real/{img_id[:-4]}.png"
    r, spec, _ = image_features(bgr, cp)
    base.update(r); base["crop_file"] = os.path.relpath(cp, OUT) if spec is not None else None
    return [base], [spec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--celeba_n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="manifest")
    a = ap.parse_args()
    for g in ["ffpp_real", "ffpp_faceswap", "celeba_real"]:
        os.makedirs(f"{OUT}/crops/{g}", exist_ok=True)

    tasks = []
    for s in ["train", "val", "test"]:
        for x, y in json.load(open(f"{FF}/splits/{s}.json")):
            tasks += [(f"{x}_{y}", s, a.frames), (f"{y}_{x}", s, a.frames)]
    part = pd.read_csv(f"{CEL}/list_eval_partition.csv")
    rng = np.random.default_rng(42)
    frac = a.celeba_n / len(part)
    cel = pd.concat([g.iloc[rng.choice(len(g), int(round(frac * len(g))), replace=False)] for _, g in part.groupby("partition")])
    ctasks = list(zip(cel.image_id, cel.partition))
    if a.limit:
        tasks, ctasks = tasks[: a.limit], ctasks[: a.limit]

    rows, specs, t0 = [], [], time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for name, fn, tl in [("ffpp", ff_unit, tasks), ("celeba", celeba_unit, ctasks)]:
            for i, (r, s) in enumerate(pool.imap_unordered(fn, tl, chunksize=1 if name == "ffpp" else 32), 1):
                rows += r; specs += s
                if i % max(1, len(tl) // 20) == 0 or i == len(tl):
                    print(f"[{name}] {i}/{len(tl)} {time.time() - t0:.0f}s", flush=True)
    assert len(rows) == len(specs)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/{a.out}.csv", index=False)
    S = np.stack([s if s is not None else np.full(NBINS, np.nan, np.float32) for s in specs])
    np.savez_compressed(f"{OUT}/{a.out}_spectra.npz", spectra=S)
    print("rows", len(df), "time", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
