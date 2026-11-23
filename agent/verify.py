"""Confere que o agente (frame original -> decisão) reproduz o manifest e as predições do Marco 2.

  python agent/verify.py [--n 24]

Para frames do teste do FF++ c23, compara: recorte (pixel a pixel), as 4 evidências, p_cnn_cal e P(fake) da rede.
"""
import argparse, os, sys, time
import cv2, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import core

PI = core.PI; D = f"{PI}/derived"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
    m = pd.read_csv(f"{D}/manifest.csv", low_memory=False, dtype={"video_id": str, "target_id": str})
    pc = pd.read_parquet(f"{D}/agent/preds_cnn.parquet").set_index("crop_file")
    te = m[(m.split == "test") & m.group.isin(["ffpp_real", "ffpp_faceswap"]) & (m.detected == 1)]
    rng = np.random.default_rng(0)
    sel = pd.concat([te[te.group == g].iloc[rng.choice((te.group == g).sum(), a.n // 2, replace=False)] for g in ["ffpp_real", "ffpp_faceswap"]])
    ag = core.Agent()
    rows = []
    for r in sel.itertuples():
        V = f"{PI}/FaceForensics/" + ("original_sequences/youtube/c23/videos" if r.group == "ffpp_real" else "manipulated_sequences/FaceSwap/c23/videos")
        _, frs = core.read_video_frames(f"{V}/{r.video_id}.mp4", 10)
        fr = dict(frs)[int(r.frame_idx)] if int(r.frame_idx) in dict(frs) else None
        if fr is None:
            continue
        t = time.time(); R = ag.analyze(fr); dt = time.time() - t
        ref = cv2.imread(f"{D}/{r.crop_file}")
        rows.append(dict(crop_file=r.crop_file, label=r.label, crop_max_abs_diff=int(np.abs(R["crop"].astype(int) - ref.astype(int)).max()) if R.get("ok") else -1,
                         d_hf=abs(R["feats"]["hf_ratio"] - r.hf_ratio), d_block=abs(R["feats"]["blockiness"] - r.blockiness),
                         d_det=abs(R["feats"]["det_prob"] - r.det_prob), d_geom=abs(R["feats"]["eye_dist_rel"] - r.eye_dist_rel),
                         d_pcnn=abs(R["p_cal"] - pc.loc[r.crop_file, "p_cnn_cal"]), p_fake=R["p_fake"], decisao=R["decision"], seg=dt))
    df = pd.DataFrame(rows)
    print(df.drop(columns="crop_file").describe().loc[["count", "mean", "max"]].round(7).to_string())
    print(df.groupby("label").decisao.value_counts().to_string())
    ok = (df.crop_max_abs_diff == 0).mean()
    print(f"recortes idênticos: {ok:.0%} | maior |Δ p_cnn_cal| = {df.d_pcnn.max():.2e} | maior |Δ evidência| = {df[['d_hf','d_block','d_det','d_geom']].max().max():.2e}")
    df.to_csv(f"{D}/agent/verify_pipeline.csv", index=False)


if __name__ == "__main__":
    main()
