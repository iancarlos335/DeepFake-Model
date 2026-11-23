"""Ajusta a Rede Bayesiana do agente e grava agent/model_params.json.

  python agent/build_model.py

- Cortes das evidências: tercis dos frames de TREINO do FF++ c23 (reaplicados sem refazer em qualquer outro conjunto).
- CNN: 5 faixas de p_cnn_cal com cortes fixos 0,1 / 0,3 / 0,7 / 0,9.
- CPTs: contagem na VALIDAÇÃO (a CNN é superconfiante no treino), prior BDeu de tamanho equivalente 5, via pgmpy.
- Estrutura: Fake -> CNN, HF, BLOCK, DET, GEOM (estrutura de domínio sem COMP, que ainda não está integrado).
"""
import json, os, sys
import numpy as np, pandas as pd

PI = os.environ.get("PI_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
D = f"{PI}/derived"
sys.path.insert(0, os.path.dirname(__file__))
from core import EVID, EVID_COL, PARAMS

CNN_BINS = [0.1, 0.3, 0.7, 0.9]
ESS = 5
COSTS = dict(c_fp=3.0, c_fn=1.0, c_inc=0.3)


def load_frames():
    m = pd.read_csv(f"{D}/manifest.csv", low_memory=False, dtype={"video_id": str})
    c = pd.read_parquet(f"{D}/agent/preds_cnn.parquet")[["crop_file", "p_cnn_cal"]]
    f = m[m.group.isin(["ffpp_real", "ffpp_faceswap"]) & (m.detected == 1) & m.crop_file.notna()].merge(c, on="crop_file")
    return f


def discretize_df(df, cuts):
    X = pd.DataFrame({"CNN": np.digitize(df.p_cnn_cal.values, CNN_BINS)}, index=df.index)
    for n in EVID:
        X[n] = np.digitize(df[EVID_COL[n]].values, cuts[n])
    return X


def main():
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    f = load_frames()
    tr, va = f[f.split == "train"], f[f.split == "val"]
    cuts = {n: np.quantile(tr[EVID_COL[n]], [1 / 3, 2 / 3]).tolist() for n in EVID}
    X = discretize_df(va, cuts); X["Fake"] = va.label.values
    nodes = ["CNN"] + EVID
    net = DiscreteBayesianNetwork([("Fake", n) for n in nodes])
    sn = {n: list(range(5 if n == "CNN" else 3)) for n in nodes}; sn["Fake"] = [0, 1]
    est = BayesianEstimator(net, X[nodes + ["Fake"]], state_names=sn)
    cpd = {n: est.estimate_cpd(n, prior_type="BDeu", equivalent_sample_size=ESS) for n in nodes + ["Fake"]}
    cpt = {n: cpd[n].values.tolist() for n in nodes}  # (estados, 2): P(estado | Fake)
    prior = np.asarray(cpd["Fake"].values).ravel().tolist()
    cal = json.load(open(f"{D}/cnn/resnet18_ft_v1/calibration.json"))
    params = dict(nodes=nodes, cnn_bins=CNN_BINS, evidence_cuts=cuts, cpt=cpt, prior_fake=prior,
                  temperature=cal["temperature"], checkpoint="derived/cnn/resnet18_ft_v1/best.pt",
                  costs=COSTS, equivalent_sample_size=ESS,
                  fit=dict(split="val", frames=int(len(va)), videos=int(va.groupby(["group", "video_id"]).ngroups),
                           cuts_from="train", train_frames=int(len(tr))),
                  note="CPTs P(estado | Fake) com colunas [real, fake]; prior_fake = [P(real), P(fake)].")
    json.dump(params, open(PARAMS, "w"), indent=1)
    print("cortes:", {k: [round(x, 5) for x in v] for k, v in cuts.items()})
    print("prior:", np.round(prior, 4).tolist(), "| CNN P(estado|Fake) (linhas = estados):")
    print(np.round(np.array(cpt["CNN"]), 4))
    print("gravado", PARAMS)


if __name__ == "__main__":
    main()
