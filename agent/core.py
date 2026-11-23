"""Núcleo do agente de detecção de deepfake facial (Marco Final).

Pipeline (cada etapa é um passo visível na interface):
  1 leitura -> 2 detecção (MTCNN) -> 3 recorte 256x256 -> 4 classificador de pixels (ResNet-18 + temperature
  scaling, com Grad-CAM) -> 5 evidências (hf_ratio, blockiness, det_prob, eye_dist_rel) -> 6 discretização ->
  7 Rede Bayesiana (pgmpy, inferência exata) -> 8 explicação em log-odds -> 9 decisão por custo esperado.

As funções de recorte e de evidências replicam eda/extract_features.py (mesmos parâmetros), para que uma imagem
nova receba o mesmo tratamento do treino; agent/verify.py confere isso contra derived/manifest.csv.
"""
import json, os, time
import cv2
import numpy as np
import torch, torch.nn as nn, torchvision
from PIL import Image

PI = os.environ.get("PI_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PARAMS = os.path.join(os.path.dirname(__file__), "model_params.json")
CROP, MARGIN = 256, 1.3
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVID = ["HF", "BLOCK", "DET", "GEOM"]
EVID_COL = {"HF": "hf_ratio", "BLOCK": "blockiness", "DET": "det_prob", "GEOM": "eye_dist_rel"}
EVID_DESC = {"HF": "energia de alta frequência (fração acima de 0,25 ciclo/pixel)",
             "BLOCK": "artefato de bloco 8×8 (marca de compressão)",
             "DET": "confiança do detector facial",
             "GEOM": "distância entre os olhos / largura do rosto"}
STATE_NAME = {0: "baixo", 1: "médio", 2: "alto"}
CNN_STATE_NAME = ["0 (p < 0,1)", "1 (0,1 a 0,3)", "2 (0,3 a 0,7)", "3 (0,7 a 0,9)", "4 (p ≥ 0,9)"]


# ----------------------------------------------------------------------------- evidências (cópia de eda/)
def radial_hf(gray):
    g = gray.astype(np.float64); g -= g.mean()
    w = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * w))) ** 2
    h, wd = P.shape; yy, xx = np.indices(P.shape)
    r = np.hypot((yy - h / 2) / h, (xx - wd / 2) / wd)
    return float(P[(r > 0.25) & (r <= 0.5)].sum() / P[(r > 0.02) & (r <= 0.5)].sum())


def blockiness(gray, x0, y0, x1, y1, B=8):
    g = gray[y0:y1, x0:x1].astype(np.float32)
    if g.shape[0] < 3 * B or g.shape[1] < 3 * B:
        return float("nan")
    dx = np.abs(np.diff(g, axis=1)); dy = np.abs(np.diff(g, axis=0))
    cx = (np.arange(dx.shape[1]) + x0) % B == B - 1
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


# ----------------------------------------------------------------------------- Rede Bayesiana
class Bayes:
    """Rede Bayesiana em estrela (Fake -> CNN, HF, BLOCK, DET, GEOM) com as CPTs de model_params.json.
    A inferência usa pgmpy (VariableElimination); a explicação em log-odds usa as mesmas CPTs em numpy e é
    conferida contra o pgmpy a cada consulta."""

    def __init__(self, params):
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
        from pgmpy.models import DiscreteBayesianNetwork
        self.nodes = params["nodes"]
        self.cpt = {n: np.array(params["cpt"][n]) for n in self.nodes}  # (estados, 2): P(estado | Fake)
        self.prior = np.array(params["prior_fake"])  # [P(real), P(fake)]
        self.net = DiscreteBayesianNetwork([("Fake", n) for n in self.nodes])
        cpds = [TabularCPD("Fake", 2, self.prior.reshape(2, 1), state_names={"Fake": [0, 1]})]
        for n in self.nodes:
            K = self.cpt[n].shape[0]
            cpds.append(TabularCPD(n, K, self.cpt[n], evidence=["Fake"], evidence_card=[2],
                                   state_names={n: list(range(K)), "Fake": [0, 1]}))
        self.net.add_cpds(*cpds); self.net.check_model()
        self.ve = VariableElimination(self.net)

    def posterior(self, evidence):
        """evidence: {nó: estado}; nós com valor None são marginalizados. Retorna (P(fake), contribuições, log-odds a priori)."""
        ev = {k: int(v) for k, v in evidence.items() if v is not None}
        q = self.ve.query(["Fake"], evidence=ev, show_progress=False)
        p = float(q.values[list(q.state_names["Fake"]).index(1)])
        contrib = {n: float(np.log(self.cpt[n][s, 1]) - np.log(self.cpt[n][s, 0])) for n, s in ev.items()}
        prior_lo = float(np.log(self.prior[1] / self.prior[0]))
        p_np = 1 / (1 + np.exp(-(prior_lo + sum(contrib.values()))))
        assert abs(p - p_np) < 1e-6, (p, p_np)
        return p, contrib, prior_lo


def decide(p, c_fp=3.0, c_fn=1.0, c_inc=0.3):
    """Custo esperado: dizer 'real' custa p*c_fn; dizer 'fake' custa (1-p)*c_fp; 'inconclusivo' custa c_inc."""
    costs = {"real": p * c_fn, "fake": (1 - p) * c_fp, "inconclusivo": c_inc}
    return min(costs, key=costs.get), costs


# ----------------------------------------------------------------------------- agente
class Agent:
    def __init__(self, ckpt=None, threads=4, device="cpu"):
        torch.set_num_threads(threads)
        self.params = json.load(open(PARAMS))
        self.T = float(self.params["temperature"])
        self.cnn_bins = self.params["cnn_bins"]
        self.cuts = {k: np.array(v) for k, v in self.params["evidence_cuts"].items()}
        self.bayes = Bayes(self.params)
        ckpt = ckpt or os.path.join(PI, self.params["checkpoint"])
        m = torchvision.models.resnet18()
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))
        m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"]); m.eval()
        self.cnn = m
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
        self._acts = {}
        m.layer4.register_forward_hook(lambda mod, i, o: self._acts.__setitem__("a", o))
        self._norm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(MEAN, STD)])

    # --- etapas isoladas (usadas também por verify.py e pelos scripts de avaliação)
    def detect(self, bgr):
        return self.mtcnn.detect(np.ascontiguousarray(bgr[..., ::-1]), landmarks=True)

    def crop_from(self, bgr, boxes, probs, lms):
        H, W = bgr.shape[:2]
        k = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        bx, lm = boxes[k], lms[k]
        x0, y0, x1, y1 = square_crop(bx, W, H)
        crop = cv2.resize(bgr[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_AREA)
        fw, fh = bx[2] - bx[0], bx[3] - bx[1]
        feats = dict(hf_ratio=radial_hf(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)),
                     blockiness=blockiness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), x0, y0, x1, y1),
                     det_prob=float(probs[k]), eye_dist_rel=float(np.hypot(*(lm[1] - lm[0])) / max(fw, 1)),
                     face_px=float(np.sqrt(max(fw, 1) * max(fh, 1))), n_faces=int(len(boxes)))
        return dict(k=k, box=bx, crop_box=(x0, y0, x1, y1), crop=crop, feats=feats)

    def cnn_prob(self, crop_bgr, cam=False):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = self._norm(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)).unsqueeze(0)
        if not cam:
            with torch.no_grad():
                return float(self.cnn(x)), None
        self.cnn.zero_grad()
        out = self.cnn(x); logit = float(out.detach())
        a = self._acts["a"]; a.retain_grad()
        out.sum().backward()
        w = a.grad.mean(dim=(2, 3), keepdim=True)
        cam_ = torch.relu((w * a).sum(1))[0].detach().numpy()
        return logit, cam_ / (cam_.max() + 1e-8)

    def discretize(self, p_cal, feats):
        st = {"CNN": int(np.digitize(p_cal, self.cnn_bins))}
        for n in EVID:
            v = feats[EVID_COL[n]]
            st[n] = None if (v is None or not np.isfinite(v)) else int(np.digitize(v, self.cuts[n]))
        return st

    def infer(self, p_cal, feats, nodes=None, costs=(3.0, 1.0, 0.3)):
        st = self.discretize(p_cal, feats)
        if nodes is not None:
            st = {k: (v if k in nodes else None) for k, v in st.items()}
        p, contrib, prior_lo = self.bayes.posterior(st)
        dec, c = decide(p, *costs)
        return dict(states=st, p_fake=p, contrib=contrib, prior_logodds=prior_lo, decision=dec, costs=c)

    # --- pipeline passo a passo (gerador): cada yield é um passo para a interface
    def steps(self, bgr, costs=(3.0, 1.0, 0.3)):
        t0 = time.time(); R = {"ok": False}
        H, W = bgr.shape[:2]
        yield dict(id=1, title="Leitura da imagem", seconds=time.time() - t0, data=dict(width=W, height=H)), R
        t = time.time()
        boxes, probs, lms = self.detect(bgr)
        if boxes is None or lms is None:
            R["error"] = ("Nenhum rosto detectado. O agente não decide sem rosto: a imagem precisa mostrar "
                          "um rosto nítido e de frente.")
            yield dict(id=2, title="Detecção do rosto (MTCNN)", seconds=time.time() - t, data=dict(n_faces=0)), R
            return
        yield dict(id=2, title="Detecção do rosto (MTCNN)", seconds=time.time() - t,
                   data=dict(n_faces=len(boxes), boxes=boxes, probs=probs)), R
        t = time.time()
        c = self.crop_from(bgr, boxes, probs, lms)
        R.update(crop=c["crop"], feats=c["feats"], box=c["box"], crop_box=c["crop_box"])
        yield dict(id=3, title="Recorte do rosto (256×256, margem 1,3×)", seconds=time.time() - t,
                   data=dict(crop=c["crop"], crop_box=c["crop_box"], box=c["box"])), R
        t = time.time()
        logit, cam_ = self.cnn_prob(c["crop"], cam=True)
        p_cal = 1 / (1 + np.exp(-logit / self.T)); p_raw = 1 / (1 + np.exp(-logit))
        R.update(logit=logit, p_raw=p_raw, p_cal=p_cal, cam=cam_)
        yield dict(id=4, title="Classificador de pixels (ResNet-18 ajustada)", seconds=time.time() - t,
                   data=dict(logit=logit, p_raw=p_raw, p_cal=p_cal, T=self.T, cam=cam_, crop=c["crop"])), R
        t = time.time()
        yield dict(id=5, title="Evidências interpretáveis", seconds=time.time() - t, data=dict(feats=c["feats"])), R
        t = time.time()
        st = self.discretize(p_cal, c["feats"]); R["states"] = st
        yield dict(id=6, title="Discretização (estados da rede)", seconds=time.time() - t,
                   data=dict(states=st, cuts=self.cuts, cnn_bins=self.cnn_bins)), R
        t = time.time()
        res = self.infer(p_cal, c["feats"], costs=costs); R.update(res)
        yield dict(id=7, title="Rede Bayesiana: P(fake | evidências)", seconds=time.time() - t,
                   data=dict(p_fake=res["p_fake"], prior=float(self.bayes.prior[1]),
                             missing=[k for k, v in st.items() if v is None])), R
        yield dict(id=8, title="Explicação: contribuição de cada evidência (log-odds)", seconds=0.0,
                   data=dict(contrib=res["contrib"], prior_logodds=res["prior_logodds"])), R
        yield dict(id=9, title="Decisão por custo esperado", seconds=0.0,
                   data=dict(decision=res["decision"], costs=res["costs"], c=costs)), R
        R["ok"] = True; R["total_seconds"] = time.time() - t0

    def analyze(self, bgr, **kw):
        R = {}
        for _, R in self.steps(bgr, **kw):
            pass
        return R


def read_video_frames(path, k=10):
    """k frames uniformemente espaçados, como na amostragem do treino: índice = int((j + 0,5) / k * n)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release(); return 0, []
    idxs = sorted({int((j + 0.5) / k * n) for j in range(k)}); want = set(idxs); out = []
    i = 0
    while i <= max(idxs):
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok:
                out.append((i, fr))
        i += 1
    cap.release()
    return n, out
