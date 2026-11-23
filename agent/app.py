"""Demonstração ao vivo: detector de deepfake facial (CNN + Rede Bayesiana).

  streamlit run agent/app.py

Anexe imagens (png/jpg) ou vídeos (mp4/avi/mov). Para cada arquivo, a página mostra cada etapa do processamento
(detecção, recorte, CNN com Grad-CAM, evidências, discretização, Rede Bayesiana, explicação) e a decisão final:
fake, real ou inconclusivo.
"""
import os, sys, tempfile, time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import core

HERE = os.path.dirname(__file__)
SAMPLES = os.path.join(HERE, "demo_samples")
st.set_page_config(page_title="Detector de deepfake facial", page_icon="🔎", layout="wide")

COLORS = {"fake": "#B03A2E", "real": "#1F618D", "inconclusivo": "#9A7D0A"}
LABEL = {"fake": "FAKE (rosto manipulado)", "real": "REAL (sem sinal de manipulação)", "inconclusivo": "INCONCLUSIVO"}


@st.cache_resource(show_spinner="Carregando modelos (MTCNN, ResNet-18, Rede Bayesiana)…")
def get_agent():
    return core.Agent(threads=4)


def bgr2rgb(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def overlay_cam(crop, cam):
    h = cv2.resize(cam, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(h, 0, 1)), cv2.COLORMAP_INFERNO)
    return bgr2rgb(cv2.addWeighted(crop, 0.55, heat, 0.45, 0))


def draw_boxes(img, boxes, probs, k):
    out = img.copy(); th = max(2, img.shape[1] // 300)
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = [int(v) for v in b]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255) if i == k else (160, 160, 160), th)
    return bgr2rgb(out)


def log_odds_chart(contrib, prior_lo):
    import matplotlib.pyplot as plt
    names = list(contrib.keys()); vals = [contrib[n] for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 0.45 * len(names) + 0.9))
    cols = ["#B03A2E" if v > 0 else "#1F618D" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("contribuição em log-odds (→ fake à direita, → real à esquerda)", fontsize=8)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + (0.08 if v >= 0 else -0.08), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    lim = max(1.0, max(abs(v) for v in vals) * 1.35); ax.set_xlim(-lim, lim)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8); fig.tight_layout()
    return fig


def explain_text(R, costs):
    c = R["contrib"]; top = sorted(c.items(), key=lambda kv: -abs(kv[1]))
    tot = R["prior_logodds"] + sum(c.values())
    parts = [f"{n} ({v:+.2f})" for n, v in top[:3]]
    return (f"log-odds total = prior ({R['prior_logodds']:+.2f}) + soma das evidências ({sum(c.values()):+.2f}) = {tot:+.2f}. "
            f"Maiores contribuições: {', '.join(parts)}.")


def render_step(s, R, sidebar):
    d, i = s["data"], s["id"]
    if i == 1:
        st.write(f"Imagem de {d['width']}×{d['height']} pixels.")
    elif i == 2:
        if d["n_faces"] == 0:
            st.error("Nenhum rosto detectado.")
        else:
            k = int(np.argmax((d["boxes"][:, 2] - d["boxes"][:, 0]) * (d["boxes"][:, 3] - d["boxes"][:, 1])))
            st.write(f"{d['n_faces']} rosto(s) detectado(s); o maior é usado (confiança do detector {d['probs'][k]:.5f}).")
            st.image(draw_boxes(R["_bgr"], d["boxes"], d["probs"], k), caption="Rosto escolhido (caixa laranja)", width=360)
    elif i == 3:
        st.image(bgr2rgb(d["crop"]), caption="Recorte 256×256 enviado aos módulos seguintes", width=200)
    elif i == 4:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(overlay_cam(d["crop"], d["cam"]), caption="Grad-CAM: regiões que mais pesaram para 'fake'", width=220)
        with c2:
            st.metric("p_cnn (calibrada)", f"{d['p_cal']:.4f}")
            st.caption(f"logit = {d['logit']:+.3f}; sem calibração {d['p_raw']:.4f}; temperatura T = {d['T']:.3f}. "
                       "O Grad-CAM mostra onde a rede olhou, não prova que ali está a manipulação.")
    elif i == 5:
        rows = [dict(Evidência=n, Significado=core.EVID_DESC[n], Valor=d["feats"][core.EVID_COL[n]]) for n in core.EVID]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif i == 6:
        rows = [dict(Nó="CNN", Estado=core.CNN_STATE_NAME[d["states"]["CNN"]])]
        for n in core.EVID:
            v = d["states"][n]
            rows.append(dict(Nó=n, Estado="(ausente: marginalizado)" if v is None else f"{v} ({core.STATE_NAME[v]})"))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Evidências: tercis dos frames de treino. CNN: cortes fixos 0,1 · 0,3 · 0,7 · 0,9.")
    elif i == 7:
        st.metric("P(fake | evidências)", f"{d['p_fake']:.4f}", delta=f"a priori {d['prior']:.2f}", delta_color="off")
        if d["missing"]:
            st.warning("Evidência ausente, marginalizada: " + ", ".join(d["missing"]))
    elif i == 8:
        st.pyplot(log_odds_chart(d["contrib"], d["prior_logodds"]), width="content")
        st.caption(explain_text(R, sidebar["costs"]))
    elif i == 9:
        dec, costs = core.decide(R["p_fake"], *sidebar["costs"])
        st.write("Custo esperado de cada resposta: " + " · ".join(f"{k} = {v:.3f}" for k, v in costs.items()))
        st.write(f"Menor custo: **{dec}**.")


def decision_card(name, p, dec, extra=""):
    col = COLORS[dec]
    st.markdown(
        f"<div style='border-left:10px solid {col};background:#F4F6F7;padding:14px 18px;border-radius:6px'>"
        f"<div style='font-size:13px;color:#555'>{name}</div>"
        f"<div style='font-size:30px;font-weight:700;color:{col}'>{LABEL[dec]}</div>"
        f"<div style='font-size:15px'>P(fake) = <b>{p:.3f}</b> {extra}</div></div>", unsafe_allow_html=True)


def process_image(name, bgr, agent, sidebar, show_steps=True):
    R_final = None
    status = st.status(f"Processando {name}…", expanded=True)
    with status:
        for s, R in agent.steps(bgr, costs=sidebar["costs"]):
            R["_bgr"] = bgr
            with st.container(border=True):
                st.markdown(f"**{s['id']}. {s['title']}** · {s['seconds'] * 1000:.0f} ms")
                render_step(s, R, sidebar)
            R_final = R
            time.sleep(sidebar["pause"])
    if R_final is None or not R_final.get("ok"):
        status.update(label=f"{name}: sem decisão", state="error", expanded=True)
        st.error(R_final.get("error", "Falha no processamento.") if R_final else "Falha no processamento.")
        return dict(arquivo=name, tipo="imagem", p_fake=np.nan, decisao="sem rosto", detalhe="nenhum rosto detectado")
    status.update(label=f"{name}: concluído em {R_final['total_seconds']:.2f} s", state="complete", expanded=False)
    dec = R_final["decision"]
    decision_card(name, R_final["p_fake"], dec, f"· CNN sozinha: {R_final['p_cal']:.3f}")
    return dict(arquivo=name, tipo="imagem", p_fake=R_final["p_fake"], p_cnn=R_final["p_cal"], decisao=dec, detalhe="")


def process_video(name, path, agent, sidebar, k):
    n, frames = core.read_video_frames(path, k)
    st.write(f"Vídeo com {n} quadros; {len(frames)} quadros amostrados de forma uniforme (como no treino).")
    bar = st.progress(0.0, text="Analisando quadros…"); rows, thumbs = [], []
    for j, (idx, fr) in enumerate(frames):
        R = agent.analyze(fr, costs=sidebar["costs"])
        if R.get("ok"):
            rows.append(dict(quadro=idx, p_cnn=R["p_cal"], p_fake=R["p_fake"], decisao=R["decision"]))
            thumbs.append((idx, bgr2rgb(R["crop"]), R["p_fake"]))
        else:
            rows.append(dict(quadro=idx, p_cnn=np.nan, p_fake=np.nan, decisao="sem rosto"))
        bar.progress((j + 1) / len(frames), text=f"Quadro {j + 1}/{len(frames)}")
        time.sleep(sidebar["pause"] / 3)
    df = pd.DataFrame(rows); ok = df.dropna(subset=["p_fake"])
    if thumbs:
        cols = st.columns(min(len(thumbs), 5))
        for i, (idx, im, p) in enumerate(thumbs):
            cols[i % len(cols)].image(im, caption=f"quadro {idx}: P(fake) {p:.2f}", width="stretch")
    st.dataframe(df, hide_index=True, width="stretch")
    if len(ok) < 3:
        st.error("Menos de 3 quadros com rosto detectado: o agente não decide sobre este vídeo.")
        return dict(arquivo=name, tipo="vídeo", p_fake=np.nan, decisao="sem rosto", detalhe=f"{len(ok)} quadros com rosto")
    p = float(ok.p_fake.mean()); dec, _ = core.decide(p, *sidebar["costs"])
    decision_card(name, p, dec, f"· média de {len(ok)} quadros (min {ok.p_fake.min():.2f}, máx {ok.p_fake.max():.2f})")
    return dict(arquivo=name, tipo="vídeo", p_fake=p, p_cnn=float(ok.p_cnn.mean()), decisao=dec, detalhe=f"{len(ok)} quadros")


def main():
    st.title("🔎 Detector de deepfake facial")
    st.caption("Projeto Integrador · classificador de pixels (ResNet-18) + Rede Bayesiana com evidências interpretáveis")
    with st.sidebar:
        st.header("Parâmetros")
        st.markdown("**Custos da decisão** (política, não verdade estatística)")
        c_fp = st.slider("C_FP: acusar um rosto real de ser fake", 0.5, 10.0, 3.0, 0.5)
        c_fn = st.slider("C_FN: deixar passar um fake", 0.5, 10.0, 1.0, 0.5)
        c_inc = st.slider("C_INC: responder 'inconclusivo'", 0.05, 2.0, 0.3, 0.05)
        k = st.slider("Quadros amostrados por vídeo", 3, 20, 10)
        pause = st.slider("Pausa entre etapas (s), para acompanhar ao vivo", 0.0, 2.0, 0.5, 0.1)
        st.divider()
        st.markdown("**Escopo do modelo**")
        st.caption("Treinado e testado em frames do FaceForensics++ (compressão c23) com troca de rosto (FaceSwap). "
                   "Fotos de outras origens, outros métodos de manipulação e compressão forte podem sair do que o modelo conhece. "
                   "A resposta é uma estimativa e deve ser revista por uma pessoa; não é prova.")
    sidebar = dict(costs=(c_fp, c_fn, c_inc), pause=pause)
    agent = get_agent()

    files = st.file_uploader("Anexe imagens ou vídeos", type=["png", "jpg", "jpeg", "bmp", "webp", "mp4", "avi", "mov", "mkv"],
                             accept_multiple_files=True)
    samples = []
    if os.path.isdir(SAMPLES):
        samples = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".png", ".jpg", ".jpeg", ".mp4")))
    chosen = st.multiselect("…ou use exemplos do conjunto de teste (agent/demo_samples)", samples) if samples else []
    go = st.button("Analisar", type="primary", disabled=not (files or chosen))

    if go:
        results = []
        items = [(f.name, f.getvalue()) for f in files] + [(s, open(os.path.join(SAMPLES, s), "rb").read()) for s in chosen]
        for name, raw in items:
            st.subheader(name)
            if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(name)[1], delete=False) as tmp:
                    tmp.write(raw); path = tmp.name
                try:
                    results.append(process_video(name, path, agent, sidebar, k))
                finally:
                    os.remove(path)
            else:
                bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    st.error("Não foi possível ler este arquivo como imagem."); continue
                results.append(process_image(name, bgr, agent, sidebar))
        st.session_state["results"] = results

    if st.session_state.get("results"):
        st.divider(); st.subheader("Resumo")
        df = pd.DataFrame(st.session_state["results"])
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Baixar resumo (CSV)", df.to_csv(index=False).encode("utf-8"), "resumo_deteccao.csv", "text/csv")

    with st.expander("Como o agente decide (resumo)"):
        st.markdown(
            "1. **MTCNN** detecta o rosto; o maior é recortado (256×256, margem 1,3×).\n"
            "2. **ResNet-18** (ajustada no FF++ c23, calibrada) dá p_cnn.\n"
            "3. Quatro **evidências interpretáveis** são medidas no recorte e discretizadas.\n"
            "4. Uma **Rede Bayesiana** (pgmpy) combina CNN e evidências em P(fake | evidências).\n"
            "5. A **explicação** é a contribuição de cada nó em log-odds.\n"
            "6. A **decisão** minimiza o custo esperado: fake se P > 0,9, real se P < 0,3 e inconclusivo no meio "
            "(com C_FP = 3, C_FN = 1, C_INC = 0,3).")


if __name__ == "__main__":
    main()
