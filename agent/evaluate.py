"""Avaliação final (c23, teste oficial do FF++): ablação, estruturas da rede, sensibilidade dos custos.

  python agent/evaluate.py c23       # ablação CNN x Naive Bayes x rede completa, estrutura Hill-Climbing/BIC, custos
  python agent/evaluate.py c40       # robustez no proxy de c40 (recompressão CRF 40) e nó COMP (precisa de make_c40_proxy.py)

Tudo com os splits oficiais por vídeo; CPTs ajustadas na validação; cortes das evidências dos tercis do treino;
IC95% por bootstrap agrupado por vídeo alvo (1.000 reamostragens, seed 0), com diferenças pareadas.
"""
import json, os, sys, itertools
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import core, build_model as bm

PI = core.PI; D = f"{PI}/derived"; OUT = f"{D}/agent"
P = json.load(open(core.PARAMS)); CUTS = {k: np.array(v) for k, v in P["evidence_cuts"].items()}
NODES = P["nodes"]; ESS = P["equivalent_sample_size"]
C_FP, C_FN, C_INC = 3.0, 1.0, 0.3


def rule(p, c_fp=C_FP, c_fn=C_FN, c_inc=C_INC):
    """0 = real, 1 = fake, -1 = inconclusivo (mesma regra de core.decide, vetorizada)."""
    cr, cf = p * c_fn, (1 - p) * c_fp
    return np.where((cr <= cf) & (cr <= c_inc), 0, np.where((cf < cr) & (cf <= c_inc), 1, -1))


def metrics(y, p, dec, c_fp=C_FP, c_fn=C_FN, c_inc=C_INC):
    d = dec != -1
    fp = ((dec == 1) & (y == 0)).sum(); fn = ((dec == 0) & (y == 1)).sum()
    return dict(cobertura=d.mean(), acc_decididos=(dec[d] == y[d]).mean() if d.any() else np.nan,
                fp=int(fp), fn=int(fn), inconclusivos=int((~d).sum()), auc=roc_auc_score(y, p),
                custo=(c_fp * fp + c_fn * fn + c_inc * (~d).sum()) / len(y))


def fit_bn(df_disc, edges, nodes_all, ess=ESS):
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    sn = {n: list(range(5 if n == "CNN" else 3)) for n in nodes_all if n != "COMP"}
    sn["Fake"] = [0, 1]
    if "COMP" in nodes_all:
        sn["COMP"] = [0, 1]
    net = DiscreteBayesianNetwork(edges); net.add_nodes_from(nodes_all)
    est = BayesianEstimator(net, df_disc[list(nodes_all)], state_names=sn)
    net.add_cpds(*[est.estimate_cpd(n, prior_type="BDeu", equivalent_sample_size=ess) for n in nodes_all])
    net.check_model()
    return net


def posterior_fake(net, df_disc, use=None, extra_hide=None):
    """P(Fake=1 | evidências) para cada linha (inferência exata por eliminação de variáveis); agrupa linhas iguais."""
    from pgmpy.inference import VariableElimination
    ve = VariableElimination(net)
    use = [c for c in df_disc.columns if c != "Fake" and (use is None or c in use) and c in net.nodes()]
    keys = df_disc[use].astype(int).apply(tuple, axis=1)
    cache = {}
    for k in keys.unique():
        q = ve.query(["Fake"], evidence=dict(zip(use, k)), show_progress=False)
        cache[k] = float(q.values[list(q.state_names["Fake"]).index(1)])
    return keys.map(cache).values


def boot(y, cluster, systems, B=1000, seed=0, base=None):
    """IC95% por bootstrap de clusters. systems: nome -> (p, dec). Devolve DataFrame com IC e diferenças vs. base."""
    rng = np.random.default_rng(seed)
    cl = pd.Series(cluster); idx = {c: np.where(cl.values == c)[0] for c in cl.unique()}; keys = np.array(list(idx))
    stats = {n: [] for n in systems}
    for _ in range(B):
        ii = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        if len(np.unique(y[ii])) < 2:
            continue
        for n, (p, dec) in systems.items():
            stats[n].append(metrics(y[ii], p[ii], dec[ii]))
    res = {}
    for n in systems:
        s = pd.DataFrame(stats[n]); res[n] = {c: (s[c].quantile(.025), s[c].quantile(.975)) for c in ("auc", "custo", "cobertura", "acc_decididos")}
        if base and n != base:
            b = pd.DataFrame(stats[base])
            res[n]["d_auc"] = ((s.auc - b.auc).quantile(.025), (s.auc - b.auc).quantile(.975))
            res[n]["d_custo"] = ((s.custo - b.custo).quantile(.025), (s.custo - b.custo).quantile(.975))
    return res


def video_level(df, p, dec_fn):
    v = df.assign(p=p).groupby(["group", "video_id"]).agg(y=("label", "first"), p=("p", "mean"), cluster=("target_id", "first")).reset_index()
    return v, dec_fn(v.p.values)


def run_c23():
    f = bm.load_frames()
    tr, va, te = (f[f.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    for d in (tr, va, te):
        d["target_id"] = d.target_id.astype(str)
    Xv, Xt = bm.discretize_df(va, CUTS), bm.discretize_df(te, CUTS)
    Xv["Fake"] = va.label.values; y = te.label.values
    star = lambda nodes: [("Fake", n) for n in nodes]
    sys_p = {}
    sys_p["CNN calibrada, limiar 0,5 (sem abstenção)"] = (te.p_cnn_cal.values, (te.p_cnn_cal.values >= .5).astype(int))
    sys_p["CNN calibrada + regra de decisão"] = (te.p_cnn_cal.values, rule(te.p_cnn_cal.values))
    nb1 = fit_bn(Xv, star(["CNN"]), ["Fake", "CNN"]); p1 = posterior_fake(nb1, Xt)
    sys_p["Rede só com o nó CNN"] = (p1, rule(p1))
    full = fit_bn(Xv, star(NODES), ["Fake"] + NODES); pf = posterior_fake(full, Xt)
    sys_p["Rede completa (CNN + 4 evidências)"] = (pf, rule(pf))
    # estrutura aprendida por Hill-Climbing + BIC (componente de busca)
    from pgmpy.estimators import HillClimbSearch, BIC
    from pgmpy.models import DiscreteBayesianNetwork
    cols = ["Fake"] + NODES
    dag = HillClimbSearch(Xv[cols].astype("category")).estimate(scoring_method="bic-d", show_progress=False)
    edges_hc = sorted(dag.edges())
    hc = fit_bn(Xv, edges_hc, cols); ph = posterior_fake(hc, Xt)
    sys_p["Rede com estrutura Hill-Climbing/BIC"] = (ph, rule(ph))
    # BIC na validação das três estruturas (menor |BIC| = melhor; pgmpy devolve BIC com sinal de log-verossimilhança)
    bic = {}
    for nm, ed, nodes in [("Naive Bayes (Fake → todos)", star(NODES), cols), ("Hill-Climbing/BIC", edges_hc, cols)]:
        net = DiscreteBayesianNetwork(ed); net.add_nodes_from(nodes); bic[nm] = float(BIC(Xv[cols].astype("category")).score(net))
    # frame
    base = "CNN calibrada + regra de decisão"
    rows = []; ci = boot(y, te.target_id.values, sys_p, base=base)
    for n, (p, dec) in sys_p.items():
        m = metrics(y, p, dec); r = dict(sistema=n, nivel="frame", **m)
        r.update({f"{k}_ic": f"{v[0]:.4f}..{v[1]:.4f}" for k, v in ci[n].items()})
        rows.append(r)
    # vídeo
    vsys, vy = {}, None
    for n, (p, dec) in sys_p.items():
        v, _ = video_level(te, p, lambda q: q)
        vy = v.y.values; vcl = v.cluster.values
        pv = v.p.values
        vdec = (pv >= .5).astype(int) if "limiar" in n else rule(pv)
        vsys[n] = (pv, vdec)
    civ = boot(vy, vcl, vsys, base=base)
    for n, (p, dec) in vsys.items():
        m = metrics(vy, p, dec); r = dict(sistema=n, nivel="vídeo", **m)
        r.update({f"{k}_ic": f"{v[0]:.4f}..{v[1]:.4f}" for k, v in civ[n].items()})
        rows.append(r)
    ab = pd.DataFrame(rows); ab.to_csv(f"{OUT}/ablacao_c23.csv", index=False)
    json.dump(dict(edges_hill_climbing=edges_hc, bic_val=bic), open(f"{OUT}/estruturas.json", "w"), indent=1)
    # evidências: leave-one-out e evidência ausente
    loo = []
    for drop in [None] + NODES[1:]:
        use = [n for n in NODES if n != drop]
        net = fit_bn(Xv, star(use), ["Fake"] + use); p = posterior_fake(net, Xt)
        m = metrics(y, p, rule(p)); loo.append(dict(sem=drop or "(nenhum)", **m))
    pd.DataFrame(loo).to_csv(f"{OUT}/leave_one_out.csv", index=False)
    rng = np.random.default_rng(1); miss = []
    for frac in (0.0, 0.25, 0.5):
        Xm = Xt.copy()
        for n in NODES[1:]:
            Xm.loc[rng.random(len(Xm)) < frac, n] = -1
        use_rows = Xm.copy()
        # marginaliza por linha: consulta com apenas as colunas presentes
        from pgmpy.inference import VariableElimination
        ve = VariableElimination(full); cache = {}; pm = []
        for _, row in use_rows.iterrows():
            ev = {n: int(row[n]) for n in NODES if row[n] != -1}; k = tuple(sorted(ev.items()))
            if k not in cache:
                q = ve.query(["Fake"], evidence=ev, show_progress=False); cache[k] = float(q.values[list(q.state_names["Fake"]).index(1)])
            pm.append(cache[k])
        pm = np.array(pm); m = metrics(y, pm, rule(pm)); miss.append(dict(frac_ausente=frac, **m))
    pd.DataFrame(miss).to_csv(f"{OUT}/evidencia_ausente.csv", index=False)
    # sensibilidade dos custos e curva cobertura x acurácia
    sens = []
    for c_inc, c_fp in itertools.product([0.1, 0.2, 0.3, 0.5, 1.0], [1, 2, 3, 5, 10]):
        for n in ("CNN calibrada + regra de decisão", "Rede completa (CNN + 4 evidências)"):
            p = sys_p[n][0]; dec = rule(p, c_fp, 1.0, c_inc); m = metrics(y, p, dec, c_fp, 1.0, c_inc)
            sens.append(dict(sistema=n, c_inc=c_inc, c_fp=c_fp, **m))
    pd.DataFrame(sens).to_csv(f"{OUT}/sensibilidade_custos.csv", index=False)
    # risco x cobertura (classificação seletiva por confiança)
    rc = []
    for n in ("CNN calibrada + regra de decisão", "Rede completa (CNN + 4 evidências)", "Rede só com o nó CNN"):
        p = sys_p[n][0]; conf = np.maximum(p, 1 - p)
        for thr in np.unique(np.round(np.concatenate([np.linspace(0.5, 0.999, 120), [0.9, 0.95, 0.99]]), 4)):
            sel = conf >= thr
            if sel.sum() >= 50:
                rc.append(dict(sistema=n, limiar_conf=thr, cobertura=sel.mean(), acc=((p[sel] >= .5) == y[sel]).mean()))
    pd.DataFrame(rc).to_csv(f"{OUT}/risco_cobertura.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(ab[["sistema", "nivel", "cobertura", "acc_decididos", "fp", "fn", "inconclusivos", "auc", "custo", "auc_ic", "custo_ic"]].round(4).to_string())
    for n in sys_p:
        if "d_auc" in ci[n]:
            print(f"{n}: Δauc vs base {ci[n]['d_auc'][0]:+.4f}..{ci[n]['d_auc'][1]:+.4f}; Δcusto {ci[n]['d_custo'][0]:+.4f}..{ci[n]['d_custo'][1]:+.4f}")
    print("arestas Hill-Climbing:", edges_hc); print("BIC val:", bic)
    print(pd.DataFrame(loo).round(4).to_string()); print(pd.DataFrame(miss).round(4).to_string())


def run_c40():
    f = bm.load_frames(); f["target_id"] = f.target_id.astype(str)
    k = pd.read_parquet(f"{OUT}/c40proxy_frames.parquet"); k = k[k.detected == 1].copy()
    k["target_id"] = k.target_id.astype(str); k["video_id"] = k.video_id.astype(str)
    v23, t23 = (f[f.split == s].reset_index(drop=True) for s in ("val", "test"))
    v40, t40 = (k[k.split == s].reset_index(drop=True) for s in ("val", "test"))
    sh = pd.DataFrame({c: [f[f.split == "test"][c].median(), t40[c].median()] for c in ("hf_ratio", "blockiness", "det_prob", "eye_dist_rel", "p_cnn_cal")}, index=["c23", "c40_proxy"])
    print("medianas no teste:\n", sh.round(5).to_string())
    X = lambda d: bm.discretize_df(d, CUTS)
    star = lambda nodes: [("Fake", n) for n in nodes]
    # redes
    Xv23 = X(v23); Xv23["Fake"] = v23.label.values
    net23 = fit_bn(Xv23, star(NODES), ["Fake"] + NODES)
    Xv40 = X(v40); Xv40["Fake"] = v40.label.values
    mix = pd.concat([Xv23.assign(COMP=0), Xv40.assign(COMP=1)], ignore_index=True)
    net_mix = fit_bn(mix, star(NODES), ["Fake"] + NODES)                      # mistura sem COMP
    net_comp = fit_bn(mix, star(NODES) + [("COMP", "CNN")], ["Fake", "COMP"] + NODES)   # COMP -> CNN (estrutura de domínio)
    out = []
    for cond, te in [("c23", t23), ("c40 (proxy)", t40)]:
        Xt = X(te); y = te.label.values; comp = 0 if cond == "c23" else 1
        Xc = Xt.assign(COMP=comp)
        S = {"CNN calibrada, limiar 0,5": (te.p_cnn_cal.values, (te.p_cnn_cal.values >= .5).astype(int)),
             "CNN calibrada + regra": (te.p_cnn_cal.values, rule(te.p_cnn_cal.values))}
        p = posterior_fake(net23, Xt); S["Rede ajustada em c23"] = (p, rule(p))
        p = posterior_fake(net_mix, Xt); S["Rede ajustada em c23+c40, sem COMP"] = (p, rule(p))
        p = posterior_fake(net_comp, Xc); S["Rede ajustada em c23+c40, com COMP observado"] = (p, rule(p))
        ci = boot(y, te.target_id.values, S, base="CNN calibrada + regra")
        for n, (p, dec) in S.items():
            m = metrics(y, p, dec); r = dict(condicao=cond, sistema=n, nivel="frame", **m)
            r.update({f"{kk}_ic": f"{v[0]:.4f}..{v[1]:.4f}" for kk, v in ci[n].items()}); out.append(r)
        vs = {}
        for n, (p, dec) in S.items():
            v, _ = video_level(te, p, lambda q: q); pv = v.p.values
            vs[n] = (pv, (pv >= .5).astype(int) if "limiar" in n else rule(pv))
        civ = boot(v.y.values, v.cluster.values, vs, base="CNN calibrada + regra")
        for n, (p, dec) in vs.items():
            m = metrics(v.y.values, p, dec); r = dict(condicao=cond, sistema=n, nivel="vídeo", **m)
            r.update({f"{kk}_ic": f"{vv[0]:.4f}..{vv[1]:.4f}" for kk, vv in civ[n].items()}); out.append(r)
    df = pd.DataFrame(out); df.to_csv(f"{OUT}/robustez_c40proxy.csv", index=False)
    sh.to_csv(f"{OUT}/deslocamento_c40proxy.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(df[["condicao", "sistema", "nivel", "cobertura", "acc_decididos", "fp", "fn", "inconclusivos", "auc", "custo", "auc_ic", "custo_ic"]].round(4).to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "c23"
    run_c23() if mode == "c23" else run_c40()
