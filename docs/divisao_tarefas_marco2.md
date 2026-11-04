# Divisão de tarefas do Projeto Integrador (Marco 2 → Marco Final) — 3 pessoas

Projeto: detecção de deepfake facial com classificador de pixels (ML) + Rede Bayesiana (IA clássica).
Hoje: 03/11/2026 · Marco 2: **08/11** · Apresentação final: **16–19/11**.
Figura de apoio: `grafo_dependencias_marco2.png` · tabela completa: `tasks_marco2.csv`.

---

## 1. Resposta curta: existe restrição de dependência para delegar a 3 pessoas?

**Não há bloqueio estrutural, mas há uma dependência crítica e três pontos de junção.**

1. **O trabalho se divide em 3 trilhas quase independentes** (modelo ML, evidências, agente/Rede Bayesiana). Todas partem só do que já existe no repositório (`derived/manifest.csv`, os crops 256×256 e o baseline do Marco 1). Nenhuma precisa esperar outra para começar.
2. **A dependência crítica é o ajuste fino da CNN (A2 → A3).** É a única tarefa longa que não dá para paralelizar nem emprestar: sem `preds_cnn.parquet` (probabilidades calibradas) não há integração real (I1), nem tabela comparativa (A4), nem modelo final (A5), nem seção 3 do documento. Ela é o caminho crítico até 08/11.
3. **Três junções (fan-in) exigem que o produtor entregue antes do consumidor:**
   - `preds_cnn.parquet` (P1) → Rede Bayesiana real (P3, tarefa I1);
   - `evidences.parquet` (P2) → Rede Bayesiana real (P3, tarefa I1) e análise ética (P2, F4);
   - seções 1, 2 e 3 (P3, P1) → fechamento do `.docx` (D2).
4. **Como contornar:** P3 não espera ninguém para construir a Rede Bayesiana. Ele desenvolve com um *stand-in* (probabilidade da regressão logística do baseline + os atributos manuais da EDA, que já estão no `manifest.csv`) e troca a fonte de dados quando os arquivos reais chegarem. Isso só funciona se os **contratos de interface da seção 5 forem congelados no kickoff (T0)**.
5. **Folga pequena.** Com a estimativa abaixo, o Marco 2 fecha em **5,5 dias de trabalho para um prazo de 6** (03/11 a 08/11 inclusive, ≈4 h úteis por pessoa por dia). Se o fine-tuning atrasar mais de ~0,5 dia, aplique o plano de contingência da seção 7.
6. **A rubrica do Marco 2 pesa arquitetura, integração e justificativa (6 dos 10 pontos).** Esses itens dependem do *contrato* e da BN com stand-in, não do número final da CNN. Isso reduz o risco: mesmo com atraso do ML, os 6 pontos não ficam bloqueados.

---

## 2. Estado de partida (lido do repositório e do Marco 1)

- Feito: dataset, EDA, pipeline de crops (`derived/manifest.csv`, `derived/crops/`), baseline de regressão logística com três representações (LR sobre espectro escolhida: AUC de validação 0,794; AUC por frame 0,77 e por vídeo 0,79 no teste), controle de viés de origem (CelebA × FF++).
- **Não encontrado em `derived/`**: pesos de CNN com ajuste fino (só o `resnet18` congelado), FF++ c40, Celeb-DF, segmentação BiSeNet, evidências de textura/ruído, Rede Bayesiana. O cronograma 5.3 do Marco 1 previa a CNN e as evidências até 25/10, então **assuma que tudo isso ainda está em aberto** e confirme no kickoff.
- Chave de junção entre todos os arquivos: a coluna `crop_file` de `derived/manifest.csv` (um valor por imagem). Unidade independente de avaliação e de qualquer divisão: o **vídeo** (`video_id`).

---

## 3. Papéis

| Pessoa | Trilha | O que entrega | Perfil sugerido |
|---|---|---|---|
| **P1** | ML / modelo | CNN ajustada e calibrada, tabela comparativa, modelo final, diagrama de arquitetura, seção 1 e 3 do documento; depois c40 e Celeb-DF | Quem tem GPU (Colab) ou Mac com MPS e mais experiência com PyTorch |
| **P2** | Evidências | Segmentação BiSeNet, evidências de textura/ruído, geometria, compressão, discretização; fechamento do `.docx`; depois análise ética (e U-Net opcional) | Quem gosta de processamento de imagem e estatística |
| **P3** | Agente / Rede Bayesiana | Especificação e implementação da BN (pgmpy), estruturas alternativas, explicação, regra de decisão, integração; seção 2; depois ablação e demo | Quem conhece melhor a parte de IA da disciplina (probabilidade, inferência) |

Carga estimada (dias de ≈4 h): Marco 2 — P1 4.75, P2 4.25, P3 4.00 (+0,25 de kickoff para todos); fase final — P1 2.00, P2 2.50, P3 2.25 (+1,0 de slides para todos). Nota: o A2 (1,5 dia) é quase todo tempo de máquina, então P1 fica livre durante o treino e por isso assume o diagrama (C5) e a seção 1.

> As durações são estimativas minhas, não medidas. Revisem-nas no kickoff e reajustem a partir de quanto tempo 1 época da CNN realmente leva no hardware de vocês.

---

## 4. Tarefas por pessoa

### 4.0 Todos
| ID | Tarefa | Janela |
|---|---|---|
| T0 | Kickoff (30 min): congelar os contratos da seção 5, decidir D1–D3 da seção 6, criar branches e solicitar acesso ao **Celeb-DF v2** (o prazo de resposta é externo) e iniciar o download do **FF++ c40** | 03/11 |
| F7 | Slides, roteiro e ensaio da apresentação | 11/11 em diante |

### 4.1 P1 — ML / modelo
| ID | Fase | Tarefa | Depende de | Janela | Entregável |
|---|---|---|---|---|---|
| A1 | Marco 2 | Script de treino da CNN (dataloader sobre manifest.csv, aumentos JPEG/blur/resize, EfficientNet-B0) | T0 | 03/11–03/11 | `train_cnn.py rodando 1 batch` |
| A2 | Marco 2 | Ajuste fino da CNN em GPU (Colab) ou MPS | A1 | 03/11–05/11 | `pesos .pt + log de treino/val` |
| A3 | Marco 2 | Temperature scaling na validação + exportar preds_cnn.parquet (p_cnn_raw, p_cnn_cal) | A2 | 05/11–05/11 | `preds_cnn.parquet` |
| A4 | Marco 2 | Tabela comparativa de modelos (LR atributos, LR espectro, LR ResNet18, CNN) com Acc/F1/AUC/Brier + IC bootstrap | A3 | 05/11–06/11 | `tabela_modelos.csv` |
| C5 | Marco 2 | Diagrama de arquitetura do agente (todos os módulos e setas, com formato dos dados) a partir de bn_spec.md | C1 | 06/11–06/11 | `arquitetura.png` |
| A5 | Marco 2 | Escolher modelo final e reportar métricas no teste (por frame e por vídeo) | A4 | 06/11–06/11 | `metricas_modelo_final.json` |
| C6a | Marco 2 | Redigir seção 1 do Marco 2 (1.1 diagrama, 1.2 descrição dos módulos, 1.3 IA clássica); P2 fornece texto dos módulos de evidência | C5,B5 | 07/11–07/11 | `Seção 1 preenchida` |
| D1 | Marco 2 | Redigir seção 3 do Marco 2 (3.1, 3.2, 3.3) | A5 | 07/11–07/11 | `Seção 3 preenchida` |
| F1 | Final | Robustez em FF++ c40 (inferência CNN + baselines) | A3 | 08/11–08/11 | `métricas c40` |
| F2 | Final | Generalização cross-dataset no Celeb-DF v2 (CNN e agente) | F1 | 09/11–09/11 | `métricas Celeb-DF` |

### 4.2 P2 — Evidências
| ID | Fase | Tarefa | Depende de | Janela | Entregável |
|---|---|---|---|---|---|
| B1 | Marco 2 | Face parsing BiSeNet nos 20k crops FF++ (máscaras pele/borda/cabelo) | T0 | 03/11–04/11 | `masks/ (npz) por imagem` |
| B2 | Marco 2 | Evidências: espectro local e resíduo de ruído, pele × faixa de borda | B1 | 04/11–05/11 | `evid_textura.parquet` |
| B3 | Marco 2 | Evidências: geometria dos landmarks, confiança do detector, estimativa de compressão (bloco 8×8 / qualidade) | T0 | 05/11–05/11 | `evid_geom_comp.parquet` |
| B4 | Marco 2 | Discretização por quantis do treino e tabela evidences.parquet conforme contrato | B2,B3 | 06/11–06/11 | `evidences.parquet + bins.json` |
| B5 | Marco 2 | Validar evidências (AUC isolada, checagem de viés de origem) — insumo das seções 2.2/2.3 | B4 | 06/11–06/11 | `evid_report.md + figura` |
| D2 | Marco 2 | Seção 4 (plano até o Marco Final), identificação, junção das seções, revisão pela rubrica e fechamento do .docx | C6a,C6b,D1,I1 | 08/11–08/11 | `Marco2.docx final` |
| F6 | Final | (Opcional) U-Net de mapa de manipulação como evidência extra | B1 | 08/11–09/11 | `unet.pt + evid_unet` |
| F4 | Final | Análise ética: FPR por atributo do CelebA (sexo, idade, tom de pele) com compressão/resolução igualadas | I1,B4 | 10/11–10/11 | `fpr_por_grupo.csv + figura` |

### 4.3 P3 — Agente / Rede Bayesiana
| ID | Fase | Tarefa | Depende de | Janela | Entregável |
|---|---|---|---|---|---|
| C1 | Marco 2 | Especificar a Rede Bayesiana: nós, estados, estrutura de domínio, compressão como pai do nó CNN | T0 | 03/11–03/11 | `bn_spec.md` |
| C2 | Marco 2 | BN em pgmpy com stand-in (p do baseline LR + atributos manuais discretizados da EDA); inferência exata | C1 | 03/11–04/11 | `bn.py + CPTs de teste` |
| C4 | Marco 2 | Explicação por contribuição em log-odds + regra de decisão por utilidade esperada com 'inconclusivo' | C2 | 04/11–05/11 | `explain.py, decide.py` |
| C3 | Marco 2 | Estruturas alternativas: Naive Bayes e Hill-Climbing/BIC (componente de busca) | C2 | 05/11–05/11 | `comparação de estruturas` |
| I1 | Marco 2 | Integração real: BN com p_cnn_cal + evidences.parquet; CPTs ajustadas na validação; avaliação no teste | A3,B4,C3,C4 | 06/11–07/11 | `preds_agent.parquet + métricas` |
| C6b | Marco 2 | Redigir seção 2 do Marco 2 (interface ML→BN com exemplo numérico, justificativa de representação/inferência, limitações) | C2,C4,B5 | 07/11–07/11 | `Seção 2 preenchida` |
| F5 | Final | Demo interativa (Streamlit/Gradio): probabilidade, contribuição por evidência, 'inconclusivo' | I1,C4 | 07/11–09/11 | `app.py` |
| F3 | Final | Ablação: CNN sozinha × Naive Bayes × BN completa (por frame e por vídeo, IC bootstrap) | I1 | 09/11–09/11 | `tabela_ablacao.csv` |

Janela = início e fim previstos pelo escalonamento com 1 tarefa por pessoa por vez (`tasks_marco2.csv`). Tarefas "Final" que aparecem antes de 08/11 só devem começar depois que as do Marco 2 daquela pessoa estiverem concluídas.

---

## 5. Contratos de interface (congelar em T0)

Todos os arquivos usam `crop_file` como chave e vão para `derived/agent/`.

**5.1 `preds_cnn.parquet` — P1 → P3 (e A4/A5)**

| Coluna | Tipo | Descrição |
|---|---|---|
| `crop_file` | str | chave do `manifest.csv` |
| `video_id`, `split`, `label` | str, str, int | copiados do manifest (label 1 = fake) |
| `p_cnn_raw` | float [0,1] | saída da CNN antes da calibração |
| `p_cnn_cal` | float [0,1] | após temperature scaling (T ajustado só na validação) |
| `model_id` | str | ex.: `effb0_ft_v1` |

Cobre treino, validação e teste do FF++ c23. Enquanto não existir, P3 usa `p_baseline` (LR sobre espectro, mesmo esquema de colunas).

**5.2 `evidences.parquet` — P2 → P3**

| Coluna | Tipo | Estados / unidade |
|---|---|---|
| `crop_file` | str | chave |
| `ev_hf`, `ev_block`, `ev_tex`, `ev_noise`, `ev_geom`, `ev_det`, `ev_comp` | int (0..K−1) | estados discretos por quantis do **treino**; K = 3 para todos, para simplificar as CPTs |
| `*_raw` (opcional) | float | valor contínuo original, para auditoria |

`bins.json` guarda os cortes por evidência (obrigatório para reaplicar no CelebA, no c40 e no Celeb-DF). `ev_comp` descreve o nível de compressão (c23/c40 ou estimado) e é o pai do nó da CNN na rede.

**5.3 `bn_spec.md` — P3 → todos**

Lista de nós, estados e arestas da estrutura de domínio. P1 e P2 só precisam ler este arquivo para saber o formato que a BN espera; P1 usa-o no diagrama (C5).

**5.4 Saída do agente — P3 → P1/P2/doc**

`preds_agent.parquet`: `crop_file`, `p_fake` (posterior), `decisao` ∈ {fake, real, inconclusivo}, `log_odds_<evidencia>` (uma coluna por evidência).

---

## 6. Decisões para tomar no kickoff

| # | Decisão | Recomendação | Por quê |
|---|---|---|---|
| D1 | Em quais dados ajustar as CPTs da BN? | Ajustar na **validação** (video-split), com temperature scaling também na validação, e avaliar só no teste | Se a BN fosse ajustada no treino, `p_cnn` estaria superconfiante lá (a CNN viu esses frames) e a BN aprenderia a confiar demais nela. Risco residual: a validação é pequena (≈280 vídeos); registrar como limitação na seção 2.3 |
| D2 | Qual CNN e onde treinar? | EfficientNet-B0, entrada 224–256, 3–5 épocas, Colab com GPU; MPS como plano B | Cabe no tempo; o Xception fica para depois se sobrar tempo |
| D3 | Classe real | Originais do FF++ (como no baseline). CelebA **não** entra no treino | O controle do Marco 1 mostrou AUC de 0,514 no FF++ quando se treina com CelebA como real |

---

## 7. Riscos e contingência

| Risco | Gatilho | Ação |
|---|---|---|
| Fine-tuning mais lento que o previsto | A2 não termina até a noite de 05/11 | Usar o último checkpoint com menos épocas; se nem isso, congelar o backbone e treinar só a cabeça (**meça** se supera o baseline antes de adotá-lo) e registrar na seção 3.2 |
| CNN sem GPU disponível | Epoch > 15 min em MPS/CPU | Reduzir para 128×128 ou subamostrar 5 frames por vídeo; documentar |
| Evidências de textura (B2) não separam as classes | `B5` mostra AUC ≈ 0,5 (a razão de textura do Marco 1 já mostrou isso) | Manter só as evidências que funcionam (espectro/HF, bloco, compressão, geometria) e relatar a evidência negativa; a BN continua válida |
| I1 atrasa | Dados reais não chegam até 06/11 | Entregar o Marco 2 com a BN no stand-in e explicitar isso em 2.3 e 4; I1 passa para a fase final |
| Celeb-DF sem acesso | Sem resposta até 09/11 | Trocar por avaliação em c40 como robustez e registrar a limitação |
| Vazamento entre splits | Qualquer divisão por frame | Todas as divisões por `video_id`; checar com um assert no início de cada script |

---

## 8. Convenções de trabalho

- Um branch por pessoa (`p1-ml`, `p2-evid`, `p3-bn`); merge em `main` só de arquivos de dados dentro de `derived/agent/` e scripts do próprio dono.
- Pastas de código separadas: `ml/`, `evidences/`, `agent/` — ninguém edita a pasta de outro sem avisar.
- Semente fixa (42) e splits oficiais do FF++ em todo script.
- Sincronização: 15 min por dia, em duas perguntas ("o que entreguei", "o que estou esperando de quem"). Pontos de checagem: **05/11 à noite** (A3 e B4 prontos?) e **07/11 de manhã** (I1 rodou?).
- Congelamento do documento: 07/11 à noite; 08/11 só revisão e entrega.

---

## 9. Mapa da rubrica do Marco 2 → responsável

| Critério (0–2) | Seção do .docx | Quem escreve | Depende de |
|---|---|---|---|
| Arquitetura do agente | 1.1, 1.2 | P1 (C5, C6a), com texto dos módulos de evidência de P2 | `bn_spec.md` (C1), contratos 5.1–5.2 |
| Componente de IA | 1.3 | P1 (C6a) | C1, C3 |
| Integração IA + ML | 2.1 | P3 (C6b) | C2, C4 (exemplo numérico com o stand-in), contrato 5.1 |
| Justificativa técnica | 2.2, 2.3 | P3 (C6b) | B5 (evidências), D1 |
| Modelo final e métricas | 3.1–3.3 | P1 (D1) | A5 |
| Plano até o Marco Final / fechamento | 4 e revisão | P2 (D2) | C6a, C6b, D1, I1 |

---

## 10. Fase final (09–15/11)

| Tarefa | Dono | Observação |
|---|---|---|
| F1 Robustez c40 | P1 | requer o download iniciado em T0 |
| F2 Celeb-DF | P1 | depende do acesso aprovado |
| F3 Ablação (CNN × NB × BN completa) | P3 | usa `preds_agent.parquet` de cada estrutura |
| F4 Ética: taxa de falsos positivos por atributo do CelebA | P2 | **igualar compressão e resolução ao FF++** antes de medir |
| F5 Demo Streamlit/Gradio | P3 | mostrar p_fake, contribuição por evidência e "inconclusivo" |
| F6 U-Net (opcional) | P2 | só se F4 estiver pronto |
| F7 Slides e ensaio | todos | ensaiar a demo ao vivo pelo menos duas vezes |

Pelo escalonamento, todo o projeto cabe em ≈9 dias de trabalho (até 11/11), deixando ≈4 dias de margem antes de 15/11. Essa margem é para atrasos de ML, não para novas funcionalidades.
