# Detecção de deepfake facial com ML + Rede Bayesiana

Projeto Integrador · Ciência da Computação, 8º semestre · Inteligência Artificial + Ciência de Dados · 2º semestre de 2026.

O sistema recebe uma imagem de rosto e devolve a probabilidade de ela ser *fake*, junto com a **justificativa**: um classificador de pixels (ML) é combinado, dentro de uma **Rede Bayesiana** (IA clássica), com evidências interpretáveis (textura, compressão, geometria dos landmarks, confiança do detector). A decisão final usa utilidade esperada e pode responder **"inconclusivo"**.

Escopo: imagem estática de rosto; fakes de *face swap* do FaceForensics++ (FaceSwap, compressão c23) e rostos reais do próprio FF++. Não cobre vídeo, áudio nem voz clonada.

## Estado atual (Marco 2, 08/11/2026)

| Componente | Estado |
|---|---|
| Dados, EDA, crops 256×256, manifest | pronto (Marco 1) |
| Baseline (regressão logística sobre espectro radial) | pronto (Marco 1) |
| Classificador de pixels (ResNet-18 ajustada + calibração) | **pronto** |
| Evidências (BiSeNet, textura/ruído, geometria, compressão) | em aberto |
| Rede Bayesiana (pgmpy), explicação e decisão | em aberto |
| Robustez c40, Celeb-DF, ablação, análise ética, demo | fase final (09–15/11) |

Resultados do classificador no teste oficial do FF++ c23 (280 vídeos, 2.800 frames, limiar 0,5, bootstrap agrupado por vídeo alvo):

| Modelo | AUC por frame | Acurácia por vídeo |
|---|---|---|
| LR + 8 atributos manuais | 0,705 | 0,668 |
| LR + espectro radial (baseline) | 0,769 | 0,732 |
| LR + ResNet-18 congelada | 0,690 | 0,675 |
| **ResNet-18 ajustada, calibrada** | **0,993** (IC95% 0,988–0,997) | **0,986** |

Isso é desempenho **em distribuição** (mesmo método e mesma compressão do treino). Robustez em c40 e generalização no Celeb-DF ainda não foram medidas. Detalhes e ressalvas: `derived/agent/marco2_secao3_rascunho.md`.

## Estrutura do repositório

```
eda/        Marco 1: extração de features/crops, EDA, baseline (baselines.py), geração do notebook
ml/         Marco 2: treino da CNN, calibração, tabela comparativa, métricas do modelo final
docs/       divisão de tarefas, grafo de dependências e tabela de tarefas do Marco 2
derived/    saídas geradas (manifest, features, pesos, predições)
  agent/    arquivos que cruzam fronteiras entre trilhas (contratos de interface)
  cnn/      execuções de treino (config, log, calibração; os .pt não são versionados)
Celeba/, FaceForensics/   dados brutos (não versionados)
```

## Dados (não versionados)

- **FaceForensics++** (Rössler et al., 2019), compressão c23: vídeos originais, FaceSwap e máscaras, com a divisão oficial em `FaceForensics/splits/{train,val,test}.json` (720/140/140 vídeos). Baixado com o script oficial (os termos de uso do FF++ proíbem redistribuição).
- **CelebA** (Liu et al., 2015), versão do Kaggle: usado só em controles e na análise ética, **nunca como classe real do treino** (a EDA mostrou viés de origem: um classificador treinado assim tem AUC 0,51 no FF++).
- **Celeb-DF v2**: cross-dataset da fase final; exige pedido de acesso aos autores.

`derived/crops/` (≈2,4 GB) e `derived/manifest.csv` são regenerados por `eda/extract_features.py` a partir dos dados brutos.

## Ambiente

Python 3.11, `torch`, `torchvision`, `scikit-learn`, `pandas`, `pyarrow`, `scipy`, `matplotlib`, `opencv-python`, `facenet-pytorch` (MTCNN). Os scripts de `ml/` aceitam `PI_DIR` para apontar a raiz do projeto; os de `eda/` ainda têm o caminho absoluto fixo no topo do arquivo.

O treino foi feito **em CPU** (sem GPU/MPS neste computador): a EfficientNet-B0 treinou a ≈2 img/s e a ResNet-18 a ≈17 img/s, por isso a ResNet-18 é o classificador atual. Os `DataLoader`s usam `num_workers=0` porque workers com memória compartilhada falharam neste ambiente.

## Como reproduzir

```bash
# 1) Marco 1: crops, manifest, features (precisa dos dados brutos)
python eda/extract_features.py
python eda/extract_cnn_features.py          # embeddings da ResNet-18 congelada

# 2) Marco 2: classificador de pixels
python ml/train_cnn.py --smoke              # valida o pipeline (2 batches)
python ml/train_cnn.py --epochs 6 --fpv 5   # ≈46 min em CPU; retoma do último checkpoint
python ml/calibrate_export.py --run resnet18_ft_v1     # T na validação -> derived/agent/preds_cnn.parquet
python ml/compare_models.py                 # tabela comparativa (reajusta as 3 LR do baseline)
python ml/final_model_metrics.py            # métricas exatas do teste + figura
```

Sementes fixas (42) e divisões oficiais do FF++ em todos os scripts.

## Contratos de interface entre trilhas

Todos os arquivos usam `crop_file` (coluna de `derived/manifest.csv`) como chave e ficam em `derived/agent/`.

| Arquivo | Produtor → consumidor | Conteúdo |
|---|---|---|
| `preds_cnn.parquet` | ML → Rede Bayesiana | `crop_file, video_id, split, label, p_cnn_raw, p_cnn_cal, logit, model_id` |
| `evidences.parquet` + `bins.json` | Evidências → Rede Bayesiana | evidências discretizadas por quantis do **treino** (3 estados) |
| `preds_agent.parquet` | Rede Bayesiana → avaliação/demo | `p_fake`, `decisao` (fake/real/inconclusivo), contribuição por evidência em log-odds |

Especificação completa e decisões de projeto: `docs/divisao_tarefas_marco2.md` (seções 5 e 6).

## Equipe e trilhas

Três trilhas quase independentes: **P1** ML/modelo, **P2** evidências, **P3** agente/Rede Bayesiana. Cronograma, dependências e riscos: `docs/divisao_tarefas_marco2.md`, `docs/tasks_marco2.csv` e `docs/grafo_dependencias_marco2.png`.

## Ética

- Os datasets usam rostos de pessoas reais (YouTube, celebridades); respeitem os termos de uso e não redistribuam imagens.
- A análise de **falsos positivos por grupo** (sexo, idade, tom de pele, no CelebA, com compressão e resolução igualadas ao FF++) é parte obrigatória da fase final. O FF++ não tem rótulos demográficos, então o viés não pode ser medido nele.
- Acusar um rosto real de ser fake e deixar passar um fake têm custos diferentes; a regra de decisão com opção "inconclusivo" existe para isso.
- Os mesmos detectores faciais servem para gerar e para detectar deepfakes; o projeto é de detecção.

## Referências principais

Rössler et al. (FaceForensics++, ICCV 2019) · Li et al. (Celeb-DF, CVPR 2020) · Dolhansky et al. (DFDC, 2020) · Zhang et al. (MTCNN, 2016) · Yu et al. (BiSeNet, ECCV 2018). Levantamento completo no documento do Marco 1.
