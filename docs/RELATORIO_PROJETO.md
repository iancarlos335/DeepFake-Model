# Relatório do projeto: detecção de deepfake facial com CNN + Rede Bayesiana

Projeto Integrador · Ciência da Computação, 8º semestre · Inteligência Artificial + Ciência de Dados · 2º semestre de 2026.

Este relatório resume o que existe no repositório, como foi feito, o que os resultados mostram e o que ficou em aberto. Todos os números vêm de arquivos gerados por código em `derived/agent/` e `derived/cnn/`; o arquivo de origem é citado junto de cada tabela. Quando algo foi aproximado, substituído ou não foi medido, isso está dito no texto.

## 1. Resumo

O agente recebe a imagem de um rosto e devolve **fake**, **real** ou **inconclusivo**, com P(fake) e a contribuição de cada evidência. Um classificador de pixels (ResNet-18 ajustada) fornece a evidência principal; uma Rede Bayesiana (pgmpy) a combina com quatro evidências interpretáveis, e uma regra de custo esperado decide a resposta.

Principais resultados no teste oficial do FF++ c23 (280 vídeos, 2.800 frames, 50% fake):

- A ResNet-18 ajustada e calibrada chega a **AUC 0,993 por frame** (IC95% 0,988 a 0,997) e **98,6% de acurácia por vídeo**, contra 0,769 e 73,2% do baseline do Marco 1.
- A Rede Bayesiana **não reduz o custo esperado** em relação à CNN com a mesma regra de decisão (0,0405 contra 0,0404; diferença pareada de −0,003 a +0,003). Ela acrescenta explicação por evidência, política de decisão explícita e tolerância a evidência ausente.
- Sob compressão forte (proxy de c40) a AUC da CNN cai de 0,993 para 0,859, e a rede não recupera.
- Falsos positivos no CelebA (rostos reais) são mais frequentes em rostos femininos e jovens (condição igualada: 1,9% contra 0,5% e 1,6% contra 0,5%).

## 2. O que foi feito, por etapa

| Etapa | O que entrou no repositório |
|---|---|
| Marco 1 | Extração de frames, recortes e atributos (`eda/extract_features.py`), EDA, baseline por regressão logística (`eda/baselines.py`), notebook e figuras |
| Marco 2 | Classificador de pixels: treino, calibração, tabela comparativa e métricas finais (`ml/`); divisão de tarefas e dependências (`docs/`); README e AGENTS |
| Marco Final | Agente completo e aplicação web (`agent/`), avaliação final (ablação, estruturas, robustez, ética), apresentação (`docs/Apresentacao_Marco_Final.pptx`) |

## 3. Estrutura do repositório

```
eda/        Marco 1: extração, EDA, baselines.py, notebook
ml/         treino da CNN, calibração, tabela de modelos, métricas finais
agent/      pipeline do agente, Rede Bayesiana, app Streamlit, avaliação final
docs/       divisão de tarefas, grafo de dependências, apresentação, este relatório
derived/    saídas geradas (agent/: contratos e resultados; cnn/: logs e configs do treino)
Celeba/, FaceForensics/   dados brutos (não versionados)
```

## 4. Dados e pré-processamento

- **FaceForensics++ (c23):** 1.000 vídeos originais (classe real) e 1.000 vídeos FaceSwap (classe fake), divisão oficial por vídeo (720/140/140 por classe). De cada vídeo, 10 frames uniformemente espaçados.
- **Detecção e recorte:** MTCNN, maior rosto, recorte quadrado com margem 1,3× e redimensionado para 256×256. Ficaram 14.398 frames de treino, 2.800 de validação e 2.800 de teste (2 frames em 20.000 sem rosto detectado).
- **CelebA:** só em controles e na análise ética; não entra no treino, porque a EDA mostrou viés de origem (treinar com CelebA como real dá AUC 0,51 no FF++).
- **Variáveis da máscara de manipulação** foram excluídas de qualquer modelo, porque a máscara não existe na hora de usar.
- **Unidade independente:** o vídeo. Divisões e intervalos de confiança (bootstrap agrupado por vídeo alvo) respeitam isso.

## 5. Classificador de pixels (ML)

ResNet-18 pré-treinada no ImageNet e ajustada em todas as camadas (`ml/train_cnn.py`): 6 épocas, 5 frames por vídeo por época, AdamW (decaimento 1e-4), taxa de aprendizado em ciclo único com máximo de 3e-4, lote 32, entrada 224×224 com média e desvio do ImageNet, perda de entropia cruzada binária. Aumentos de dados só no treino: espelhamento, redução e ampliação, desfoque gaussiano e recompressão JPEG (qualidade 50 a 95). Treino em CPU, cerca de 46 minutos. Checkpoint escolhido pela maior AUC de validação (época 6, 0,9972); o teste não foi usado em nenhuma escolha. A saída foi calibrada por temperature scaling, com T = 1,158 ajustada só na validação.

A EfficientNet-B0 prevista no Marco 1 foi descartada: sem GPU treinava a cerca de 2 imagens por segundo contra cerca de 17 da ResNet-18.

Teste oficial do FF++ c23 (`derived/agent/tabela_modelos.csv`):

| Modelo | AUC por frame (IC95%) | Acurácia por frame | Acurácia por vídeo |
|---|---|---|---|
| LR, 8 atributos manuais | 0,705 (0,665 a 0,747) | 0,655 | 0,668 |
| LR, espectro radial (baseline) | 0,769 (0,731 a 0,805) | 0,725 | 0,732 |
| LR, ResNet-18 congelada | 0,690 (0,663 a 0,721) | 0,640 | 0,675 |
| **ResNet-18 ajustada, calibrada** | **0,993 (0,988 a 0,997)** | **0,968** | **0,986** |

Modelo final (`metricas_modelo_final.json`): por frame, acurácia 0,9675, F1 0,9677, AUC 0,9930, Brier 0,0269, ECE 0,0161; por vídeo, acurácia 0,9857 (138 de 140 em cada classe) e AUC 0,9987. O temperature scaling quase não mudou a calibração (ECE de 0,018 para 0,016). Aplicada a 984 rostos reais do CelebA (teste), a CNN classificou 9,8% como fake (`derived/cnn/resnet18_ft_v1/celeba_real_check.json`).

## 6. Agente (IA clássica + ML)

Pipeline em `agent/core.py`, com nove etapas visíveis na aplicação:

1. leitura; 2. detecção do rosto (MTCNN); 3. recorte 256×256; 4. CNN com Grad-CAM; 5. quatro evidências (`hf_ratio`, `blockiness`, `det_prob`, `eye_dist_rel`); 6. discretização; 7. Rede Bayesiana; 8. explicação em log-odds; 9. decisão por custo esperado.

- **Discretização:** a saída da CNN vira 5 faixas fixas (0,1, 0,3, 0,7, 0,9); cada evidência vira 3 estados (baixo, médio, alto) pelos tercis dos frames de treino, calculados sem usar o rótulo e reaplicados sem refazer. Cortes em `agent/model_params.json`.
- **Rede:** Fake → CNN, HF, BLOCK, DET, GEOM. Tabelas por contagem na **validação** (2.800 frames, 280 vídeos) com prior BDeu de tamanho equivalente 5, porque no treino a CNN é superconfiante (perda 0,013 no treino contra 0,068 na validação). Inferência exata por eliminação de variáveis (pgmpy); a explicação em log-odds é conferida contra o pgmpy a cada consulta.
- **Decisão:** custo esperado com C_FP = 3, C_FN = 1 e C_INC = 0,3, o que dá fake se P > 0,9, real se P < 0,3 e inconclusivo no meio. É uma política definida antes de olhar o teste, e não uma verdade estatística; os custos são ajustáveis no app.
- **Verificação (`agent/verify.py`, 21 frames do teste):** os recortes do agente são idênticos aos do manifest (100%), as evidências coincidem exatamente e a diferença máxima em p_cnn_cal é 1,7e-06.

### Aplicação

`streamlit run agent/app.py`. Aceita imagens e vídeos, mostra cada etapa e a decisão final, e baixa um resumo em CSV. O teste foi feito de forma automática, com o framework de testes do Streamlit: 4 imagens, 1 vídeo e 1 imagem sem rosto, sem erros, e as decisões coincidem com o gabarito. **Não foi aberta num navegador** no ambiente de desenvolvimento, que não permite abrir portas; vale um ensaio na máquina da apresentação. As amostras de demonstração (`agent/demo_samples`) não são versionadas.

## 7. Avaliação final (teste do FF++ c23)

Por frame, com a mesma regra de decisão (`derived/agent/ablacao_c23.csv`):

| Sistema | Cobertura | Acurácia nos decididos | FP | FN | Inconclusivos | AUC | Custo (IC95%) |
|---|---|---|---|---|---|---|---|
| CNN, limiar 0,5 (sem abstenção) | 1,000 | 0,9675 | 56 | 35 | 0 | 0,9930 | 0,0725 (0,044 a 0,106) |
| CNN + regra de decisão | 0,951 | 0,9842 | 15 | 27 | 137 | 0,9930 | 0,0404 (0,025 a 0,058) |
| Rede só com o nó CNN | 0,951 | 0,9842 | 15 | 27 | 137 | 0,9862 | 0,0404 |
| Rede completa (CNN + 4 evidências) | 0,965 | 0,9800 | 15 | 39 | 98 | 0,9887 | 0,0405 (0,024 a 0,059) |
| Rede com estrutura Hill-Climbing/BIC | 0,965 | 0,9800 | 15 | 39 | 98 | 0,9888 | 0,0405 |

- Abster-se reduz o custo de 0,072 para 0,040. A rede completa não o reduz mais; a AUC dela é um pouco menor que a da CNN, em parte pela discretização em 5 faixas.
- **Estrutura aprendida (`estruturas.json`):** o Hill-Climbing com BIC ligou CNN→Fake, Fake→HF, Fake→BLOCK e DET–GEOM. BIC na validação: −15.028 contra −15.126 do Naive Bayes. A decisão no teste é a mesma.
- **Retirar uma evidência (`leave_one_out.csv`):** a AUC fica entre 0,988 e 0,989 e o custo entre 0,0405 e 0,0408 em todos os casos; nenhuma das quatro evidências provisórias muda o resultado.
- **Evidência ausente (`evidencia_ausente.csv`):** com 25% e 50% das evidências sorteadas como ausentes, a AUC é 0,9877 e 0,9887 (contra 0,9887) e o custo 0,0402 e 0,0407.

### Robustez à compressão (proxy de c40)

Os vídeos c23 de validação e teste foram recomprimidos em H.264 CRF 40 (`agent/make_c40_proxy.py`). **Isto é um proxy, não o c40 oficial**, que parte do vídeo bruto (`robustez_c40proxy.csv`):

| Sistema (frame, c40 proxy) | Cobertura | Acurácia nos decididos | AUC | Custo |
|---|---|---|---|---|
| CNN, limiar 0,5 | 1,000 | 0,763 | 0,859 | 0,371 |
| CNN + regra | 0,722 | 0,816 | 0,859 | 0,244 |
| Rede ajustada em c23 | 0,848 | 0,738 | 0,829 | 0,296 |
| Rede ajustada em c23+c40, sem COMP | 0,658 | 0,838 | 0,833 | 0,237 |
| Rede ajustada em c23+c40, com COMP observado | 0,581 | 0,841 | 0,833 | 0,239 |

O nó de compressão (COMP) aumenta a acurácia nos frames decididos, mas à custa de cobertura e sem reduzir o custo. No teste usou-se a compressão como conhecida; estimá-la automaticamente não foi implementado. A mediana de `blockiness` passa de 1,026 (c23) para 1,235 (proxy), o que desloca os estados discretizados.

### Falsos positivos por grupo (CelebA)

Todas as imagens do CelebA são rostos reais; logo, acusar de fake é sempre falso positivo (`fpr_por_grupo.csv`, rede completa). Na condição "igualada" o rosto foi redimensionado para cerca de 160 px e recomprimido em H.264 CRF 23, uma aproximação do FF++:

| Grupo (condição igualada) | n | Falsos positivos (IC95%) |
|---|---|---|
| Todos (amostra) | 4.252 | 1,4% (1,1 a 1,8) |
| Feminino / Masculino | 2.576 / 1.676 | 1,9% (1,5 a 2,6) / 0,5% (0,3 a 1,0) |
| Jovem / Não jovem | 3.378 / 874 | 1,6% (1,3 a 2,1) / 0,5% (0,2 a 1,2) |
| Pele pálida: sim / não | 442 / 3.810 | 1,6% (0,8 a 3,2) / 1,4% (1,0 a 1,8) |

Sem diferença detectável para pele pálida. Os atributos do CelebA são binários, grosseiros e correlacionados, e não medem tom de pele, então os resultados não provam causa. O FF++ não tem rótulos demográficos, de modo que o viés não pode ser medido nele.

## 8. Desvios do plano e decisões

- **ResNet-18 em vez de EfficientNet-B0**, por velocidade de treino em CPU.
- **c40 por proxy** e **Celeb-DF não avaliado** (sem acesso aos dados).
- **Evidências definitivas** (segmentação BiSeNet, textura, ruído, compressão estimada) especificadas no Marco 2, mas **não integradas**; o agente usa as quatro evidências provisórias do Marco 1.
- **Hill-Climbing/BIC** foi executado na avaliação, mas não está no aplicativo.
- A rede em `agent/` reproduz exatamente os números da Seção 2 do Marco 2 (texto da trilha da Rede Bayesiana); o código original dessa trilha não estava no repositório.

## 9. Limitações

- Desempenho medido **em distribuição** (FaceSwap, c23); a generalização para outros métodos e para o Celeb-DF não foi medida.
- **Uma única execução** de treino (semente 42), sem variância entre sementes. Sem ablação dos aumentos de dados nem busca de hiperparâmetros.
- Início de sobreajuste: depois da época 3 a perda de treino continua caindo e a de validação oscila entre 0,068 e 0,095.
- 280 vídeos de validação; frames do mesmo vídeo são quase repetidos, então o tamanho efetivo da amostra é menor que 2.800.
- Os tercis dependem do c23, e DET quase não varia (os cortes caem na quarta e na quinta casa decimal).
- O ponto de partida é o ImageNet, que não é de rostos. Não treinamos com pesos aleatórios, então não sabemos quanto o pré-treino ajudou.
- A saída é uma estimativa para triagem, com decisão humana; não é prova.

## 10. Reprodução

```bash
python eda/extract_features.py && python eda/extract_cnn_features.py   # dados e baselines (precisa dos dados brutos)
python ml/train_cnn.py --epochs 6 --fpv 5                               # treino (~46 min em CPU)
python ml/calibrate_export.py --run resnet18_ft_v1                      # temperature scaling e preds_cnn.parquet
python ml/compare_models.py && python ml/final_model_metrics.py         # tabela e métricas finais
python agent/build_model.py                                             # CPTs da rede -> agent/model_params.json
python agent/verify.py                                                  # confere o pipeline contra o manifest
python agent/evaluate.py c23                                            # ablação, estrutura, evidência ausente, custos
python agent/make_c40_proxy.py && python agent/evaluate.py c40          # proxy de c40
python agent/ethics_celeba.py                                           # falsos positivos por grupo
streamlit run agent/app.py                                              # aplicação
```

Não versionados: dados brutos, `derived/crops`, `derived/manifest.csv`, `*.npz`, pesos (`*.pt`, `*.pth`, incluindo `derived/cnn/resnet18_ft_v1/best.pt`, necessário para o app) e `agent/demo_samples`. Veja `README.md` e `AGENTS.md` para as regras do projeto.

## 11. Pendências

1. Integrar as evidências definitivas e estimar a compressão automaticamente.
2. Validar no c40 oficial e no Celeb-DF.
3. Repetir o treino com várias sementes e mais compressão nos aumentos, em GPU.
4. Levar o Hill-Climbing/BIC ao aplicativo.
5. Ensaiar a demonstração no computador da apresentação e gravar o vídeo de plano B.
