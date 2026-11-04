# AGENTS.md — diretrizes para agentes de IA e colaboradores

Leia o `README.md` antes. Este arquivo define as regras do projeto: o que nunca pode ser feito, onde cada coisa fica e como entregar mudanças.

## 1. Contexto em uma frase

Detector de deepfake facial: classificador de pixels (ML) + Rede Bayesiana (IA clássica) que combina a saída da CNN com evidências interpretáveis e pode responder "inconclusivo". Marco 2 em 08/11/2026; apresentação final em 16–19/11/2026. A nota depende de arquitetura, integração IA+ML, justificativa técnica e métricas honestas.

## 2. Regras científicas (não negociáveis)

1. **Divisão por vídeo.** A unidade independente é o vídeo (`video_id`), não o frame. Use sempre os splits oficiais do FF++ (`FaceForensics/splits`, coluna `split` do manifest). Qualquer script novo começa com um `assert` de que nenhum vídeo aparece em dois splits.
2. **Teste é só para relatar.** Escolha de hiperparâmetros, época, limiar, temperatura e discretização usa treino/validação. Nunca olhe o teste para decidir nada.
3. **Classe real = originais do FF++.** O CelebA não entra no treino (viés de origem comprovado na EDA: AUC 0,51 no FF++). Ele só serve para controles e para a análise ética, depois de igualar compressão e resolução.
4. **Sem variáveis derivadas da máscara.** A máscara de manipulação não existe na inferência (`mask_frac`, `tex_logratio`, `color_dist` ficam fora de qualquer modelo ou evidência).
5. **Probabilidades da CNN no treino são superconfiantes.** Não ajuste as tabelas da Rede Bayesiana nos frames de treino da CNN; use a validação (decisão D1 em `docs/divisao_tarefas_marco2.md`). A temperatura é ajustada só na validação.
6. **Discretização por quantis do treino.** Os cortes ficam em `bins.json` e são reaplicados, sem refit, em validação, teste, c40, Celeb-DF e CelebA.
7. **Intervalos de confiança** por bootstrap agrupado por vídeo alvo (`eda/baselines.py: cluster_ci`, 2.000 reamostragens, seed 0). Reutilize `baselines.metrics`, `video_level` e `cluster_ci`; não reimplemente métricas.
8. **Sementes fixas** (42). Resultado de uma única execução deve ser descrito como tal, sem alegar estabilidade entre sementes.
9. **Todo número reportado vem de um arquivo gerado por código** (`derived/agent/*.csv|json`). Não digite métricas de memória em documentos; gere o texto a partir dos arquivos (como `marco2_secao3_rascunho.md`).
10. **Relate o que não funcionou.** Evidência que não separa as classes (ex.: razão de textura pelo Laplaciano na EDA) entra no texto como resultado negativo, não é descartada em silêncio.
11. **Desempenho em distribuição não é generalização.** Toda métrica em FF++ c23/FaceSwap deve dizer isso até que c40 e Celeb-DF tenham sido medidos.

## 3. Estrutura e responsabilidades

| Pasta | Conteúdo | Trilha |
|---|---|---|
| `eda/` | Marco 1: extração, EDA, `baselines.py` | histórico; só corrija bugs |
| `ml/` | treino, calibração, tabela comparativa, métricas finais | P1 |
| `evidences/` (a criar) | BiSeNet, textura/ruído, geometria, compressão, discretização | P2 |
| `agent/` (a criar) | Rede Bayesiana (pgmpy), explicação, decisão, demo | P3 |
| `derived/agent/` | **contratos de interface** entre trilhas | quem produz o arquivo |
| `docs/` | planejamento, tarefas, dependências | todos |

Não edite a pasta de outra trilha sem avisar. Mudar o esquema de um arquivo de `derived/agent/` exige atualizar o contrato em `docs/divisao_tarefas_marco2.md` (seção 5) e avisar os consumidores.

**Chave de junção:** `crop_file` (coluna de `derived/manifest.csv`). Nunca use o índice de linha do manifest como chave entre arquivos.

## 4. Ambiente e comandos

- Python 3.11 com `torch`, `torchvision`, `scikit-learn`, `pandas`, `pyarrow`, `scipy`, `matplotlib`, `opencv-python`, `facenet-pytorch`; para a rede bayesiana, `pgmpy`.
- **Sem GPU/MPS neste computador.** Estime o tempo antes de treinar: ResNet-18 ≈17 img/s em CPU, EfficientNet-B0 ≈2 img/s. Rode treinos longos em segundo plano e com retomada por checkpoint.
- `DataLoader` com `num_workers=0` (workers com memória compartilhada falham aqui).
- Scripts de `ml/` usam `PI_DIR` (padrão: caminho local do projeto). Scripts de `eda/` têm o caminho absoluto fixo; ao tocar neles, troque por `PI_DIR`.
- Antes de qualquer treino: `python ml/train_cnn.py --smoke`. Depois de mudar a CNN: refaça, nessa ordem, `calibrate_export.py`, `compare_models.py`, `final_model_metrics.py`.

## 5. O que nunca vai para o git

Dados brutos (`Celeba/`, `FaceForensics/`), `derived/crops/`, pesos (`*.pt`, `*.pth`), `*.npz`, `derived/manifest.csv`, backups `.docx`, `.DS_Store`, credenciais e links de download do FF++/Celeb-DF. Versione código, configs, logs pequenos e os arquivos de contrato em `derived/agent/` (hoje < 1 MB). Se um arquivo passar de ≈5 MB, pare e pergunte.

## 6. Ética e uso dos dados

- Os rostos são de pessoas reais. Não redistribua imagens, não tente identificar pessoas e não gere deepfakes: o projeto é de detecção.
- Respeite os termos do FF++, CelebA e Celeb-DF; não publique recortes de rosto em relatórios além do necessário e sem rótulo de identidade.
- A análise de falsos positivos por atributo (sexo, idade, tom de pele) é obrigatória antes da apresentação final; diga explicitamente quando um grupo tem poucas amostras.
- A decisão final deve poder dizer "inconclusivo"; nunca apresente a probabilidade como veredito certo.

## 7. Estilo de código e documentos

- Código em Python, identificadores em inglês, docstrings e comentários em português (como no restante do repositório). Funções pequenas, sem dependências desnecessárias.
- Todo script de `ml/` tem docstring de topo com uso, entradas e saídas, e argumentos por `argparse`.
- Documentos e textos entregues (`.md`, `.docx`) em português do Brasil, tom técnico e direto.
- Figuras: título que afirma o resultado verdadeiro (verifique contra os dados), eixos legíveis, cores colorblind-safe, sem vermelho/verde como par opositor.
- Não sobrescreva os `.docx` dos marcos sem backup; trabalhe em cópia e junte o texto na entrega.

## 8. Fluxo de trabalho com git

- Branches por trilha: `p1-ml`, `p2-evid`, `p3-bn`; integração em `main` por merge/PR curto.
- Commits pequenos, no imperativo e em português, com prefixo da área: `ml: calibra CNN com temperature scaling`, `agent: adiciona estrutura Naive Bayes`, `docs: atualiza contrato de evidences`.
- Não faça `push --force`, não reescreva histórico compartilhado e não commite em `main` código que quebre `--smoke`.
- Antes de commitar, rode `git status` e confira que nada da seção 5 entrou.

## 9. Definição de pronto de uma tarefa

1. O script roda do zero com os comandos do README e gera o arquivo do contrato.
2. O arquivo de saída tem o esquema documentado e a chave `crop_file` sem NaN.
3. Os números do texto batem com o arquivo gerado.
4. As ressalvas (distribuição, tamanho da amostra, uma semente) estão escritas.
5. O consumidor da saída confirmou que consegue ler o arquivo.
