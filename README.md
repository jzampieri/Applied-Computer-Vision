# CP5 — Detecção de Ações de Pessoas com YOLO + U-Net + SlowFast

Applied Computer Vision 2026 — CheckPoint 5 (Prof. Dr. Paulo Sergio Rodrigues).

Pipeline que lê um vídeo curto e mostra, em cada trecho, **onde está a pessoa**
(YOLO), **quais pixels pertencem a ela** (U-Net) e **qual ação está realizando**
(SlowFast). Todos os modelos são **pré-treinados** — nenhum treinamento
adicional é feito, conforme a restrição do enunciado.

## Resultado

| Entregável | Arquivo |
|---|---|
| Vídeo de demonstração com as três saídas sobrepostas | `outputs/<video>/overlay.mp4` |
| Relatório (≤3 páginas) | `report/CP5_relatorio.pdf` |
| Métricas por momento analisado | `report/metrics.csv` |
| Agregados por vídeo | `report/summary.json` |
| Painéis de comparação e gráficos | `report/panels/` |

## Modelos

| Etapa | Modelo | Pesos |
|---|---|---|
| Detecção | **YOLO11n** (`ultralytics`), classe `person`, tracker ByteTrack | `yolo11n.pt` |
| Segmentação | **U²-Net** `u2net_human_seg` via ONNX Runtime | `u2net_human_seg.onnx` |
| Ação | **SlowFast R50 8×8** (`pytorchvideo`), Kinetics-400 | `SLOWFAST_8x8_R50.pyth` |

**Por que U²-Net como "a U-Net".** U²-Net é arquitetura da família U-Net —
encoder–decoder com *skip connections*, construído com blocos RSU em estrutura
U aninhada — e o checkpoint `u2net_human_seg` é pré-treinado especificamente
para segmentação de pessoas. A alternativa óbvia (`Unet` do
`segmentation_models_pytorch`) tem apenas o *encoder* pré-treinado no ImageNet e
o *decoder* inicializado aleatoriamente: sem treinar, produz ruído — exatamente
o que o enunciado proíbe. O estágio é plugável
(`src/stage_segment.py`), bastando trocar o backend caso haja um checkpoint
U-Net específico da disciplina.

## Instalação

Ambiente de referência: Ubuntu 26.04, Python 3.14, **CPU-only**, 8 núcleos,
6 GB de RAM. Nenhuma GPU é necessária.

```bash
bash scripts/setup_env.sh          # cria .venv, instala tudo e roda o smoke test
source .venv/bin/activate
python scripts/fetch_weights.py    # ~460 MB de pesos pré-treinados
python scripts/fetch_videos.py     # baixa, recorta e normaliza os vídeos
```

`scripts/setup_env.sh` instala o `pytorchvideo` a partir do repositório
(`--no-deps`): o release do PyPI (0.1.5, de 2022) importa
`torchvision.transforms.functional_tensor`, removido no torchvision moderno. O
pré-processamento do SlowFast é implementado em `src/stage_action.py`, então
`pytorchvideo.transforms` nunca é usado.

## Execução

```bash
python -m src.cli all      --video archery           # A→B→C(person+full)→D
python -m src.cli all      --video all               # os três vídeos
python -m src.cli analyze                            # métricas, momentos e gráficos
python -m src.cli report                             # report/CP5_relatorio.pdf
```

Estágios isolados (o enunciado pede que os três modelos rodem separadamente):

```bash
python -m src.cli detect  --video pushups
python -m src.cli segment --video pushups --threshold 0.5
python -m src.cli action  --video pushups --roi person   # ou --roi full
python -m src.cli render  --video pushups
```

## Arquitetura

Cada estágio grava artefatos em disco e pode ser reexecutado sozinho. Isso
mantém **um modelo por vez na memória** — necessário numa máquina sem GPU e com
6 GB de RAM — e torna barato reprocessar só a etapa alterada.

```
data/videos/<nome>.mp4
  │
  ├─ A. stage_detect ── YOLO11n + ByteTrack ─────────► detections.json
  │                     bbox, confiança e track_id por frame
  │
  ├─ B. stage_segment ─ U²-Net no recorte de cada bbox (padding 10%),
  │                     máscara reprojetada no frame ► masks/%06d.png + mask_stats.json
  │
  ├─ C. stage_action ── SlowFast em janelas de 32 frames, passo 16
  │                     --roi person | full ─────────► actions_person.json / actions_full.json
  │
  └─ D. stage_render ── bbox + máscara + rótulo de ação + faixa temporal ► overlay.mp4
```

Detalhes que importam:

- **Reprojeção da máscara.** O recorte da bbox é expandido 10% antes de ir para
  a U-Net. Sem essa folga a máscara encosta na borda do recorte e o contorno sai
  cortado, porque a rede não vê o entorno do sujeito.
- **Janelas temporais.** 32 frames por janela com passo 16 (50% de
  sobreposição). O *fast pathway* recebe os 32 frames; o *slow pathway* recebe a
  subamostragem por `alpha=4` — é esse par `[slow, fast]` que o modelo espera.
- **Rótulo por frame.** Cada frame herda a ação da janela que o contém; nas
  regiões de sobreposição vence a janela de maior confiança.
- **`--roi person`.** Recorta a união das bboxes da janela, de forma que o
  sujeito preencha o quadro. A comparação contra `--roi full` é a extensão
  opcional do enunciado.

## Vídeos

Nenhum vídeo era pré-selecionado. Os três escolhidos são de domínio público ou
Creative Commons, com procedência registrada em `data/videos/LICENSES.md`:

| Vídeo | Papel |
|---|---|
| `archery` | Caso limpo: ação do vocabulário Kinetics-400, sujeito estável |
| `pushups` | Fronteiras de ação: séries alternadas com descanso |
| `skate` | Casos de falha: múltiplas pessoas, oclusão, movimento rápido |

O enunciado exige dois vídeos; `archery` e `pushups` cumprem a exigência e
`skate` entra como evidência extra de falha.

## Custo de execução (CPU, sem GPU)

Medido na máquina de referência (Ubuntu 26.04, 8 núcleos, 6 GB de RAM, sem GPU),
em vídeos de 10–12 s a 640 px e 30 fps:

| Vídeo | Quadros | Pessoas/quadro | YOLO | U²-Net | SlowFast (por ROI) | Render | Total |
|---|---|---|---|---|---|---|---|
| `archery` | 300 | 2,0 | 12 s | 259 s | 20 s | 2,4 s | 5,2 min |
| `pushups` | 360 | 1,0 | 15 s | 179 s | 26 s | 2,2 s | 4,1 min |
| `skate` | 360 | 4,4 | 13 s | 773 s | 25 s | 1,9 s | 13,9 min |

A segmentação domina o custo porque roda **uma vez por pessoa por quadro** — é o
estágio a otimizar primeiro se o pipeline precisar escalar.

## Estrutura

```
src/config.py         caminhos, hiperparâmetros, catálogo de vídeos
src/video_io.py       leitura/escrita de vídeo e primitivas de desenho
src/stage_detect.py   A — YOLO
src/stage_segment.py  B — U-Net (backend plugável)
src/stage_action.py   C — SlowFast
src/stage_render.py   D — composição do overlay
src/analyze.py        métricas, momentos, painéis e gráficos
src/report_text.py    texto analítico do relatório
src/make_report.py    montagem do PDF
src/cli.py            interface de linha de comando
```
