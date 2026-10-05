"""Caminhos, hiperparametros e catalogo de videos do CP5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "videos"
WEIGHTS_DIR = ROOT / "weights"
OUTPUTS_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
PANELS_DIR = REPORT_DIR / "panels"

DEVICE = "cpu"  # a maquina do projeto nao tem GPU

# ---------------------------------------------------------------- normalizacao
# Todos os videos sao reamostrados para este formato antes do pipeline, para
# manter o custo de CPU previsivel (ver orcamento no plano).
TARGET_WIDTH = 640
TARGET_FPS = 30
TARGET_SECONDS = 12

# ---------------------------------------------------------------------- YOLO
YOLO_WEIGHTS = WEIGHTS_DIR / "yolo11n.pt"
YOLO_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
YOLO_CONF = 0.35
YOLO_IMGSZ = 640
PERSON_CLASS_ID = 0  # COCO

# --------------------------------------------------------------------- U2-Net
# U2-Net e arquitetura da familia U-Net (encoder-decoder com skip connections,
# blocos RSU em estrutura U aninhada). O checkpoint u2net_human_seg e
# pre-treinado para segmentacao de pessoas -> satisfaz "modelos ja treinados,
# sem treinamento adicional".
UNET_WEIGHTS = WEIGHTS_DIR / "u2net_human_seg.onnx"
UNET_WEIGHTS_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"
)
UNET_INPUT_SIZE = 320
UNET_THRESHOLD = 0.5
# Padding relativo aplicado ao crop da bbox antes de segmentar. Sem ele a
# mascara encosta na borda do crop e o contorno sai cortado.
UNET_BBOX_PADDING = 0.10

# ------------------------------------------------------------------- SlowFast
SLOWFAST_WEIGHTS = WEIGHTS_DIR / "SLOWFAST_8x8_R50.pyth"
SLOWFAST_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R50.pyth"
)
KINETICS_CLASSNAMES = WEIGHTS_DIR / "kinetics_classnames.json"
KINETICS_CLASSNAMES_URL = (
    "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
)
SLOWFAST_WINDOW = 32  # frames por janela (fast pathway)
SLOWFAST_STRIDE = 16  # deslocamento entre janelas -> 50% de sobreposicao
SLOWFAST_ALPHA = 4  # fast:slow = 4:1
SLOWFAST_SIDE_SIZE = 256
SLOWFAST_CROP_SIZE = 256
SLOWFAST_MEAN = (0.45, 0.45, 0.45)
SLOWFAST_STD = (0.225, 0.225, 0.225)
SLOWFAST_TOPK = 5

# ---------------------------------------------------------------------- videos
@dataclass(frozen=True)
class VideoSpec:
    """Um video do experimento, com procedencia e o papel que cumpre."""

    name: str
    url: str
    license: str
    author: str
    source_page: str
    expected_action: str  # rotulo observado por nos, usado como verdade no relatorio
    role: str
    start: float = 0.0  # segundo inicial do recorte
    duration: float = TARGET_SECONDS


VIDEOS: dict[str, VideoSpec] = {
    "archery": VideoSpec(
        name="archery",
        url="https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4",
        license="Amostra de pesquisa do PyTorchVideo (clipe do Kinetics-400)",
        author="PyTorchVideo / Facebook AI Research",
        source_page="https://pytorchvideo.org/",
        expected_action="archery",
        role="Caso limpo: uma pessoa, acao presente no vocabulario Kinetics-400.",
        start=0.0,
        duration=10.0,
    ),
    "pushups": VideoSpec(
        name="pushups",
        url="https://upload.wikimedia.org/wikipedia/commons/9/98/Interval_Push-ups.webm",
        license="CC BY-SA 4.0",
        author="Wikimedia Commons",
        source_page="https://commons.wikimedia.org/wiki/File:Interval_Push-ups.webm",
        expected_action="push up",
        role=(
            "Fronteiras de acao: series alternadas com descanso, para medir se a "
            "confianca cai ou a classe muda quando a acao comeca e termina."
        ),
        start=2.0,
        duration=12.0,
    ),
    "skate": VideoSpec(
        name="skate",
        url=(
            "https://upload.wikimedia.org/wikipedia/commons/4/4d/"
            "Day_of_the_Town_%282017-09-09%29_-_Skateboarding_-_0022.webm"
        ),
        license="CC BY-SA 4.0",
        author="Wikimedia Commons",
        source_page=(
            "https://commons.wikimedia.org/wiki/File:Day_of_the_Town_(2017-09-09)"
            "_-_Skateboarding_-_0022.webm"
        ),
        expected_action="skateboarding",
        role=(
            "Casos de falha: multiplas pessoas, oclusao mutua, movimento rapido e "
            "sujeitos pequenos no quadro."
        ),
        start=0.0,
        duration=12.0,
    ),
}

# Os dois videos que o PDF exige; 'skate' entra como evidencia extra de falha.
REQUIRED_VIDEOS = ("archery", "pushups")
ALL_VIDEOS = tuple(VIDEOS)


def video_path(name: str) -> Path:
    return DATA_DIR / f"{name}.mp4"


def run_dir(name: str) -> Path:
    return OUTPUTS_DIR / name
