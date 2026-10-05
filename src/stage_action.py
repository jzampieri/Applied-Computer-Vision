"""Estagio C - SlowFast: classifica a acao em janelas sucessivas de quadros.

Modelo: `slowfast_r50` do PyTorchVideo, pesos oficiais SLOWFAST_8x8_R50
treinados no Kinetics-400. Nenhum treinamento adicional e feito.

Janelas: `SLOWFAST_WINDOW` frames (fast pathway) com passo `SLOWFAST_STRIDE`
-> 50% de sobreposicao. O slow pathway e a subamostragem temporal da mesma
janela por `alpha=4`, que e o contrato de entrada do modelo: ele recebe a
lista [slow, fast].

Pre-processamento (feito aqui, nao via pytorchvideo.transforms, que depende de
APIs removidas do torchvision moderno): BGR->RGB, escala [0,1], resize do lado
curto para 256, center-crop 256, normalizacao Kinetics.

ROI (`--roi`):
  * `person` - recorta a uniao das bboxes da janela (o sujeito preenche o quadro);
  * `full`   - usa o quadro inteiro.
A comparacao entre os dois e a extensao opcional pedida no PDF.

Saida: outputs/<video>/actions_<roi>.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config, video_io


def load_kinetics_labels() -> list[str]:
    """Lista de 400 rotulos indexada pelo id da classe.

    O JSON oficial mapeia nome->id e muitas chaves vem com aspas embutidas
    (ex.: '"eating ice cream"'), por isso o strip.
    """
    raw = json.loads(config.KINETICS_CLASSNAMES.read_text(encoding="utf-8"))
    labels = [""] * (max(raw.values()) + 1)
    for name, idx in raw.items():
        labels[idx] = name.strip('"').strip()
    return labels


def load_model() -> torch.nn.Module:
    """Carrega o slowfast_r50 com os pesos locais do Kinetics-400."""
    from pytorchvideo.models.hub import slowfast_r50

    model = slowfast_r50(pretrained=False)
    checkpoint = torch.load(config.SLOWFAST_WEIGHTS, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return model.eval()


def union_box(
    boxes_per_frame: list[list[dict]], width: int, height: int, margin: float = 0.08
) -> tuple[int, int, int, int]:
    """Bbox que cobre todas as pessoas da janela, com margem, limitada ao frame."""
    xs1, ys1, xs2, ys2 = [], [], [], []
    for boxes in boxes_per_frame:
        for b in boxes:
            x1, y1, x2, y2 = b["xyxy"]
            xs1.append(x1), ys1.append(y1), xs2.append(x2), ys2.append(y2)
    if not xs1:
        return (0, 0, width, height)
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        int(max(0, round(x1 - mw))),
        int(max(0, round(y1 - mh))),
        int(min(width, round(x2 + mw))),
        int(min(height, round(y2 + mh))),
    )


def _preprocess(frames_bgr: list[np.ndarray]) -> list[torch.Tensor]:
    """Lista de frames BGR -> [slow, fast] prontos para o modelo."""
    side, crop = config.SLOWFAST_SIDE_SIZE, config.SLOWFAST_CROP_SIZE
    processed = []
    for frame in frames_bgr:
        h, w = frame.shape[:2]
        scale = side / min(h, w)
        img = cv2.resize(frame, (max(crop, int(round(w * scale))), max(crop, int(round(h * scale)))))
        h2, w2 = img.shape[:2]
        top, left = (h2 - crop) // 2, (w2 - crop) // 2
        img = img[top : top + crop, left : left + crop]
        processed.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    clip = np.stack(processed).astype(np.float32) / 255.0  # T,H,W,C
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2)  # C,T,H,W
    mean = torch.tensor(config.SLOWFAST_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(config.SLOWFAST_STD).view(3, 1, 1, 1)
    clip = (clip - mean) / std

    fast = clip.unsqueeze(0)  # 1,C,T,H,W
    slow = fast[:, :, :: config.SLOWFAST_ALPHA]
    return [slow, fast]


def run(video: str, roi: str = "person") -> Path:
    if roi not in {"person", "full"}:
        raise ValueError("roi deve ser 'person' ou 'full'")

    src = config.video_path(video)
    out_dir = config.run_dir(video)
    detections = video_io.read_json(out_dir / "detections.json")
    by_index = {f["index"]: f["boxes"] for f in detections["frames"]}
    width, height, fps = detections["width"], detections["height"], detections["fps"]

    labels = load_kinetics_labels()
    model = load_model()

    # Os frames do vídeo normalizado (12 s a 640 px) cabem na RAM como uint8:
    # ~450 x 640 x 360 x 3 = 310 MB. Carregar de uma vez evita reabrir o
    # arquivo a cada janela deslizante.
    frames = [f for _, f in video_io.iter_frames(src)]
    n = len(frames)

    started = time.perf_counter()
    windows: list[dict] = []
    win, stride = config.SLOWFAST_WINDOW, config.SLOWFAST_STRIDE
    start = 0
    while start < n:
        end = min(start + win, n)
        idxs = list(range(start, end))
        # ultima janela: repete o ultimo frame para completar o tamanho fixo
        while len(idxs) < win:
            idxs.append(n - 1)

        clip_frames = [frames[i] for i in idxs]
        if roi == "person":
            bx1, by1, bx2, by2 = union_box([by_index.get(i, []) for i in idxs], width, height)
            if bx2 - bx1 < 16 or by2 - by1 < 16:  # sem pessoa util: cai no quadro inteiro
                bx1, by1, bx2, by2 = 0, 0, width, height
            clip_frames = [f[by1:by2, bx1:bx2] for f in clip_frames]
        else:
            bx1, by1, bx2, by2 = 0, 0, width, height

        inputs = _preprocess(clip_frames)
        with torch.no_grad():
            logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)[0]
        topk = torch.topk(probs, config.SLOWFAST_TOPK)

        windows.append(
            {
                "window": len(windows),
                "frame_start": start,
                "frame_end": end - 1,
                "t_start": round(start / fps, 3),
                "t_end": round((end - 1) / fps, 3),
                "roi_xyxy": [bx1, by1, bx2, by2],
                "top1": labels[int(topk.indices[0])],
                "top1_conf": round(float(topk.values[0]), 4),
                "topk": [
                    {"label": labels[int(i)], "conf": round(float(v), 4)}
                    for v, i in zip(topk.values, topk.indices)
                ],
            }
        )
        w = windows[-1]
        print(
            f"  janela {w['window']:2d} [{w['t_start']:5.2f}-{w['t_end']:5.2f}s] "
            f"{w['top1']:<28s} {w['top1_conf']:.3f}"
        )
        if end >= n:
            break
        start += stride

    elapsed = time.perf_counter() - started
    out_path = out_dir / f"actions_{roi}.json"
    video_io.write_json(
        out_path,
        {
            "video": video,
            "model": "slowfast_r50 (PyTorchVideo, Kinetics-400, SLOWFAST_8x8_R50)",
            "roi": roi,
            "window": win,
            "stride": stride,
            "alpha": config.SLOWFAST_ALPHA,
            "crop_size": config.SLOWFAST_CROP_SIZE,
            "fps": fps,
            "n_frames": n,
            "elapsed_s": round(elapsed, 2),
            "windows": windows,
        },
    )
    print(f"[action] {video} roi={roi}: {len(windows)} janelas em {elapsed:.1f}s -> {out_path}")
    return out_path
