"""Estagio D - composicao do video de demonstracao.

Sobrepoe as tres saidas no mesmo quadro:
  * bbox do YOLO (cor por track_id) com confianca;
  * mascara da U-Net preenchida com alpha + contorno;
  * rotulo de acao do SlowFast com confianca, mais uma faixa temporal no
    rodape mostrando as janelas e onde a classe muda.

Cada frame herda a acao da janela que o contem; em regiao de sobreposicao
(as janelas tem 50% de overlap) vence a janela de maior confianca.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from . import config, video_io

MASK_COLOR = (90, 220, 160)  # BGR
TEXT_PRIMARY = (255, 255, 255)
TEXT_DIM = (190, 190, 190)


def action_per_frame(windows: list[dict], n_frames: int) -> list[dict | None]:
    """Resolve, para cada frame, qual janela o descreve (maior confianca vence)."""
    chosen: list[dict | None] = [None] * n_frames
    for w in windows:
        for i in range(w["frame_start"], min(w["frame_end"] + 1, n_frames)):
            current = chosen[i]
            if current is None or w["top1_conf"] > current["top1_conf"]:
                chosen[i] = w
    return chosen


def _timeline_segments(windows: list[dict], n_frames: int) -> list[tuple[float, float, tuple]]:
    """Uma faixa por janela, colorida por classe (cores ciclicas por rotulo)."""
    palette = [(200, 120, 60), (60, 160, 200), (120, 200, 90), (180, 90, 200), (90, 200, 200)]
    order: dict[str, int] = {}
    segments = []
    for w in windows:
        label = w["top1"]
        if label not in order:
            order[label] = len(order)
        color = palette[order[label] % len(palette)]
        segments.append((w["frame_start"] / n_frames, (w["frame_end"] + 1) / n_frames, color))
    return segments


def run(video: str, roi: str = "person") -> Path:
    src = config.video_path(video)
    out_dir = config.run_dir(video)
    masks_dir = out_dir / "masks"

    detections = video_io.read_json(out_dir / "detections.json")
    actions = video_io.read_json(out_dir / f"actions_{roi}.json")
    by_index = {f["index"]: f["boxes"] for f in detections["frames"]}
    n_frames, fps = detections["n_frames"], detections["fps"]

    per_frame = action_per_frame(actions["windows"], n_frames)
    segments = _timeline_segments(actions["windows"], n_frames)

    out_path = out_dir / "overlay.mp4"
    started = time.perf_counter()
    with video_io.VideoWriter(out_path, fps) as writer:
        for idx, frame in video_io.iter_frames(src):
            mask_path = masks_dir / f"{idx:06d}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    video_io.overlay_mask(frame, mask, MASK_COLOR, alpha=0.40)

            boxes = by_index.get(idx, [])
            win = per_frame[idx] if idx < len(per_frame) else None
            lines = [
                (f"YOLO11n: {len(boxes)} pessoa(s)", TEXT_PRIMARY),
                ("U-Net (U2-Net human-seg): mascara", MASK_COLOR),
            ]
            if win is not None:
                lines.append((f"SlowFast: {win['top1']} ({win['top1_conf']:.2f})", TEXT_PRIMARY))
                runner_up = win["topk"][1]
                lines.append(
                    (f"  2o: {runner_up['label']} ({runner_up['conf']:.2f})", TEXT_DIM)
                )
                lines.append(
                    (f"  janela {win['window']} | {win['t_start']:.1f}-{win['t_end']:.1f}s | roi={roi}", TEXT_DIM)
                )
            else:
                lines.append(("SlowFast: -", TEXT_DIM))
            # o painel e desenhado antes das caixas, e seu retangulo e passado
            # adiante para que nenhum rotulo de bbox fique ilegivel atras dele
            panel_rect = video_io.draw_panel(frame, lines)
            for box in boxes:
                tid = box["track_id"]
                label = (
                    f"pessoa#{tid} {box['conf']:.2f}"
                    if tid is not None
                    else f"pessoa {box['conf']:.2f}"
                )
                video_io.draw_box(
                    frame, box["xyxy"], label, video_io.track_color(tid), avoid=panel_rect
                )
            video_io.draw_timeline(frame, idx / max(1, n_frames), segments)
            writer.write(frame)

    print(f"[render] {video} roi={roi} em {time.perf_counter() - started:.1f}s -> {out_path}")
    return out_path
