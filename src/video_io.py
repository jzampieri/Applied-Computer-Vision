"""Leitura/escrita de video e primitivas de desenho do overlay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"nao foi possivel abrir o video: {path}")
    try:
        return VideoMeta(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Itera (indice, frame BGR) em streaming - nunca carrega o video inteiro."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"nao foi possivel abrir o video: {path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_frame(path: Path, index: int) -> np.ndarray:
    """Le um frame especifico (usado pelos paineis de analise)."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise IndexError(f"frame {index} fora do video {path}")
        return frame
    finally:
        cap.release()


class VideoWriter:
    """Writer mp4 que descobre a resolucao no primeiro frame."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"nao foi possivel abrir o writer em {self.path}")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ------------------------------------------------------------------- json util
def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- desenho
# Paleta estavel por track_id (BGR).
_TRACK_COLORS = [
    (80, 200, 120),
    (60, 160, 240),
    (200, 120, 240),
    (240, 200, 80),
    (120, 240, 240),
    (240, 120, 120),
]


def track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (160, 160, 160)
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
    avoid: tuple[int, int, int, int] | None = None,
) -> None:
    """Desenha a bbox e, opcionalmente, um rotulo.

    `avoid` e um retangulo (x1,y1,x2,y2) que o rotulo nao deve cobrir - usado
    para o rotulo nao sumir atras do painel de texto do overlay. Se houver
    colisao, o rotulo desce para logo abaixo do painel.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, tht), base = cv2.getTextSize(label, font, scale, th)
    box_h = tht + base + 4
    top = max(0, y1 - box_h)
    if avoid is not None and _overlaps((x1, top, x1 + tw + 6, top + box_h), avoid):
        top = min(frame.shape[0] - box_h, avoid[3] + 4)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + box_h), color, -1)
    cv2.putText(
        frame, label, (x1 + 3, top + tht + 2), font, scale, (20, 20, 20), th, cv2.LINE_AA
    )


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
    contour: bool = True,
) -> np.ndarray:
    """Pinta `mask` (uint8 0/255, mesmo HxW do frame) sobre o frame."""
    if mask.max() == 0:
        return frame
    sel = mask > 127
    tint = np.empty_like(frame)
    tint[:] = color
    frame[sel] = (frame[sel] * (1.0 - alpha) + tint[sel] * alpha).astype(np.uint8)
    if contour:
        contours, _ = cv2.findContours(
            (sel.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, color, 2, cv2.LINE_AA)
    return frame


def draw_panel(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int] = (10, 10),
    scale: float = 0.52,
) -> tuple[int, int, int, int]:
    """Caixa semitransparente com linhas de texto (texto, cor) sobre o frame.

    Devolve o retangulo ocupado, para que os rotulos das bboxes possam desvia-lo.
    """
    if not lines:
        return (0, 0, 0, 0)
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 1
    pad, gap = 8, 6
    sizes = [cv2.getTextSize(t, font, scale, th)[0] for t, _ in lines]
    box_w = max(w for w, _ in sizes) + 2 * pad
    line_h = max(h for _, h in sizes)
    box_h = len(lines) * (line_h + gap) + 2 * pad - gap
    x0, y0 = origin
    x1, y1 = min(frame.shape[1], x0 + box_w), min(frame.shape[0], y0 + box_h)

    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = (roi * 0.35).astype(np.uint8)  # escurece o fundo do painel

    y = y0 + pad + line_h
    for (text, color) in lines:
        cv2.putText(frame, text, (x0 + pad, y), font, scale, color, th, cv2.LINE_AA)
        y += line_h + gap
    return (x0, y0, x1, y1)


def draw_timeline(
    frame: np.ndarray,
    position: float,
    segments: list[tuple[float, float, tuple[int, int, int]]],
    height: int = 14,
) -> None:
    """Faixa temporal no rodape: `segments` em fracoes [0,1] e o cursor atual."""
    h, w = frame.shape[:2]
    top = h - height - 6
    cv2.rectangle(frame, (0, top), (w, top + height), (40, 40, 40), -1)
    for start, end, color in segments:
        x1 = int(round(start * w))
        x2 = max(x1 + 1, int(round(end * w)))
        cv2.rectangle(frame, (x1, top), (x2, top + height), color, -1)
    cursor = int(round(position * w))
    cv2.line(frame, (cursor, top - 3), (cursor, top + height + 3), (255, 255, 255), 2)
