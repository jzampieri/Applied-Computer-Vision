"""Estagio B - U-Net: mascara dos pixels da pessoa dentro de cada bbox.

Modelo: U2-Net `u2net_human_seg` (familia U-Net: encoder-decoder com skip
connections, blocos RSU em estrutura U aninhada), pre-treinado para
segmentacao de pessoas e executado via ONNX Runtime.

Procedimento por bbox (reprojecao):
  1. recorta a bbox com padding de 10% (sem o padding o contorno sai cortado
     na borda do crop, porque a U-Net nao ve o entorno do sujeito);
  2. redimensiona o recorte para 320x320, normaliza e roda a rede;
  3. limiariza o mapa de saliencia e redimensiona de volta ao tamanho do crop;
  4. cola a mascara na posicao original do frame.

Saida: outputs/<video>/masks/%06d.png  (mascara uniao, 0/255, HxW do frame)
       outputs/<video>/mask_stats.json (area por bbox, para as metricas)
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import config, video_io


class U2NetHumanSegmenter:
    """Wrapper ONNX do u2net_human_seg."""

    def __init__(self, weights: Path, size: int = config.UNET_INPUT_SIZE) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0  # deixa o ORT escolher conforme os cores
        self.session = ort.InferenceSession(
            str(weights), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.size = size
        # Normalizacao do U2-Net: a imagem e primeiro escalada por max(), depois
        # normalizada com as medias/desvios do repositorio original.
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        peak = img.max()
        img = img / peak if peak > 0 else img
        img = (img - self._mean) / self._std
        return np.transpose(img, (2, 0, 1))[None].astype(np.float32)

    def saliency(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Mapa de saliencia em [0,1] no tamanho do crop original."""
        if crop_bgr.size == 0:
            return np.zeros(crop_bgr.shape[:2], dtype=np.float32)
        out = self.session.run(None, {self.input_name: self._preprocess(crop_bgr)})[0]
        sal = np.asarray(out)[0, 0].astype(np.float32)
        lo, hi = float(sal.min()), float(sal.max())
        sal = (sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)
        h, w = crop_bgr.shape[:2]
        return cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)


def pad_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    pad: float = config.UNET_BBOX_PADDING,
) -> tuple[int, int, int, int]:
    """Expande a bbox por `pad` (fracao de cada lado), limitada ao frame."""
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
    return (
        int(max(0, round(x1 - dw))),
        int(max(0, round(y1 - dh))),
        int(min(width, round(x2 + dw))),
        int(min(height, round(y2 + dh))),
    )


def run(video: str, threshold: float | None = None) -> Path:
    threshold = config.UNET_THRESHOLD if threshold is None else threshold
    src = config.video_path(video)
    out_dir = config.run_dir(video)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    detections = video_io.read_json(out_dir / "detections.json")
    by_index = {f["index"]: f["boxes"] for f in detections["frames"]}
    width, height = detections["width"], detections["height"]

    segmenter = U2NetHumanSegmenter(config.UNET_WEIGHTS)

    started = time.perf_counter()
    stats: list[dict] = []
    for idx, frame in video_io.iter_frames(src):
        union = np.zeros((height, width), dtype=np.uint8)
        per_box = []
        for slot, box in enumerate(by_index.get(idx, [])):
            px1, py1, px2, py2 = pad_box(box["xyxy"], width, height)
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                per_box.append({"slot": slot, "track_id": box["track_id"], "mask_area": 0})
                continue
            sal = segmenter.saliency(crop)
            mask = (sal >= threshold).astype(np.uint8) * 255
            # cola de volta no frame (reprojecao)
            region = union[py1:py2, px1:px2]
            np.maximum(region, mask, out=region)
            bbox_area = max(1.0, (box["xyxy"][2] - box["xyxy"][0]) * (box["xyxy"][3] - box["xyxy"][1]))
            per_box.append(
                {
                    "slot": slot,
                    "track_id": box["track_id"],
                    "mask_area": int((mask > 0).sum()),
                    "bbox_area": round(bbox_area, 1),
                    "fill_ratio": round(float((mask > 0).sum()) / bbox_area, 4),
                }
            )

        cv2.imwrite(str(masks_dir / f"{idx:06d}.png"), union)
        stats.append({"index": idx, "boxes": per_box, "union_area": int((union > 0).sum())})
        if idx % 60 == 0:
            print(f"  frame {idx:4d}  area_uniao={(union > 0).sum()}")

    elapsed = time.perf_counter() - started
    video_io.write_json(
        out_dir / "mask_stats.json",
        {
            "video": video,
            "model": "u2net_human_seg (U2-Net / familia U-Net)",
            "weights": config.UNET_WEIGHTS.name,
            "input_size": config.UNET_INPUT_SIZE,
            "threshold": threshold,
            "bbox_padding": config.UNET_BBOX_PADDING,
            "elapsed_s": round(elapsed, 2),
            "frames": stats,
        },
    )
    print(f"[segment] {video}: {len(stats)} mascaras em {elapsed:.1f}s -> {masks_dir}")
    return masks_dir
