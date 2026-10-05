"""Estagio A - YOLO: detecta pessoas e produz bounding boxes rastreadas.

Saida: outputs/<video>/detections.json
  {
    "video": "...", "width": W, "height": H, "fps": F, "n_frames": N,
    "model": "yolo11n.pt", "conf": 0.35,
    "frames": [ {"index": 0, "boxes": [{"xyxy": [...], "conf": .., "track_id": ..}]}, ... ]
  }

O tracker (ByteTrack) e usado para dar IDs estaveis: sem eles nao da para
responder "a caixa acompanha a pessoa?" - so se saberia que houve *alguma*
deteccao, nao que e a mesma pessoa.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import config, video_io


def run(video: str, conf: float | None = None) -> Path:
    from ultralytics import YOLO

    conf = config.YOLO_CONF if conf is None else conf
    src = config.video_path(video)
    meta = video_io.probe(src)
    out_path = config.run_dir(video) / "detections.json"

    model = YOLO(str(config.YOLO_WEIGHTS))

    started = time.perf_counter()
    frames: list[dict] = []
    # persist=True mantem o estado do tracker entre chamadas frame a frame.
    for idx, frame in video_io.iter_frames(src):
        result = model.track(
            frame,
            persist=True,
            classes=[config.PERSON_CLASS_ID],
            conf=conf,
            imgsz=config.YOLO_IMGSZ,
            tracker="bytetrack.yaml",
            device=config.DEVICE,
            verbose=False,
        )[0]

        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else [None] * len(confs)
            )
            for box, c, tid in zip(xyxy, confs, ids):
                boxes.append(
                    {
                        "xyxy": [round(float(v), 2) for v in box],
                        "conf": round(float(c), 4),
                        "track_id": None if tid is None else int(tid),
                    }
                )
            # maior area primeiro: o "sujeito principal" fica no indice 0
            boxes.sort(key=lambda b: -(b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1]))

        frames.append({"index": idx, "boxes": boxes})
        if idx % 60 == 0:
            print(f"  frame {idx:4d}  pessoas={len(boxes)}")

    elapsed = time.perf_counter() - started
    detected = sum(1 for f in frames if f["boxes"])
    payload = {
        "video": video,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "n_frames": len(frames),
        "model": config.YOLO_WEIGHTS.name,
        "conf": conf,
        "imgsz": config.YOLO_IMGSZ,
        "tracker": "bytetrack.yaml",
        "elapsed_s": round(elapsed, 2),
        "frames_with_person": detected,
        "frames": frames,
    }
    video_io.write_json(out_path, payload)
    print(
        f"[detect] {video}: {detected}/{len(frames)} frames com pessoa "
        f"em {elapsed:.1f}s -> {out_path}"
    )
    return out_path
