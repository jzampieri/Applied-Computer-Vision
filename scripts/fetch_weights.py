"""Baixa os pesos pre-treinados dos tres modelos (nenhum treinamento e feito)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

UA = {"User-Agent": "cp05-vision-ai/1.0 (trabalho academico)"}

TARGETS = [
    ("YOLO11n", config.YOLO_WEIGHTS_URL, config.YOLO_WEIGHTS),
    ("U2-Net human-seg", config.UNET_WEIGHTS_URL, config.UNET_WEIGHTS),
    ("SlowFast 8x8 R50 (Kinetics-400)", config.SLOWFAST_WEIGHTS_URL, config.SLOWFAST_WEIGHTS),
    ("Kinetics-400 classnames", config.KINETICS_CLASSNAMES_URL, config.KINETICS_CLASSNAMES),
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r    {done / 1e6:7.1f}/{total / 1e6:.1f} MB ({pct:5.1f}%)", end="")
    print()
    tmp.replace(dest)


def main() -> int:
    for label, url, dest in TARGETS:
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[ja existe] {label} -> {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[baixando ] {label} -> {dest.name}")
        download(url, dest)
    print("\nPesos prontos em", config.WEIGHTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
