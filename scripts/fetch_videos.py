"""Baixa os videos do experimento, normaliza com ffmpeg e registra as licencas.

Normalizacao (ver src/config.py): recorte temporal, 640 px de largura e 30 fps.
Isso mantem o custo de CPU dentro do orcamento do projeto (maquina sem GPU).
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

UA = {"User-Agent": "cp05-vision-ai/1.0 (trabalho academico)"}


def download_raw(spec: config.VideoSpec, raw_dir: Path) -> Path:
    suffix = Path(spec.url.split("?")[0]).suffix or ".mp4"
    dest = raw_dir / f"{spec.name}_raw{suffix}"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [ja existe] {dest.name}")
        return dest
    print(f"  [baixando ] {dest.name}")
    req = urllib.request.Request(spec.url, headers=UA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def normalize(raw: Path, spec: config.VideoSpec) -> Path:
    out = config.video_path(spec.name)
    if out.exists() and out.stat().st_size > 0:
        print(f"  [ja existe] {out.name}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(spec.start), "-i", str(raw), "-t", str(spec.duration),
        "-an",  # audio nao e usado em nenhum estagio
        "-vf", f"scale={config.TARGET_WIDTH}:-2:flags=bicubic,fps={config.TARGET_FPS}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    print(f"  [ffmpeg   ] {out.name}  ({spec.start}s +{spec.duration}s)")
    subprocess.run(cmd, check=True)
    return out


def write_licenses() -> None:
    lines = [
        "# Procedencia e licenca dos videos",
        "",
        "Todos os videos usados no CP5 sao de dominio publico ou licenca Creative Commons.",
        "Nenhum video foi editado alem do recorte temporal e do redimensionamento",
        f"({config.TARGET_WIDTH} px de largura, {config.TARGET_FPS} fps) feitos por",
        "`scripts/fetch_videos.py`.",
        "",
    ]
    for spec in config.VIDEOS.values():
        lines += [
            f"## `{spec.name}.mp4`",
            "",
            f"- **Papel no experimento:** {spec.role}",
            f"- **Acao observada (verdade de referencia):** {spec.expected_action}",
            f"- **Autor / creditos:** {spec.author}",
            f"- **Licenca:** {spec.license}",
            f"- **Pagina da fonte:** {spec.source_page}",
            f"- **URL do arquivo:** {spec.url}",
            f"- **Recorte aplicado:** inicio {spec.start}s, duracao {spec.duration}s",
            "",
        ]
    (config.DATA_DIR / "LICENSES.md").write_text("\n".join(lines), encoding="utf-8")
    print("LICENSES.md atualizado")


def main() -> int:
    raw_dir = config.DATA_DIR / "raw"
    for spec in config.VIDEOS.values():
        print(f"== {spec.name}")
        raw = download_raw(spec, raw_dir)
        normalize(raw, spec)
    write_licenses()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
