"""CLI do pipeline CP5.

    python -m src.cli detect  --video archery
    python -m src.cli segment --video archery
    python -m src.cli action  --video archery --roi person
    python -m src.cli render  --video archery
    python -m src.cli all     --video archery          # A->B->C(person+full)->D
    python -m src.cli analyze                          # metricas + paineis
    python -m src.cli report                           # PDF final

Os estagios gravam artefatos em disco (outputs/<video>/) e podem ser
reexecutados isoladamente. Isso atende o item 1 do roteiro do PDF ("executar
os tres modelos separadamente") e mantem um modelo por vez na memoria, o que
importa numa maquina sem GPU e com 6 GB de RAM.
"""

from __future__ import annotations

import argparse
import sys

from . import config


def _videos(args: argparse.Namespace) -> list[str]:
    if args.video == "all":
        return list(config.ALL_VIDEOS)
    if args.video not in config.VIDEOS:
        raise SystemExit(f"video desconhecido: {args.video} (opcoes: {', '.join(config.ALL_VIDEOS)}, all)")
    return [args.video]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cp05", description="Pipeline YOLO + U-Net + SlowFast")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_video(p: argparse.ArgumentParser) -> None:
        p.add_argument("--video", default="all", help="nome do video ou 'all'")

    p_detect = sub.add_parser("detect", help="estagio A - YOLO")
    add_video(p_detect)
    p_detect.add_argument("--conf", type=float, default=None)

    p_seg = sub.add_parser("segment", help="estagio B - U-Net")
    add_video(p_seg)
    p_seg.add_argument("--threshold", type=float, default=None)

    p_act = sub.add_parser("action", help="estagio C - SlowFast")
    add_video(p_act)
    p_act.add_argument("--roi", choices=["person", "full"], default="person")

    p_ren = sub.add_parser("render", help="estagio D - overlay")
    add_video(p_ren)
    p_ren.add_argument("--roi", choices=["person", "full"], default="person")

    p_all = sub.add_parser("all", help="A -> B -> C(person e full) -> D")
    add_video(p_all)

    sub.add_parser("analyze", help="metricas, momentos e paineis")
    sub.add_parser("report", help="gera o PDF do relatorio")

    args = parser.parse_args(argv)

    if args.cmd == "analyze":
        from . import analyze

        analyze.run()
        return 0

    if args.cmd == "report":
        from . import make_report

        make_report.run()
        return 0

    for video in _videos(args):
        print(f"\n=== {video} :: {args.cmd} ===")
        if args.cmd == "detect":
            from . import stage_detect

            stage_detect.run(video, conf=args.conf)
        elif args.cmd == "segment":
            from . import stage_segment

            stage_segment.run(video, threshold=args.threshold)
        elif args.cmd == "action":
            from . import stage_action

            stage_action.run(video, roi=args.roi)
        elif args.cmd == "render":
            from . import stage_render

            stage_render.run(video, roi=args.roi)
        elif args.cmd == "all":
            from . import stage_action, stage_detect, stage_render, stage_segment

            stage_detect.run(video)
            stage_segment.run(video)
            stage_action.run(video, roi="person")
            stage_action.run(video, roi="full")
            stage_render.run(video, roi="person")
    return 0


if __name__ == "__main__":
    sys.exit(main())
