"""Gera report/CP5_relatorio.pdf (maximo 3 paginas) a partir das metricas reais.

Os numeros vem de report/summary.json e report/metrics.csv (produzidos por
`python -m src.cli analyze`). O texto analitico vive em src/report_text.py.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import config, report_text, video_io

INK = colors.HexColor("#0b0b0b")
INK_SOFT = colors.HexColor("#52514e")
RULE = colors.HexColor("#dedddb")
ACCENT = colors.HexColor("#2a78d6")
BAND = colors.HexColor("#f2f1ef")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=base["Title"], fontSize=14, leading=17, textColor=INK, spaceAfter=2
        ),
        "subtitle": ParagraphStyle(
            "st", parent=base["Normal"], fontSize=8, leading=11, textColor=INK_SOFT
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=9.5,
            leading=12,
            textColor=ACCENT,
            spaceBefore=7,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "b",
            parent=base["Normal"],
            fontSize=7.6,
            leading=10,
            textColor=INK,
            alignment=TA_JUSTIFY,
        ),
        "small": ParagraphStyle(
            "s", parent=base["Normal"], fontSize=6.4, leading=8, textColor=INK_SOFT
        ),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=6.3, leading=7.6, textColor=INK),
        "cellh": ParagraphStyle(
            "ch", parent=base["Normal"], fontSize=6.3, leading=7.6, textColor=colors.white
        ),
    }


def _table(data: list[list], widths: list[float], st: dict, align_left_first=True) -> Table:
    body = [[Paragraph(str(c), st["cellh"]) for c in data[0]]]
    body += [[Paragraph(str(c), st["cell"]) for c in row] for row in data[1:]]
    t = Table(body, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
    ]
    t.setStyle(TableStyle(style))
    return t


def _fit_image(path: Path, max_w: float) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    scale = max_w / w
    return Image(str(path), width=max_w, height=h * scale)


def build(summary: dict) -> list:
    st = _styles()
    flow: list = []
    W = A4[0] - 24 * mm  # largura util

    # ------------------------------------------------------------ cabecalho
    flow.append(Paragraph(report_text.TITULO, st["title"]))
    flow.append(Paragraph(report_text.SUBTITULO, st["subtitle"]))
    flow.append(Spacer(1, 3))
    autores = [["Membro", "RM", "Responsabilidade", "Parecer (resumo)"]]
    autores += [[m["nome"], m["rm"], m["responsabilidade"], m["parecer"]] for m in report_text.MEMBROS]
    flow.append(_table(autores, [W * 0.22, W * 0.10, W * 0.26, W * 0.42], st))
    flow.append(Spacer(1, 5))

    # ------------------------------------------------------------- pipeline
    flow.append(Paragraph("1. Objetivo e pipeline", st["h2"]))
    flow.append(Paragraph(report_text.OBJETIVO, st["body"]))
    flow.append(Spacer(1, 3))
    modelos = [["Etapa", "Modelo (pré-treinado, sem treino adicional)", "Papel no pipeline"]]
    modelos += report_text.MODELOS
    flow.append(_table(modelos, [W * 0.14, W * 0.38, W * 0.48], st))
    flow.append(Spacer(1, 3))
    flow.append(Paragraph(report_text.INTEGRACAO, st["body"]))

    # -------------------------------------------------- tabela de resultados
    flow.append(Paragraph("2. Tabela de resultados", st["h2"]))
    head = [
        "Video",
        "Ação observada",
        "Cobertura YOLO",
        "Conf. YOLO",
        "IoU temp.",
        "IDs / trocas do sujeito principal",
        "Fill ratio da máscara",
        "Acerto top-1",
        "Acerto top-5",
        "Conf. SlowFast",
    ]
    rows = [head]
    for name, r in summary.items():
        d, m, a = r["deteccao"], r["mascara"], r["acao_person"]
        rows.append(
            [
                name,
                r["acao_observada"],
                f"{d['coverage'] * 100:.0f}%",
                f"{d['conf_media']:.2f}",
                f"{d['iou_temporal_media']:.2f}",
                f"{d['track_ids']} / {d['trocas_sujeito_principal']}",
                f"{m['fill_ratio_medio']:.2f} ({m['fill_ratio_min']:.2f}-{m['fill_ratio_max']:.2f})",
                f"{a['acerto_top1'] * 100:.0f}%",
                f"{a['acerto_top5'] * 100:.0f}%",
                f"{a['conf_media']:.2f}",
            ]
        )
    flow.append(_table(rows, [W * 0.09, W * 0.12, W * 0.09, W * 0.08, W * 0.08, W * 0.09, W * 0.16, W * 0.09, W * 0.09, W * 0.11], st))
    flow.append(Spacer(1, 2))
    flow.append(Paragraph(report_text.LEGENDA_TABELA, st["small"]))

    # --------------------------------------------------- custo de execucao
    flow.append(Paragraph("2.1. Custo de execução (CPU, sem GPU)", st["h2"]))
    chead = ["Video", "Quadros", "YOLO", "U-Net", "SlowFast (por ROI)", "Total dos modelos"]
    crows = [chead]
    for name, r in summary.items():
        d, m = r["deteccao"], r["mascara"]
        a = r["acao_person"]["elapsed_s"] or 0
        total = (d["elapsed_s"] or 0) + (m["elapsed_s"] or 0) + 2 * a
        crows.append(
            [
                name,
                d["n_frames"],
                f"{d['elapsed_s']:.0f} s",
                f"{m['elapsed_s']:.0f} s",
                f"{a:.0f} s",
                f"{total / 60:.1f} min",
            ]
        )
    flow.append(_table(crows, [W * 0.16, W * 0.14, W * 0.14, W * 0.14, W * 0.21, W * 0.21], st))
    flow.append(Spacer(1, 2))
    flow.append(Paragraph(report_text.CUSTO, st["small"]))

    flow.append(PageBreak())

    # ------------------------------------------------- momentos + perguntas
    flow.append(Paragraph("3. Momentos analisados e as quatro perguntas", st["h2"]))
    flow.append(Paragraph(report_text.PERGUNTAS_INTRO, st["body"]))
    flow.append(Spacer(1, 3))

    mhead = ["Video", "t (s)", "Motivo do recorte", "YOLO", "Máscara", "SlowFast top-1", "Conf.", "OK?"]
    mrows = [mhead]
    for name, r in summary.items():
        for mo in r["momentos"]:
            mrows.append(
                [
                    name,
                    mo["tempo_s"],
                    mo["motivo"],
                    f"{mo['pessoas']}p conf {mo['yolo_conf']}",
                    f"{mo['mascara_area_px']} px",
                    mo["slowfast_top1"],
                    mo["slowfast_conf"],
                    mo["acerto_top1"],
                ]
            )
    flow.append(_table(mrows, [W * 0.09, W * 0.06, W * 0.27, W * 0.13, W * 0.09, W * 0.18, W * 0.07, W * 0.06], st))
    flow.append(Spacer(1, 4))

    for titulo, texto in report_text.RESPOSTAS:
        flow.append(Paragraph(f"<b>{titulo}</b> {texto}", st["body"]))
        flow.append(Spacer(1, 2))

    # grafico de confianca (small multiples: um painel por video)
    chart = config.PANELS_DIR / "conf_todos.png"
    if chart.exists():
        flow.append(Spacer(1, 3))
        flow.append(_fit_image(chart, W * 0.78))

    flow.append(PageBreak())

    # ----------------------------------------------------- erros explicados
    flow.append(Paragraph("4. Erros observados e explicados", st["h2"]))
    for erro in report_text.ERROS:
        flow.append(Paragraph(f"<b>{erro['titulo']}</b>", st["body"]))
        flow.append(Paragraph(erro["texto"], st["body"]))
        painel = config.PANELS_DIR / erro["painel"]
        if painel.exists():
            flow.append(Spacer(1, 2))
            flow.append(_fit_image(painel, W * 0.74))
            flow.append(Paragraph(erro["legenda"], st["small"]))
        else:
            print(f"[report] AVISO: painel ausente para este erro: {erro['painel']}")
        flow.append(Spacer(1, 3))

    # --------------------------------------------------- extensao opcional
    flow.append(Paragraph("5. Extensão opcional: SlowFast no quadro inteiro vs no recorte", st["h2"]))
    ehead = ["Video", "Concordância top-1", "Acerto top-1 (pessoa)", "Acerto top-1 (quadro)", "Conf. média (pessoa)", "Conf. média (quadro)"]
    erows = [ehead]
    for name, r in summary.items():
        c = r["comparacao_roi"]
        erows.append(
            [
                name,
                f"{c['concordancia_top1'] * 100:.0f}%",
                f"{c['acerto_top1_person'] * 100:.0f}%",
                f"{c['acerto_top1_full'] * 100:.0f}%",
                f"{c['conf_media_person']:.2f}",
                f"{c['conf_media_full']:.2f}",
            ]
        )
    flow.append(_table(erows, [W * 0.14, W * 0.17, W * 0.18, W * 0.18, W * 0.17, W * 0.16], st))
    flow.append(Spacer(1, 3))
    flow.append(Paragraph(report_text.EXTENSAO, st["body"]))

    flow.append(Paragraph("6. Conclusão", st["h2"]))
    flow.append(Paragraph(report_text.CONCLUSAO, st["body"]))
    flow.append(Spacer(1, 3))
    flow.append(Paragraph(report_text.CREDITOS, st["small"]))
    return flow


def run() -> Path:
    summary = video_io.read_json(config.REPORT_DIR / "summary.json")
    out = config.REPORT_DIR / "CP5_relatorio.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title="CP5 - Deteccao de Acoes de Pessoas com YOLO, U-Net e SlowFast",
    )
    doc.build(build(summary))
    print(f"[report] -> {out}")
    return out


if __name__ == "__main__":
    run()
