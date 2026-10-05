"""Metricas, momentos de comparacao e graficos dos experimentos.

O PDF pede comparar, em pelo menos tres momentos de cada video, a saida dos
tres modelos, respondendo quatro perguntas. Cada pergunta e traduzida aqui numa
metrica calculavel:

  1. "A caixa acompanha a pessoa?"     -> cobertura de deteccao, confianca media,
                                          numero de track_ids, trocas de id,
                                          IoU entre bboxes consecutivas do mesmo id.
  2. "A mascara acompanha o contorno?" -> fill_ratio (area da mascara / area da
                                          bbox), variacao quadro a quadro da area,
                                          frames com bbox mas mascara vazia.
  3. "A acao prevista corresponde?"    -> top-1 vs acao observada, acerto no top-5.
  4. "A confianca cai / a classe muda
      quando a acao comeca ou termina?" -> serie temporal da confianca e janelas
                                          onde o top-1 troca.

Saidas:
  report/metrics.csv       uma linha por momento analisado
  report/summary.json      agregados por video + comparacao person vs full
  report/panels/*.png      painel 1x4 (original | bbox | mascara | overlay)
  report/panels/conf_*.png grafico da confianca ao longo do tempo
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from . import config, stage_render, video_io

# Paleta categorica validada (dataviz: todos os checks PASS em modo claro).
SERIES_PERSON = "#2a78d6"  # slot 1 - azul
SERIES_FULL = "#eb6834"  # slot 2 - laranja
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#dedddb"
SURFACE = "#fcfcfb"


# --------------------------------------------------------------------- helpers
def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def label_matches(predicted: str, expected: str) -> bool:
    """Compara rotulo Kinetics com a acao observada, tolerando variacoes.

    Os rotulos do Kinetics-400 nem sempre batem palavra a palavra com a
    descricao natural do trecho ("push up" vs "pushing up"), por isso a
    comparacao e por sobreposicao de radicais em vez de igualdade estrita.
    """
    p, e = predicted.lower().strip(), expected.lower().strip()
    if p == e:
        return True
    pt = {w[:5] for w in p.replace("-", " ").split() if len(w) > 2}
    et = {w[:5] for w in e.replace("-", " ").split() if len(w) > 2}
    return bool(et) and et.issubset(pt)


# ------------------------------------------------------------------ agregados
def detection_metrics(detections: dict) -> dict:
    frames = detections["frames"]
    n = len(frames)
    with_person = [f for f in frames if f["boxes"]]
    confs = [b["conf"] for f in frames for b in f["boxes"]]
    ids = {b["track_id"] for f in frames for b in f["boxes"] if b["track_id"] is not None}

    # IoU temporal do sujeito principal (maior bbox) entre frames vizinhos.
    # `switches` conta quantas vezes a *maior* bbox muda de track_id: com duas
    # pessoas de tamanho parecido isso alterna sem que o tracker tenha errado,
    # por isso a metrica se chama "trocas do sujeito principal" e nao
    # "falhas de tracking".
    ious, switches = [], 0
    prev_box, prev_id = None, None
    for f in frames:
        if not f["boxes"]:
            prev_box, prev_id = None, None
            continue
        main = f["boxes"][0]
        if prev_box is not None:
            ious.append(iou(main["xyxy"], prev_box))
            if prev_id is not None and main["track_id"] != prev_id:
                switches += 1
        prev_box, prev_id = main["xyxy"], main["track_id"]

    return {
        "n_frames": n,
        "coverage": round(len(with_person) / n, 4) if n else 0.0,
        "frames_sem_pessoa": n - len(with_person),
        "conf_media": round(float(np.mean(confs)), 4) if confs else 0.0,
        "conf_min": round(float(np.min(confs)), 4) if confs else 0.0,
        "pessoas_por_frame": round(float(np.mean([len(f["boxes"]) for f in frames])), 2),
        "track_ids": len(ids),
        "trocas_sujeito_principal": switches,
        "iou_temporal_media": round(float(np.mean(ious)), 4) if ious else 0.0,
        "elapsed_s": detections.get("elapsed_s"),
    }


def mask_metrics(mask_stats: dict, detections: dict) -> dict:
    frames = mask_stats["frames"]
    fills = [b["fill_ratio"] for f in frames for b in f["boxes"] if "fill_ratio" in b]
    areas = np.array([f["union_area"] for f in frames], dtype=float)
    vazias = sum(
        1
        for f, d in zip(frames, detections["frames"])
        if d["boxes"] and f["union_area"] == 0
    )
    # variacao relativa da area entre frames vizinhos: proxy de estabilidade
    nz = areas[areas > 0]
    jitter = (
        float(np.mean(np.abs(np.diff(nz)) / nz[:-1])) if len(nz) > 1 else 0.0
    )
    return {
        "fill_ratio_medio": round(float(np.mean(fills)), 4) if fills else 0.0,
        "fill_ratio_min": round(float(np.min(fills)), 4) if fills else 0.0,
        "fill_ratio_max": round(float(np.max(fills)), 4) if fills else 0.0,
        "area_media_px": int(areas.mean()) if len(areas) else 0,
        "variacao_area_media": round(jitter, 4),
        "frames_bbox_sem_mascara": vazias,
        "elapsed_s": mask_stats.get("elapsed_s"),
    }


def action_metrics(actions: dict, expected: str) -> dict:
    windows = actions["windows"]
    top1 = [w["top1"] for w in windows]
    confs = [w["top1_conf"] for w in windows]
    acertos = [label_matches(w["top1"], expected) for w in windows]
    top5 = [
        any(label_matches(c["label"], expected) for c in w["topk"]) for w in windows
    ]
    trocas = [i for i in range(1, len(top1)) if top1[i] != top1[i - 1]]
    # confianca nas janelas de troca vs nas janelas estaveis
    conf_troca = [confs[i] for i in trocas]
    conf_estavel = [c for i, c in enumerate(confs) if i not in trocas]
    return {
        "n_janelas": len(windows),
        "classes_distintas": len(set(top1)),
        "top1_mais_frequente": max(set(top1), key=top1.count),
        "acerto_top1": round(sum(acertos) / len(acertos), 4) if acertos else 0.0,
        "acerto_top5": round(sum(top5) / len(top5), 4) if top5 else 0.0,
        "conf_media": round(float(np.mean(confs)), 4) if confs else 0.0,
        "conf_min": round(float(np.min(confs)), 4) if confs else 0.0,
        "janelas_de_troca": trocas,
        "conf_media_nas_trocas": round(float(np.mean(conf_troca)), 4) if conf_troca else None,
        "conf_media_estavel": round(float(np.mean(conf_estavel)), 4) if conf_estavel else None,
        "elapsed_s": actions.get("elapsed_s"),
    }


def roi_comparison(person: dict, full: dict, expected: str) -> dict:
    """Extensao opcional do PDF: SlowFast no recorte da pessoa vs quadro inteiro."""
    pw, fw = person["windows"], full["windows"]
    pairs = list(zip(pw, fw))
    concorda = sum(1 for p, f in pairs if p["top1"] == f["top1"])
    return {
        "janelas_comparadas": len(pairs),
        "concordancia_top1": round(concorda / len(pairs), 4) if pairs else 0.0,
        "acerto_top1_person": action_metrics(person, expected)["acerto_top1"],
        "acerto_top1_full": action_metrics(full, expected)["acerto_top1"],
        "conf_media_person": round(float(np.mean([w["top1_conf"] for w in pw])), 4),
        "conf_media_full": round(float(np.mean([w["top1_conf"] for w in fw])), 4),
    }


# --------------------------------------------------------- momentos e paineis
def pick_moments(
    detections: dict, actions: dict, mask_stats: dict, max_momentos: int = 5
) -> list[dict]:
    """Escolhe os momentos a comparar, no maximo `max_momentos` por video.

    O enunciado pede pelo menos tres momentos. Os tres primeiros sao fixos
    (inicio, meio, fim); as vagas restantes vao para os momentos em que o
    pipeline de fato falha, porque e isso que o relatorio precisa mostrar: o
    quadro de menor area de mascara e as trocas de classe de menor confianca
    (as mais informativas, ja que confianca baixa marca a fronteira da acao).
    Vídeos com muitas trocas, como `skate`, teriam dezenas de linhas sem esse
    limite e a tabela deixaria de caber no relatorio de tres paginas.
    """
    n = detections["n_frames"]
    windows = actions["windows"]
    chosen: dict[int, str] = {}

    for frac, nome in ((0.15, "início"), (0.50, "meio"), (0.85, "fim")):
        chosen[int(n * frac)] = nome

    # quadro de menor area de mascara entre os que tem pessoa detectada
    com_pessoa = [
        (f["union_area"], f["index"])
        for f, d in zip(mask_stats["frames"], detections["frames"])
        if d["boxes"]
    ]
    if com_pessoa and len(chosen) < max_momentos:
        _, worst = min(com_pessoa)
        chosen.setdefault(worst, "menor área de máscara")

    # trocas de classe, da menor para a maior confianca
    top1 = [w["top1"] for w in windows]
    trocas = [i for i in range(1, len(top1)) if top1[i] != top1[i - 1]]
    for i in sorted(trocas, key=lambda j: windows[j]["top1_conf"]):
        if len(chosen) >= max_momentos:
            break
        frame = windows[i]["frame_start"]
        chosen.setdefault(frame, f"troca de classe ({top1[i - 1]} \u2192 {top1[i]})")

    return [{"frame": min(f, n - 1), "motivo": chosen[f]} for f in sorted(chosen)]


def build_panel(video: str, frame_idx: int, roi: str = "person") -> Path:
    """Painel 1x4: original | bbox do YOLO | mascara da U-Net | overlay final."""
    src = config.video_path(video)
    out_dir = config.run_dir(video)
    detections = video_io.read_json(out_dir / "detections.json")
    actions = video_io.read_json(out_dir / f"actions_{roi}.json")
    boxes = {f["index"]: f["boxes"] for f in detections["frames"]}.get(frame_idx, [])

    original = video_io.read_frame(src, frame_idx)

    com_bbox = original.copy()
    for b in boxes:
        tid = b["track_id"]
        video_io.draw_box(
            com_bbox, b["xyxy"], f"#{tid} {b['conf']:.2f}", video_io.track_color(tid)
        )

    mask_path = out_dir / "masks" / f"{frame_idx:06d}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
    so_mascara = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    final = original.copy()
    video_io.overlay_mask(final, mask, stage_render.MASK_COLOR, alpha=0.40)
    for b in boxes:
        video_io.draw_box(final, b["xyxy"], "", video_io.track_color(b["track_id"]))
    per_frame = stage_render.action_per_frame(actions["windows"], detections["n_frames"])
    win = per_frame[frame_idx] if frame_idx < len(per_frame) else None
    if win is not None:
        video_io.draw_panel(
            final, [(f"{win['top1']} ({win['top1_conf']:.2f})", (255, 255, 255))], scale=0.5
        )

    # rotula cada quadro do painel
    tiles = []
    for img, titulo in (
        (original, "1. original"),
        (com_bbox, "2. YOLO bbox"),
        (so_mascara, "3. U-Net mascara"),
        (final, "4. integrado"),
    ):
        tile = img.copy()
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (25, 25, 25), -1)
        cv2.putText(
            tile, titulo, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA
        )
        tiles.append(tile)

    panel = cv2.hconcat(tiles)
    config.PANELS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PANELS_DIR / f"{video}_f{frame_idx:06d}.png"
    cv2.imwrite(str(out), panel)
    return out


# ------------------------------------------------------------------- graficos
def plot_confidence_grid(resultados: dict) -> Path:
    """Confianca do top-1 ao longo do tempo, um painel por video.

    Forma: mudanca-ao-longo-do-tempo com duas series -> linhas. Como a mesma
    medida e comparada entre tres videos, usa-se small multiples com eixo y
    compartilhado, em vez de tres figuras soltas - fica mais compacto e a
    comparacao entre videos passa a ser direta. Legenda sempre presente (duas
    series) e as trocas de classe recebem rotulo direto, para que a identidade
    nunca dependa so da cor.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nomes = list(resultados)
    fig, axes = plt.subplots(
        len(nomes), 1, figsize=(7.0, 1.55 * len(nomes)), dpi=200, sharey=True
    )
    if len(nomes) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for ax, nome in zip(axes, nomes):
        dados = resultados[nome]["_janelas"]
        ax.set_facecolor(SURFACE)
        for chave, color, rotulo in (
            ("person", SERIES_PERSON, "roi = pessoa"),
            ("full", SERIES_FULL, "roi = quadro inteiro"),
        ):
            janelas = dados[chave]
            ax.plot(
                [w["t_start"] for w in janelas],
                [w["top1_conf"] for w in janelas],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=rotulo,
                zorder=3,
            )

        # marca as janelas em que o top-1 muda (a pergunta 4 do enunciado)
        top1 = [w["top1"] for w in dados["person"]]
        trocas = [i for i in range(1, len(top1)) if top1[i] != top1[i - 1]]
        for ordem, i in enumerate(trocas):
            w = dados["person"][i]
            ax.axvline(w["t_start"], color=INK_SECONDARY, linewidth=0.8, linestyle=":", zorder=1)
            # alterna o deslocamento vertical para rotulos proximos nao colidirem
            ax.annotate(
                top1[i],
                xy=(w["t_start"], w["top1_conf"]),
                xytext=(2, -11 if ordem % 2 == 0 else 7),
                textcoords="offset points",
                fontsize=5.5,
                color=INK_SECONDARY,
            )

        ax.set_ylim(0, 1.08)
        ax.set_ylabel("confiança top-1", fontsize=7, color=INK_SECONDARY)
        ax.set_title(
            f"{nome} \u2014 ação observada: \u201c{resultados[nome]['acao_observada']}\u201d",
            fontsize=8,
            color=INK_PRIMARY,
            loc="left",
            pad=5,
        )
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(labelsize=6.5, colors=INK_SECONDARY)

    axes[-1].set_xlabel("tempo (s)", fontsize=7, color=INK_SECONDARY)
    leg = axes[0].legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)
    for texto in leg.get_texts():
        texto.set_color(INK_SECONDARY)

    fig.tight_layout(h_pad=0.8)
    config.PANELS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PANELS_DIR / "conf_todos.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------- orquestrador
CSV_FIELDS = [
    "video",
    "frame",
    "tempo_s",
    "motivo",
    "pessoas",
    "yolo_conf",
    "track_id",
    "iou_frame_anterior",
    "mascara_area_px",
    "fill_ratio",
    "slowfast_top1",
    "slowfast_conf",
    "slowfast_top2",
    "acerto_top1",
    "acao_observada",
    "painel",
]


def analyze_video(video: str) -> dict:
    spec = config.VIDEOS[video]
    out_dir = config.run_dir(video)
    detections = video_io.read_json(out_dir / "detections.json")
    mask_stats = video_io.read_json(out_dir / "mask_stats.json")
    person = video_io.read_json(out_dir / "actions_person.json")
    full = video_io.read_json(out_dir / "actions_full.json")

    fps = detections["fps"]
    boxes_by_idx = {f["index"]: f["boxes"] for f in detections["frames"]}
    masks_by_idx = {f["index"]: f for f in mask_stats["frames"]}
    per_frame = stage_render.action_per_frame(person["windows"], detections["n_frames"])

    rows = []
    for moment in pick_moments(detections, person, mask_stats):
        idx = moment["frame"]
        boxes = boxes_by_idx.get(idx, [])
        main = boxes[0] if boxes else None
        prev = boxes_by_idx.get(idx - 1, [])
        mstat = masks_by_idx.get(idx, {})
        win = per_frame[idx] if idx < len(per_frame) else None
        panel = build_panel(video, idx)
        fill = next((b.get("fill_ratio") for b in mstat.get("boxes", [])), None)

        rows.append(
            {
                "video": video,
                "frame": idx,
                "tempo_s": round(idx / fps, 2),
                "motivo": moment["motivo"],
                "pessoas": len(boxes),
                "yolo_conf": main["conf"] if main else "",
                "track_id": main["track_id"] if main else "",
                "iou_frame_anterior": (
                    round(iou(main["xyxy"], prev[0]["xyxy"]), 4) if main and prev else ""
                ),
                "mascara_area_px": mstat.get("union_area", ""),
                "fill_ratio": fill if fill is not None else "",
                "slowfast_top1": win["top1"] if win else "",
                "slowfast_conf": win["top1_conf"] if win else "",
                "slowfast_top2": win["topk"][1]["label"] if win else "",
                "acerto_top1": (
                    "sim" if win and label_matches(win["top1"], spec.expected_action) else "nao"
                ),
                "acao_observada": spec.expected_action,
                "painel": panel.name,
            }
        )

    return {
        "video": video,
        "papel": spec.role,
        "acao_observada": spec.expected_action,
        "licenca": spec.license,
        "deteccao": detection_metrics(detections),
        "mascara": mask_metrics(mask_stats, detections),
        "acao_person": action_metrics(person, spec.expected_action),
        "acao_full": action_metrics(full, spec.expected_action),
        "comparacao_roi": roi_comparison(person, full, spec.expected_action),
        "momentos": rows,
        # usado apenas para montar o grafico; removido antes de gravar o summary
        "_janelas": {"person": person["windows"], "full": full["windows"]},
    }


def _garantir_paineis_do_relatorio(resultados: dict) -> None:
    """Constroi os paineis que o texto do relatorio cita mas que nao sairam dos momentos.

    `pick_moments` limita quantos momentos entram na tabela, entao um quadro
    citado na analise escrita pode nao ter painel gerado. Sem esta checagem a
    figura some do PDF em silencio - foi o que aconteceu com o quadro 112 do
    `skate`, citado no erro 3.
    """
    from . import report_text

    for erro in report_text.ERROS:
        nome = erro.get("painel", "")
        destino = config.PANELS_DIR / nome
        if not nome or destino.exists():
            continue
        try:
            video, _, frame = nome.removesuffix(".png").rpartition("_f")
            idx = int(frame)
        except ValueError:
            print(f"[analyze] AVISO: painel '{nome}' nao segue o padrao <video>_f<quadro>.png")
            continue
        if video not in resultados:
            print(f"[analyze] AVISO: painel '{nome}' pede o video '{video}', que nao foi analisado")
            continue
        print(f"[analyze] painel citado no relatorio: {nome}")
        build_panel(video, idx)


def run() -> dict:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    resultados = {}
    todas_as_linhas = []

    for video in config.ALL_VIDEOS:
        if not (config.run_dir(video) / "actions_full.json").exists():
            print(f"[analyze] pulando {video}: pipeline ainda nao rodou")
            continue
        print(f"[analyze] {video}")
        res = analyze_video(video)
        resultados[video] = res
        todas_as_linhas.extend(res["momentos"])

    if resultados:
        grafico = plot_confidence_grid(resultados)
        print(f"[analyze] grafico -> {grafico}")
    for res in resultados.values():
        res.pop("_janelas", None)

    _garantir_paineis_do_relatorio(resultados)

    csv_path = config.REPORT_DIR / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(todas_as_linhas)

    video_io.write_json(config.REPORT_DIR / "summary.json", resultados)
    print(f"[analyze] {len(todas_as_linhas)} momentos -> {csv_path}")
    print(f"[analyze] agregados -> {config.REPORT_DIR / 'summary.json'}")
    return resultados
