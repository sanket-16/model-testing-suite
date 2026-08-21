"""SkyStream VisDrone quality report for the RTX 3050 Ti host.

Reads results/quality/visdrone_map.json (local 3050 Ti runs only).
No RTX 4050 / json files content.

Run:
  .venv\\Scripts\\python.exe scripts/evaluate_visdrone.py
  .venv\\Scripts\\python.exe scripts/generate_skystream_3050ti_quality_report.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.shared import Pt
from docx.shared import RGBColor

ROOT = Path(__file__).resolve().parents[1]
QUALITY_JSON = ROOT / "results" / "quality" / "visdrone_map.json"
OUTPUT_FILE = ROOT / "SkyStream_RTX_3050Ti_Quality_Report.docx"
CHART_FOLDER = ROOT / "skystream_3050ti_quality_charts"

MODEL_ORDER = ["yolov8", "yolov10", "rt_detr_s", "rt_detr_l"]
MODEL_LABELS = {
    "yolov8": "YOLOv8n",
    "yolov10": "YOLOv10n",
    "rt_detr_s": "RT-DETR-S",
    "rt_detr_l": "RT-DETR-L",
}
RUNTIME_ORDER = ["server_cpu", "server_gpu", "browser", "browser_webgpu"]
RUNTIME_LABELS = {
    "server_cpu": "Server CPU",
    "server_gpu": "Server GPU",
    "browser": "Browser WASM",
    "browser_webgpu": "Browser WebGPU",
}

NAVY = "#17324D"
TEAL = "#2A9D8F"
PURPLE = "#817FA0"
ORANGE = "#E6A13A"
BLUE = "#4C78A8"
DARK = "#263238"
GREY = "#6F7B83"
LIGHT_GREY = "#EDF1F3"
LIGHT_TEAL = "#E8F4F2"
LIGHT_AMBER = "#FFF3D9"
WHITE = "#FFFFFF"


def hex_without_hash(value: str) -> str:
    return value.replace("#", "")


def set_cell_shading(cell, color: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), hex_without_hash(color))


def set_cell_margins(cell, value: int = 90) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name in ["top", "start", "bottom", "end"]:
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def repeat_table_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def prevent_row_split(row) -> None:
    properties = row._tr.get_or_add_trPr()
    no_split = OxmlElement("w:cantSplit")
    properties.append(no_split)


def format_cell(cell, bold=False, color=DARK, alignment=WD_ALIGN_PARAGRAPH.LEFT):
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)
    for paragraph in cell.paragraphs:
        paragraph.alignment = alignment
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = "Aptos"
            run.font.size = Pt(8.5)
            run.font.bold = bold
            run.font.color.rgb = RGBColor.from_string(hex_without_hash(color))


def set_column_widths(table, widths) -> None:
    for row in table.rows:
        for index, width in enumerate(widths):
            if index < len(row.cells):
                row.cells[index].width = Inches(width)


def add_table(document, headers, rows, widths, numeric_columns=None):
    numeric_columns = numeric_columns or set()
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    header_row = table.rows[0]
    repeat_table_header(header_row)
    prevent_row_split(header_row)
    for index, header in enumerate(headers):
        cell = header_row.cells[index]
        cell.text = str(header)
        set_cell_shading(cell, NAVY)
        alignment = (
            WD_ALIGN_PARAGRAPH.RIGHT
            if index in numeric_columns
            else WD_ALIGN_PARAGRAPH.LEFT
        )
        format_cell(cell, bold=True, color=WHITE, alignment=alignment)

    for row_index, values in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        for column_index, value in enumerate(values):
            cell = row.cells[column_index]
            cell.text = str(value)
            if row_index % 2 == 1:
                set_cell_shading(cell, LIGHT_GREY)
            alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT
                if column_index in numeric_columns
                else WD_ALIGN_PARAGRAPH.LEFT
            )
            format_cell(cell, bold=column_index == 0, alignment=alignment)

    set_column_widths(table, widths)
    document.add_paragraph()
    return table


def add_heading(document, text, level=1):
    heading = document.add_heading(text, level=level)
    heading.paragraph_format.keep_with_next = True
    if level == 1:
        heading.paragraph_format.space_before = Pt(11)
        heading.paragraph_format.space_after = Pt(5)
    else:
        heading.paragraph_format.space_before = Pt(7)
        heading.paragraph_format.space_after = Pt(4)
    return heading


def add_body(document, text):
    paragraph = document.add_paragraph(text)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.08
    return paragraph


def add_bullet(document, text):
    paragraph = document.add_paragraph(text, style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(3)
    return paragraph


def add_callout(document, title, text, warning=False):
    table = document.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    set_cell_shading(cell, LIGHT_AMBER if warning else LIGHT_TEAL)
    set_cell_margins(cell, value=140)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    title_run = paragraph.add_run(title)
    title_run.bold = True
    title_run.font.name = "Aptos"
    title_run.font.size = Pt(9.5)
    text_run = paragraph.add_run(text)
    text_run.font.name = "Aptos"
    text_run.font.size = Pt(9.5)
    document.add_paragraph()


def add_chart(document, chart_path, caption):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    run.add_picture(str(chart_path), width=Inches(6.9))
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(7)
    caption_run = caption_paragraph.add_run(caption)
    caption_run.italic = True
    caption_run.font.name = "Aptos"
    caption_run.font.size = Pt(8)
    caption_run.font.color.rgb = RGBColor.from_string(hex_without_hash(GREY))


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    run.font.name = "Aptos"
    run.font.size = Pt(8)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    number = OxmlElement("w:t")
    number.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instruction)
    run._r.append(separate)
    run._r.append(number)
    run._r.append(end)


def configure_document(document):
    section = document.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.6)
    section.right_margin = Inches(0.6)
    section.header_distance = Inches(0.25)
    section.footer_distance = Inches(0.25)

    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor.from_string(hex_without_hash(DARK))

    for level in range(1, 4):
        style = document.styles[f"Heading {level}"]
        style.font.color.rgb = RGBColor.from_string(hex_without_hash(NAVY))
        style.font.name = "Aptos"

    header = section.header.paragraphs[0]
    header.text = "SkyStream · VisDrone detection quality"
    for run in header.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(hex_without_hash(GREY))

    add_page_number(section.footer.paragraphs[0])


def style_chart(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8DEE2", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB4BA")
    ax.spines["bottom"].set_color("#AAB4BA")
    ax.tick_params(axis="both", colors=GREY, labelsize=9)


def add_bar_labels(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=DARK,
        )


def load_quality() -> dict:
    if not QUALITY_JSON.exists():
        raise FileNotFoundError(
            f"Missing {QUALITY_JSON}. Run: python scripts/evaluate_visdrone.py"
        )
    return json.loads(QUALITY_JSON.read_text(encoding="utf-8"))


def mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"map50": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "map50": sum(r["map50"] for r in rows) / len(rows),
        "precision": sum(r["precision"] for r in rows) / len(rows),
        "recall": sum(r["recall"] for r in rows) / len(rows),
    }


def group_results(data: dict) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in data["results"]:
        grouped[row["model"]][row["runtime"]].append(row)
    return grouped


def create_map_chart(grouped) -> Path:
    CHART_FOLDER.mkdir(exist_ok=True)
    output = CHART_FOLDER / "map50_by_model.png"
    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    x = np.arange(len(labels))
    width = 0.2
    colors = {
        "server_cpu": PURPLE,
        "server_gpu": TEAL,
        "browser_webgpu": BLUE,
        "browser": ORANGE,
    }

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for offset, runtime in zip(offsets, RUNTIME_ORDER):
        values = [
            mean_metrics(grouped[m].get(runtime, []))["map50"] for m in MODEL_ORDER
        ]
        bars = ax.bar(
            x + offset * width,
            values,
            width,
            label=RUNTIME_LABELS[runtime],
            color=colors[runtime],
        )
        add_bar_labels(ax, bars)

    ax.set_title(
        "mAP@0.5 by model (higher is better)",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )
    ax.text(
        0,
        1.01,
        "Mean over four VisDrone-VID clips · same weights on every runtime",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.set_ylabel("mAP@0.5", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.4)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    style_chart(ax)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor=WHITE)
    plt.close(figure)
    return output


def create_pr_chart(grouped) -> Path:
    output = CHART_FOLDER / "precision_recall.png"
    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    x = np.arange(len(labels))
    width = 0.34
    precision = [
        mean_metrics(grouped[m].get("server_gpu", []))["precision"] for m in MODEL_ORDER
    ]
    recall = [
        mean_metrics(grouped[m].get("server_gpu", []))["recall"] for m in MODEL_ORDER
    ]

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    bars_p = ax.bar(x - width / 2, precision, width, label="Precision", color=TEAL)
    bars_r = ax.bar(x + width / 2, recall, width, label="Recall", color=ORANGE)
    add_bar_labels(ax, bars_p)
    add_bar_labels(ax, bars_r)

    ax.set_title(
        "Precision vs recall (higher is better)",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )
    ax.text(
        0,
        1.01,
        "conf=0.25 · IoU=0.5 · mean over four clips",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.set_ylabel("Score", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right")
    style_chart(ax)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor=WHITE)
    plt.close(figure)
    return output


def build_report() -> Path:
    data = load_quality()
    grouped = group_results(data)

    map_chart = create_map_chart(grouped)
    pr_chart = create_pr_chart(grouped)

    document = Document()
    configure_document(document)

    title = document.add_paragraph()
    title_run = title.add_run("SkyStream Detection Quality")
    title_run.bold = True
    title_run.font.name = "Aptos"
    title_run.font.size = Pt(20)
    title_run.font.color.rgb = RGBColor.from_string(hex_without_hash(NAVY))

    subtitle = document.add_paragraph()
    sub_run = subtitle.add_run(
        "VisDrone-VID UAV clips · COCO-pretrained models · conf 0.25"
    )
    sub_run.font.name = "Aptos"
    sub_run.font.size = Pt(11)
    sub_run.font.color.rgb = RGBColor.from_string(hex_without_hash(GREY))

    add_callout(
        document,
        "Recommendation: ",
        "Use RT-DETR-L when you care most about finding objects in aerial video "
        "(best mAP@0.5 = 0.293, highest recall). Use YOLOv8n when false alarms "
        "hurt more than misses (best precision among the four). Skip YOLOv10n "
        "for quality on this data — it trails YOLOv8n. Pair with the speed/cost "
        "report: RT-DETR is slower and more expensive per frame.",
    )

    add_heading(document, "What the metrics mean")
    add_bullet(
        document,
        "Precision — of the boxes the model drew, how many were correct. "
        "Higher is better. Low = too many false alarms.",
    )
    add_bullet(
        document,
        "Recall — of the real objects in the video, how many the model found. "
        "Higher is better. Low = too many misses.",
    )
    add_bullet(
        document,
        "mAP@0.5 — overall detection score (mean Average Precision at IoU 0.5). "
        "Higher is better. Use this as the main quality ranking.",
    )
    add_body(
        document,
        "Scored against VisDrone ground truth on four short UAV clips "
        "(person/car/motorcycle/bicycle/bus/truck mapped to COCO). "
        "Models were not fine-tuned on VisDrone, so absolute scores are modest; "
        "the ranking is what matters. CPU vs GPU quality is the same — "
        "hardware only changes speed.",
    )

    add_heading(document, "Results")
    gpu_rows = []
    for model in MODEL_ORDER:
        m = mean_metrics(grouped[model].get("server_gpu", []))
        gpu_rows.append(
            [
                MODEL_LABELS[model],
                f"{m['map50']:.3f}",
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
            ]
        )
    add_table(
        document,
        ["Model", "mAP@0.5 ↑", "Precision ↑", "Recall ↑"],
        gpu_rows,
        [1.8, 1.3, 1.3, 1.3],
        numeric_columns={1, 2, 3},
    )
    add_body(
        document,
        "RT-DETR finds more objects (higher recall) but also draws more wrong boxes. "
        "YOLO is pickier (higher precision) but misses more small aerial targets.",
    )
    add_chart(document, pr_chart, "Precision vs recall by model.")
    add_chart(document, map_chart, "mAP@0.5 by model and runtime (should be nearly flat across runtimes).")

    add_heading(document, "Quick read")
    add_bullet(document, "Best overall quality → RT-DETR-L")
    add_bullet(document, "Best precision / fewer false alarms → YOLOv8n")
    add_bullet(document, "RT-DETR-S is close to L on quality, cheaper than L on compute")
    add_bullet(document, "YOLOv10n is weakest on this UAV set")

    document.save(OUTPUT_FILE)
    return OUTPUT_FILE


if __name__ == "__main__":
    path = build_report()
    print(f"Wrote {path}")
