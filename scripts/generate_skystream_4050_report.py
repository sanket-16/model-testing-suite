"""SkyStream report for the RTX 4050 / Ryzen 7 7840HS host only.

Covers server CPU, server GPU, browser WASM, and browser WebGPU.
Source: json files/ on LEGION. No RTX 3050 Ti content.

Run:
  .venv\\Scripts\\python.exe scripts/generate_skystream_4050_report.py
"""

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


OUTPUT_FILE = Path("SkyStream_RTX_4050_Benchmark_Report.docx")
CHART_FOLDER = Path("skystream_4050_report_charts")

GOOGLE_CPU_RATE = 0.3744
GOOGLE_GPU_RATE = 1.04652
AWS_GPU_RATE = 0.805
MONTHLY_HOURS = 730

MODELS = ["YOLOv8n", "YOLOv10n", "RT-DETR-S", "RT-DETR-L"]

# AMD Ryzen 7 7840HS · NVIDIA GeForce RTX 4050 (LEGION, json files/)
CPU_LATENCY = [24.2, 26.9, 145.4, 326.1]
GPU_LATENCY = [10.7, 14.5, 23.1, 33.9]
WASM_LATENCY = [328.4, 280.6, 1844.2, 3193.0]
WEBGPU_LATENCY = [58.7, 60.8, 165.1, 244.7]

CPU_FPS = [1000.0 / value for value in CPU_LATENCY]
GPU_FPS = [1000.0 / value for value in GPU_LATENCY]
WASM_FPS = [1000.0 / value for value in WASM_LATENCY]
WEBGPU_FPS = [1000.0 / value for value in WEBGPU_LATENCY]

COST_MODELS = ["YOLOv8n", "YOLOv10n", "RT-DETR-S", "RT-DETR-L"]
COST_LATENCY = [10.7, 14.5, 23.1, 33.9]

INVALID_GPU_INDEX = None

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


def hours_per_million(latency_ms):
    return latency_ms / 3.6


def cost_per_million(latency_ms, hourly_rate):
    return hours_per_million(latency_ms) * hourly_rate


def money(value):
    return f"US${value:,.2f}"


GOOGLE_COSTS = [
    cost_per_million(latency, GOOGLE_GPU_RATE) for latency in COST_LATENCY
]
AWS_COSTS = [
    cost_per_million(latency, AWS_GPU_RATE) for latency in COST_LATENCY
]


def create_chart_folder():
    CHART_FOLDER.mkdir(exist_ok=True)


def style_chart(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8DEE2", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB4BA")
    ax.spines["bottom"].set_color("#AAB4BA")
    ax.tick_params(axis="both", colors=GREY, labelsize=9)


def add_bar_labels(ax, bars, currency=False):
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        label = f"${height:.2f}" if currency else f"{height:.1f}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=DARK,
        )


def create_throughput_chart():
    output = CHART_FOLDER / "gpu_throughput_4050.png"
    x = np.arange(len(MODELS))
    width = 0.2

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    bars = [
        ax.bar(x - 1.5 * width, CPU_FPS, width, label="Server CPU", color=PURPLE),
        ax.bar(x - 0.5 * width, GPU_FPS, width, label="Server GPU", color=TEAL),
        ax.bar(x + 0.5 * width, WEBGPU_FPS, width, label="Browser WebGPU", color=BLUE),
        ax.bar(x + 1.5 * width, WASM_FPS, width, label="Browser WASM", color=ORANGE),
    ]
    for group in bars:
        add_bar_labels(ax, group)

    ax.set_title(
        "RTX 4050 throughput by runtime",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )
    ax.text(
        0,
        1.01,
        "Higher is better · Ryzen 7 7840HS host",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.set_ylabel("Frames per second", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    style_chart(ax)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor=WHITE)
    plt.close(figure)
    return output


def create_latency_chart():
    output = CHART_FOLDER / "runtime_latency_4050.png"
    x = np.arange(len(MODELS))
    width = 0.2

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    bars = [
        ax.bar(x - 1.5 * width, CPU_LATENCY, width, label="Server CPU", color=PURPLE),
        ax.bar(x - 0.5 * width, GPU_LATENCY, width, label="Server GPU", color=TEAL),
        ax.bar(x + 0.5 * width, WEBGPU_LATENCY, width, label="Browser WebGPU", color=BLUE),
        ax.bar(x + 1.5 * width, WASM_LATENCY, width, label="Browser WASM", color=ORANGE),
    ]
    for group in bars:
        add_bar_labels(ax, group)

    ax.set_title(
        "RTX 4050 latency by runtime",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )
    ax.text(
        0,
        1.01,
        "Lower is better · log scale · ms per frame",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.set_ylabel("Latency (ms / frame)", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_yscale("log")
    ax.set_ylim(5, 6000)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    style_chart(ax)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor=WHITE)
    plt.close(figure)
    return output


def create_browser_chart():
    output = CHART_FOLDER / "browser_wasm_vs_webgpu_4050.png"
    x = np.arange(len(MODELS))
    width = 0.34

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    bars_webgpu = ax.bar(
        x - width / 2, WEBGPU_FPS, width, label="Browser WebGPU", color=BLUE
    )
    bars_wasm = ax.bar(
        x + width / 2, WASM_FPS, width, label="Browser WASM", color=ORANGE
    )
    add_bar_labels(ax, bars_webgpu)
    add_bar_labels(ax, bars_wasm)

    ax.set_title(
        "Browser WebGPU vs WASM throughput",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )
    ax.text(
        0,
        1.01,
        "Higher is better · Chromium ONNX Runtime Web on Ryzen 7 7840HS",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.set_ylabel("Frames per second", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 20)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    style_chart(ax)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor=WHITE)
    plt.close(figure)
    return output


def create_cost_chart():
    output = CHART_FOLDER / "cloud_cost_4050.png"
    x = np.arange(len(COST_MODELS))
    width = 0.34

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    google_bars = ax.bar(
        x - width / 2, GOOGLE_COSTS, width, label="Google Cloud Run L4", color=TEAL
    )
    aws_bars = ax.bar(
        x + width / 2, AWS_COSTS, width, label="AWS EC2 g6.xlarge", color=ORANGE
    )
    add_bar_labels(ax, google_bars, currency=True)
    add_bar_labels(ax, aws_bars, currency=True)

    ax.set_title(
        "Estimated GPU cost per 1 million frames",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )
    ax.text(
        0,
        1.01,
        "Workload-normalised from RTX 4050 GPU latency · lower is better",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.set_ylabel("Estimated cost in US dollars", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(COST_MODELS)
    ax.set_ylim(0, 12)
    ax.legend(frameon=False, loc="upper left")
    style_chart(ax)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor=WHITE)
    plt.close(figure)
    return output


def hex_without_hash(value):
    return value.replace("#", "")


def set_cell_shading(cell, color):
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), hex_without_hash(color))


def set_cell_margins(cell, value=90):
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


def repeat_table_header(row):
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def prevent_row_split(row):
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


def set_column_widths(table, widths):
    for row in table.rows:
        for index, width in enumerate(widths):
            if index < len(row.cells):
                row.cells[index].width = Inches(width)


def add_table(
    document,
    headers,
    rows,
    widths,
    invalid_rows=None,
    numeric_columns=None,
):
    invalid_rows = invalid_rows or set()
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
            if row_index in invalid_rows:
                set_cell_shading(cell, LIGHT_AMBER)
            elif row_index % 2 == 1:
                set_cell_shading(cell, LIGHT_GREY)
            alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT
                if column_index in numeric_columns
                else WD_ALIGN_PARAGRAPH.LEFT
            )
            format_cell(cell, bold=column_index == 0, alignment=alignment)

    set_column_widths(table, widths)
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(0)
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
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor.from_string(hex_without_hash(DARK))

    title = document.styles["Title"]
    title.font.name = "Aptos Display"
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(hex_without_hash(NAVY))

    subtitle = document.styles["Subtitle"]
    subtitle.font.name = "Aptos"
    subtitle.font.size = Pt(11)
    subtitle.font.color.rgb = RGBColor.from_string(hex_without_hash(TEAL))

    heading_1 = document.styles["Heading 1"]
    heading_1.font.name = "Aptos Display"
    heading_1.font.size = Pt(15)
    heading_1.font.bold = True
    heading_1.font.color.rgb = RGBColor.from_string(hex_without_hash(NAVY))

    heading_2 = document.styles["Heading 2"]
    heading_2.font.name = "Aptos Display"
    heading_2.font.size = Pt(11.5)
    heading_2.font.bold = True
    heading_2.font.color.rgb = RGBColor.from_string(hex_without_hash(TEAL))

    header = section.header.paragraphs[0]
    header.text = "SkyStream | RTX 4050 benchmark review"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(hex_without_hash(TEAL))

    add_page_number(section.footer.paragraphs[0])


def add_title(document):
    title = document.add_paragraph(style="Title")
    title.add_run("SkyStream RTX 4050 Benchmark Report")

    subtitle = document.add_paragraph(style="Subtitle")
    subtitle.add_run(
        "AMD Ryzen 7 7840HS · RTX 4050 · "
        "Server CPU / GPU · Browser WASM / WebGPU"
    )

    detail = document.add_paragraph()
    detail.paragraph_format.space_after = Pt(8)
    run = detail.add_run(
        "VisDrone-VID · Image size 640 · YOLOv8n, YOLOv10n, RT-DETR-S, RT-DETR-L · "
        "Source: json files/"
    )
    run.bold = True
    run.font.size = Pt(9)


def add_summary(document):
    add_heading(document, "Decision snapshot")

    add_callout(
        document,
        "Bottom line: ",
        (
            "On this Ryzen 7 7840HS / RTX 4050 laptop, YOLOv8n server GPU is the "
            "live-detection pick at about 93.5 FPS. Browser WebGPU reaches about "
            "17 FPS for YOLOv8n — roughly 5.6× faster than WASM, still below "
            "server CPU and far below server GPU."
        ),
    )

    rows = [
        [
            "Fastest result",
            "YOLOv8n server GPU",
            "93.5 FPS at 10.7 ms per frame",
        ],
        [
            "Best browser path",
            "YOLOv8n WebGPU",
            "17.0 FPS — about 5.6× faster than WASM",
        ],
        [
            "Valid RT-DETR-S GPU",
            "RT-DETR-S server GPU",
            "43.3 FPS with a clean 6.3× CPU speedup",
        ],
        [
            "WASM path",
            "Compatibility only",
            "About 3 FPS for YOLO nanos; sub-1 FPS for RT-DETR",
        ],
    ]
    add_table(document, ["Decision", "Result", "Reason"], rows, [1.55, 1.85, 3.65])


def add_test_basis(document):
    add_heading(document, "1. Test basis")
    add_body(
        document,
        (
            "All numbers in this document come from one host: AMD Ryzen 7 7840HS "
            "with an NVIDIA GeForce RTX 4050 (LEGION). Results are taken from "
            "json files/ for server CPU, server GPU, browser WASM and browser "
            "WebGPU. Benchmarks use VisDrone-VID UAV clips (uav_seq_a–d), image "
            "size 640, confidence 0.25, first five seconds per clip."
        ),
    )


def add_performance(document, throughput_chart, latency_chart):
    add_heading(document, "2. Measured performance")

    rows = []
    for index, model in enumerate(MODELS):
        rows.append(
            [
                model,
                f"{CPU_LATENCY[index]:.1f} ms",
                f"{CPU_FPS[index]:.1f}",
                f"{GPU_LATENCY[index]:.1f} ms",
                f"{GPU_FPS[index]:.1f}",
                f"{WEBGPU_LATENCY[index]:.1f} ms",
                f"{WEBGPU_FPS[index]:.1f}",
                f"{WASM_LATENCY[index]:.1f} ms",
                f"{WASM_FPS[index]:.2f}",
            ]
        )

    add_table(
        document,
        [
            "Model",
            "CPU lat.",
            "CPU FPS",
            "GPU lat.",
            "GPU FPS",
            "WebGPU lat.",
            "WebGPU FPS",
            "WASM lat.",
            "WASM FPS",
        ],
        rows,
        [0.95, 0.85, 0.75, 0.85, 0.75, 0.95, 0.9, 0.9, 0.8],
        numeric_columns={1, 2, 3, 4, 5, 6, 7, 8},
    )

    add_chart(
        document,
        throughput_chart,
        "Figure 1. Throughput on the RTX 4050 host across four runtimes.",
    )
    add_chart(
        document,
        latency_chart,
        "Figure 2. Latency on a log scale. Browser WASM remains the slowest path.",
    )

    add_body(
        document,
        (
            "Ranking on this machine is server GPU ≫ server CPU ≳ browser WebGPU "
            "≫ browser WASM for the YOLO nanos. RT-DETR-S and RT-DETR-L both have "
            "valid server GPU paths here (43.3 FPS and 29.5 FPS)."
        ),
    )


def add_browser(document, browser_chart):
    add_heading(document, "3. Browser WASM vs WebGPU")

    add_body(
        document,
        (
            "Both browser paths use ONNX Runtime Web in Chromium against the same "
            "clips. WASM is CPU-bound in the browser process. WebGPU uses the "
            "WebGPU API (backed by the discrete GPU through the browser)."
        ),
    )

    add_callout(
        document,
        "Takeaway: ",
        (
            "WebGPU is clearly better than WASM for every model (YOLOv8n about "
            "5.6×), but it does not match server GPU. Use WebGPU for demos and "
            "client-side trials; use server GPU for live multi-stream UAV "
            "detection."
        ),
    )

    rows = []
    for index, model in enumerate(MODELS):
        speedup = WASM_LATENCY[index] / WEBGPU_LATENCY[index]
        vs_cpu = WEBGPU_LATENCY[index] / CPU_LATENCY[index]
        vs_gpu = WEBGPU_LATENCY[index] / GPU_LATENCY[index]
        rows.append(
            [
                model,
                f"{WASM_LATENCY[index]:.1f} ms",
                f"{WASM_FPS[index]:.2f}",
                f"{WEBGPU_LATENCY[index]:.1f} ms",
                f"{WEBGPU_FPS[index]:.1f}",
                f"{speedup:.1f}x",
                f"{vs_cpu:.1f}x",
                f"{vs_gpu:.1f}x",
            ]
        )

    add_table(
        document,
        [
            "Model",
            "WASM lat.",
            "WASM FPS",
            "WebGPU lat.",
            "WebGPU FPS",
            "vs WASM",
            "vs CPU",
            "vs GPU",
        ],
        rows,
        [1.05, 1.0, 0.9, 1.05, 1.0, 0.85, 0.8, 0.8],
        numeric_columns={1, 2, 3, 4, 5, 6, 7},
    )

    add_chart(
        document,
        browser_chart,
        "Figure 3. Browser WebGPU versus WASM throughput on the RTX 4050 host.",
    )


def add_costs(document, cost_chart):
    add_heading(document, "4. Cloud cost per 1 million frames")

    add_body(
        document,
        (
            "Workload-normalised estimates using this laptop's server GPU "
            "latencies. They are not direct NVIDIA L4 benchmarks."
        ),
    )

    rate_rows = [
        [
            "Google Cloud Run CPU",
            "4 vCPU and 16 GiB",
            money(GOOGLE_CPU_RATE),
            money(GOOGLE_CPU_RATE * MONTHLY_HOURS),
        ],
        [
            "Google Cloud Run L4",
            "L4, 4 vCPU and 16 GiB",
            money(GOOGLE_GPU_RATE),
            money(GOOGLE_GPU_RATE * MONTHLY_HOURS),
        ],
        [
            "AWS EC2 g6.xlarge",
            "L4, 4 vCPU and 16 GiB",
            money(AWS_GPU_RATE),
            money(AWS_GPU_RATE * MONTHLY_HOURS),
        ],
    ]
    add_table(
        document,
        ["Reference", "Configuration", "Hourly", "730-hour month"],
        rate_rows,
        [1.65, 2.55, 1.2, 1.45],
        numeric_columns={2, 3},
    )

    scenario_rows = []
    for index, model in enumerate(COST_MODELS):
        latency = COST_LATENCY[index]
        hours = hours_per_million(latency)
        scenario_rows.append(
            [
                model,
                f"{latency:.1f} ms",
                f"{hours:.2f} h",
                money(GOOGLE_COSTS[index]),
                money(AWS_COSTS[index]),
            ]
        )

    add_heading(document, "RTX 4050 GPU workload-normalised scenario", level=2)
    add_table(
        document,
        [
            "Model",
            "Local GPU latency",
            "GPU h / 1M",
            "Google / 1M",
            "AWS / 1M",
        ],
        scenario_rows,
        [1.25, 1.5, 1.25, 1.4, 1.3],
        numeric_columns={1, 2, 3, 4},
    )

    add_chart(
        document,
        cost_chart,
        (
            "Figure 4. Estimated L4 processing cost from RTX 4050 GPU "
            "latency. Browser paths are local and have no cloud GPU cost."
        ),
    )


def add_recommendation(document):
    add_heading(document, "5. Recommendation")
    rows = [
        [
            "Live UAV detection",
            "YOLOv8n server GPU",
            "93.5 FPS on this host; lowest estimated cloud cost",
        ],
        [
            "Newer nano YOLO",
            "YOLOv10n server GPU",
            "69 FPS with solid realtime headroom",
        ],
        [
            "Browser demo",
            "YOLOv8n or YOLOv10n WebGPU",
            "~16–17 FPS; much better than WASM",
        ],
        [
            "Browser fallback",
            "WASM only if WebGPU unavailable",
            "Too slow for live multi-stream",
        ],
        [
            "Transformer option",
            "RT-DETR-S server GPU",
            "Valid 43.3 FPS on this host",
        ],
        [
            "Heavier transformer",
            "RT-DETR-L server GPU",
            "About 29.5 FPS; quality over maximum rate",
        ],
    ]
    add_table(document, ["Use case", "Choice", "Reason"], rows, [1.75, 1.95, 3.35])
    add_body(
        document,
        (
            "Prefer server GPU for SkyStream live work on this class of laptop. "
            "Promote browser WebGPU for demos and privacy-sensitive client trials. "
            "Treat WASM as a compatibility fallback only."
        ),
    )


def add_licensing(document):
    add_heading(document, "6. Model licensing")
    add_callout(
        document,
        "Commercial note: ",
        (
            "Ultralytics YOLOv8 and Ultralytics-packaged RT-DETR-L are AGPL-3.0 "
            "unless a commercial licence is purchased. Apache-2.0 RT-DETR "
            "checkpoints avoid that copyleft path, but training-data provenance "
            "may still need review."
        ),
        warning=True,
    )
    rows = [
        [
            "YOLOv8",
            "AGPL-3.0 / Ultralytics commercial",
            "Requires Ultralytics commercial licence for closed-source products",
            "https://yolov8.com/",
        ],
        [
            "YOLOv10",
            "AGPL-3.0",
            "Products that distribute it generally need to be open-sourced under AGPL",
            "https://github.com/THU-MIG/yolov10",
        ],
        [
            "RT-DETR-S / rtdetr_r18vd",
            "Apache 2.0",
            "Free to use; COCO training-data provenance may still raise questions",
            "https://huggingface.co/PekingU/rtdetr_r18vd",
        ],
        [
            "RT-DETR-L (raw / Apache)",
            "Apache 2.0",
            "Completely free under Apache terms",
            "—",
        ],
        [
            "RT-DETR-L (Ultralytics)",
            "AGPL-3.0 / Ultralytics commercial",
            "Requires Ultralytics commercial licence for closed-source products",
            "—",
        ],
    ]
    add_table(
        document,
        ["Model", "Licence", "Notes", "Reference"],
        rows,
        [1.55, 1.85, 2.55, 1.55],
    )


def add_limitations(document):
    add_heading(document, "7. Limitations")
    items = [
        "This report covers only the Ryzen 7 7840HS / RTX 4050 host.",
        "Figures are imported from json files/ on the LEGION machine.",
        "Cloud costs are workload-normalised from local GPU latency, not L4 benches.",
        "Browser WebGPU depends on Chromium, drivers and a headed GPU-capable session.",
        "Licence notes are a working summary, not legal advice.",
        "Storage, networking, egress and logging are excluded from cost estimates.",
    ]
    for item in items:
        add_bullet(document, item)
    add_body(
        document,
        (
            "Compute hours per 1 million frames = latency_ms / 3.6. "
            "Cost = those hours × selected hourly rate."
        ),
    )


def build_document():
    create_chart_folder()
    throughput_chart = create_throughput_chart()
    latency_chart = create_latency_chart()
    browser_chart = create_browser_chart()
    cost_chart = create_cost_chart()

    document = Document()
    configure_document(document)
    document.core_properties.title = "SkyStream RTX 4050 Benchmark Report"
    document.core_properties.author = "SkyStream"
    document.core_properties.subject = (
        "Ryzen 7 7840HS / RTX 4050 CPU, GPU, browser WASM and WebGPU review"
    )

    add_title(document)
    add_summary(document)
    add_test_basis(document)
    add_performance(document, throughput_chart, latency_chart)
    document.add_page_break()
    add_browser(document, browser_chart)
    add_costs(document, cost_chart)
    add_recommendation(document)
    add_licensing(document)
    add_limitations(document)

    document.save(OUTPUT_FILE)
    print(f"Created document: {OUTPUT_FILE.resolve()}")
    print(f"Created charts: {CHART_FOLDER.resolve()}")


if __name__ == "__main__":
    build_document()