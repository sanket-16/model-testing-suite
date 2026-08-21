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


OUTPUT_FILE = Path("SkyStream_CPU_vs_GPU_Cost_Comparison.docx")
CHART_FOLDER = Path("skystream_report_charts")

GOOGLE_CPU_RATE = 0.3744
GOOGLE_GPU_RATE = 1.04652
AWS_GPU_RATE = 0.805
MONTHLY_HOURS = 730

MODELS = ["YOLOv8n", "YOLOv10n", "RT-DETR-S", "RT-DETR-L"]

# RTX 3050 Ti (Ryzen 9 5900HX) — GPU FPS; RT-DETR-S invalid CUDA fallback
GPU_FPS_3050 = [59.2, 44.2, np.nan, 11.2]

# RTX 4050 (LEGION) — updated json files/ run including browser
GPU_FPS_4050 = [93.5, 69.0, 43.3, 29.5]

# RTX 4050 mean GPU latency ms/frame (updated)
LATENCY_4050 = [10.7, 14.5, 23.1, 33.9]

# Browser / WASM mean latency ms/frame
BROWSER_LATENCY_3050 = [355.9, 317.6, 1990.8, 3491.2]
BROWSER_LATENCY_4050 = [328.4, 280.6, 1844.2, 3193.0]

# Browser WebGPU mean latency ms/frame (RTX 3050 Ti host, 2026-08-21)
WEBGPU_LATENCY_3050 = [79.6, 87.7, 287.5, 660.7]

BROWSER_FPS_3050 = [1000.0 / value for value in BROWSER_LATENCY_3050]
BROWSER_FPS_4050 = [1000.0 / value for value in BROWSER_LATENCY_4050]
WEBGPU_FPS_3050 = [1000.0 / value for value in WEBGPU_LATENCY_3050]

# RTX 4050 server CPU latency for runtime comparison chart
CPU_LATENCY_4050 = [24.2, 26.9, 145.4, 326.1]
CPU_LATENCY_3050 = [48.7, 65.4, 261.7, 629.5]
GPU_LATENCY_3050 = [16.9, 22.6, np.nan, 89.2]

# Index of RT-DETR-S in MODELS (3050 Ti GPU invalid)
INVALID_GPU_INDEX = 2

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


def percentage_change(old_value, new_value):
    return ((new_value / old_value) - 1) * 100


GOOGLE_COSTS_4050 = [
    cost_per_million(latency, GOOGLE_GPU_RATE)
    for latency in LATENCY_4050
]

AWS_COSTS_4050 = [
    cost_per_million(latency, AWS_GPU_RATE)
    for latency in LATENCY_4050
]


def create_chart_folder():
    CHART_FOLDER.mkdir(exist_ok=True)


def style_chart(ax):
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color="#D8DEE2",
        linewidth=0.8,
        alpha=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB4BA")
    ax.spines["bottom"].set_color("#AAB4BA")

    ax.tick_params(
        axis="both",
        colors=GREY,
        labelsize=9,
    )


def add_bar_labels(ax, bars, currency=False):
    for bar in bars:
        height = bar.get_height()

        if not np.isfinite(height):
            continue

        if currency:
            label = f"${height:.2f}"
        else:
            label = f"{height:.1f}"

        ax.annotate(
            label,
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=DARK,
        )


def create_throughput_chart():
    output = CHART_FOLDER / "gpu_throughput_comparison.png"

    x = np.arange(len(MODELS))
    width = 0.34

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)

    bars_3050 = ax.bar(
        x - width / 2,
        GPU_FPS_3050,
        width,
        label="RTX 3050 Ti",
        color=PURPLE,
    )

    bars_4050 = ax.bar(
        x + width / 2,
        GPU_FPS_4050,
        width,
        label="RTX 4050",
        color=TEAL,
    )

    add_bar_labels(ax, bars_3050)
    add_bar_labels(ax, bars_4050)

    invalid_x = x[INVALID_GPU_INDEX] - width / 2

    ax.annotate(
        "Invalid\nCUDA fallback",
        xy=(invalid_x, 1.5),
        xytext=(invalid_x, 18),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#A05A00",
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": ORANGE,
            "linewidth": 1.2,
        },
    )

    ax.set_title(
        "Measured GPU throughput",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )

    ax.text(
        0,
        1.01,
        "Higher is better · updated 4050 results from json files/",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )

    ax.set_ylabel("Frames per second", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 110)

    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=2,
    )

    style_chart(ax)

    figure.tight_layout()
    figure.savefig(
        output,
        bbox_inches="tight",
        facecolor=WHITE,
    )
    plt.close(figure)

    return output


def create_cost_chart():
    output = CHART_FOLDER / "cloud_cost_comparison.png"

    x = np.arange(len(MODELS))
    width = 0.34

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)

    google_bars = ax.bar(
        x - width / 2,
        GOOGLE_COSTS_4050,
        width,
        label="Google Cloud Run L4",
        color=TEAL,
    )

    aws_bars = ax.bar(
        x + width / 2,
        AWS_COSTS_4050,
        width,
        label="AWS EC2 g6.xlarge",
        color=ORANGE,
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
        "RTX 4050 updated latency · workload-normalised · lower is better",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )

    ax.set_ylabel("Estimated cost in US dollars", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 12)

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    style_chart(ax)

    figure.tight_layout()
    figure.savefig(
        output,
        bbox_inches="tight",
        facecolor=WHITE,
    )
    plt.close(figure)

    return output


def create_browser_chart():
    output = CHART_FOLDER / "browser_throughput_comparison.png"

    x = np.arange(len(MODELS))
    width = 0.34

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)

    bars_3050 = ax.bar(
        x - width / 2,
        BROWSER_FPS_3050,
        width,
        label="RTX 3050 Ti host (WASM)",
        color=PURPLE,
    )

    bars_4050 = ax.bar(
        x + width / 2,
        BROWSER_FPS_4050,
        width,
        label="RTX 4050 host (WASM)",
        color=BLUE,
    )

    add_bar_labels(ax, bars_3050)
    add_bar_labels(ax, bars_4050)

    ax.set_title(
        "Measured Browser / WASM throughput",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )

    ax.text(
        0,
        1.01,
        "Higher is better · Chromium ONNX Runtime Web (WASM)",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )

    ax.set_ylabel("Frames per second", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 4.5)

    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=2,
    )

    style_chart(ax)

    figure.tight_layout()
    figure.savefig(
        output,
        bbox_inches="tight",
        facecolor=WHITE,
    )
    plt.close(figure)

    return output


def create_webgpu_chart():
    """WASM vs WebGPU on the RTX 3050 Ti host only."""
    output = CHART_FOLDER / "browser_webgpu_vs_wasm_3050ti.png"

    x = np.arange(len(MODELS))
    width = 0.34

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)

    bars_webgpu = ax.bar(
        x - width / 2,
        WEBGPU_FPS_3050,
        width,
        label="Browser WebGPU",
        color=TEAL,
    )
    bars_wasm = ax.bar(
        x + width / 2,
        BROWSER_FPS_3050,
        width,
        label="Browser WASM",
        color=ORANGE,
    )

    add_bar_labels(ax, bars_webgpu)
    add_bar_labels(ax, bars_wasm)

    ax.set_title(
        "RTX 3050 Ti browser WebGPU vs WASM",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=15,
    )
    ax.text(
        0,
        1.01,
        "Higher is better · Ryzen 9 5900HX host · headed Chrome",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.set_ylabel("Frames per second", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 15)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    style_chart(ax)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor=WHITE)
    plt.close(figure)
    return output


def create_runtime_latency_chart():
    """CPU vs GPU vs Browser latency on the updated RTX 4050 system."""
    output = CHART_FOLDER / "runtime_latency_comparison.png"

    x = np.arange(len(MODELS))
    width = 0.25

    figure, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)

    bars_cpu = ax.bar(
        x - width,
        CPU_LATENCY_4050,
        width,
        label="Server CPU",
        color=PURPLE,
    )
    bars_gpu = ax.bar(
        x,
        LATENCY_4050,
        width,
        label="Server GPU",
        color=TEAL,
    )
    bars_browser = ax.bar(
        x + width,
        BROWSER_LATENCY_4050,
        width,
        label="Browser / WASM",
        color=ORANGE,
    )

    add_bar_labels(ax, bars_cpu)
    add_bar_labels(ax, bars_gpu)
    add_bar_labels(ax, bars_browser)

    ax.set_title(
        "RTX 4050 runtime latency by path",
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

    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=3,
    )

    style_chart(ax)

    figure.tight_layout()
    figure.savefig(
        output,
        bbox_inches="tight",
        facecolor=WHITE,
    )
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


def format_cell(
    cell,
    bold=False,
    color=DARK,
    alignment=WD_ALIGN_PARAGRAPH.LEFT,
):
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
            run.font.color.rgb = RGBColor.from_string(
                hex_without_hash(color)
            )


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

    table = document.add_table(
        rows=1,
        cols=len(headers),
    )

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

        format_cell(
            cell,
            bold=True,
            color=WHITE,
            alignment=alignment,
        )

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

            format_cell(
                cell,
                bold=column_index == 0,
                alignment=alignment,
            )

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
    paragraph = document.add_paragraph(
        text,
        style="List Bullet",
    )
    paragraph.paragraph_format.space_after = Pt(3)
    return paragraph


def add_callout(document, title, text, warning=False):
    table = document.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)

    color = LIGHT_AMBER if warning else LIGHT_TEAL
    set_cell_shading(cell, color)
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
    run.add_picture(
        str(chart_path),
        width=Inches(6.9),
    )

    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(7)

    caption_run = caption_paragraph.add_run(caption)
    caption_run.italic = True
    caption_run.font.name = "Aptos"
    caption_run.font.size = Pt(8)
    caption_run.font.color.rgb = RGBColor.from_string(
        hex_without_hash(GREY)
    )


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
    normal.font.color.rgb = RGBColor.from_string(
        hex_without_hash(DARK)
    )

    title = document.styles["Title"]
    title.font.name = "Aptos Display"
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(
        hex_without_hash(NAVY)
    )

    subtitle = document.styles["Subtitle"]
    subtitle.font.name = "Aptos"
    subtitle.font.size = Pt(11)
    subtitle.font.color.rgb = RGBColor.from_string(
        hex_without_hash(TEAL)
    )

    heading_1 = document.styles["Heading 1"]
    heading_1.font.name = "Aptos Display"
    heading_1.font.size = Pt(15)
    heading_1.font.bold = True
    heading_1.font.color.rgb = RGBColor.from_string(
        hex_without_hash(NAVY)
    )

    heading_2 = document.styles["Heading 2"]
    heading_2.font.name = "Aptos Display"
    heading_2.font.size = Pt(11.5)
    heading_2.font.bold = True
    heading_2.font.color.rgb = RGBColor.from_string(
        hex_without_hash(TEAL)
    )

    header = section.header.paragraphs[0]
    header.text = "SkyStream | Compute benchmark and cost review"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    for run in header.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(
            hex_without_hash(TEAL)
        )

    add_page_number(section.footer.paragraphs[0])


def add_title(document):
    title = document.add_paragraph(style="Title")
    title.add_run("SkyStream CPU vs GPU Cost Comparison")

    subtitle = document.add_paragraph(style="Subtitle")
    subtitle.add_run(
        "RTX 3050 Ti and RTX 4050 findings | "
        "YOLOv8n, YOLOv10n, RT-DETR-S and RT-DETR-L"
    )

    detail = document.add_paragraph()
    detail.paragraph_format.space_after = Pt(8)

    run = detail.add_run(
        "VisDrone-VID | Image size 640 | "
        "Updated 4050 results with browser | "
        "Workload-normalised cloud estimates"
    )
    run.bold = True
    run.font.size = Pt(9)


def add_summary(document):
    add_heading(document, "Decision snapshot")

    add_callout(
        document,
        "Bottom line: ",
        (
            "YOLOv8n on GPU remains the best fit for live SkyStream detection. "
            "On the updated RTX 4050 run it reached 93.5 FPS. YOLOv10n is also "
            "strong at 69.0 FPS. Browser WebGPU on the 3050 Ti host reaches "
            "about 12.6 FPS for YOLOv8n — much better than WASM, still not "
            "live multi-stream."
        ),
    )

    rows = [
        [
            "Fastest result",
            "YOLOv8n on RTX 4050",
            "93.5 FPS at 10.7 ms per frame",
        ],
        [
            "Strong nano alternative",
            "YOLOv10n on RTX 4050",
            "69.0 FPS at 14.5 ms per frame",
        ],
        [
            "Best browser result",
            "YOLOv8n WebGPU on 3050 Ti",
            "12.6 FPS — about 4.5× faster than WASM on the same host",
        ],
        [
            "Largest valid speedup",
            "RT-DETR-L on RTX 4050",
            "9.6x faster than its paired CPU result",
        ],
        [
            "Resolved RT-DETR-S",
            "RT-DETR-S on RTX 4050",
            "43.3 FPS with a valid 6.3x GPU speedup",
        ],
        [
            "Current cloud fit",
            "Google Cloud Run",
            "Firestore integration and scale to zero",
        ],
        [
            "Always-on L4",
            "AWS g6.xlarge",
            "Lower referenced hourly price",
        ],
        [
            "Licence caution",
            "YOLOv8 / Ultralytics RT-DETR-L",
            "AGPL-3.0 unless a commercial Ultralytics licence is purchased",
        ],
    ]

    add_table(
        document,
        ["Decision", "Result", "Reason"],
        rows,
        [1.55, 1.85, 3.65],
    )


def add_test_basis(document):
    add_heading(document, "1. Test basis")

    add_body(
        document,
        (
            "The benchmarks use VisDrone-VID clips at an image size of 640. "
            "The RTX 3050 Ti run used an AMD Ryzen 9 5900HX host and included "
            "CPU, GPU and Browser/WASM results for YOLOv8n, YOLOv10n, "
            "RT-DETR-S and RT-DETR-L. The updated RTX 4050 LEGION run in "
            "json files/ also includes CPU, GPU and Browser/WASM for all "
            "four models."
        ),
    )

    add_callout(
        document,
        "Invalid result: ",
        (
            "The RTX 3050 Ti RT-DETR-S GPU test failed under CUDA 13 "
            "(missing cublasLt64_13.dll) and reported CPU fallback. It is "
            "excluded from GPU performance and cost calculations. The RTX "
            "4050 RT-DETR-S GPU result is valid."
        ),
        warning=True,
    )


def add_performance(document, throughput_chart):
    add_heading(document, "2. Measured performance")

    rows_3050 = [
        [
            "YOLOv8n",
            "48.7 ms",
            "20.5",
            "16.9 ms",
            "59.2",
            "79.6 ms",
            "12.6",
            "2.81",
        ],
        [
            "YOLOv10n",
            "65.4 ms",
            "15.3",
            "22.6 ms",
            "44.2",
            "87.7 ms",
            "11.4",
            "3.15",
        ],
        [
            "RT-DETR-S",
            "261.7 ms",
            "3.82",
            "Invalid",
            "Invalid",
            "287.5 ms",
            "3.5",
            "0.50",
        ],
        [
            "RT-DETR-L",
            "629.5 ms",
            "1.59",
            "89.2 ms",
            "11.2",
            "660.7 ms",
            "1.5",
            "0.29",
        ],
    ]

    add_heading(document, "RTX 3050 Ti system", level=2)

    add_table(
        document,
        [
            "Model",
            "CPU latency",
            "CPU FPS",
            "GPU latency",
            "GPU FPS",
            "WebGPU lat.",
            "WebGPU FPS",
            "WASM FPS",
        ],
        rows_3050,
        [0.95, 1.0, 0.75, 1.0, 0.75, 1.0, 0.9, 0.85],
        invalid_rows={2},
        numeric_columns={1, 2, 3, 4, 5, 6, 7},
    )

    rows_4050 = [
        [
            "YOLOv8n",
            "24.2 ms",
            "41.3",
            "10.7 ms",
            "93.5",
            "3.0",
            "2.3x",
        ],
        [
            "YOLOv10n",
            "26.9 ms",
            "37.2",
            "14.5 ms",
            "69.0",
            "3.6",
            "1.9x",
        ],
        [
            "RT-DETR-S",
            "145.4 ms",
            "6.9",
            "23.1 ms",
            "43.3",
            "0.5",
            "6.3x",
        ],
        [
            "RT-DETR-L",
            "326.1 ms",
            "3.1",
            "33.9 ms",
            "29.5",
            "0.3",
            "9.6x",
        ],
    ]

    add_heading(document, "RTX 4050 system (updated)", level=2)

    add_table(
        document,
        [
            "Model",
            "CPU latency",
            "CPU FPS",
            "GPU latency",
            "GPU FPS",
            "Browser FPS",
            "GPU vs CPU",
        ],
        rows_4050,
        [1.0, 1.1, 0.8, 1.1, 0.8, 1.0, 1.0],
        numeric_columns={1, 2, 3, 4, 5, 6},
    )

    add_chart(
        document,
        throughput_chart,
        (
            "Figure 1. Measured GPU throughput. The RTX 3050 Ti "
            "RT-DETR-S result is excluded because it was CPU fallback. "
            "RTX 4050 figures use the updated json files/ dataset."
        ),
    )

    yolo8_gain = percentage_change(59.2, 93.5)
    yolo10_gain = percentage_change(44.2, 69.0)
    rtdetr_l_gain = percentage_change(11.2, 29.5)

    add_body(
        document,
        (
            f"Against the RTX 3050 Ti system, the updated RTX 4050 run "
            f"increased YOLOv8n GPU throughput by about {yolo8_gain:.0f}%, "
            f"YOLOv10n by about {yolo10_gain:.0f}%, and RT-DETR-L by about "
            f"{rtdetr_l_gain:.0f}%. RT-DETR-S cannot be compared on the "
            f"3050 Ti GPU path because that earlier test was invalid. An "
            f"earlier incomplete 4050 export had shown only about 17.5 FPS "
            f"for RT-DETR-L; the refreshed run is stable near 29.5 FPS."
        ),
    )


def add_browser(document, browser_chart, runtime_chart, webgpu_chart):
    add_heading(document, "3. Browser WASM and WebGPU results")

    add_body(
        document,
        (
            "Browser inference uses ONNX Runtime Web in Chromium against the "
            "same VisDrone UAV clips and image size 640. WASM results exist for "
            "both hosts. WebGPU results are from the Ryzen 9 5900HX / RTX 3050 Ti "
            "host (headed Chrome)."
        ),
    )

    add_callout(
        document,
        "Takeaway: ",
        (
            "On the 3050 Ti host, browser WebGPU is about 4–5× faster than WASM "
            "for YOLO models (YOLOv8n ~12.6 FPS vs ~2.8 FPS) but still slower than "
            "server CPU and far slower than server GPU. Prefer WebGPU for demos; "
            "keep WASM as a fallback."
        ),
    )

    browser_rows = []

    for index, model in enumerate(MODELS):
        lat_3050 = BROWSER_LATENCY_3050[index]
        lat_4050 = BROWSER_LATENCY_4050[index]
        fps_3050 = BROWSER_FPS_3050[index]
        fps_4050 = BROWSER_FPS_4050[index]
        vs_gpu = lat_4050 / LATENCY_4050[index]
        vs_cpu = lat_4050 / CPU_LATENCY_4050[index]

        browser_rows.append(
            [
                model,
                f"{lat_3050:.1f} ms",
                f"{fps_3050:.2f}",
                f"{lat_4050:.1f} ms",
                f"{fps_4050:.2f}",
                f"{vs_cpu:.1f}x",
                f"{vs_gpu:.0f}x",
            ]
        )

    add_heading(document, "Browser WASM latency and throughput", level=2)

    add_table(
        document,
        [
            "Model",
            "3050 Ti lat.",
            "3050 Ti FPS",
            "4050 lat.",
            "4050 FPS",
            "vs CPU",
            "vs GPU",
        ],
        browser_rows,
        [1.05, 1.1, 1.05, 1.05, 1.0, 0.85, 0.85],
        numeric_columns={1, 2, 3, 4, 5, 6},
    )

    webgpu_rows = []
    for index, model in enumerate(MODELS):
        wasm_lat = BROWSER_LATENCY_3050[index]
        webgpu_lat = WEBGPU_LATENCY_3050[index]
        gpu_lat = GPU_LATENCY_3050[index]
        vs_gpu = (
            "n/a"
            if not np.isfinite(gpu_lat)
            else f"{webgpu_lat / gpu_lat:.1f}x"
        )
        webgpu_rows.append(
            [
                model,
                f"{wasm_lat:.1f} ms",
                f"{BROWSER_FPS_3050[index]:.2f}",
                f"{webgpu_lat:.1f} ms",
                f"{WEBGPU_FPS_3050[index]:.1f}",
                f"{wasm_lat / webgpu_lat:.1f}x",
                f"{webgpu_lat / CPU_LATENCY_3050[index]:.1f}x",
                vs_gpu,
            ]
        )

    add_heading(document, "RTX 3050 Ti browser WebGPU vs WASM", level=2)

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
        webgpu_rows,
        [1.0, 0.95, 0.9, 1.05, 1.0, 0.85, 0.8, 0.8],
        numeric_columns={1, 2, 3, 4, 5, 6, 7},
    )

    add_chart(
        document,
        browser_chart,
        (
            "Figure 2. Browser / WASM throughput on both host systems. "
            "Nano YOLO models reach about 3 FPS; RT-DETR stays below 1 FPS."
        ),
    )

    add_chart(
        document,
        webgpu_chart,
        (
            "Figure 3. RTX 3050 Ti browser WebGPU versus WASM. WebGPU is "
            "clearly faster but still below server GPU."
        ),
    )

    add_chart(
        document,
        runtime_chart,
        (
            "Figure 4. RTX 4050 CPU vs GPU vs Browser WASM latency on a log "
            "scale. WASM remains the slowest path for every model."
        ),
    )

    add_body(
        document,
        (
            "WASM results are close across the two laptops for the YOLO models, "
            "which is expected because WASM is CPU-bound in the browser process. "
            "WebGPU on the 3050 Ti host closes much of that gap versus WASM but "
            "does not replace a server GPU service for live multi-stream UAV "
            "detection."
        ),
    )


def add_costs(document, cost_chart):
    add_heading(document, "4. Cloud cost comparison")

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

    for index, model in enumerate(MODELS):
        latency = LATENCY_4050[index]
        hours = hours_per_million(latency)

        scenario_rows.append(
            [
                model,
                f"{latency:.1f} ms",
                f"{hours:.2f} h",
                money(GOOGLE_COSTS_4050[index]),
                money(AWS_COSTS_4050[index]),
            ]
        )

    add_heading(
        document,
        "RTX 4050 workload-normalised scenario",
        level=2,
    )

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
            "Figure 5. Estimated processing cost using updated RTX 4050 "
            "GPU latency and the supplied Google and AWS L4 hourly rates. "
            "Browser inference is local and is not included in cloud cost."
        ),
    )

    add_body(
        document,
        (
            f"At these rates, YOLOv8n costs about "
            f"{money(GOOGLE_COSTS_4050[0])} per 1 million frames on Google "
            f"and {money(AWS_COSTS_4050[0])} on AWS. YOLOv10n is about "
            f"{money(GOOGLE_COSTS_4050[1])} / {money(AWS_COSTS_4050[1])}. "
            f"RT-DETR-S is about {money(GOOGLE_COSTS_4050[2])} / "
            f"{money(AWS_COSTS_4050[2])}. RT-DETR-L is about "
            f"{money(GOOGLE_COSTS_4050[3])} / {money(AWS_COSTS_4050[3])}. "
            "AWS remains cheaper while an L4 stays active; Google Cloud Run "
            "can still win for intermittent UAV workloads when scale to zero "
            "avoids idle GPU time."
        ),
    )


def add_recommendation(document):
    add_heading(document, "5. Recommendation")

    rows = [
        [
            "Live UAV detection",
            "YOLOv8n on GPU",
            "Fastest result (93.5 FPS) and lowest estimated cost",
        ],
        [
            "Newer nano YOLO",
            "YOLOv10n on GPU",
            "69.0 FPS on RTX 4050 with solid headroom",
        ],
        [
            "Browser demo / offline client",
            "YOLOv8n WebGPU",
            "~12.6 FPS on 3050 Ti; much better than WASM",
        ],
        [
            "Transformer option",
            "RT-DETR-S on GPU",
            "Valid 43.3 FPS result on RTX 4050",
        ],
        [
            "Heavier model",
            "RT-DETR-L on GPU",
            "Strong acceleration; now about 29.5 FPS on 4050",
        ],
        [
            "Intermittent cloud use",
            "Google Cloud Run",
            "Existing integration and scale to zero",
        ],
        [
            "Continuous cloud use",
            "AWS g6.xlarge",
            "Lower referenced active L4 price",
        ],
    ]

    add_table(
        document,
        ["Use case", "Choice", "Reason"],
        rows,
        [1.75, 1.75, 3.55],
    )

    add_body(
        document,
        (
            "Start with YOLOv8n on a GPU service for the current SkyStream "
            "workload. It still provides the best live throughput and the "
            "lowest estimated processing cost. Use YOLOv10n when you want a "
            "newer nano detector with similar realtime behaviour. Keep "
            "Browser / WASM for demos and client-side trials only. Use "
            "RT-DETR-S when its detection results justify the extra compute. "
            "Keep RT-DETR-L for workloads where model quality matters more "
            "than maximum frame rate."
        ),
    )


def add_licensing(document):
    add_heading(document, "6. Model licensing")

    add_body(
        document,
        (
            "Benchmark speed does not decide commercial readiness. The table "
            "below summarises the licence posture for the detectors used in "
            "this review. Confirm current terms with legal before shipping."
        ),
    )

    add_callout(
        document,
        "Commercial note: ",
        (
            "Ultralytics YOLOv8 and Ultralytics-packaged RT-DETR-L are "
            "AGPL-3.0 unless an Ultralytics commercial licence is purchased. "
            "Apache-2.0 RT-DETR checkpoints avoid that copyleft path, but "
            "training-data provenance (for example COCO scraping concerns) "
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
            "Free to use; COCO training-data provenance may still raise policy questions",
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

    add_body(
        document,
        (
            "For SkyStream product work, prefer an Apache-2.0 RT-DETR "
            "checkpoint when transformer quality is required and AGPL is "
            "undesirable. If Ultralytics YOLO remains the performance pick, "
            "budget for a commercial licence or keep the deployment in an "
            "AGPL-compliant open-source form."
        ),
    )


def add_limitations(document):
    add_heading(document, "7. Limitations")

    items = [
        (
            "Cloud costs are workload-normalised estimates from local "
            "latency results. They are not direct NVIDIA L4 benchmarks."
        ),
        (
            "The RTX 3050 Ti and RTX 4050 tests used different host systems, "
            "so this is not a controlled GPU-only comparison."
        ),
        (
            "The RTX 4050 CPU model, GPU power limit and software details "
            "were not supplied."
        ),
        (
            "Browser / WASM results depend on Chromium, ONNX Runtime Web and "
            "host CPU behaviour. They do not use the discrete NVIDIA GPU."
        ),
        (
            "Browser WebGPU results are currently available for the RTX 3050 Ti "
            "host only (headed Chrome)."
        ),
        (
            "Licence notes are a working summary, not legal advice. Upstream "
            "terms and training-data provenance should be re-checked before "
            "release."
        ),
        (
            "An earlier incomplete 4050 export showed unstable RT-DETR-L "
            "GPU latency on some clips; the updated json files/ run is the "
            "source used in this document."
        ),
        (
            "Storage, Firestore operations, S3, networking, egress and "
            "logging are excluded. Browser inference has no cloud GPU cost."
        ),
        (
            "YOLOv8n and YOLOv10n results must not be relabelled as larger "
            "YOLO variants."
        ),
    ]

    for item in items:
        add_bullet(document, item)

    add_body(
        document,
        (
            "Compute hours per 1 million frames are calculated as latency in "
            "milliseconds divided by 3.6. Cost is calculated by multiplying "
            "those compute hours by the selected hourly rate."
        ),
    )


def build_document():
    create_chart_folder()

    throughput_chart = create_throughput_chart()
    browser_chart = create_browser_chart()
    webgpu_chart = create_webgpu_chart()
    runtime_chart = create_runtime_latency_chart()
    cost_chart = create_cost_chart()

    document = Document()
    configure_document(document)

    document.core_properties.title = (
        "SkyStream CPU vs GPU Cost Comparison"
    )
    document.core_properties.author = "SkyStream"
    document.core_properties.subject = (
        "RTX 3050 Ti and updated RTX 4050 CPU, GPU, browser WASM and "
        "WebGPU benchmark and cloud cost review"
    )

    add_title(document)
    add_summary(document)
    add_test_basis(document)
    add_performance(document, throughput_chart)

    document.add_page_break()

    add_browser(document, browser_chart, runtime_chart, webgpu_chart)
    add_costs(document, cost_chart)
    add_recommendation(document)
    add_licensing(document)
    add_limitations(document)

    document.save(OUTPUT_FILE)

    print(f"Created document: {OUTPUT_FILE.resolve()}")
    print(f"Created charts: {CHART_FOLDER.resolve()}")


if __name__ == "__main__":
    build_document()
