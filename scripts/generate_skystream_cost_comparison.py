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

MODELS = ["YOLOv8n", "RT-DETR-S", "RT-DETR-L"]

GPU_FPS_3050 = [59.2, np.nan, 11.2]
GPU_FPS_4050 = [76.9, 40.2, 17.5]

LATENCY_4050 = [13.0, 24.9, 57.1]

NAVY = "#17324D"
TEAL = "#2A9D8F"
PURPLE = "#817FA0"
ORANGE = "#E6A13A"
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

    figure, ax = plt.subplots(figsize=(10, 4.8), dpi=180)

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

    invalid_x = x[1] - width / 2

    ax.annotate(
        "Invalid\nCUDA fallback",
        xy=(invalid_x, 1.5),
        xytext=(invalid_x, 12),
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
        "Higher is better",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )

    ax.set_ylabel("Frames per second", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 90)

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

    figure, ax = plt.subplots(figsize=(10, 4.8), dpi=180)

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
        "RTX 4050 workload-normalised scenario, lower is better",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )

    ax.set_ylabel("Estimated cost in US dollars", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 20)

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
        "YOLOv8n, RT-DETR-S and RT-DETR-L"
    )

    detail = document.add_paragraph()
    detail.paragraph_format.space_after = Pt(8)

    run = detail.add_run(
        "VisDrone-VID | Image size 640 | "
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
            "YOLOv8n on GPU is the best fit for live SkyStream detection. "
            "The RTX 4050 reached 76.9 FPS. RT-DETR-S is also viable at "
            "40.2 FPS. RT-DETR-L benefits heavily from GPU acceleration "
            "but remains the slower option at 17.5 FPS."
        ),
    )

    rows = [
        [
            "Fastest result",
            "YOLOv8n on RTX 4050",
            "76.9 FPS at 13.0 ms per frame",
        ],
        [
            "Largest valid speedup",
            "RT-DETR-L on RTX 4050",
            "7.6x faster than its paired CPU result",
        ],
        [
            "Resolved test",
            "RT-DETR-S on RTX 4050",
            "40.2 FPS with a valid 6.1x speedup",
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
            "CPU, GPU and Browser/WASM results. The RTX 4050 run included CPU "
            "and GPU results."
        ),
    )

    add_callout(
        document,
        "Invalid result: ",
        (
            "The RTX 3050 Ti RT-DETR-S GPU test failed under CUDA 13 and "
            "reported CPU fallback. It is excluded from GPU performance and "
            "cost calculations."
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
            "2.81",
        ],
        [
            "RT-DETR-S",
            "261.7 ms",
            "3.82",
            "Invalid",
            "Invalid",
            "0.50",
        ],
        [
            "RT-DETR-L",
            "629.5 ms",
            "1.59",
            "89.2 ms",
            "11.2",
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
            "Browser FPS",
        ],
        rows_3050,
        [1.1, 1.25, 0.9, 1.25, 0.9, 1.15],
        invalid_rows={1},
        numeric_columns={1, 2, 3, 4, 5},
    )

    rows_4050 = [
        [
            "YOLOv8n",
            "26.0 ms",
            "38.5",
            "13.0 ms",
            "76.9",
            "2.0x",
        ],
        [
            "RT-DETR-S",
            "150.9 ms",
            "6.6",
            "24.9 ms",
            "40.2",
            "6.1x",
        ],
        [
            "RT-DETR-L",
            "435.8 ms",
            "2.3",
            "57.1 ms",
            "17.5",
            "7.6x",
        ],
    ]

    add_heading(document, "RTX 4050 system", level=2)

    add_table(
        document,
        [
            "Model",
            "CPU latency",
            "CPU FPS",
            "GPU latency",
            "GPU FPS",
            "GPU vs CPU",
        ],
        rows_4050,
        [1.1, 1.25, 0.9, 1.25, 0.9, 1.15],
        numeric_columns={1, 2, 3, 4, 5},
    )

    add_chart(
        document,
        throughput_chart,
        (
            "Figure 1. Measured GPU throughput. The RTX 3050 Ti "
            "RT-DETR-S result is excluded because it was CPU fallback."
        ),
    )

    add_body(
        document,
        (
            "The RTX 4050 increased YOLOv8n throughput by about 30 percent "
            "and RT-DETR-L throughput by about 56 percent compared with the "
            "RTX 3050 Ti system. RT-DETR-S cannot be compared across the two "
            "GPUs because the earlier test was invalid."
        ),
    )


def add_costs(document, cost_chart):
    add_heading(document, "3. Cloud cost comparison")

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
            "Figure 2. Estimated processing cost using RTX 4050 latency "
            "and the supplied Google and AWS L4 hourly rates."
        ),
    )

    add_body(
        document,
        (
            "AWS has the lower active L4 rate in every scenario. Google "
            "Cloud Run remains a strong fit for intermittent UAV workloads "
            "because the service can scale to zero instead of paying for an "
            "idle GPU."
        ),
    )


def add_recommendation(document):
    add_heading(document, "4. Recommendation")

    rows = [
        [
            "Live UAV detection",
            "YOLOv8n on GPU",
            "Fastest result and lowest estimated cost",
        ],
        [
            "Transformer option",
            "RT-DETR-S on GPU",
            "Valid 40.2 FPS result on RTX 4050",
        ],
        [
            "Heavier model",
            "RT-DETR-L on GPU",
            "Strong acceleration, but below 20 FPS",
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
            "workload. It provides the best live throughput and the lowest "
            "estimated processing cost. Use RT-DETR-S when its detection "
            "results justify the extra compute. Keep RT-DETR-L for workloads "
            "where model quality matters more than real-time speed."
        ),
    )


def add_limitations(document):
    add_heading(document, "5. Limitations")

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
            "Browser/WASM results were available only for the RTX 3050 Ti "
            "test."
        ),
        (
            "Storage, Firestore operations, S3, networking, egress and "
            "logging are excluded."
        ),
        (
            "YOLOv8n results must not be relabelled as YOLOv8s or YOLOv8l."
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
    cost_chart = create_cost_chart()

    document = Document()
    configure_document(document)

    document.core_properties.title = (
        "SkyStream CPU vs GPU Cost Comparison"
    )
    document.core_properties.author = "SkyStream"
    document.core_properties.subject = (
        "RTX 3050 Ti and RTX 4050 benchmark and cloud cost review"
    )

    add_title(document)
    add_summary(document)
    add_test_basis(document)
    add_performance(document, throughput_chart)

    document.add_page_break()

    add_costs(document, cost_chart)
    add_recommendation(document)
    add_limitations(document)

    document.save(OUTPUT_FILE)

    print(f"Created document: {OUTPUT_FILE.resolve()}")
    print(f"Created charts: {CHART_FOLDER.resolve()}")


if __name__ == "__main__":
    build_document()
