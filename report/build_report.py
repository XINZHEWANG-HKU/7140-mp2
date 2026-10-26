"""Render the English report from the accompanying Markdown source."""
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               PageBreak, Preformatted, Flowable)
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = A4
MARGIN = 18 * mm
CONTENT = WIDTH - 2 * MARGIN
NAVY = colors.HexColor("#16324F")
TEAL = colors.HexColor("#087E8B")
PALE = colors.HexColor("#EEF4F7")
GRAY = colors.HexColor("#485463")

styles = {
    "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=10.1, leading=13.7,
                           spaceAfter=6.5, textColor=colors.HexColor("#202C38")),
    "h1": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=25, leading=29,
                         textColor=NAVY, spaceAfter=14),
    "h2": ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=15, leading=19,
                         textColor=NAVY, spaceBefore=5, spaceAfter=8, keepWithNext=True),
    "h3": ParagraphStyle("Subheading", fontName="Helvetica-Bold", fontSize=11.2, leading=15,
                         textColor=TEAL, spaceBefore=6, spaceAfter=5, keepWithNext=True),
    "cell": ParagraphStyle("Cell", fontName="Helvetica", fontSize=9.25, leading=12,
                           textColor=colors.HexColor("#202C38")),
    "th": ParagraphStyle("TableHeading", fontName="Helvetica-Bold", fontSize=9.25, leading=12,
                         textColor=colors.white),
    "code": ParagraphStyle("Code", fontName="Courier", fontSize=8.5, leading=11.7,
                           backColor=PALE, borderPadding=7, spaceBefore=3, spaceAfter=10),
}


def inline(text):
    text = escape(text)
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)


class Pipeline(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width, self.height = CONTENT, 86

    def draw(self):
        canvas = self.canv
        boxes = [(0, 20, 140, 46, ["Texts + fixed embeddings", "Training-only normalization"]),
                 (165, 45, 155, 29, ["3 neural rankers"]),
                 (165, 5, 155, 29, ["15 depth-4 trees"]),
                 (345, 20, CONTENT - 345, 46, ["50:50 probability mix", "Select one candidate"])]
        for x, y, w, h, lines in boxes:
            canvas.setFillColor(PALE)
            canvas.setStrokeColor(TEAL)
            canvas.roundRect(x, y, w, h, 5, fill=1, stroke=1)
            canvas.setFillColor(NAVY)
            for index, line in enumerate(lines):
                canvas.setFont("Helvetica-Bold" if index == 0 else "Helvetica", 8.3)
                canvas.drawCentredString(x + w / 2, y + h / 2 + (len(lines) - 1) * 5 - index * 11 - 3, line)
        canvas.setStrokeColor(TEAL)
        for start, end in (((140, 43), (165, 59)), ((140, 43), (165, 19)),
                           ((320, 59), (345, 43)), ((320, 19), (345, 43))):
            canvas.line(*start, *end)


def table(rows):
    count = len(rows[0])
    fractions = {2: [.51, .49], 3: [.55, .225, .225], 4: [.48, .18, .17, .17]}[count]
    if rows[0][0] == "Split":
        fractions = [.18, .18, .64]
    if rows[0][0] == "Official metric":
        fractions = [.43, .20, .20, .17]
    if rows[0][0] == "Final-method variant":
        fractions = [.43, .19, .19, .19]
    cells = [[Paragraph(inline(value), styles["th" if row_index == 0 else "cell"])
              for value in row] for row_index, row in enumerate(rows)]
    result = Table(cells, colWidths=[CONTENT * fraction for fraction in fractions], hAlign="LEFT", repeatRows=1)
    commands = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
                ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#CBD7DF"))]
    for row in range(1, len(rows)):
        if row % 2:
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    result.setStyle(TableStyle(commands))
    return [result, Spacer(1, 10)]


def parse_page(source):
    lines = source.strip().splitlines()
    output = []
    position = 0
    while position < len(lines):
        line = lines[position].strip()
        if not line:
            position += 1
            continue
        if line == "[[method_pipeline]]":
            output.extend([Pipeline(), Spacer(1, 5)])
            position += 1
        elif line.startswith("```"):
            block = []
            position += 1
            while position < len(lines) and not lines[position].startswith("```"):
                block.append(lines[position])
                position += 1
            output.append(Preformatted("\n".join(block), styles["code"]))
            position += 1
        elif line.startswith("|"):
            rows = []
            while position < len(lines) and lines[position].strip().startswith("|"):
                row = [cell.strip() for cell in lines[position].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in row):
                    rows.append(row)
                position += 1
            output.extend(table(rows))
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            output.append(Paragraph(inline(line.lstrip("#").strip()), styles[f"h{min(level, 3)}"]))
            position += 1
        else:
            block = [line]
            position += 1
            while position < len(lines) and lines[position].strip():
                block.append(lines[position].strip())
                position += 1
            output.append(Paragraph(inline(" ".join(block)), styles["body"]))
    return output


def decoration(canvas, document):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawString(MARGIN, HEIGHT - 30, "DASE7506  /  MINI-PROJECT 2")
    canvas.setFont("Helvetica", 8.5)
    canvas.drawRightString(WIDTH - MARGIN, HEIGHT - 30, "v4-wide | Final submission")
    canvas.setStrokeColor(colors.HexColor("#CBD7DF"))
    canvas.line(MARGIN, HEIGHT - 38, WIDTH - MARGIN, HEIGHT - 38)
    canvas.line(MARGIN, 38, WIDTH - MARGIN, 38)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(MARGIN, 25, "Selecting reliable mathematical solutions | 21 October 2026")
    canvas.drawRightString(WIDTH - MARGIN, 25, str(document.page))
    canvas.restoreState()


def main():
    output = ROOT / "MP2_Report.pdf"
    source = (ROOT / "report.md").read_text(encoding="utf-8")
    pages = source.split("<!-- page -->")
    story = []
    for index, page in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.extend(parse_page(page))
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=MARGIN, leftMargin=MARGIN,
                                 topMargin=53, bottomMargin=51,
                                 title="Selecting Reliable Mathematical Solutions with Compact Neural and Tree Rankers",
                                 author="DASE7506 MP2 submission", subject="Frozen v4-wide method and reproducible evaluation")
    document.build(story, onFirstPage=decoration, onLaterPages=decoration)
    reader = PdfReader(output)
    page_text = [page.extract_text() or "" for page in reader.pages]
    if len(reader.pages) > 10 or len(reader.pages) != len(pages):
        raise RuntimeError(f"Expected {len(pages)} pages within coursework limit; got {len(reader.pages)}")
    if any(len(text.strip()) < 100 for text in page_text):
        raise RuntimeError("Unexpected near-empty page")
    for fragment in ("82.87%", "387/467", "5.0066", "2.4475", "Codex", "References"):
        if fragment not in "\n".join(page_text):
            raise RuntimeError(f"Expected report evidence missing: {fragment}")
    print({"output": str(output), "pages": len(reader.pages), "words": len(source.split()),
           "page_text_characters": [len(text) for text in page_text]})


if __name__ == "__main__":
    main()
