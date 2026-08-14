"""
Generate the synthetic two-page PDF used by section_3_bedrock/06_extraction.py.

    uv run make_sample_pdf.py

Entirely invented content — no client data anywhere in this repo. It is built to
exercise the things that make real extraction hard: a table, a total that must be
summed, two date formats, a missing field, and a second page with a different
layout.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

OUT = Path(__file__).resolve().parent / "sample_docs" / "expenses.pdf"

ROWS = [
    ["Date", "Description", "Category", "Amount (SGD)"],
    ["03/02/26", "Rail pass — team offsite", "Travel", "48.00"],
    ["11/02/2026", "Printing, workshop handouts", "Materials", "126.50"],
    ["17/02/26", "Coffee, client meeting", "Hospitality", "22.80"],
    ["28/02/26", "Replacement keyboard", "Equipment", "89.00"],
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            topMargin=22 * mm, bottomMargin=22 * mm)
    story = []

    story.append(Paragraph("Northgate Consulting Pte Ltd", styles["Title"]))
    story.append(Paragraph("Expense Claim — February 2026", styles["Heading2"]))
    story.append(Spacer(1, 6 * mm))
    # Cost centre is deliberately ABSENT — not written as "not stated", which the
    # model would (correctly) extract as a literal string. Genuine absence is
    # what exercises the "do not invent" instruction.
    story.append(Paragraph(
        "Claimant: A. Tan &nbsp;&nbsp;|&nbsp;&nbsp; Staff ID: NC-0417",
        styles["Normal"]))
    story.append(Spacer(1, 8 * mm))

    t = Table(ROWS, colWidths=[28 * mm, 72 * mm, 32 * mm, 32 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (3, 1), (3, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 6 * mm))
    # Deliberately no printed total: the model has to add it up.
    story.append(Paragraph(
        "<i>Total is not printed on this form. Approver to verify against "
        "attached receipts.</i>", styles["Normal"]))

    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("Approval", styles["Heading3"]))
    story.append(Paragraph(
        "Approved by: R. Menon (Finance Manager)<br/>"
        "Approval date: 05/03/26<br/>"
        "Payment status: <b>paid</b>", styles["Normal"]))

    # Page two, different shape on purpose — an explicit break, because a Spacer
    # large enough to overflow is not guaranteed to and the whole point is that
    # the model must read past page one.
    story.append(PageBreak())
    story.append(Paragraph("Notes to the claim", styles["Heading2"]))
    story.append(Spacer(1, 4 * mm))
    for note in [
        "The rail pass covers four staff travelling to the Jurong site.",
        "Printing was quoted at 140.00 and invoiced at 126.50 after a discount.",
        "The replacement keyboard is a capital item under 100.00 and is expensed.",
        "No receipt is attached for the coffee expense.",
    ]:
        story.append(Paragraph(f"• {note}", styles["Normal"]))
        story.append(Spacer(1, 3 * mm))

    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
