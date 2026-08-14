"""
TAKE-HOME 1 — PDF to structured records, end to end.

    uv run make_sample_pdf.py       # once, to create the sample
    uv run section_3_bedrock/06_extraction.py

This is the lab most likely to be useful at work: it combines the multi-modal
section (pages as images) with the extraction section (a Pydantic schema doing
the prompt engineering, plus validation inside the schema).

NOTE ON PyMuPDF vs pdf2image
  The slides show pdf2image, because that is what our real code uses. This lab
  uses PyMuPDF instead: pdf2image shells out to poppler's pdftoppm, so a pip
  install alone is not enough — you also need the poppler-utils system package.
  PyMuPDF ships a self-contained wheel. The content blocks you build from the
  rendered pages are byte-identical either way.

# ======================================================================
# TODO(student) 1 — Run at DPI = 60, 100, 150, 200 and 300, and write down
#   the reported input token count each time. Two things to notice: this
#   document still extracts correctly even at 60 DPI, and the token count
#   STOPS RISING above ~150. Work out why before reading the solution.
#   (Then try it on a real scan of your own, where 60 DPI will not survive.)
#
# TODO(student) 2 — The form deliberately has NO printed total. Check
#   whether `total_amount` was summed correctly. Then add a `Reasoning` field
#   as the FIRST field of ExpenseClaim and re-run — forcing the model to
#   think on paper before committing usually fixes arithmetic.
#
# TODO(student) 3 — `cost_centre` is genuinely absent from the document.
#   Confirm it comes back as the "not found" default rather than invented.
#   Then delete the "do not invent" instruction from the prompt and see what
#   happens.
# ======================================================================

"""

import base64
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

import _bootstrap  # noqa: F401

from _common import banner, chat_model, report_usage

PDF = Path(__file__).resolve().parent / "sample_docs" / "expenses.pdf"
DPI = 150          # TODO 1: measure the token cost of resolution


# --------------------------------------------------------------------------
# The schema IS the prompt. Every description= is an instruction the model
# reads, and the nesting tells it one document yields many line items.
# --------------------------------------------------------------------------

class ExpenseLine(BaseModel):
    date: str | None = Field(
        None, description="Date of the expense in DD/MM/YY format")
    description: str | None = Field(
        None, description="What the expense was for, as written on the form")
    category: str | None = Field(
        None, description="Expense category, e.g. 'Travel', 'Materials'")
    amount: float = Field(
        0.0, description="Line amount in SGD. Use 0.0 if genuinely absent.")

    @field_validator("date")
    @classmethod
    def check_date(cls, v):
        """Accept both DD/MM/YY and DD/MM/YYYY.

        A validator should reject what is genuinely wrong and tolerate what is
        merely inconsistent. The model will produce both of these forms and
        either is unambiguous, so accept both — a validator stricter than your
        requirements is just an outage with good intentions.
        """
        if v is None:
            return v
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                datetime.strptime(v, fmt)
                return v
            except ValueError:
                continue
        raise ValueError(f"date {v!r} must be DD/MM/YY or DD/MM/YYYY")


class ExpenseClaim(BaseModel):
    # TODO 2: add `reasoning: str` here, as the FIRST field.
    claimant: str | None = Field(None, description="Full name of the claimant")
    staff_id: str | None = Field(None, description="Claimant's staff or employee ID")
    cost_centre: str | None = Field(
        None, description="Cost centre code. Null if not stated on the form.")
    period: str | None = Field(None, description="Claim period, e.g. 'February 2026'")
    lines: list[ExpenseLine] = Field(
        default_factory=list, description="Every expense line in the table")
    total_amount: float = Field(
        0.0, description="Sum of all line amounts. Compute it if not printed.")
    is_paid: bool = Field(
        False, description="True if the form states the claim has been paid")
    missing_receipts: list[str] = Field(
        default_factory=list,
        description="Descriptions of any expenses noted as lacking a receipt")


PROMPT = """You extract structured data from expense forms.

Rules:
- Read every page before answering. Page two often qualifies page one.
- Use the exact strings on the form for descriptions.
- If a field is genuinely absent, use the schema's default. Do not invent
  information that is not in the document.
- If a total is not printed, compute it from the line items.
"""


def page_images(pdf: Path, dpi: int) -> list[dict]:
    """Render each page to PNG and wrap as LangChain image_url content blocks.

    This is the whole multi-modal pipeline: page -> bitmap -> base64 -> block.
    """
    import fitz  # PyMuPDF

    blocks = []
    with fitz.open(pdf) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
    return blocks


def main() -> None:
    if not PDF.exists():
        sys.exit(f"{PDF} not found — run: uv run make_sample_pdf.py")

    images = page_images(PDF, DPI)
    banner(f"RENDERED {len(images)} PAGE(S) AT {DPI} DPI")
    for i, b in enumerate(images, 1):
        kb = len(b["image_url"]["url"]) * 3 // 4 // 1024
        print(f"  page {i}: ~{kb} KB of base64")

    # Text first or images first? Put the question AFTER the material it refers
    # to — models attend better that way. Here the instruction is the system
    # prompt and the pages are the user turn.
    msg = HumanMessage(content=[{"type": "text", "text": PROMPT}] + images)

    model = chat_model().with_structured_output(ExpenseClaim, include_raw=True)

    banner("EXTRACTING")
    result = model.invoke([msg])

    if result.get("parsing_error"):
        print(f"  !! validation failed: {result['parsing_error']}")
        print("     A raise from a field_validator lands here. In production this "
              "is where you retry with the error fed back to the model.")
        return

    claim: ExpenseClaim = result["parsed"]
    report_usage("extract", result["raw"].usage_metadata or {})

    banner("RESULT")
    print(f"  claimant     {claim.claimant}")
    print(f"  staff_id     {claim.staff_id}")
    print(f"  cost_centre  {claim.cost_centre}   <- absent from the form (TODO 3)")
    print(f"  period       {claim.period}")
    print(f"  is_paid      {claim.is_paid}")
    print(f"  lines        {len(claim.lines)}")
    for ln in claim.lines:
        print(f"    {ln.date or '??':>11}  {(ln.description or '')[:38]:38}  "
              f"{ln.amount:>8.2f}")

    computed = round(sum(ln.amount for ln in claim.lines), 2)
    flag = "OK" if abs(computed - claim.total_amount) < 0.01 else "MISMATCH"
    print(f"\n  total_amount {claim.total_amount:.2f}  "
          f"(lines sum to {computed:.2f})  <- {flag}  (TODO 2)")
    print(f"  missing_receipts {claim.missing_receipts}")

    banner("Why this shape works")
    print("""  - the schema carried the instructions; the prompt is only 6 lines
  - field_validator ran BEFORE your code saw the data
  - defaults that mean "not found" (0.0, False, None) keep absence unambiguous
  - list[ExpenseLine] lets one document produce N records
  - include_raw=True gives you the token usage as well as the parsed object
  - resolution has a ceiling: Bedrock downscales images past ~1568px on the
    long edge, so beyond roughly 150 DPI on A4 you send more bytes for the
    same tokens and the same accuracy. Measure it — that is TODO 1.""")


if __name__ == "__main__":
    main()
