from io import BytesIO

from typing import Optional

import pdfplumber

from PyPDF2 import PdfReader

from docx import Document


def _extract_pdf_with_pdfplumber(buffer: BytesIO) -> Optional[str]:
    try:
        text_parts = []

        with pdfplumber.open(buffer) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)

        return "\n".join(text_parts)

    except Exception:
        return None


def _extract_pdf_with_pypdf2(buffer: BytesIO) -> Optional[str]:
    try:
        reader = PdfReader(buffer)

        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    except Exception:
        return None


def _extract_docx(buffer: BytesIO) -> str:
    doc = Document(buffer)

    text_parts = []

    # Extract normal paragraphs
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)

    # Extract text from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text_parts.append(cell.text)

    return "\n".join(text_parts)


def extract_text_from_file(filename: str, buffer: BytesIO) -> str:
    """Extract plain text from PDF, DOCX, or TXT uploads."""

    lower = filename.lower()

    if lower.endswith(".pdf"):
        # Try pdfplumber first, then fallback
        text = _extract_pdf_with_pdfplumber(buffer)

        if text is None or not text.strip():
            buffer.seek(0)
            text = _extract_pdf_with_pypdf2(buffer)

        if text is None:
            raise ValueError("Unable to extract text from PDF")

        return text

    if lower.endswith(".docx"):
        return _extract_docx(buffer)

    if lower.endswith(".txt"):
        return buffer.read().decode("utf-8", errors="ignore")

    raise ValueError(
        "Unsupported file type. Use PDF, DOCX, or TXT."
    )