import io
import re
from zipfile import BadZipFile
from pathlib import Path
from typing import Any

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
import pymupdf


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                text = "\n".join(page.get_text() for page in document)
        except (pymupdf.FileDataError, ValueError) as error:
            raise ValueError("Could not parse the PDF file.") from error
    elif extension == ".docx":
        try:
            document = Document(io.BytesIO(content))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            for table in document.tables:
                text += "\n" + "\n".join(
                    " ".join(cell.text for cell in row.cells) for row in table.rows
                )
        except (BadZipFile, PackageNotFoundError, KeyError) as error:
            raise ValueError("Could not parse the DOCX file.") from error
    elif extension in {".txt", ".md"}:
        text = content.decode("utf-8-sig")
    else:
        raise ValueError(f"Unsupported file type: {extension or 'no extension'}.")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    document_id: str,
    filename: str,
    uploaded_at: str,
) -> list[dict[str, Any]]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
    if not text.strip():
        raise ValueError("The uploaded file contains no extractable text.")

    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk": len(chunks) + 1,
                    "text": chunk,
                    "uploaded_at": uploaded_at,
                }
            )
        if end >= len(text):
            break
        start = max(start + 1, end - chunk_overlap)
    return chunks
