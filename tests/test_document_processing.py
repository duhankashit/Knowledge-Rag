from io import BytesIO
import unittest

import pymupdf
from docx import Document

from app.services.document_processing import chunk_text, extract_text


class DocumentProcessingTests(unittest.TestCase):
    def test_extracts_utf8_text_file(self) -> None:
        self.assertEqual(extract_text("notes.txt", b"  hello\n  rag  "), "hello rag")

    def test_extracts_pdf_text(self) -> None:
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "RAG from PDF")
        content = pdf.tobytes()
        pdf.close()

        self.assertEqual(extract_text("notes.pdf", content), "RAG from PDF")

    def test_extracts_docx_text(self) -> None:
        document = Document()
        document.add_paragraph("RAG from DOCX")
        output = BytesIO()
        document.save(output)

        self.assertEqual(extract_text("notes.docx", output.getvalue()), "RAG from DOCX")

    def test_rejects_malformed_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not parse the PDF"):
            extract_text("broken.pdf", b"not a pdf")
        with self.assertRaisesRegex(ValueError, "Could not parse the DOCX"):
            extract_text("broken.docx", b"not a docx")

    def test_rejects_unsupported_file_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            extract_text("notes.csv", b"a,b")

    def test_chunks_text_with_metadata_and_overlap(self) -> None:
        source = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = chunk_text(
            source,
            chunk_size=20,
            chunk_overlap=5,
            document_id="doc-1",
            filename="notes.txt",
            uploaded_at="2026-01-01T00:00:00+00:00",
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual([chunk["chunk"] for chunk in chunks], list(range(1, len(chunks) + 1)))
        self.assertTrue(all(chunk["document_id"] == "doc-1" for chunk in chunks))
        self.assertTrue(all(chunk["filename"] == "notes.txt" for chunk in chunks))
        self.assertTrue(all(chunk["text"] for chunk in chunks))
        self.assertIn("theta", chunks[-1]["text"])

    def test_rejects_empty_text_and_invalid_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "no extractable text"):
            chunk_text("", 100, 10, "doc-1", "empty.txt", "now")
        with self.assertRaisesRegex(ValueError, "smaller than chunk size"):
            chunk_text("some text", 10, 10, "doc-1", "notes.txt", "now")


if __name__ == "__main__":
    unittest.main()
