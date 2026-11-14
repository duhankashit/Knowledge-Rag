from io import BytesIO
import unittest

import pymupdf
import tiktoken
from docx import Document

from app.services.document_processing import chunk_text, extract_text


class DocumentProcessingTests(unittest.TestCase):
    def test_extracts_utf8_text_file(self) -> None:
        self.assertEqual(extract_text("notes.txt", b"  hello\n  rag  "), "hello rag")

    def test_extracts_text_while_preserving_paragraph_breaks(self) -> None:
        self.assertEqual(
            extract_text("notes.txt", b"First paragraph.\n\nSecond\nparagraph."),
            "First paragraph.\n\nSecond paragraph.",
        )

    def test_extracts_markdown_while_preserving_structure_lines(self) -> None:
        self.assertEqual(
            extract_text("notes.md", b"# Heading\r\n- first\r\n- second"),
            "# Heading\n- first\n- second",
        )

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

    def test_supports_sentence_paragraph_recursive_and_sliding_strategies(self) -> None:
        source = "First sentence. Second sentence.\n\nThird paragraph is longer."
        for strategy in ("recursive", "sentence", "paragraph", "sliding_window"):
            with self.subTest(strategy=strategy):
                chunks = chunk_text(
                    source,
                    chunk_size=24,
                    chunk_overlap=4,
                    document_id="doc-1",
                    filename="notes.txt",
                    uploaded_at="now",
                    strategy=strategy,
                )
                self.assertGreater(len(chunks), 1)
                self.assertTrue(all(len(chunk["text"]) <= 24 for chunk in chunks))
                self.assertIn("Third", "".join(chunk["text"] for chunk in chunks))

    def test_semantic_strategy_splits_at_embedding_similarity_change(self) -> None:
        source = "Apples are red. Apples are green. Cars are fast."
        vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        chunks = chunk_text(
            source,
            chunk_size=100,
            chunk_overlap=0,
            document_id="doc-1",
            filename="notes.txt",
            uploaded_at="now",
            strategy="semantic",
            embed_texts=lambda texts: vectors,
            similarity_threshold=0.55,
        )

        self.assertEqual(len(chunks), 2)
        self.assertIn("Apples are green.", chunks[0]["text"])
        self.assertEqual(chunks[1]["text"], "Cars are fast.")

    def test_agentic_strategy_uses_and_validates_model_boundaries(self) -> None:
        requests: list[list[dict[str, str]]] = []

        def generate(messages: list[dict[str, str]]) -> str:
            requests.append(messages)
            return '{"boundaries": [10, 20]}'

        chunks = chunk_text(
            "0123456789abcdefghijKLMNO",
            chunk_size=12,
            chunk_overlap=2,
            document_id="doc-1",
            filename="notes.txt",
            uploaded_at="now",
            strategy="agentic",
            max_chunk_tokens=12,
            token_counter=lambda value: len(value),
            prompt="Group related numeric and alphabetic content.",
            generate=generate,
        )

        self.assertEqual(len(requests), 1)
        self.assertIn("Group related numeric", requests[0][0]["content"])
        self.assertTrue(all(len(chunk["text"]) <= 12 for chunk in chunks))
        self.assertIn("KLMNO", "".join(chunk["text"] for chunk in chunks))

    def test_agentic_chunking_preserves_paragraph_boundaries(self) -> None:
        source = "Alpha section.\n\nBeta section.\n\nGamma section."
        chunks = chunk_text(
            source,
            chunk_size=40,
            chunk_overlap=0,
            document_id="doc-1",
            filename="notes.txt",
            uploaded_at="now",
            strategy="agentic",
            max_chunk_tokens=20,
            token_counter=lambda value: len(value),
            preserve_structure=True,
            generate=lambda messages: '{"boundaries": [15, 30]}',
        )

        self.assertEqual(
            [chunk["text"] for chunk in chunks],
            ["Alpha section.", "Beta section.", "Gamma section."],
        )

    def test_agentic_maximum_tokens_is_enforced_with_model_tokenizer(self) -> None:
        source = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        chunks = chunk_text(
            source,
            chunk_size=100,
            chunk_overlap=0,
            document_id="doc-1",
            filename="notes.txt",
            uploaded_at="now",
            strategy="agentic",
            max_chunk_tokens=4,
            llm_model="unknown-model",
            generate=lambda messages: '{"boundaries": []}',
        )
        encoding = tiktoken.get_encoding("cl100k_base")

        self.assertTrue(
            all(
                len(encoding.encode(chunk["text"])) <= 4
                for chunk in chunks
            )
        )

    def test_sliding_window_uses_window_size_and_stride(self) -> None:
        source = "0123456789ABCDEFGHIJ"
        chunks = chunk_text(
            source,
            chunk_size=10,
            chunk_overlap=6,
            document_id="doc-1",
            filename="notes.txt",
            uploaded_at="now",
            strategy="sliding_window",
            window_size=10,
            window_stride=4,
        )

        self.assertEqual([chunk["text"] for chunk in chunks], [
            source[0:10],
            source[4:14],
            source[8:18],
            source[12:20],
        ])

    def test_semantic_chunking_merges_short_chunks_up_to_minimum(self) -> None:
        vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        chunks = chunk_text(
            "Apples are red. Apples are green. Cars are fast.",
            chunk_size=100,
            chunk_overlap=0,
            document_id="doc-1",
            filename="notes.txt",
            uploaded_at="now",
            strategy="semantic",
            min_chunk_size=40,
            embed_texts=lambda texts: vectors,
        )

        self.assertEqual(len(chunks), 1)

    def test_semantic_and_agentic_strategies_require_their_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an embedding model"):
            chunk_text("some text", 10, 2, "doc-1", "notes.txt", "now", strategy="semantic")
        with self.assertRaisesRegex(ValueError, "requires an LLM"):
            chunk_text("some text", 10, 2, "doc-1", "notes.txt", "now", strategy="agentic")

    def test_agentic_strategy_rejects_invalid_model_boundaries(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid boundaries"):
            chunk_text(
                "short text",
                chunk_size=10,
                chunk_overlap=2,
                document_id="doc-1",
                filename="notes.txt",
                uploaded_at="now",
                strategy="agentic",
                generate=lambda messages: '{"boundaries": [99]}',
            )


if __name__ == "__main__":
    unittest.main()
