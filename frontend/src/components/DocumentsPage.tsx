import {
  AlertCircle,
  Check,
  File,
  FileText,
  FileUp,
  FileX2,
  LoaderCircle,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";

import {
  deleteDocument,
  getChunkingOptions,
  getDocuments,
  uploadDocument,
} from "../api";
import type { ChunkingOptions, ChunkingStrategy, DocumentInfo } from "../types";

const ACCEPTED_TYPES = [".pdf", ".docx", ".txt", ".md"];
const MAX_FILE_BYTES = 20 * 1024 * 1024;

function prettySize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Date unavailable"
    : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(date);
}

function extension(filename: string): string {
  return filename.split(".").pop()?.toLowerCase() ?? "";
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [busyDocument, setBusyDocument] = useState("");
  const [search, setSearch] = useState("");
  const [chunkingStrategy, setChunkingStrategy] = useState<ChunkingStrategy>("basic");
  const [chunkSize, setChunkSize] = useState(900);
  const [minChunkSize, setMinChunkSize] = useState(100);
  const [chunkOverlap, setChunkOverlap] = useState(150);
  const [useOverlap, setUseOverlap] = useState(true);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.55);
  const [maxChunkTokens, setMaxChunkTokens] = useState(512);
  const [chunkingPrompt, setChunkingPrompt] = useState("Prefer coherent, self-contained chunks.");
  const [preserveStructure, setPreserveStructure] = useState(true);
  const [windowSize, setWindowSize] = useState(900);
  const [windowStride, setWindowStride] = useState(750);
  const [chunkingOptions, setChunkingOptions] = useState<ChunkingOptions>({
    embedding_models: [],
    llm_models: [],
  });
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const loadDocuments = useCallback(async (signal?: AbortSignal) => {
    setError("");
    try {
      const [result, options] = await Promise.all([
        getDocuments(signal),
        getChunkingOptions(signal),
      ]);
      setDocuments(result);
      setChunkingOptions(options);
      setEmbeddingModel(options.embedding_models[0] ?? "");
      setLlmModel(options.llm_models[0] ?? "");
    } catch (requestError) {
      if (signal?.aborted) return;
      if (requestError instanceof DOMException && requestError.name === "AbortError") return;
      setError(requestError instanceof Error ? requestError.message : "Could not load documents.");
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadDocuments(controller.signal);
    return () => controller.abort();
  }, [loadDocuments]);

  async function addFile(file: File) {
    if (busy) return;
    const ext = `.${extension(file.name)}`;
    if (!ACCEPTED_TYPES.includes(ext)) {
      setError(`"${file.name}" is not supported. Choose a PDF, DOCX, TXT, or MD file.`);
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setError(`"${file.name}" is larger than the 20 MB upload limit.`);
      return;
    }
    if (file.size === 0) {
      setError(`"${file.name}" is empty.`);
      return;
    }
    const usesCharacterSize = ["basic", "recursive", "sentence", "paragraph", "semantic"].includes(chunkingStrategy);
    if (usesCharacterSize && (!Number.isInteger(chunkSize) || chunkSize <= 0)) {
      setError("Chunk size must be a positive whole number.");
      return;
    }
    const usesCharacterOverlap = ["basic", "recursive"].includes(chunkingStrategy) ||
      (["sentence", "paragraph"].includes(chunkingStrategy) && useOverlap);
    if (usesCharacterOverlap &&
      (!Number.isInteger(chunkOverlap) || chunkOverlap < 0 || chunkOverlap >= chunkSize)
    ) {
      setError("Overlap must be a non-negative whole number smaller than the chunk size.");
      return;
    }
    if (
      ["semantic", "sentence", "paragraph"].includes(chunkingStrategy) &&
      (!Number.isInteger(minChunkSize) || minChunkSize <= 0 || minChunkSize > chunkSize)
    ) {
      setError("Minimum chunk size must be positive and no larger than the maximum.");
      return;
    }
    if (chunkingStrategy === "sliding_window" &&
      (!Number.isInteger(windowSize) || !Number.isInteger(windowStride) ||
        windowSize <= 0 || windowStride <= 0 || windowStride > windowSize)
    ) {
      setError("Stride must be positive and no larger than the window size.");
      return;
    }
    if (!Number.isFinite(similarityThreshold) || similarityThreshold < 0 || similarityThreshold > 1) {
      setError("Similarity threshold must be between 0 and 1.");
      return;
    }
    if (chunkingStrategy === "agentic" &&
      (!Number.isInteger(maxChunkTokens) || maxChunkTokens <= 0)
    ) {
      setError("Maximum chunk tokens must be a positive whole number.");
      return;
    }
    if (chunkingStrategy === "semantic" && !embeddingModel) {
      setError("The configured embedding model is unavailable.");
      return;
    }
    if (chunkingStrategy === "agentic" && !llmModel) {
      setError("The configured LLM is unavailable.");
      return;
    }

    setBusy(true);
    setError("");
    setNotice("");
    try {
      const document = await uploadDocument(file, {
        strategy: chunkingStrategy,
        chunkSize: chunkingStrategy === "agentic" ? 900 : chunkSize,
        chunkOverlap: chunkingStrategy === "sliding_window"
          ? windowSize - windowStride
          : ["sentence", "paragraph"].includes(chunkingStrategy)
            ? useOverlap ? chunkOverlap : 0
            : ["semantic", "agentic"].includes(chunkingStrategy) ? 0 : chunkOverlap,
        minChunkSize: ["semantic", "sentence", "paragraph"].includes(chunkingStrategy)
          ? minChunkSize
          : 1,
        similarityThreshold,
        maxChunkTokens: chunkingStrategy === "agentic" ? maxChunkTokens : 512,
        prompt: chunkingPrompt,
        preserveStructure,
        windowSize: chunkingStrategy === "sliding_window" ? windowSize : 900,
        windowStride: chunkingStrategy === "sliding_window" ? windowStride : 750,
        embeddingModel,
        llmModel,
      });
      setDocuments((current) => [
        document,
        ...current.filter((item) => item.document_id !== document.document_id),
      ]);
      setNotice(`${document.filename} is ready — ${document.chunk_count} ${document.chunk_count === 1 ? "chunk" : "chunks"} indexed.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Upload failed.");
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void addFile(file);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void addFile(file);
  }

  async function removeDocument(document: DocumentInfo) {
    if (!window.confirm(`Remove "${document.filename}" from your knowledge base?`)) return;
    setBusyDocument(document.document_id);
    setError("");
    setNotice("");
    try {
      await deleteDocument(document.document_id);
      setDocuments((current) => current.filter((item) => item.document_id !== document.document_id));
      setNotice(`${document.filename} was removed.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not remove the file.");
    } finally {
      setBusyDocument("");
    }
  }

  const visibleDocuments = documents.filter((document) =>
    document.filename.toLowerCase().includes(search.trim().toLowerCase()),
  );
  const totalChunks = documents.reduce((sum, document) => sum + document.chunk_count, 0);

  return (
    <section className="documents-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow"><span /> YOUR KNOWLEDGE BASE</div>
          <h1>Files &amp; sources</h1>
          <p>Manage the documents used to answer your questions.</p>
        </div>
        <div className="storage-badge"><ShieldCheck size={15} /> Stored locally</div>
      </div>

      {error && (
        <div className="notice notice-error" role="alert">
          <AlertCircle size={17} /><span>{error}</span>
          <button onClick={() => setError("")} aria-label="Dismiss error"><X size={15} /></button>
        </div>
      )}
      {notice && (
        <div className="notice notice-success" role="status">
          <Check size={17} /><span>{notice}</span>
          <button onClick={() => setNotice("")} aria-label="Dismiss message"><X size={15} /></button>
        </div>
      )}

      <div
        className={`upload-zone ${dragging ? "dragging" : ""} ${busy ? "uploading" : ""}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
        }}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          onChange={handleFileChange}
          aria-label="Choose a document to upload"
        />
        <div className="upload-icon">{busy ? <LoaderCircle className="spin" size={22} /> : <FileUp size={22} />}</div>
        <div className="upload-copy">
          <strong>{busy ? "Adding document to your library..." : "Drop a file to add it to your library"}</strong>
          <span>PDF, DOCX, TXT, or Markdown <i /> Up to 20 MB</span>
        </div>
        <button className="button-primary" onClick={() => inputRef.current?.click()} disabled={busy} type="button">
          <Upload size={15} /> Browse files
        </button>
        <div className="upload-hint">Files are indexed by your RAG backend for search.</div>
      </div>

      <div className="chunk-options">
        <label>
          <span>Chunking method</span>
          <select
            value={chunkingStrategy}
            onChange={(event) => setChunkingStrategy(event.target.value as ChunkingStrategy)}
            disabled={busy}
          >
            <option value="basic">Basic character</option>
            <option value="recursive">Recursive character</option>
            <option value="sentence">Sentence-based</option>
            <option value="paragraph">Paragraph-based</option>
            <option value="semantic">Semantic (embedding-based)</option>
            <option value="agentic">Agentic (LLM-based)</option>
            <option value="sliding_window">Sliding window</option>
          </select>
        </label>
        {chunkingStrategy === "semantic" && (
          <>
            <label>
              <span>Embedding model</span>
              <select value={embeddingModel} onChange={(event) => setEmbeddingModel(event.target.value)} disabled={busy}>
                {chunkingOptions.embedding_models.map((model) => <option key={model} value={model}>{model}</option>)}
              </select>
            </label>
            <label>
              <span>Similarity threshold ({similarityThreshold.toFixed(2)})</span>
              <input type="range" min="0" max="1" step="0.01" value={similarityThreshold} onChange={(event) => setSimilarityThreshold(Number(event.target.value))} disabled={busy} />
            </label>
          </>
        )}
        {["semantic", "sentence", "paragraph"].includes(chunkingStrategy) && (
          <>
            <label>
              <span>Maximum chunk size (characters)</span>
              <input type="number" min="1" step="1" value={chunkSize} onChange={(event) => setChunkSize(Number(event.target.value))} disabled={busy} />
            </label>
            <label>
              <span>Minimum chunk size (characters)</span>
              <input type="number" min="1" step="1" value={minChunkSize} onChange={(event) => setMinChunkSize(Number(event.target.value))} disabled={busy} />
            </label>
            {chunkingStrategy !== "semantic" && (
              <label className="chunk-checkbox">
                <input type="checkbox" checked={useOverlap} onChange={(event) => setUseOverlap(event.target.checked)} disabled={busy} />
                <span>Use overlap</span>
              </label>
            )}
          </>
        )}
        {["basic", "recursive"].includes(chunkingStrategy) && (
          <>
            <label>
              <span>Maximum chunk size (characters)</span>
              <input type="number" min="1" step="1" value={chunkSize} onChange={(event) => setChunkSize(Number(event.target.value))} disabled={busy} />
            </label>
            <label>
              <span>Overlap (characters)</span>
              <input type="number" min="0" step="1" value={chunkOverlap} onChange={(event) => setChunkOverlap(Number(event.target.value))} disabled={busy} />
            </label>
          </>
        )}
        {chunkingStrategy === "agentic" && (
          <>
            <label>
              <span>LLM model</span>
              <select value={llmModel} onChange={(event) => setLlmModel(event.target.value)} disabled={busy}>
                {chunkingOptions.llm_models.map((model) => <option key={model} value={model}>{model}</option>)}
              </select>
            </label>
            <label>
              <span>Maximum chunk tokens</span>
              <input type="number" min="1" step="1" value={maxChunkTokens} onChange={(event) => setMaxChunkTokens(Number(event.target.value))} disabled={busy} />
            </label>
            <label className="chunk-prompt">
              <span>Chunking prompt</span>
              <textarea value={chunkingPrompt} maxLength={4000} onChange={(event) => setChunkingPrompt(event.target.value)} disabled={busy} />
            </label>
            <label className="chunk-checkbox">
              <input type="checkbox" checked={preserveStructure} onChange={(event) => setPreserveStructure(event.target.checked)} disabled={busy} />
              <span>Preserve document structure</span>
            </label>
          </>
        )}
        {chunkingStrategy === "sliding_window" && (
          <>
            <label>
              <span>Window size (characters)</span>
              <input type="number" min="1" step="1" value={windowSize} onChange={(event) => setWindowSize(Number(event.target.value))} disabled={busy} />
            </label>
            <label>
              <span>Step size / stride (characters)</span>
              <input type="number" min="1" step="1" value={windowStride} onChange={(event) => setWindowStride(Number(event.target.value))} disabled={busy} />
            </label>
            <label>
              <span>Overlap (characters)</span>
              <input type="number" value={Math.max(0, windowSize - windowStride)} readOnly />
            </label>
          </>
        )}
        <p>
          {chunkingStrategy === "semantic"
            ? "Semantic boundaries are selected from adjacent sentence embedding similarity."
            : chunkingStrategy === "agentic"
              ? "Agentic chunking sends document text to the configured Groq model and may incur API usage. Token limits use the selected model encoding when supported."
              : chunkingStrategy === "sliding_window"
                ? "Overlap equals window size minus stride."
                : "Chunk sizes and overlap are measured in characters."}
        </p>
      </div>

      <div className="library-toolbar">
        <div className="library-title">
          <div className="library-title-icon"><FileText size={17} /></div>
          <div>
            <h2>Indexed documents</h2>
            <p>{documents.length} {documents.length === 1 ? "file" : "files"} <span>·</span> {totalChunks} searchable {totalChunks === 1 ? "chunk" : "chunks"}</p>
          </div>
        </div>
        <div className="library-actions">
          <label className="search-field">
            <Search size={15} />
            <input
              type="search"
              placeholder="Search files"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              aria-label="Search files"
            />
            {search && <button onClick={() => setSearch("")} aria-label="Clear search"><X size={14} /></button>}
          </label>
          <button className="icon-button" onClick={() => { setLoading(true); void loadDocuments(); }} title="Refresh list" aria-label="Refresh documents">
            <RefreshCw size={15} />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="library-state"><LoaderCircle className="spin" size={22} /><span>Loading your documents...</span></div>
      ) : documents.length === 0 ? (
        <div className="library-state empty-library">
          <div className="empty-library-icon"><FileX2 size={21} /></div>
          <strong>Your library is waiting</strong>
          <span>Add a document above and it will appear here.</span>
        </div>
      ) : visibleDocuments.length === 0 ? (
        <div className="library-state empty-library">
          <Search size={20} />
          <strong>No files match “{search}”</strong>
          <span>Try another filename.</span>
        </div>
      ) : (
        <div className="document-list">
          <div className="document-list-header">
            <span>NAME</span><span>TYPE</span><span>CHUNKS</span><span>ADDED</span><span />
          </div>
          {visibleDocuments.map((document) => {
            const ext = extension(document.filename);
            return (
              <article className="document-row" key={document.document_id}>
                <div className="document-name-cell">
                  <div className={`file-icon file-${ext === "pdf" ? "pdf" : ext === "docx" ? "doc" : "text"}`}>
                    {ext === "pdf" ? <File size={18} /> : <FileText size={18} />}
                  </div>
                  <div className="document-name-copy">
                    <strong title={document.filename}>{document.filename}</strong>
                    <span>Added {formatDate(document.uploaded_at)}</span>
                  </div>
                </div>
                <div className="document-type-cell"><span className="type-pill">{ext.toUpperCase()}</span></div>
                <div className="document-chunks">{document.chunk_count}</div>
                <div className="document-date">{formatDate(document.uploaded_at)}</div>
                <div className="document-remove">
                  <button
                    className="delete-button"
                    onClick={() => void removeDocument(document)}
                    disabled={busyDocument === document.document_id}
                    title={`Remove ${document.filename}`}
                    aria-label={`Remove ${document.filename}`}
                  >
                    {busyDocument === document.document_id ? <LoaderCircle className="spin" size={16} /> : <Trash2 size={16} />}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      )}

      <div className="library-footnote">
        <FileUp size={14} />
        <span>Supported: PDF, DOCX, TXT, MD</span>
        <span className="footnote-divider" />
        <span>Maximum file size: {prettySize(MAX_FILE_BYTES)}</span>
      </div>
    </section>
  );
}
