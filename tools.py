import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))

import io
import json
import math
import random
import re
import statistics
import contextlib
import asyncio
import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Optional

import pymupdf as fitz  # `import fitz` still works but is deprecated upstream
from fastembed import TextEmbedding

from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.embeddings import Embeddings
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Phase 6 fix — GoogleGenerativeAIEmbeddings resource leak / quota exhaustion.
#
# Phase 4 made `process_and_index_documents` successfully extract the FULL
# text of large digital PDFs via PyMuPDF instead of hanging in Docling OCR —
# but that unlocked a new problem: a 50MB textbook now chunks into 600-1200+
# pieces, and `Chroma.from_texts(..., embedding=GoogleGenerativeAIEmbeddings(...))`
# issued ONE BILLED API REQUEST PER CHUNK with no batching or cap — 1000+
# requests and 400K+ tokens in 15 seconds on a single upload, instantly
# exhausting the API quota.
#
# Fix: embed entirely locally, for $0, with FastEmbed (ONNX-runtime-based).
# This project's `.venv` ALREADY has `onnxruntime` installed transitively
# via Docling/RapidOCR, so this adds essentially no new heavy runtime — just
# the thin `fastembed` wrapper package plus a one-time model download
# (cached locally afterward, then fully offline). Embedding a document now
# never makes a network call at all, so there is no per-chunk request, no
# token metering, and no way for a single large upload to exhaust an
# external quota ever again.
#
# Model choice: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~220MB)
# rather than an English-only model like `bge-small-en-v1.5` — this app's
# `STRICT_LANGUAGE_RULE` (agents.py) explicitly supports Azerbaijani, and
# uploaded documents/queries are not guaranteed to be English.
# ---------------------------------------------------------------------------
_FASTEMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class _FastEmbedEmbeddings(Embeddings):
    """Minimal `langchain_core.embeddings.Embeddings` adapter around
    `fastembed.TextEmbedding`. Deliberately NOT
    `langchain_community.embeddings.FastEmbedEmbeddings` — that class lives
    in a package LangChain has marked for sunset; wrapping the raw
    `fastembed` API ourselves is a handful of lines, has one fewer
    dependency layer to break under future version drift, and avoids a
    deprecation warning on every single embed call."""

    def __init__(self, model_name: str = _FASTEMBED_MODEL_NAME):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


# Constructed once, at import time, not lazily per-request: this is where
# the ONNX model is loaded into memory (and, on this machine's very first
# run, downloaded — a few hundred MB, cached under `~/.cache/fastembed`
# afterward). Paying that cost once at process startup — rather than on the
# first user request — is a deliberate trade-off in exchange for every
# document-processing call afterward being instant, local, and free.
embeddings = _FastEmbedEmbeddings()

_thread_auth: dict[str, tuple[str, str]] = {}


def remember_thread_auth(thread_id: str, user_id: str, access_token: str) -> None:
    if thread_id and user_id and access_token:
        _thread_auth[thread_id] = (user_id, access_token)


def _supabase_rest_env() -> tuple[str, str]:
    url = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").rstrip("/")
    key = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or ""
    )
    return url, key


def _pg_request(method: str, path: str, token: str, body=None, prefer: str = "return=representation") -> bytes:
    import urllib.error
    import urllib.request

    url, api_key = _supabase_rest_env()
    missing = []
    if not url:
        missing.append("project URL")
    if not api_key:
        missing.append("API key")
    if not token:
        missing.append("user access token")
    if missing:
        raise RuntimeError(
            "Supabase is not configured for document chunks (missing " + ", ".join(missing) + ")"
        )
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/rest/v1/{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": api_key,
            "Content-Type": "application/json",
            "Prefer": prefer,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"pgvector {method} {path} failed: {exc.code} {detail}") from exc


def _quote_filter(value: str) -> str:
    import urllib.parse
    return urllib.parse.quote(value, safe="")


def delete_document_chunks(thread_key: str) -> None:
    auth = _thread_auth.get(thread_key)
    if not auth:
        return
    user_id, token = auth
    path = (
        "document_chunks"
        f"?user_id=eq.{_quote_filter(user_id)}"
        f"&thread_id=eq.{_quote_filter(thread_key)}"
    )
    _pg_request("DELETE", path, token, prefer="return=minimal")


def insert_document_chunks(thread_key: str, file_path: str, chunks: list[str], vectors: list[list[float]]) -> None:
    auth = _thread_auth.get(thread_key)
    if not auth:
        raise RuntimeError("Signed-in user is required to store document chunks")
    user_id, token = auth
    delete_document_chunks(thread_key)
    rows = [
        {
            "user_id": user_id,
            "thread_id": thread_key,
            "file_path": file_path,
            "chunk_index": index,
            "content": text,
            "embedding": vector,
        }
        for index, (text, vector) in enumerate(zip(chunks, vectors))
    ]
    for start in range(0, len(rows), 40):
        _pg_request("POST", "document_chunks", token, rows[start:start + 40], prefer="return=minimal")


def search_document_chunks(thread_key: str, query: str, limit: int = 5) -> list[str]:
    auth = _thread_auth.get(thread_key)
    if not auth:
        return []
    user_id, token = auth
    vector = embeddings.embed_query(query)
    raw = _pg_request(
        "POST",
        "rpc/match_document_chunks",
        token,
        {
            "query_embedding": vector,
            "match_user": user_id,
            "match_thread": thread_key,
            "match_count": limit,
        },
    )
    rows = json.loads(raw.decode("utf-8") or "[]")
    if not isinstance(rows, list):
        return []
    rows.sort(key=lambda row: int(row.get("chunk_index") or 0))
    return [str(row.get("content") or "") for row in rows if row.get("content")]

# ---------------------------------------------------------------------------
# Thread-scoped document memory.
#
# These USED to be single, module-level globals (`vector_store = None`,
# `indexed_chunks = []`) shared by every request — a correctness bug under
# any concurrent-session load: Session A indexing a PDF would silently
# clobber (or be clobbered by) Session B's vector store, and Session B's
# `search_in_document` could return Session A's document content.
#
# They are now dicts keyed by LangGraph `thread_id`, so each conversation
# thread gets its own isolated vector store / chunk list. `converter` and
# `embeddings` stay as shared singletons above — they're stateless model/
# client handles, not per-session data, so sharing them is safe and avoids
# re-instantiating a client per request.
# ---------------------------------------------------------------------------
_vector_stores: dict[str, Chroma] = {}
_indexed_chunks: dict[str, list[str]] = {}
_embeddings_ready: dict[str, bool] = {}

# --- Embedding lifecycle (see `start_embed_warmup`) -------------------------
# `_doc_generation` is bumped whenever a thread's document is cleared, so a job
# or writer started for the old document can detect it is stale. The per-thread
# lock serializes every pgvector delete/insert for that thread.
_doc_generation: dict[str, int] = {}
_vector_write_locks: dict[str, threading.Lock] = {}
_vector_locks_guard = threading.Lock()
_embed_gate: Optional[asyncio.Semaphore] = None
EMBED_BATCH_SIZE = 32
EMBED_JOB_BUDGET_SECONDS = 150


@dataclass
class _EmbedJob:
    task: "asyncio.Task"
    loop: "asyncio.AbstractEventLoop"
    generation: int
    owner_request_id: str
    user_id: str


_embed_jobs: dict[str, _EmbedJob] = {}


def _vector_write_lock(thread_key: str) -> threading.Lock:
    with _vector_locks_guard:
        lock = _vector_write_locks.get(thread_key)
        if lock is None:
            lock = _vector_write_locks[thread_key] = threading.Lock()
        return lock


_DEFAULT_THREAD_KEY = "default_session"

# Idempotency guard: which exact `file_path` was last successfully indexed
# for a given thread. QA on a 1-page PDF exposed `doc_worker` calling
# `process_and_index_documents` TWICE in the same turn (once up front, once
# again mid-loop "just in case") — each call re-runs the full extract ->
# chunk -> embed pipeline (~5s even for a tiny document), so a second call
# for the SAME file is pure wasted latency, not a correctness fix. Checked
# at the top of `process_and_index_documents` below.
_indexed_file_paths: dict[str, str] = {}
# Absolute page count even when `_page_texts` is empty (scanned PDFs).
_pdf_page_counts: dict[str, int] = {}
# True when the PDF has no usable text layer — TOC + lazy OCR/vision only.
_scan_mode: dict[str, bool] = {}
# Fast-OCR cache: thread -> {absolute 1-indexed page -> text}.
_ocr_page_cache: dict[str, dict[int, str]] = {}
# Last Gemini extract per thread (stem + normalized boxes for Phase 3).
_vision_extracts: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Two-Tier Retrieval Architecture — Tier 1 (structural navigation) memory.
#
# `_vector_stores`/`_indexed_chunks` above are Tier 2: semantic similarity
# search over an EMBEDDED, CHUNK-CAPPED (`MAX_INDEXED_CHUNKS`) copy of the
# document. That is fundamentally the wrong tool for "find Chapter 22" —
# embeddings are approximate, and worse, a chapter past the chunk cap was
# never embedded at all, so no similarity search can ever find it.
#
# `_page_texts`/`_document_outlines` are Tier 1: a page-indexed copy of the
# RAW text (one string per PDF page) plus a small structural map (chapter/
# section title -> page range), built once at indexing time and NEVER
# passed through chunking/embedding/Chroma. A book's outline is always tiny
# (dozens of entries) regardless of page count, so Tier 1 navigation is
# completely unaffected by `MAX_INDEXED_CHUNKS` — a chapter Tier 2 had to
# truncate away is still reachable here via direct page extraction.
#
# Both are scoped per-thread, same rationale as `_vector_stores` above.
# `_page_texts[thread_key]` is empty for scanned PDFs (no text layer) and
# for Docling-routed office files. Scanned PDFs still get a TOC outline and
# `_pdf_page_counts`; question text is filled lazily via fast OCR + Gemini
# Flash inside `locate_marker_in_range`. Office files stay on Tier 2.
# ---------------------------------------------------------------------------
_page_texts: dict[str, list[str]] = {}
_document_outlines: dict[str, list[dict]] = {}
# Last successful `resolve_chapter_target` for this thread (absolute pages).
# `locate_marker_in_range` uses this to decide whether a bare "10." heading
# is safe to match — only inside an already-narrowed question bank.
_resolved_targets: dict[str, dict] = {}

# Short-document hard bypass. A 1-5 page worksheet has no TOC worth
# walking and nothing for semantic search to usefully narrow; letting
# `doc_worker` call these anyway is what produced the 363s / $burn
# recursion loop after the previous per-tool-name caps were removed.
SHORT_DOC_MAX_PAGES = 5
SHORT_DOC_BLOCKED_TOOLS = frozenset({
    "search_in_document",
    "locate_marker_in_range",
    "get_document_outline",
    "resolve_chapter_target",
})
# Blank/"həll et" uploads: auto-solve only when the PDF is this short.
# Larger files (or unreadable/non-PDF → count 0) fail closed and ask
# for a page/question instead of dumping the whole book into math_worker.
EMPTY_PROMPT_AUTOSOLVE_MAX_PAGES = 3


def count_pdf_pages(file_path: str) -> int:
    """Cheap PyMuPDF page count. Returns 0 on missing path, non-PDF, or
    any error so callers can fail-closed (treat 0 as too large to auto-solve)."""
    if not file_path:
        return 0
    path = str(file_path).strip()
    if not path.lower().endswith(".pdf"):
        return 0
    try:
        doc = fitz.open(path)
        try:
            return int(doc.page_count or 0)
        finally:
            doc.close()
    except Exception as e:
        print(f"[PDF] count_pdf_pages failed: {e}", flush=True)
        return 0

# ---------------------------------------------------------------------------
# Logical (printed) <-> absolute page mapping (Tier 1).
#
# `_page_texts[thread_key][i]` is indexed by ABSOLUTE position in the PDF's
# page array (0-indexed there, 1-indexed everywhere else in this file). A
# student never thinks in those terms — they say "page 740", meaning the
# number actually PRINTED on the page, which is frequently offset from the
# absolute index by however many unnumbered/roman-numeral front-matter pages
# (cover, copyright, table of contents, preface) precede printed page 1.
# Every tool argument/response involving a page number (`read_page_range`,
# `locate_marker_in_range`, `get_document_outline`'s displayed ranges) is
# defined to be in this PRINTED/logical space — `_printed_to_absolute_page`/
# `_absolute_to_printed_label` below are the ONLY place that boundary is
# crossed, so the LLM never has to reason about the distinction at all.
#
# Populated once at indexing time (`process_and_index_documents`), scoped
# per-thread like `_page_texts` above.
# ---------------------------------------------------------------------------
_page_label_maps: dict[str, dict[int, str]] = {}      # absolute page -> printed label
_page_label_reverse: dict[str, dict[str, int]] = {}   # printed label -> absolute page
_printed_page_offsets: dict[str, Optional[int]] = {}  # constant fallback offset, or None

# ---------------------------------------------------------------------------
# Bilingual structural markers (Tier 1).
#
# Uploaded textbooks and the STRICT_LANGUAGE_RULE-driven user queries
# (agents.py) may each independently be in English or Azerbaijani — a
# student asking in Azerbaijani may still be reading an English-authored
# textbook, and vice versa — so both the chapter-heading fallback regex and
# `locate_marker_in_range`'s question-marker search must recognize BOTH
# vocabularies regardless of which language the current turn is in.
#
# Azerbaijani chapter/question numbers are just as often written
# NUMBER-FIRST with an ordinal suffix ("22-ci Fəsil", "5-ci Sual") as
# KEYWORD-FIRST ("Fəsil 22", "Sual 5") — `_AZ_ORDINAL_SUFFIX` covers the
# vowel-harmony variants of that suffix so both orders match.
# ---------------------------------------------------------------------------
_AZ_ORDINAL_SUFFIX = r"(?:-?(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü|ci|cı|cu|cü))?"

_CHAPTER_KEYWORDS = ["Chapter", "Fəsil", "Bölmə"]
_QUESTION_KEYWORDS = [
    "Question", "Sual", "Məsələ", "Tapşırıq", "Problem", "Exercise",
]

# Nested bookmark depth kept from `fitz.get_toc()` — Part / Chapter /
# Summary / Problems live at levels 1–3 in textbooks like Serway. Deeper
# 22.3.1-style nodes are dropped so the tree stays navigable.
MAX_OUTLINE_LEVEL = 4
# If the full tree is longer than this, `get_document_outline` shows only
# Parts + Chapters and tells the model to call `resolve_chapter_target`.
_OUTLINE_COMPACT_MAX = 80

_PROBLEMS_TITLE_RE = re.compile(
    r"\b(?:problems?(?:\s+and\s+exercises)?|exercises?|end[- ]of[- ]chapter|"
    r"məsələlər|həll\s+ediləcək|задачи|aufgaben|exercices?)\b",
    re.IGNORECASE,
)
_CONCEPTUAL_TITLE_RE = re.compile(
    r"\b(?:conceptual|konseptual)\b",
    re.IGNORECASE,
)
_OBJECTIVE_TITLE_RE = re.compile(
    r"\b(?:objective|obyektiv|multiple[- ]choice|quick\s+quiz(?:zes)?)\b",
    re.IGNORECASE,
)
_PART_TITLE_RE = re.compile(r"^\s*part\s+\d", re.IGNORECASE)
_CHAPTER_IN_TITLE_RE = re.compile(
    r"(?:ch(?:apter)?|fəsil|bölmə|section)\s*[:.\-]?\s*(\d{1,3})\b",
    re.IGNORECASE,
)
_LEADING_CH_NUM_RE = re.compile(
    r"^\s*(?:ch(?:apter)?\.?\s*)?(\d{1,3})(?!\.\d)(?:\s*[:.\-]|\s*$)"
)
# "1.1 Average velocity" / "22.3" — subsection, not a chapter bookmark.
_SUBSECTION_TITLE_RE = re.compile(r"^\s*\d{1,3}\.\d")
# Major number for block bounds: "1.1 …" and "1. Kinematics" both group as 1.
_MAJOR_OUTLINE_NUM_RE = re.compile(r"^\s*(\d{1,3})(?:\.\d+)*\b")
_SUBSECTION_LINE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"^(?:problems|exercises|end[- ]of[- ]chapter(?:\s+problems)?|məsələlər)\b",
        re.IGNORECASE,
    ), "problems"),
    (re.compile(
        r"^(?:conceptual\s+questions?|konseptual(?:\s+suallar)?)\b",
        re.IGNORECASE,
    ), "conceptual"),
    (re.compile(
        r"^(?:objective\s+questions?|multiple[- ]choice|obyektiv(?:\s+suallar)?|quick\s+quizzes?)\b",
        re.IGNORECASE,
    ), "objective"),
]
_BANK_HEADING_IN_TEXT = re.compile(
    r"(?im)^.{0,3}((?:conceptual\s+questions?|objective\s+questions?|"
    r"problems|exercises|konseptual(?:\s+suallar)?|obyektiv(?:\s+suallar)?|"
    r"məsələlər).{0,60})$",
)

# Hard guardrails for the new Tier 1 tools (see MAX_INDEXED_CHUNKS above for
# the equivalent Tier 2 guardrail) — bounds worst-case context/token cost
# per call regardless of how the caller (the LLM) invokes them.
MAX_PAGE_RANGE = 25
MARKER_SNIPPET_CHARS = 3000


def _compile_marker_pattern(keywords: list[str]) -> "re.Pattern":
    """Builds a regex matching either `<keyword> <number>` (e.g. "Chapter
    22", "Fəsil 22") or `<number><AZ ordinal suffix> <keyword>` (e.g.
    "22-ci Fəsil"), capturing the matched number under `num1`/`num2`."""
    keyword_alt = "|".join(re.escape(k) for k in keywords)
    pattern = (
        rf"(?:(?P<kw_first>{keyword_alt})\s*[:.\-]?\s*(?P<num1>\d{{1,4}})"
        rf"|(?P<num2>\d{{1,4}}){_AZ_ORDINAL_SUFFIX}\s+(?P<kw_second>{keyword_alt}))"
    )
    return re.compile(pattern, re.IGNORECASE)


def _compile_numbered_marker_pattern(keywords: list[str], number: str) -> "re.Pattern":
    """Same shape as `_compile_marker_pattern`, but pinned to one specific
    number — used by `locate_marker_in_range` to search for e.g. "Question
    8"/"Sual 8"/"8-ci Sual" once the target number is already known."""
    keyword_alt = "|".join(re.escape(k) for k in keywords)
    escaped_number = re.escape(number)
    pattern = (
        rf"(?:(?:{keyword_alt})\s*[:.\-]?\s*{escaped_number}\b"
        rf"|\b{escaped_number}{_AZ_ORDINAL_SUFFIX}\s+(?:{keyword_alt}))"
    )
    return re.compile(pattern, re.IGNORECASE)


def _extract_marker_number(marker: str) -> Optional[str]:
    """Pulls the first number out of a caller-supplied marker string (e.g.
    "Question 8" -> "8"), so `locate_marker_in_range` matches the NUMBER
    against BOTH English and Azerbaijani question vocabulary regardless of
    which language/keyword the caller happened to use."""
    match = re.search(r"\d{1,4}", marker)
    return match.group(0) if match else None


_CHAPTER_PATTERN = _compile_marker_pattern(_CHAPTER_KEYWORDS)

# Generic (any-number) question-marker pattern — used by `locate_marker_in_range`
# to find the NEXT question heading after a match, so a question's content
# can be bounded precisely (see `_is_heading_like`/`locate_marker_in_range`
# below) instead of relying on a fixed character count that might cut off
# mid-question OR bleed into the following question.
_QUESTION_PATTERN = _compile_marker_pattern(_QUESTION_KEYWORDS)


def _is_heading_like(text: str, match_start: int) -> bool:
    """True if `match_start` sits at (or very near) the beginning of its own
    line. Filters out casual in-text cross-references — e.g. "...as shown in
    Question 5 above, we already found..." — which mention a question number
    mid-sentence but are NOT that question's actual heading/content. Without
    this check, `locate_marker_in_range` could match the FIRST mention of a
    number anywhere in the range instead of the real question, which is
    exactly the "grabs the wrong question" failure mode this guards against."""
    line_start = text.rfind("\n", 0, match_start) + 1
    return (match_start - line_start) <= 3


_BANK_CANONICAL_TITLE = {
    "problems": "Problems",
    "conceptual": "Conceptual Questions",
    "objective": "Objective Questions",
}


def _match_subsection_line(stripped: str) -> Optional[str]:
    """Returns 'problems' / 'conceptual' / 'objective' if `stripped` looks
    like an end-of-chapter bank heading, else None."""
    for pattern, kind in _SUBSECTION_LINE_PATTERNS:
        if pattern.search(stripped):
            return kind
    return None


def _outline_bank_title(kind: str, stripped: str) -> str:
    """A short bank label. Cross-reference sentences that continue after the
    heading word ('Problems 45, 48, ... can also be') are not headings."""
    canonical = _BANK_CANONICAL_TITLE.get(kind) or stripped
    for pattern, pattern_kind in _SUBSECTION_LINE_PATTERNS:
        if pattern_kind != kind:
            continue
        match = pattern.search(stripped)
        if not match:
            continue
        rest = stripped[match.end():].strip(" :-–—.")
        return canonical if rest else stripped
    return canonical


def _section_kind_of_title(title: str) -> Optional[str]:
    if _CONCEPTUAL_TITLE_RE.search(title):
        return "conceptual"
    if _OBJECTIVE_TITLE_RE.search(title):
        return "objective"
    if _PROBLEMS_TITLE_RE.search(title) and not _CONCEPTUAL_TITLE_RE.search(title):
        return "problems"
    return None


_QUESTION_KEYWORD_ALT = "|".join(re.escape(k) for k in _QUESTION_KEYWORDS)
# Next-heading bound inside a question bank: keyword markers AND Serway-style
# bare "11." / "P11" so a match on "10." stops before the following item.
_ANY_NUMBERED_HEADING = re.compile(
    rf"(?:(?:{_QUESTION_KEYWORD_ALT})\s*[:.\-]?\s*\d{{1,4}}\b"
    rf"|\b\d{{1,4}}{_AZ_ORDINAL_SUFFIX}\s+(?:{_QUESTION_KEYWORD_ALT})"
    rf"|(?<!\d)\d{{1,4}}\s*[\.)]\s+\S"
    rf"|(?<!\d)(?:P|Q)\d{{1,4}}\b)",
    re.IGNORECASE,
)


def _compile_bare_number_heading(number: str) -> "re.Pattern":
    """Serway-style bank headings: '10. A Carnot...', '10)', 'P10', 'Q10'.
    Only used when the search range is already a question bank — never on a
    full chapter (that would hit §22.10 or Conceptual 10 first)."""
    n = re.escape(number)
    return re.compile(
        rf"(?:(?<!\d){n}\s*[\.)]\s+\S|(?<!\d)(?:P|Q){n}\b)",
        re.IGNORECASE,
    )


def _clip_to_bank_heading(combined_text: str, want_kind: str) -> tuple[int, int]:
    """If Conceptual / Objective / Problems headings share one blob, restrict
    the search to the requested bank so Conceptual 10 is not returned as
    Problems 10."""
    if want_kind not in {"problems", "conceptual", "objective"}:
        return 0, len(combined_text)
    hits = list(_BANK_HEADING_IN_TEXT.finditer(combined_text))
    if not hits:
        return 0, len(combined_text)
    start = 0
    found = False
    for match in hits:
        if _section_kind_of_title(match.group(1) or "") == want_kind:
            start = match.start()
            found = True
            break
    if not found:
        return 0, len(combined_text)
    end = len(combined_text)
    for match in hits:
        kind = _section_kind_of_title(match.group(1) or "")
        if (
            match.start() > start
            and kind in {"problems", "conceptual", "objective"}
            and kind != want_kind
        ):
            end = match.start()
            break
    return start, end


def _chapter_number_from_title(title: str) -> Optional[str]:
    if _PART_TITLE_RE.match(title or ""):
        return None
    if _is_subsection_title(title):
        return None
    match = _CHAPTER_IN_TITLE_RE.search(title or "")
    if match:
        return match.group(1)
    match = _LEADING_CH_NUM_RE.match(title or "")
    return match.group(1) if match else None


def _is_subsection_title(title: str) -> bool:
    return bool(_SUBSECTION_TITLE_RE.match(title or ""))


def _major_outline_number(title: str) -> Optional[str]:
    """Groups 'Chapter 1', '1. Kinematics', and '1.1 …' under major number 1."""
    if _PART_TITLE_RE.match(title or ""):
        return None
    found = _chapter_number_from_title(title)
    if found:
        return found
    match = _MAJOR_OUTLINE_NUM_RE.match(title or "")
    return match.group(1) if match else None


def _assign_sibling_page_ends(entries: list[dict], total_pages: int) -> None:
    """Each node's page_end is the page before the next sibling/uncle
    (next entry with level <= this node's level), NOT the next row — so a
    Part spans all of its chapters and a Chapter's Problems child does not
    steal the Part's range."""
    for i, entry in enumerate(entries):
        level = int(entry.get("level") or 1)
        next_start = total_pages + 1
        for later in entries[i + 1:]:
            if int(later.get("level") or 1) <= level:
                next_start = later["page_start"]
                break
        entry["page_end"] = max(entry["page_start"], next_start - 1)


def _assign_parent_indices(entries: list[dict]) -> None:
    stack: list[tuple[int, int]] = []
    for i, entry in enumerate(entries):
        level = int(entry.get("level") or 1)
        while stack and stack[-1][1] >= level:
            stack.pop()
        entry["parent_index"] = stack[-1][0] if stack else None
        stack.append((i, level))


def _descendant_indices(entries: list[dict], parent_i: int) -> list[int]:
    parent_level = int(entries[parent_i].get("level") or 1)
    out: list[int] = []
    for j in range(parent_i + 1, len(entries)):
        if int(entries[j].get("level") or 1) <= parent_level:
            break
        out.append(j)
    return out


def _scan_subsection_headings(
    page_texts: list[str], page_start: int, page_end: int, child_level: int,
) -> list[dict]:
    found: list[dict] = []
    seen_kinds: set[str] = set()
    last = min(page_end, len(page_texts))
    for abs_page in range(max(1, page_start), last + 1):
        for line in page_texts[abs_page - 1].splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 80:
                continue
            kind = _match_subsection_line(stripped)
            if not kind or kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            found.append({
                "title": _outline_bank_title(kind, stripped),
                "page_start": abs_page,
                "level": child_level,
            })
    return found


def _inject_scanned_subsections(entries: list[dict], page_texts: list[str]) -> list[dict]:
    """For chapter nodes that have no Problems/Conceptual/Objective child in
    the bookmark tree, scan the chapter's pages for those headings and insert
    them so `resolve_chapter_target` still works."""
    extras: list[dict] = []
    for i, entry in enumerate(entries):
        if _chapter_number_from_title(entry.get("title") or "") is None:
            continue
        descendants = _descendant_indices(entries, i)
        if any(_section_kind_of_title(entries[j].get("title") or "") for j in descendants):
            continue
        extras.extend(_scan_subsection_headings(
            page_texts,
            entry["page_start"],
            entry.get("page_end") or entry["page_start"],
            int(entry.get("level") or 1) + 1,
        ))
    if not extras:
        return entries
    merged = entries + extras
    merged.sort(key=lambda e: (e["page_start"], int(e.get("level") or 1)))
    return merged


def _regex_fallback_outline(page_texts: list[str]) -> list[dict]:
    """Best-effort structural map for PDFs with NO embedded bookmarks
    (common for scanned/badly-authored academic PDFs) — scans each page's
    text line-by-line for a chapter/section heading matching
    `_CHAPTER_PATTERN`, then for Problems/Conceptual/Objective bank
    headings. Deliberately conservative to avoid false positives from
    mid-sentence cross-references ("...as discussed in Chapter 22..."):
    only lines where the match starts at (or very near) column 0, and are
    short enough to plausibly BE a heading rather than a paragraph, count.
    This is a heuristic, not a guarantee — `_build_document_outline`'s
    caller (`get_document_outline`) explicitly tells the user when nothing
    could be detected, rather than fabricating structure."""
    entries: list[dict] = []
    seen_numbers: set[str] = set()
    seen_kinds_this_chapter: set[str] = set()

    for page_idx, page_text in enumerate(page_texts):
        page_number = page_idx + 1
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 120:
                continue

            match = _CHAPTER_PATTERN.search(stripped)
            if match and match.start() <= 3:
                number = match.group("num1") or match.group("num2")
                if number not in seen_numbers:
                    seen_numbers.add(number)
                    seen_kinds_this_chapter = set()
                    entries.append({"title": stripped, "page_start": page_number, "level": 1})
                    continue

            kind = _match_subsection_line(stripped)
            if kind and kind not in seen_kinds_this_chapter and entries:
                seen_kinds_this_chapter.add(kind)
                entries.append({
                    "title": _outline_bank_title(kind, stripped),
                    "page_start": page_number,
                    "level": 2,
                })

    return entries


def _build_document_outline(file_path: str, page_texts: Optional[list[str]] = None) -> list[dict]:
    """Builds the structural map (chapter/section title -> page range) that
    `get_document_outline` / `resolve_chapter_target` serve — this is what
    lets the LLM resolve "Chapter 22, Problems" to an exact page range
    WITHOUT touching embeddings/Chroma at all.

    Tier 1a: the PDF's own embedded outline/bookmarks (`fitz.get_toc()`) —
    exact, free, instant, when present. Works on scanned textbooks that
    still have bookmarks even with no text layer.
    Tier 1b (fallback): `_regex_fallback_outline` when there are no bookmarks
    AND `page_texts` is available (digital PDFs only).

    `page_end` is the next sibling/uncle, not the next row.

    Runs synchronously; the caller (`process_and_index_documents`) is
    responsible for offloading this via `asyncio.to_thread`.
    """
    native_entries: list[dict] = []
    total_pages = len(page_texts) if page_texts else 0
    try:
        doc = fitz.open(file_path)
        try:
            if not total_pages:
                total_pages = int(doc.page_count or 0)
            raw_toc = doc.get_toc()
        finally:
            doc.close()
        for level, title, page in raw_toc:
            cleaned = (title or "").strip()
            if not cleaned or not (1 <= page <= total_pages):
                continue
            if level < 1 or level > MAX_OUTLINE_LEVEL:
                continue
            native_entries.append({
                "title": cleaned,
                "page_start": page,
                "level": int(level),
            })
    except Exception as e:
        print(f"[DOC OUTLINE] fitz.get_toc() failed for '{file_path}': {e}", flush=True)

    if native_entries:
        entries = native_entries
    elif page_texts:
        entries = _regex_fallback_outline(page_texts)
    else:
        entries = []
    if not total_pages:
        return entries
    _assign_sibling_page_ends(entries, total_pages)
    _assign_parent_indices(entries)
    if native_entries and page_texts:
        entries = _inject_scanned_subsections(entries, page_texts)
        _assign_sibling_page_ends(entries, total_pages)
        _assign_parent_indices(entries)
    print(
        f"[DOC OUTLINE] {len(entries)} nodes "
        f"(native={len(native_entries)}, pages={total_pages})",
        flush=True,
    )
    return entries


# Standalone printed page-number line (typically a header/footer, on its
# own line with nothing else) — used by `_detect_printed_page_offset` below.
_PRINTED_PAGE_NUMBER_LINE = re.compile(r"^\s*(\d{1,4})\s*$")


def _build_page_label_maps(file_path: str, total_pages: int) -> tuple[dict[int, str], dict[str, int]]:
    """Tier A (exact): reads the PDF's own `/PageLabels` numbering scheme via
    PyMuPDF's `page.get_label()`, iterating every page to build the absolute
    <-> printed-label mapping. `get_label()` returns "" for every page when
    the PDF defines no such scheme at all (the common case for most
    real-world PDFs) — an empty result here is the expected signal for
    `process_and_index_documents` to fall back to `_detect_printed_page_offset`
    (Tier B) instead, not an error.
    """
    absolute_to_label: dict[int, str] = {}
    label_to_absolute: dict[str, int] = {}
    try:
        doc = fitz.open(file_path)
        try:
            for i in range(min(doc.page_count, total_pages)):
                label = doc.load_page(i).get_label().strip()
                if not label:
                    continue
                absolute_page = i + 1
                absolute_to_label[absolute_page] = label
                label_to_absolute.setdefault(label, absolute_page)
        finally:
            doc.close()
    except Exception as e:
        print(f"[PAGE MAPPING] get_label() scan failed for '{file_path}': {e}", flush=True)
    return absolute_to_label, label_to_absolute


def _detect_printed_page_offset(page_texts: list[str], sample_size: int = 40) -> Optional[int]:
    """Tier B (heuristic fallback), used only when the PDF has no
    `/PageLabels` scheme at all. Real textbooks print their page number
    somewhere in the header/footer of nearly every content page; this scans
    a sample of pages for a line that is JUST a bare number near the top or
    bottom of the extracted text, and looks for a single constant
    `absolute_page - printed_number` offset that keeps recurring across
    several different pages.

    Deliberately conservative: `best_votes < 3` returns None (no offset
    applied) rather than trusting a single stray digit-only line (a lone
    equation number, a footnote count) — an unproven guessed offset would
    silently point every subsequent page lookup at the wrong page, which is
    worse than the honest "no mapping available" fallback in
    `_printed_to_absolute_page`/`_absolute_to_printed_label`.
    """
    offset_votes: dict[int, int] = {}
    total_pages = len(page_texts)

    for page_idx in range(min(sample_size, total_pages)):
        lines = [ln.strip() for ln in page_texts[page_idx].splitlines() if ln.strip()]
        if not lines:
            continue
        candidate_lines = (lines[:2] + lines[-2:]) if len(lines) > 4 else lines

        for line in candidate_lines:
            match = _PRINTED_PAGE_NUMBER_LINE.match(line)
            if not match:
                continue
            printed_number = int(match.group(1))
            if printed_number <= 0 or printed_number > total_pages + 200:
                continue
            offset = (page_idx + 1) - printed_number
            offset_votes[offset] = offset_votes.get(offset, 0) + 1

    if not offset_votes:
        return None

    best_offset, best_votes = max(offset_votes.items(), key=lambda kv: kv[1])
    return best_offset if best_votes >= 3 else None


def _printed_to_absolute_page(thread_key: str, printed_page: int, total_pages: int) -> int:
    """Converts a PRINTED/logical page number (what the user said, e.g.
    "740") into the absolute 1-indexed position that actually lines up with
    `_page_texts[thread_key]`. Tries the exact `/PageLabels` reverse map
    first, then the heuristic constant offset, then finally assumes
    printed == absolute (today's original behavior) if neither mapping
    signal is available for this document — always returns SOME usable
    page number rather than failing outright."""
    reverse_map = _page_label_reverse.get(thread_key) or {}
    absolute = reverse_map.get(str(printed_page))
    if absolute is not None:
        return absolute

    offset = _printed_page_offsets.get(thread_key)
    if offset is not None:
        mapped = printed_page + offset
        if 1 <= mapped <= total_pages:
            return mapped

    return printed_page


def _absolute_to_printed_label(thread_key: str, absolute_page: int) -> str:
    """The reverse of `_printed_to_absolute_page` — used only to format
    page numbers BACK into printed/logical terms for tool output, so
    everything the LLM ever sees (and echoes back into later tool calls,
    including the persisted `active_navigation` cursor in graph.py) stays
    consistently in printed-page space end to end."""
    label = (_page_label_maps.get(thread_key) or {}).get(absolute_page)
    if label:
        return label

    offset = _printed_page_offsets.get(thread_key)
    if offset is not None:
        printed = absolute_page - offset
        if printed >= 1:
            return str(printed)

    return str(absolute_page)

# ---------------------------------------------------------------------------
# Phase 4 fix — Docling 50MB PDF hang (heavy OCR on natively digital PDFs).
#
# `process_and_index_documents` used to hand EVERY PDF straight to Docling's
# default converter, which runs its full OCR pipeline unconditionally
# regardless of whether the PDF already has a real, extractable text layer.
# For a 50MB born-digital textbook, that meant minutes of RapidOCR grinding
# over every page image (logging "RapidOCR returned empty result!" the
# whole time, since there was no actual scanned text for it to find) before
# ever reaching the trivial text-extraction result PyMuPDF gets in
# milliseconds.
#
# The fix is a hybrid loader (`_load_document_text` below):
#   - PDFs that already have a text layer -> PyMuPDF (`fitz`), synchronous
#     text extraction only, NO layout analysis, NO OCR.
#   - PDFs with no usable text layer (scanned/image-based) skip Docling.
#     Indexing stores TOC + page count only; `locate_marker_in_range` does
#     a bounded fast OCR pass then Gemini Flash on pages N/N+1. Full-file
#     RapidOCR was the 90s timeout.
#   - Non-PDF office files (.docx/.pptx) still use Docling.
#   - Every remaining Docling call is wrapped in a hard `asyncio.wait_for`
#     timeout so a pathological office conversion cannot hang the graph.
#
# Phase 5 fix — resilient digital extraction, without ever retrying Docling
# OCR on a document that has already proven to be a problem.
#
# A real 50MB textbook PDF smoke-tested as PyMuPDF raising
# `MuPDF error: syntax error: truncated sample function stream` — i.e. the
# file itself is malformed/truncated, not merely "scanned". The naive fix
# would be "if PyMuPDF fails, fall back to Docling" — but Docling (via
# RapidOCR) had ALREADY failed on this exact file earlier (empty OCR
# results, the original Phase 4 hang). Retrying the same broken bytes
# through the heavier OCR pipeline would just reproduce that hang, not
# recover anything. So on a PyMuPDF failure we instead try a SECOND,
# independent digital extractor (`PyPDFLoader`/`pypdf`, different parsing
# tolerances than PyMuPDF/MuPDF — a file that trips one up may still yield
# text from the other) and NEVER escalate to Docling OCR from this branch.
# If that also fails, `_load_document_text` returns `CORRUPTED_DOCUMENT_MESSAGE`
# (a plain string, not an exception) so `process_and_index_documents` can
# short-circuit gracefully instead of indexing garbage or crashing the
# backend — see its handling of that sentinel below.
# ---------------------------------------------------------------------------
DOCLING_TIMEOUT_SECONDS = 90
# Hard cap for PyMuPDF extract / FastEmbed+Chroma / outline stages. A hang
# here used to freeze the whole uvicorn process with no log line (the 3rd
# consecutive upload stall). `asyncio.wait_for` cannot kill the worker
# thread, but it DOES unblock the graph so the tool can return an error
# instead of waiting forever.
INDEX_STAGE_TIMEOUT_SECONDS = 30

CORRUPTED_DOCUMENT_MESSAGE = (
    "SYSTEM ERROR: The PDF file is corrupted or malformed and cannot be read. "
    "Do not attempt to re-index it — inform the user that the file could not be processed."
)

# Office files only (docx/pptx). PDFs never go through Docling — digital
# text uses PyMuPDF; scans are indexed as TOC + path and extracted lazily.
converter = DocumentConverter()

# ---------------------------------------------------------------------------
# Scientific-notation extraction repair. Many textbook PDFs (this app's
# Serway-style physics books included) embed the "×" multiplication sign and
# superscript exponent digits via custom font glyph mappings that PyMuPDF's
# plain `get_text()` sometimes resolves incorrectly — the "×" glyph collides
# with an unrelated digit in that font's cmap, and the superscript "^4" loses
# its caret and glues directly onto the preceding "10", producing exactly the
# corruption reported in QA: "2.00 × 10^4 N/C" extracted as "2.00 3 104 N/C".
# Applied uniformly to every extracted page (regardless of which extractor
# path produced it) so `doc_worker`'s LLM only ever sees ALREADY-CORRECT
# notation and never has a chance to inconsistently mis-transcribe it while
# "faithfully" quoting it back (which is what caused a SECOND occurrence of
# "10^5" in the SAME response to render correctly while this one didn't —
# the ambiguity lived in the raw extracted text itself, not the LLM).
# ---------------------------------------------------------------------------

# Matches "<mantissa> <isolated garbled operator digit> 10<exponent digits>",
# e.g. "2.00 3 104" -> mantissa="2.00", exponent="4". The isolated digit
# between two numeric tokens (bounded by `\b`/`\s`, never absorbing extra
# digits) is what makes this pattern specific enough to avoid false-positives
# on ordinary prose numbers — a real "×" almost never gets extracted as a
# lone digit sitting between two OTHER numbers unless this exact corruption
# occurred.
_GARBLED_SCI_NOTATION_RE = re.compile(
    r"(?P<mantissa>\d+(?:[.,]\d+)?)\s+\d\s+10(?P<exp>\d{1,2})\b(?!\.\d)"
)

# Matches "<mantissa> 10^<exponent>" with the multiplication operator missing
# entirely (rather than garbled into a digit) but the caret itself intact —
# a lower-risk, narrower pattern since it requires the literal "^" to already
# be present, e.g. "2.00 10^4" -> "2.00 × 10^4".
_MISSING_MULT_SCI_NOTATION_RE = re.compile(
    r"(?P<mantissa>\d+(?:[.,]\d+)?)\s+10\^(?P<exp>-?\d{1,2})\b"
)


def _normalize_math_notation(text: str) -> str:
    """Repairs the specific PDF-font-cmap scientific-notation corruption
    patterns described above. Idempotent and conservative: only rewrites
    text matching these two narrow, specific shapes; everything else in the
    page is left byte-for-byte untouched. Safe to run on every extracted
    page regardless of source (PyMuPDF, PyPDFLoader, or Docling's OCR
    markdown) since the same glyph-substitution patterns can surface from
    any of them."""
    if not text:
        return text

    def _reconstruct(match: "re.Match") -> str:
        return f"{match.group('mantissa')} \u00d7 10^{match.group('exp')}"

    text = _GARBLED_SCI_NOTATION_RE.sub(_reconstruct, text)
    text = _MISSING_MULT_SCI_NOTATION_RE.sub(_reconstruct, text)
    return text


def _pdf_has_text_layer(file_path: str, sample_pages: int = 5, min_chars_per_page: int = 40) -> bool:
    """Cheap heuristic, meant to run via `asyncio.to_thread`: opens the PDF
    with PyMuPDF and checks whether a handful of pages already carry a real
    text layer. Digital/born-text PDFs (the vast majority of academic
    textbooks/slides) return substantial text here almost instantly;
    scanned/image-only PDFs return empty/near-empty strings — that absence
    is the signal to enter scan mode (TOC + lazy OCR/vision), not a
    file-extension check. May raise on a malformed/truncated PDF — the
    caller (`_load_document_text`) is responsible for catching that."""
    doc = fitz.open(file_path)
    try:
        page_indices = range(min(sample_pages, doc.page_count))
        total_chars = sum(len(doc.load_page(i).get_text().strip()) for i in page_indices)
        pages_checked = max(len(page_indices), 1)
        return (total_chars / pages_checked) >= min_chars_per_page
    finally:
        doc.close()


def _extract_pages_fast(file_path: str) -> list[str]:
    """Fast, OCR-free PER-PAGE text extraction for digital/born-text PDFs
    via PyMuPDF, meant to run via `asyncio.to_thread`. Orders of magnitude
    faster than routing these through Docling's full layout+OCR pipeline,
    which is simply unnecessary work for a document that already has a
    real, machine-readable text layer.

    Returns ONE STRING PER PAGE (index i = page i, 0-indexed) rather than a
    single flattened blob — this page-level shape is the prerequisite for
    Tier 1 structural navigation (`get_document_outline`/`read_page_range`/
    `locate_marker_in_range` below), which needs to map a resolved chapter/
    question back to an exact page number. May raise on a malformed/
    truncated PDF — the caller (`_load_document_text`) is responsible for
    catching that and trying the secondary extractor instead."""
    doc = fitz.open(file_path)
    try:
        return [doc.load_page(i).get_text() for i in range(doc.page_count)]
    finally:
        doc.close()


def _extract_pages_pypdf(file_path: str) -> list[str]:
    """Secondary, independent digital-text extractor, meant to run via
    `asyncio.to_thread`. ONLY ever invoked after PyMuPDF itself has failed
    on a malformed/truncated PDF (see the Phase 5 note above) — deliberately
    NOT a Docling/OCR fallback. Returns one string per page, for the same
    reason as `_extract_pages_fast` above. Raises if it produces no usable
    text either, so the caller can treat "both digital extractors failed" as
    a single, unambiguous failure case."""
    pages = [page.page_content for page in PyPDFLoader(file_path).load()]
    if not "".join(pages).strip():
        raise ValueError("PyPDFLoader produced no extractable text either.")
    return pages


async def _run_docling_conversion(active_converter: DocumentConverter, file_path: str) -> str:
    """Runs a Docling conversion off the event loop, hard-capped by
    `DOCLING_TIMEOUT_SECONDS` (raises `asyncio.TimeoutError` past that) so a
    pathological OCR run can never hang the graph run forever."""
    def _convert() -> str:
        return active_converter.convert(file_path).document.export_to_markdown()

    return await asyncio.wait_for(asyncio.to_thread(_convert), timeout=DOCLING_TIMEOUT_SECONDS)


async def _load_document_text(file_path: str) -> tuple[str, Optional[list[str]]]:
    """Routes document loading between the fast PyMuPDF path, a secondary
    digital-extraction fallback, scan-mode metadata, and Docling for office
    files — per the hybrid strategy described in the Phase 4/Phase 5 notes
    above.

    Returns `(full_text_for_chunking, page_texts)`. `page_texts` is a list
    of one string per PDF page whenever a digital extractor succeeded.
    For scanned PDFs it is `None` and chunking text is `""` — the caller
    then indexes TOC + page count only (no Docling, no Chroma).
    Office files still go through Docling (`page_texts` None, markdown blob).

    Returns `(CORRUPTED_DOCUMENT_MESSAGE, None)` only when the PDF cannot
    be opened at all.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            has_text_layer = await _await_index_stage(
                "text_layer_probe",
                asyncio.to_thread(_pdf_has_text_layer, file_path),
            )
        except asyncio.TimeoutError:
            print(f"[DOC LOADER] PyMuPDF inspect timed out on '{file_path}'", flush=True)
            return await _extract_pages_pypdf_or_fail(file_path)
        except Exception as e:
            print(f"[DOC LOADER] PyMuPDF failed to inspect '{file_path}': {e}", flush=True)
            return await _extract_pages_pypdf_or_fail(file_path)

        if has_text_layer:
            try:
                pages = await _await_index_stage(
                    "pymupdf_extract",
                    asyncio.to_thread(_extract_pages_fast, file_path),
                )
                pages = [_normalize_math_notation(p) for p in pages]
                return "\n\n".join(pages), pages
            except asyncio.TimeoutError:
                print(f"[DOC LOADER] PyMuPDF extract timed out on '{file_path}'", flush=True)
                return await _extract_pages_pypdf_or_fail(file_path)
            except Exception as e:
                print(f"[DOC LOADER] PyMuPDF failed to extract text from '{file_path}': {e}", flush=True)
                return await _extract_pages_pypdf_or_fail(file_path)

        print(
            "[DOC LOADER] scan PDF — skipping Docling full-file OCR "
            "(lazy OCR + Gemini at locate time)",
            flush=True,
        )
        return "", None

    print("[DOC LOADER] stage=docling_office START", flush=True)
    converted_text = await _run_docling_conversion(converter, file_path)
    print("[DOC LOADER] stage=docling_office END", flush=True)
    return _normalize_math_notation(converted_text), None


async def _extract_pages_pypdf_or_fail(file_path: str) -> tuple[str, Optional[list[str]]]:
    """Last-resort digital extraction after PyMuPDF has already failed.
    Never falls through to Docling OCR. If pypdf also fails but the PDF
    still opens, treat it as a scan (metadata index) rather than corrupted.
    """
    try:
        pages = await _await_index_stage(
            "pypdf_extract",
            asyncio.to_thread(_extract_pages_pypdf, file_path),
        )
        pages = [_normalize_math_notation(p) for p in pages]
        return "\n\n".join(pages), pages
    except Exception as e:
        print(f"[DOC LOADER] Secondary extractor (PyPDFLoader) also failed on '{file_path}': {e}", flush=True)
        n = count_pdf_pages(file_path)
        if n > 0:
            print(f"[DOC LOADER] treating as scan ({n} pages, no text layer)", flush=True)
            return "", None
        return CORRUPTED_DOCUMENT_MESSAGE, None


# Phase 6 defense-in-depth: local embeddings make indexing $0, but a
# pathological upload (a scanned 2000-page archive, a mis-uploaded dataset
# dump, etc.) could still produce an unbounded number of chunks, each one
# living in memory for the life of the process (`_indexed_chunks`,
# `_vector_stores`) and costing CPU time to embed. This hard cap bounds
# worst-case latency/memory per document regardless of which embedding
# backend is behind `embeddings`.
#
# 800 chunks at the current chunk_size=1800 covers roughly 350-400 dense
# textbook pages — generous for any legitimate single upload this app
# expects. Benchmarked on this machine, the 12-layer multilingual model
# above costs ~150-200ms/chunk for a full 1800-char chunk (CPU, ONNX), so
# 800 chunks is a worst case of ~2-3 minutes of one-time, background-thread
# (non-event-loop-blocking) indexing — still bounded and always $0, unlike
# the incident this phase fixes. If that latency proves too slow on a given
# deployment's hardware, lower this value, or trade `_FASTEMBED_MODEL_NAME`
# down to a smaller/faster 6-layer model (e.g. `BAAI/bge-small-en-v1.5`) in
# exchange for weaker non-English (incl. Azerbaijani) retrieval quality.
MAX_INDEXED_CHUNKS = 800


def _resolve_thread_key(config: RunnableConfig) -> str:
    """Extracts the LangGraph `thread_id` from the run's injected config so
    document memory can be scoped per-conversation.

    `config` is auto-injected by LangChain at tool-call time (it's excluded
    from the schema shown to the LLM — the model never sees or fills this
    argument) as long as the caller forwards its own run config into
    `.ainvoke(tool_args, config=...)`. Falls back to a shared default bucket
    only when a tool is invoked with no run config at all (e.g. an ad-hoc
    script or test), matching today's un-scoped behavior in that edge case.
    """
    configurable = (config or {}).get("configurable") or {}
    return configurable.get("thread_id") or _DEFAULT_THREAD_KEY


def _thread_page_count(thread_key: str) -> int:
    n = _pdf_page_counts.get(thread_key) or 0
    if n:
        return n
    return len(_page_texts.get(thread_key) or [])


def get_indexed_page_count(config: RunnableConfig) -> int:
    """How many pages the current thread's document has. Uses stored
    `_pdf_page_counts` so scanned PDFs (empty `_page_texts`) still count."""
    return _thread_page_count(_resolve_thread_key(config))


def get_last_vision_extract(config: RunnableConfig) -> Optional[dict]:
    """Last Gemini extract for this thread (stem + 0–1 boxes). Phase 3
    canvas figure ops read this; the tool message itself stays stem-only."""
    return _vision_extracts.get(_resolve_thread_key(config))


def consume_vision_figure_urls(config: RunnableConfig) -> list[str]:
    """Pop cropped figure data-URLs once so composer emits them this turn only."""
    payload = _vision_extracts.get(_resolve_thread_key(config))
    if not isinstance(payload, dict):
        return []
    urls = payload.pop("image_urls", None) or []
    return [u for u in urls if isinstance(u, str) and u.startswith("data:image")]


def store_vision_figure_urls(config: RunnableConfig, urls: list) -> None:
    """Write cropped figure data-URLs onto this thread without replacing the
    rest of a prior PDF vision payload. Empty `urls` clears leftovers."""
    thread_key = _resolve_thread_key(config)
    payload = _vision_extracts.get(thread_key)
    if not isinstance(payload, dict):
        payload = {}
        _vision_extracts[thread_key] = payload
    payload["image_urls"] = [
        u for u in (urls or []) if isinstance(u, str) and u.startswith("data:image")
    ]


def is_short_indexed_document(config: RunnableConfig) -> bool:
    """True only when this thread has a page-aware index of 1..SHORT_DOC_MAX_PAGES
    pages. Large textbooks and OCR-only documents return False so their
    outline/search tools stay available."""
    page_count = get_indexed_page_count(config)
    return 0 < page_count <= SHORT_DOC_MAX_PAGES


def lookup_outline_chapter(config: RunnableConfig, printed_page: Optional[int]) -> Optional[str]:
    """Returns the outline heading that contains `printed_page` (printed /
    logical numbering, same space as `active_navigation`), or None.
    Used by `graph._doc_worker_finish_update` to compose the Active Problem
    header without a second LLM call."""
    if printed_page is None:
        return None
    try:
        page_num = int(printed_page)
    except (TypeError, ValueError):
        return None

    thread_key = _resolve_thread_key(config)
    outline = _document_outlines.get(thread_key) or []
    total_pages = _thread_page_count(thread_key)
    if not outline or not total_pages:
        return None

    absolute = _printed_to_absolute_page(thread_key, page_num, total_pages)
    matches = [
        entry for entry in outline
        if entry.get("page_start", 0) <= absolute <= entry.get("page_end", entry.get("page_start", 0))
    ]
    if not matches:
        return None
    matches.sort(key=lambda e: (int(e.get("level") or 1), e.get("page_start", 0)))
    deepest = matches[-1]
    title = str(deepest.get("title") or "").strip()
    if title and _section_kind_of_title(title):
        current = deepest
        seen: set[int] = set()
        while isinstance(current, dict) and id(current) not in seen:
            seen.add(id(current))
            parent_i = current.get("parent_index")
            if not isinstance(parent_i, int) or not (0 <= parent_i < len(outline)):
                break
            current = outline[parent_i]
            parent_title = str(current.get("title") or "").strip()
            if (
                parent_title
                and not _section_kind_of_title(parent_title)
                and _chapter_number_from_title(parent_title)
            ):
                return parent_title
    return title or None


def _short_doc_block_message(config: RunnableConfig, tool_name: str) -> Optional[str]:
    if not is_short_indexed_document(config):
        return None
    n = get_indexed_page_count(config)
    return (
        f"Xəta: Bu sənəd QISADIR ({n} səhifə). '{tool_name}' bu sənəd üçün BLOK EDİLİB. "
        f"YALNIZ 'read_page_range' çağır (start_page=1, end_page={n}) və dərhal yekun cavab ver — "
        "başqa alət çağırma."
    )


async def _await_index_stage(stage: str, awaitable, timeout: float = INDEX_STAGE_TIMEOUT_SECONDS):
    """Runs one indexing sub-step with START/END/TIMEOUT logs and a hard
    asyncio timeout so a wedged ONNX/PyMuPDF call cannot stall the graph
    indefinitely. The underlying thread may keep running after timeout;
    the tool call itself still returns."""
    t0 = time.time()
    print(f"[DOC LOADER] stage={stage} START", flush=True)
    try:
        result = await asyncio.wait_for(awaitable, timeout=timeout)
        print(f"[DOC LOADER] stage={stage} END took={time.time() - t0:.2f}s", flush=True)
        return result
    except asyncio.TimeoutError:
        print(f"[DOC LOADER] stage={stage} TIMEOUT after {timeout}s", flush=True)
        raise


def _dispose_thread_vector_store(thread_key: str) -> None:
    """Deletes this thread's pgvector chunks so a new upload replaces them."""
    try:
        with _vector_write_lock(thread_key):
            delete_document_chunks(thread_key)
        print(f"[DOC LOADER] Cleared pgvector chunks for thread={thread_key}", flush=True)
    except Exception as e:
        print(f"[DOC LOADER] pgvector delete failed for thread={thread_key}: {e}", flush=True)


def clear_thread_document_cache(thread_key: str) -> None:
    """Drop every per-thread document map so a new upload cannot inherit
    the previous file's page count, outline, OCR/vision cache, or Chroma store."""
    _indexed_file_paths.pop(thread_key, None)
    _pdf_page_counts.pop(thread_key, None)
    _scan_mode.pop(thread_key, None)
    _ocr_page_cache.pop(thread_key, None)
    _vision_extracts.pop(thread_key, None)
    _page_texts.pop(thread_key, None)
    _document_outlines.pop(thread_key, None)
    _resolved_targets.pop(thread_key, None)
    _page_label_maps.pop(thread_key, None)
    _page_label_reverse.pop(thread_key, None)
    _printed_page_offsets.pop(thread_key, None)
    _indexed_chunks.pop(thread_key, None)
    _embeddings_ready.pop(thread_key, None)
    # Order matters: invalidate any in-flight writer first, then stop it, then
    # delete under the write lock so a stale insert can never land afterwards.
    _doc_generation[thread_key] = _doc_generation.get(thread_key, 0) + 1
    _cancel_embed_job(thread_key)
    _dispose_thread_vector_store(thread_key)
    print(f"[DOC LOADER] Cleared thread document cache for thread={thread_key}", flush=True)


def get_indexed_file_path(config: RunnableConfig) -> Optional[str]:
    """Absolute path last successfully indexed for this thread, or None."""
    return _indexed_file_paths.get(_resolve_thread_key(config))


_INDEX_WIPE_SKIP_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def reset_thread_document_if_new_file(config: RunnableConfig, file_path: Optional[str]) -> bool:
    """Wipe the thread document cache when this turn's File path differs
    from the last indexed path. Returns True when a prior document was discarded.
    Images never wipe the textbook index — a screenshot after a PDF must not
    discard Chroma / page texts."""
    if not file_path:
        return False
    lowered = str(file_path).lower()
    if any(lowered.endswith(ext) for ext in _INDEX_WIPE_SKIP_EXTS):
        return False
    thread_key = _resolve_thread_key(config)
    previous = _indexed_file_paths.get(thread_key)
    if previous == file_path:
        return False
    if previous is None and not _pdf_page_counts.get(thread_key):
        return False
    print(
        f"[DOC LOADER] New upload replaces prior index: '{previous}' -> '{file_path}'",
        flush=True,
    )
    clear_thread_document_cache(thread_key)
    return previous is not None


def _chroma_collection_name(thread_key: str, file_path: str) -> str:
    digest = hashlib.md5(f"{thread_key}:{file_path}".encode("utf-8")).hexdigest()[:20]
    return f"lnl_{digest}"


def _index_ready_message(
    thread_key: str,
    skipped: bool,
    chunks: int = 0,
    was_truncated: bool = False,
    embedded: bool = True,
) -> str:
    page_count = _thread_page_count(thread_key)
    is_scan = bool(_scan_mode.get(thread_key))
    outline_count = len(_document_outlines.get(thread_key) or [])
    if skipped:
        if is_scan:
            prefix = (
                f"SKAN REJİMİ: sənəd artıq indekslənib ({page_count} səhifə, "
                "tam-fayl OCR SKIP). Yenidən emal etməyə ehtiyac yoxdur."
            )
        else:
            prefix = (
                "Sənəd artıq bu söhbətdə indekslənib — yenidən emal etməyə ehtiyac yoxdur."
            )
            if page_count:
                prefix += f" ({page_count} səhifə mövcuddur.)"
    elif is_scan:
        prefix = f"SKAN REJİMİ: {page_count} səhifə, tam-fayl OCR SKIP EDİLDİ. "
    elif chunks and not embedded:
        prefix = (
            "Fayl uğurla emal olundu. Səhifə mətni və fəsil xəritəsi hazırdır. "
            "Vektor axtarışı ilk search_in_document çağırışında qurulur və təxminən bir dəqiqə çəkə bilər."
        )
    else:
        prefix = (
            f"Fayl uğurla emal olundu"
            + (f" və {chunks} parçaya bölünərək vektor bazasına yükləndi." if chunks else ".")
        )
    if 0 < page_count <= SHORT_DOC_MAX_PAGES:
        return (
            f"{prefix} Qısa sənəd. "
            f"'get_document_outline'/'resolve_chapter_target'/'locate_marker_in_range'/"
            f"'search_in_document' BLOK EDİLİB. "
            f"YALNIZ 'read_page_range(1, {page_count})' çağır və dərhal cavab ver."
        )
    if is_scan:
        nav = (
            f" Bookmark xəritəsi: {outline_count} fəsil/bölmə."
            if outline_count else
            " Bookmark yoxdur — çap olunmuş səhifə nömrəsi istə; search_in_document İŞLƏMİR."
        )
        return (
            f"{prefix}{nav} Nömrələnmiş məsələlər üçün "
            "'resolve_chapter_target' sonra 'locate_marker_in_range' çağır. "
            "search_in_document İSTİFADƏ ETMƏ."
        )
    extra = ""
    if outline_count:
        extra += f" Struktur naviqasiya üçün {page_count} səhifə indekslənib ({outline_count} fəsil/bölmə aşkar edildi)."
    elif page_count:
        extra += f" Struktur naviqasiya üçün {page_count} səhifə indekslənib, lakin heç bir fəsil başlığı aşkar edilmədi."
    if was_truncated:
        extra += (
            f" QEYD: yalnız ilk {MAX_INDEXED_CHUNKS} parça SEMANTIK axtarış üçündür; "
            "'resolve_chapter_target' / 'read_page_range' tam sənədi əhatə edir."
        )
    extra += (
        " Birbaşa 'resolve_chapter_target' (nömrələnmiş məsələlər), "
        "'get_document_outline', 'read_page_range' və ya 'search_in_document' "
        "vasitəsilə davam et."
    )
    return prefix + extra


@tool
async def process_and_index_documents(file_path: str, config: RunnableConfig) -> str:
    """
    Reads the document from file_path and prepares BOTH retrieval tiers,
    scoped to the current conversation thread:
      - Tier 2 (semantic fallback): splits the text into chunks, embeds
        them, and stores them in a vector database (capped at
        MAX_INDEXED_CHUNKS).
      - Tier 1 (structural navigation): builds a page-indexed text cache
        plus a chapter/section outline (`get_document_outline`), entirely
        independent of the Tier 2 chunk cap — so chapters far past
        MAX_INDEXED_CHUNKS remain reachable via `read_page_range`/
        `locate_marker_in_range` even though they were never embedded.
    Use this tool FIRST when a new file path is provided.
    """
    thread_key = _resolve_thread_key(config)
    configurable = config.get("configurable") or {}
    remember_thread_auth(
        thread_key,
        str(configurable.get("user_id") or ""),
        str(configurable.get("access_token") or ""),
    )

    # Idempotency guard (see `_indexed_file_paths` above): the SAME file for
    # this thread was already fully indexed — skip the entire extract ->
    # chunk -> embed pipeline instead of redoing ~5s of work for nothing.
    if _indexed_file_paths.get(thread_key) == file_path:
        print(f"[TIMING] Tool Doc Indexing SKIPPED (already indexed): '{file_path}'", flush=True)
        return _index_ready_message(thread_key, skipped=True)

    indexed = _indexed_file_paths.get(thread_key) or ""
    lowered = str(file_path or "").lower()
    is_image = any(lowered.endswith(ext) for ext in _INDEX_WIPE_SKIP_EXTS)
    if is_image or not (file_path and os.path.isfile(file_path)):
        if indexed and os.path.isfile(indexed):
            print(
                f"[TIMING] Tool Doc Indexing SKIPPED (kept textbook index): '{file_path}'",
                flush=True,
            )
            return _index_ready_message(thread_key, skipped=True)
        if is_image:
            return (
                "Xəta: Şəkil faylı əsas sənəd kimi indekslənmir. "
                "Əsas PDF yolunu istifadə edin."
            )
        return f"Xəta: Fayl tapılmadı və indeks toxunulmaz qaldı: {file_path}"

    print("[TIMING] Tool Doc Indexing Started...", flush=True)
    t0 = time.time()
    clear_thread_document_cache(thread_key)
    try:
        context, page_texts = await _load_document_text(file_path)

        if context == CORRUPTED_DOCUMENT_MESSAGE:
            return context

        ext = os.path.splitext(file_path)[1].lower()
        is_scan = ext == ".pdf" and page_texts is None and not context
        page_count = len(page_texts) if page_texts else count_pdf_pages(file_path)
        _pdf_page_counts[thread_key] = page_count
        _scan_mode[thread_key] = is_scan

        _dispose_thread_vector_store(thread_key)

        is_short_doc = 0 < page_count <= SHORT_DOC_MAX_PAGES
        chunks: list[str] = []
        was_truncated = False
        if is_scan or is_short_doc or not context.strip():
            print(
                f"[DOC LOADER] stage=embed SKIPPED "
                f"(scan={is_scan}, short={is_short_doc}, pages={page_count})",
                flush=True,
            )
            _indexed_chunks[thread_key] = []
            _embeddings_ready[thread_key] = False
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)
            chunks = text_splitter.split_text(context)
            was_truncated = len(chunks) > MAX_INDEXED_CHUNKS
            if was_truncated:
                print(
                    f"[DOC LOADER] '{file_path}' produced {len(chunks)} chunks; "
                    f"capping at {MAX_INDEXED_CHUNKS} (see MAX_INDEXED_CHUNKS).",
                    flush=True,
                )
                chunks = chunks[:MAX_INDEXED_CHUNKS]
            _indexed_chunks[thread_key] = chunks
            _embeddings_ready[thread_key] = False
            print("[DOC LOADER] stage=embed DEFERRED until search_in_document", flush=True)

        _page_texts[thread_key] = page_texts or []
        if page_count > 0 and ext == ".pdf":
            _document_outlines[thread_key] = await _await_index_stage(
                "outline",
                asyncio.to_thread(_build_document_outline, file_path, page_texts),
            )
            absolute_to_label, label_to_absolute = await _await_index_stage(
                "page_labels",
                asyncio.to_thread(_build_page_label_maps, file_path, page_count),
            )
            _page_label_maps[thread_key] = absolute_to_label
            _page_label_reverse[thread_key] = label_to_absolute
            _printed_page_offsets[thread_key] = (
                None if label_to_absolute
                else (_detect_printed_page_offset(page_texts) if page_texts else None)
            )
        else:
            _document_outlines[thread_key] = []
            _page_label_maps[thread_key] = {}
            _page_label_reverse[thread_key] = {}
            _printed_page_offsets[thread_key] = None

        _indexed_file_paths[thread_key] = file_path
        return _index_ready_message(
            thread_key,
            skipped=False,
            chunks=len(chunks),
            was_truncated=was_truncated,
            embedded=False,
        )

    except asyncio.TimeoutError:
        return (
            "Xəta: Sənədin indekslənməsi vaxt limitini keçdi (OCR, mətn çıxarışı və ya "
            "embedding mərhələsi cavab vermədi). Zəhmət olmasa sənədi yenidən yükləyin "
            "və ya daha kiçik bir fayl istifadə edin."
        )
    except Exception as e:
        return f"Sənədin indekslənməsi zamanı xəta baş verdi: {str(e)}"
    finally:
        print(f"[TIMING] Tool Doc Indexing Ended - Took {time.time() - t0:.2f}s", flush=True)


@tool
def get_document_outline(config: RunnableConfig) -> str:
    """
    TIER 1 — STRUCTURAL NAVIGATION. Returns the document's nested bookmark
    map (Part → Chapter → Problems) with printed page ranges. For a numbered
    end-of-chapter problem, do NOT hunt the whole chapter: call
    `resolve_chapter_target` (defaults to the Problems/Exercises bank).

    All page numbers shown here are PRINTED/logical page numbers — the same
    numbers actually printed on the page, and the same numbers you must pass
    straight into `read_page_range`/`locate_marker_in_range` (they perform
    their own internal conversion; NEVER adjust these numbers yourself).
    Call this AT MOST ONCE per user query.
    """
    blocked = _short_doc_block_message(config, "get_document_outline")
    if blocked:
        return blocked

    thread_key = _resolve_thread_key(config)
    outline = _document_outlines.get(thread_key)

    if outline is None:
        return "Xəta: Hələ heç bir fayl indekslənməyib. Əvvəlcə 'process_and_index_documents' tool-unu çağırın."

    if not outline:
        if _scan_mode.get(thread_key):
            return (
                "Bu skan edilmiş sənəddə bookmark/fəsil xəritəsi yoxdur. "
                "Çap olunmuş səhifə nömrəsini istifadəçi-dən öyrən və "
                "'locate_marker_in_range' (start_page=həmin səhifə, end_page=həmin+1) çağır. "
                "search_in_document İSTİFADƏ ETMƏ."
            )
        return (
            "Bu sənəd üçün struktur xəritə (fəsil/bölmə) aşkar edilmədi — sənəddə nə bookmark, "
            "nə də aşkar fəsil başlıqları var. "
            "Bunun əvəzinə 'search_in_document' (Tier 2, semantik axtarış) alətini istifadə edin."
        )

    lines = []
    compact = len(outline) > _OUTLINE_COMPACT_MAX
    for entry in outline:
        level = int(entry.get("level") or 1)
        if compact and level > 2:
            continue
        indent = "  " * max(0, level - 1)
        start = _absolute_to_printed_label(thread_key, entry["page_start"])
        end = _absolute_to_printed_label(thread_key, entry["page_end"])
        lines.append(f"{indent}- {entry['title']} (Səhifə {start}-{end})")
    header = (
        "SƏNƏDİN STRUKTUR XƏRİTƏSİ (ixtiyari səviyyə: Part → Fəsil → Problems; "
        "çap olunmuş səhifə nömrələri):\n"
    )
    note = ""
    if compact:
        note = (
            "\n(Qısa indeks: yalnız Part/Fəsil göstərilir. Konkret sual bankı üçün "
            "`resolve_chapter_target(chapter=\"22\", section_kind=\"problems\")` çağır — "
            "TAM fəsil aralığında axtarma.)"
        )
    else:
        note = (
            "\nNömrələnmiş məsələlər üçün TAM fəsili oxuma — "
            "`resolve_chapter_target` ilə Problems/Exercises alt bölməsini götür."
        )
    return header + "\n".join(lines) + note


def _find_chapter_index(outline: list[dict], number: str) -> Optional[int]:
    """Best TOC node for a chapter number — skips Part N and '1.1' subsections."""
    exact: list[int] = []
    weak: list[int] = []
    for i, entry in enumerate(outline):
        title = entry.get("title") or ""
        if _PART_TITLE_RE.match(title) or _is_subsection_title(title):
            continue
        found = _chapter_number_from_title(title)
        if found == number:
            exact.append(i)
        elif re.search(rf"\b{re.escape(number)}\b", title) and not re.search(
            rf"\b{re.escape(number)}\.\d", title
        ):
            weak.append(i)

    def _rank(i: int) -> tuple:
        title = outline[i].get("title") or ""
        has_word = 0 if _CHAPTER_IN_TITLE_RE.search(title) else 1
        level = int(outline[i].get("level") or 1)
        return (has_word, level, i)

    if exact:
        exact.sort(key=_rank)
        return exact[0]
    if weak:
        weak.sort(key=_rank)
        return weak[0]
    return None


def _span_to_next_chapter(
    outline: list[dict], chapter_i: int, total_pages: int
) -> tuple[int, int]:
    """Start of this chapter → start of the next peer chapter (inclusive)."""
    start = int(outline[chapter_i]["page_start"])
    level = int(outline[chapter_i].get("level") or 1)
    next_start = total_pages
    for later in outline[chapter_i + 1:]:
        if int(later.get("level") or 1) <= level:
            next_start = int(later.get("page_start") or next_start)
            break
    end = min(total_pages, max(start, next_start))
    return start, end


def _chapter_block_pages(
    outline: list[dict], number: str, total_pages: int
) -> Optional[tuple[int, int]]:
    """When the TOC has no 'Chapter N' parent — only 1.1 / 1.2 nodes — span
    from the first bookmark in that major number to the first bookmark of
    the next major number."""
    starts: list[int] = []
    next_starts: list[int] = []
    try:
        want = int(number)
    except ValueError:
        return None
    for entry in outline:
        ident = _major_outline_number(entry.get("title") or "")
        if ident is None:
            continue
        try:
            n = int(ident)
        except ValueError:
            continue
        page = int(entry.get("page_start") or 0)
        if not page:
            continue
        if n == want:
            starts.append(page)
        elif n > want and starts:
            next_starts.append(page)
    if not starts:
        return None
    start = min(starts)
    end = min(next_starts) if next_starts else total_pages
    return start, min(total_pages, max(start, end))


def _pick_bank_node(
    outline: list[dict],
    chapter_i: Optional[int],
    section_kind: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
) -> Optional[int]:
    """PROBLEMS / EXERCISES: prefer a direct child, then any descendant
    (shallower first), then a sibling, then a PROBLEMS parent."""
    if section_kind not in {"problems", "conceptual", "objective"}:
        return None

    def _is_kind(j: int) -> bool:
        return _section_kind_of_title(outline[j].get("title") or "") == section_kind

    if chapter_i is not None:
        descendants = _descendant_indices(outline, chapter_i)
        hits = [j for j in descendants if _is_kind(j)]
        if hits:
            hits.sort(key=lambda j: (
                int(outline[j].get("level") or 1),
                int(outline[j].get("page_start") or 0),
            ))
            return hits[0]
        parent = outline[chapter_i].get("parent_index")
        siblings = [
            j for j, entry in enumerate(outline)
            if j != chapter_i and entry.get("parent_index") == parent and _is_kind(j)
        ]
        if siblings:
            siblings.sort(key=lambda j: int(outline[j].get("page_start") or 0))
            return siblings[0]
        if parent is not None and _is_kind(int(parent)):
            return int(parent)

    hits = [j for j in range(len(outline)) if _is_kind(j)]
    if page_start is not None and page_end is not None:
        hits = [
            j for j in hits
            if page_start <= int(outline[j].get("page_start") or 0) <= page_end
        ]
    if not hits:
        return None
    hits.sort(key=lambda j: int(outline[j].get("page_start") or 0))
    return hits[0]


@tool
def resolve_chapter_target(
    chapter: str,
    section_kind: str = "problems",
    config: RunnableConfig = None,
) -> str:
    """
    TIER 1 — resolve a named chapter to the printed page range of a specific
    end-of-chapter bank using the PDF bookmark tree (NOT the vector store).

    `chapter`: "22", "Chapter 22", "Section 22", "Fəsil 22".
    `section_kind`:
      - "problems" (DEFAULT) — Problems / Exercises / Məsələlər. Use this
        whenever the user asks for numbered questions without saying
        conceptual/objective.
      - "conceptual" — Conceptual Questions / Konseptual suallar.
      - "objective" — Objective / multiple-choice / Quick Quiz.
      - "chapter" — the whole chapter (theory + every bank). Avoid this
        for numbered problem extraction.

    Returns printed start/end pages to pass UNCHANGED into
    `locate_marker_in_range`. The range is this chapter through the start
    of the next chapter (not the first 1.1 subsection). If a PROBLEMS /
    EXERCISES bookmark exists, Fast OCR starts there. Call this AT MOST
    ONCE per named chapter per turn.
    """
    blocked = _short_doc_block_message(config, "resolve_chapter_target")
    if blocked:
        return blocked

    thread_key = _resolve_thread_key(config)
    outline = _document_outlines.get(thread_key)
    if outline is None:
        return "Xəta: Hələ heç bir fayl indekslənməyib. Əvvəlcə 'process_and_index_documents' tool-unu çağırın."
    if not outline:
        if _scan_mode.get(thread_key):
            return (
                "Xəta: Bu skan edilmiş sənəddə bookmark/fəsil xəritəsi yoxdur. "
                "Çap olunmuş səhifə nömrəsini istifadəçidən öyrən və "
                "'locate_marker_in_range(start_page=həmin, end_page=həmin+1, marker=...)' çağır. "
                "search_in_document İSTİFADƏ ETMƏ."
            )
        return (
            "Xəta: Bu sənəd üçün struktur xəritə yoxdur. "
            "Bunun əvəzinə 'search_in_document' (Tier 2, semantik axtarış) alətini istifadə edin."
        )

    kind = (section_kind or "problems").strip().lower()
    if kind not in {"problems", "conceptual", "objective", "chapter"}:
        kind = "problems"

    number_match = re.search(r"\d{1,3}", chapter or "")
    if not number_match:
        return f"Xəta: '{chapter}' daxilində fəsil nömrəsi tapılmadı (məs. 'Chapter 22')."
    number = number_match.group(0)

    total_pages = _thread_page_count(thread_key) or 1
    chapter_i = _find_chapter_index(outline, number)
    chapter_entry: dict
    if chapter_i is not None:
        chapter_entry = outline[chapter_i]
        ch_start, ch_end = _span_to_next_chapter(outline, chapter_i, total_pages)
    else:
        block = _chapter_block_pages(outline, number, total_pages)
        if block is None:
            return (
                f"Xəta: Konturda Chapter/Fəsil {number} tapılmadı. "
                "`get_document_outline` nəticəsinə bax və adı dəqiqləşdir."
            )
        ch_start, ch_end = block
        chapter_entry = {"title": f"Chapter {number}"}

    bank_i = (
        _pick_bank_node(outline, chapter_i, kind, ch_start, ch_end)
        if kind != "chapter" else None
    )
    if bank_i is not None and kind != "chapter":
        bank = outline[bank_i]
        bank_start = int(bank.get("page_start") or ch_start)
        abs_start = bank_start if ch_start <= bank_start <= ch_end else ch_start
        abs_end = ch_end
        matched_kind = kind
        fallback = False
        target_title = bank.get("title") or ""
    else:
        abs_start, abs_end = ch_start, ch_end
        matched_kind = "chapter" if kind == "chapter" else kind
        fallback = kind != "chapter"
        target_title = chapter_entry.get("title") or ""

    abs_start = max(1, min(abs_start, total_pages))
    abs_end = max(abs_start, min(abs_end, total_pages))
    printed_start = _absolute_to_printed_label(thread_key, abs_start)
    printed_end = _absolute_to_printed_label(thread_key, abs_end)
    try:
        printed_start_i = int(str(printed_start).replace("x", "0") or abs_start)
    except ValueError:
        printed_start_i = abs_start
    try:
        printed_end_i = int(str(printed_end).replace("x", "0") or abs_end)
    except ValueError:
        printed_end_i = abs_end

    is_bank = kind in {"problems", "conceptual", "objective"}
    _resolved_targets[thread_key] = {
        "page_start": abs_start,
        "page_end": abs_end,
        "section_kind": matched_kind,
        "section_title": target_title,
        "is_bank": is_bank,
        "printed_start": printed_start_i if str(printed_start).isdigit() else abs_start,
        "printed_end": printed_end_i if str(printed_end).isdigit() else abs_end,
    }

    warn = ""
    if fallback and kind != "chapter":
        warn = (
            f" NOTE: no '{kind}' bookmark; range is the chapter through the next chapter. "
            "Fast OCR skips theory pages and stops at the numbered heading."
        )

    print(
        f"[RESOLVE] ch={number} requested={kind} got={matched_kind} "
        f"abs={abs_start}-{abs_end} printed={printed_start}-{printed_end}",
        flush=True,
    )
    return (
        f"[RESOLVED] kind={matched_kind} pages={printed_start}-{printed_end} title={target_title}\n"
        f"Chapter: {chapter_entry.get('title')}\n"
        f"Use locate_marker_in_range(start_page={printed_start}, end_page={printed_end}, "
        f"marker=\"<N>\") for each requested number. Pass this range UNCHANGED."
        f"{warn}"
    )


@tool
async def read_page_range(start_page: int, end_page: int, config: RunnableConfig) -> str:
    """
    TIER 1 — STRUCTURAL NAVIGATION. Directly extracts raw text from an
    exact, already-resolved page range of the currently indexed PDF via
    PyMuPDF — NOT the vector database. Use this AFTER
    `resolve_chapter_target` or `get_document_outline` has resolved a
    chapter/section to a page range, to read its actual content (theory,
    formulas, questions). Capped at MAX_PAGE_RANGE pages per call — for a
    wider chapter, narrow via `resolve_chapter_target` or use
    `locate_marker_in_range` to jump straight to one question.

    `start_page`/`end_page` are ALWAYS PRINTED/logical page numbers — the
    same numbers the user says and `get_document_outline` displays (e.g. the
    user's literal "page 740" -> pass 740 here directly). NEVER guess or
    apply your own offset; this tool converts to the PDF's real internal
    page position internally.
    Call this AT MOST TWICE per user query (combined with
    `locate_marker_in_range`).
    """
    thread_key = _resolve_thread_key(config)
    if thread_key not in _indexed_file_paths:
        return (
            "Xəta: Bu sənəd üçün səhifə-əsaslı naviqasiya mövcud deyil (hələ indekslənməyib). "
            "Əvvəlcə 'process_and_index_documents' tool-unu çağırın."
        )

    total_pages = _thread_page_count(thread_key)
    if total_pages < 1:
        return (
            "Xəta: Bu sənəd üçün səhifə-əsaslı naviqasiya mövcud deyil "
            "(ofis faylı). Bunun əvəzinə 'search_in_document' istifadə edin."
        )
    if start_page < 1 or end_page < start_page:
        return f"Xəta: Yanlış səhifə aralığı ({start_page}-{end_page})."

    absolute_start = _printed_to_absolute_page(thread_key, start_page, total_pages)
    absolute_end = _printed_to_absolute_page(thread_key, end_page, total_pages)
    absolute_start = max(1, absolute_start)
    absolute_end = min(absolute_end, total_pages)
    if absolute_end < absolute_start:
        absolute_end = absolute_start
    if absolute_end - absolute_start + 1 > MAX_PAGE_RANGE:
        absolute_end = absolute_start + MAX_PAGE_RANGE - 1

    page_texts = _page_texts.get(thread_key) or []
    selected = page_texts[absolute_start - 1:absolute_end] if page_texts else []
    selected_text = "\n\n".join(selected).strip() if selected else ""
    printed_start = _absolute_to_printed_label(thread_key, absolute_start)
    printed_end = _absolute_to_printed_label(thread_key, absolute_end)

    if selected_text:
        return f"[Səhifə {printed_start}-{printed_end}]:\n{selected_text}"

    file_path = _indexed_file_paths.get(thread_key)
    if not file_path:
        return f"Səhifə {printed_start}-{printed_end} aralığında mətn tapılmadı."

    span = absolute_end - absolute_start + 1
    if span <= SHORT_DOC_MAX_PAGES:
        from vision_extract import vision_extract_pages, crop_diagrams_to_data_urls
        pages = list(range(absolute_start, absolute_end + 1))
        extract = await vision_extract_pages(file_path, pages, marker="worksheet")
        payload = extract.model_dump()
        payload["abs_page"] = absolute_start
        payload["file_path"] = file_path
        if extract.diagrams:
            payload["image_urls"] = crop_diagrams_to_data_urls(
                file_path, absolute_start, extract.diagrams, page_count=total_pages,
            )
        _vision_extracts[thread_key] = payload
        if extract.stem.strip():
            return f"[Səhifə {printed_start}-{printed_end}]:\n{extract.stem.strip()}"

    ocr_pages = await _ocr_page_range(thread_key, file_path, absolute_start, absolute_end)
    selected_text = "\n\n".join(ocr_pages).strip()
    if not selected_text:
        return f"Səhifə {printed_start}-{printed_end} aralığında mətn tapılmadı."
    return f"[Səhifə {printed_start}-{printed_end}]:\n{selected_text}"


def _pages_are_sparse(page_texts: list[str], abs_start: int, abs_end: int) -> bool:
    if not page_texts:
        return True
    selected = page_texts[abs_start - 1:abs_end]
    if not selected:
        return True
    chars = sum(len((p or "").strip()) for p in selected)
    return chars < 40 * max(len(selected), 1)


def _in_resolved_bank(thread_key: str, abs_start: int, abs_end: int, section_kind: str) -> tuple[bool, str]:
    kind_arg = (section_kind or "").strip().lower()
    in_bank = kind_arg in {"problems", "conceptual", "objective"}
    stored = _resolved_targets.get(thread_key) or {}
    if not in_bank and stored.get("is_bank"):
        stored_start, stored_end = stored.get("page_start"), stored.get("page_end")
        if stored_start and stored_end and abs_start >= stored_start - 1 and abs_end <= stored_end + 1:
            in_bank = True
            kind_arg = stored.get("section_kind") or kind_arg
    if kind_arg not in {"problems", "conceptual", "objective"}:
        kind_arg = "problems"
        in_bank = True
    return in_bank, kind_arg


def _find_marker_in_page_texts(
    selected_pages: list[str],
    abs_start: int,
    number: str,
    in_bank: bool,
    kind_arg: str,
) -> Optional[tuple[int, str]]:
    """Returns (absolute 1-indexed page, snippet) or None."""
    if not selected_pages:
        return None
    page_start_offsets = []
    cursor = 0
    for page_text in selected_pages:
        page_start_offsets.append(cursor)
        cursor += len(page_text) + 2
    combined_text = "\n\n".join(selected_pages)

    def _page_for_offset(offset: int) -> int:
        page_index = 0
        for i, page_offset in enumerate(page_start_offsets):
            if page_offset <= offset:
                page_index = i
            else:
                break
        return abs_start + page_index

    search_text = combined_text
    search_origin = 0
    if in_bank and kind_arg in {"problems", "conceptual", "objective"}:
        clip_start, clip_end = _clip_to_bank_heading(combined_text, kind_arg)
        search_text = combined_text[clip_start:clip_end]
        search_origin = clip_start

    keyword_pattern = _compile_numbered_marker_pattern(_QUESTION_KEYWORDS, number)
    bare_pattern = _compile_bare_number_heading(number) if in_bank else None

    def _first_heading(pattern: "re.Pattern", text: str):
        search_from = 0
        while True:
            candidate = pattern.search(text, search_from)
            if not candidate:
                return None
            if _is_heading_like(text, candidate.start()):
                return candidate
            search_from = candidate.end()

    match = _first_heading(keyword_pattern, search_text)
    if match is None and bare_pattern is not None:
        match = _first_heading(bare_pattern, search_text)
    if match is None:
        return None

    global_start = search_origin + match.start()
    global_end_of_match = search_origin + match.end()
    found_abs = _page_for_offset(global_start)

    next_pat = _ANY_NUMBERED_HEADING if in_bank else _QUESTION_PATTERN
    next_heading_start = None
    search_from = global_end_of_match
    while True:
        candidate = next_pat.search(combined_text, search_from)
        if not candidate:
            break
        if _is_heading_like(combined_text, candidate.start()):
            next_heading_start = candidate.start()
            break
        search_from = candidate.end()

    bank_end = search_origin + len(search_text)
    end_bound = next_heading_start if next_heading_start is not None else bank_end
    end_bound = min(end_bound, global_start + MARKER_SNIPPET_CHARS, bank_end)
    snippet = combined_text[global_start:end_bound].strip()
    if not snippet:
        return None
    return found_abs, snippet


async def _ocr_page_range(
    thread_key: str,
    file_path: str,
    abs_start: int,
    abs_end: int,
) -> list[str]:
    from vision_extract import FAST_OCR_MAX_PAGES, ocr_pdf_page

    cache = _ocr_page_cache.setdefault(thread_key, {})
    end = min(abs_end, abs_start + FAST_OCR_MAX_PAGES - 1)
    pages: list[str] = []
    for abs_page in range(abs_start, end + 1):
        if abs_page in cache:
            pages.append(cache[abs_page])
            continue
        try:
            text = await ocr_pdf_page(file_path, abs_page)
        except Exception as e:
            print(f"[VISION] OCR page {abs_page} failed: {e}", flush=True)
            text = ""
        cache[abs_page] = text
        pages.append(text)
    return pages


LOCATE_BLOB_MAX_CHARS = 80_000


async def _collect_span_pages(
    thread_key: str,
    file_path: str,
    page_texts: list[str],
    abs_start: int,
    abs_end: int,
    use_ocr: bool,
) -> list[str]:
    """Full resolved span (capped at FAST_OCR_MAX_PAGES). No per-page early return."""
    from vision_extract import FAST_OCR_MAX_PAGES

    end = min(abs_end, abs_start + FAST_OCR_MAX_PAGES - 1)
    if use_ocr or not page_texts:
        return await _ocr_page_range(thread_key, file_path, abs_start, end)
    selected = page_texts[abs_start - 1:end]
    if len(selected) < (end - abs_start + 1):
        # Sparse tail: OCR the missing remainder, keep existing text pages.
        ocr_pages = await _ocr_page_range(thread_key, file_path, abs_start, end)
        merged: list[str] = []
        for i, ocr in enumerate(ocr_pages):
            text = selected[i] if i < len(selected) else ""
            merged.append(text if (text or "").strip() else ocr)
        return merged
    return selected


def _build_tagged_chapter_blob(
    thread_key: str,
    pages: list[str],
    abs_start: int,
    max_chars: int = LOCATE_BLOB_MAX_CHARS,
) -> str:
    parts: list[str] = []
    used = 0
    for i, text in enumerate(pages):
        abs_page = abs_start + i
        printed = _absolute_to_printed_label(thread_key, abs_page)
        block = f"=== PAGE printed={printed} abs={abs_page} ===\n{(text or '').strip()}"
        extra = (2 if parts else 0) + len(block)
        if parts and used + extra > max_chars:
            print(
                f"[VISION] locate blob capped at {i} pages / {used} chars",
                flush=True,
            )
            break
        parts.append(block)
        used += extra
    return "\n\n".join(parts)


PIPELINE_STATUS_INDEX = "Indexing document (large textbooks may take up to 1 minute)..."
PIPELINE_STATUS_LOCATE = "Locating target question in document..."
PIPELINE_STATUS_VISION = "Page located, analyzing visual context..."
PIPELINE_STATUS_MATH = "Executing mathematical engine & formulating solution..."
PIPELINE_STATUS_RENDER = "Rendering output to whiteboard..."

_pipeline_status: dict[str, str] = {}


def set_pipeline_status(thread_key: str, status: str) -> None:
    if thread_key and status:
        _pipeline_status[thread_key] = status


def get_pipeline_status(thread_key: str) -> str:
    return _pipeline_status.get(thread_key) or ""


async def _phase2_vision_and_figures(
    thread_key: str,
    file_path: str,
    found_abs: int,
    total_pages: int,
    number: str,
    found_page: str,
    snippet: str,
) -> str:
    """N/N+1 (N-1) vision extract. Crops go to `_vision_extracts.image_urls`
    so composer can emit `canvas_op: figure`. Stem is the tool return."""
    set_pipeline_status(thread_key, PIPELINE_STATUS_VISION)
    from vision_extract import vision_extract_question, crop_diagrams_to_data_urls

    extract = await vision_extract_question(file_path, found_abs, total_pages, number)
    payload = extract.model_dump()
    payload["abs_page"] = found_abs
    payload["file_path"] = file_path
    if extract.diagrams:
        payload["image_urls"] = crop_diagrams_to_data_urls(
            file_path, found_abs, extract.diagrams, page_count=total_pages,
        )
    _vision_extracts[thread_key] = payload
    if extract.found and extract.stem.strip():
        return f"[Tapıldı - Səhifə {found_page}]:\n{extract.stem.strip()}"
    print("[VISION] Gemini stem empty — returning locator snippet", flush=True)
    if snippet:
        return f"[Tapıldı - Səhifə {found_page}]:\n{snippet}"
    return (
        f"'{number}' Səhifə {found_page} aralığında BAŞLIQ kimi tapılmadı."
    )


@tool
async def locate_marker_in_range(
    start_page: int,
    end_page: int,
    marker: str,
    section_kind: str = "problems",
    config: RunnableConfig = None,
) -> str:
    """
    TIER 1 — STRUCTURAL NAVIGATION. Scans an already-resolved page range for
    a specific NUMBERED problem heading — NOT the vector database.

    Pass the printed range from `resolve_chapter_target` UNCHANGED. That
    span is the chapter through the next chapter (not the first 1.1
    subsection). A cheap text model reads the tagged chapter OCR and
    returns the page of THIS number in the requested bank.

    `marker` can be "10", "Question 10", "Problem 10", or "Sual 10" — the
    NUMBER is what is matched.

    `section_kind` is REQUIRED. Default "problems" (Problems/Exercises/
    Məsələlər). Pass "conceptual" or "objective" ONLY when the user named
    that bank. English "Question 12" in a homework / TASK START extract
    is NOT Conceptual Questions — it is Problems unless they said so.

    Page-text regex locates the page first. Flash locate runs only when
    that match misses. Gemini Vision then extracts pages
    N and N+1 (and N-1 if the figure is behind). For `section_kind=
    "problems"` this vision pass ALWAYS runs — including born-digital
    PDFs — so diagram numerals (e.g. l = 0.350 m) enter the stem. Cropped
    figures are stored for the canvas `figure` op; this tool still
    returns stem text only.

    `start_page`/`end_page` are PRINTED/logical page numbers.
    Do NOT apply MAX_PAGE_RANGE here — the resolved bank span is already short.
    """
    blocked = _short_doc_block_message(config, "locate_marker_in_range")
    if blocked:
        return blocked

    thread_key = _resolve_thread_key(config)
    set_pipeline_status(thread_key, PIPELINE_STATUS_LOCATE)
    stored = _vision_extracts.get(thread_key)
    if isinstance(stored, dict):
        stored.pop("image_urls", None)
    if thread_key not in _indexed_file_paths:
        return (
            "Xəta: Bu sənəd üçün səhifə-əsaslı naviqasiya mövcud deyil (hələ indekslənməyib). "
            "Əvvəlcə 'process_and_index_documents' tool-unu çağırın."
        )

    total_pages = _thread_page_count(thread_key)
    if total_pages < 1:
        return (
            "Xəta: Bu sənəd üçün səhifə-əsaslı naviqasiya mövcud deyil "
            "(ofis faylı). Bunun əvəzinə 'search_in_document' istifadə edin."
        )

    printed_start_requested, printed_end_requested = start_page, end_page
    abs_start = _printed_to_absolute_page(thread_key, start_page, total_pages)
    abs_end = _printed_to_absolute_page(thread_key, end_page, total_pages)
    abs_start = max(1, abs_start)
    abs_end = min(abs_end, total_pages)
    if abs_end < abs_start:
        return f"Xəta: Yanlış səhifə aralığı ({printed_start_requested}-{printed_end_requested})."

    number = _extract_marker_number(marker)
    if number is None:
        return f"Xəta: '{marker}' daxilində heç bir nömrə tapılmadı (məs. 'Question 8')."

    _, kind_arg = _in_resolved_bank(thread_key, abs_start, abs_end, section_kind)
    page_texts = _page_texts.get(thread_key) or []
    file_path = _indexed_file_paths[thread_key]
    use_ocr = bool(
        _scan_mode.get(thread_key) or _pages_are_sparse(page_texts, abs_start, abs_end)
    )

    selected_pages = await _collect_span_pages(
        thread_key, file_path, page_texts, abs_start, abs_end, use_ocr,
    )
    tagged = _build_tagged_chapter_blob(thread_key, selected_pages, abs_start)
    span_end = abs_start + len(selected_pages) - 1 if selected_pages else abs_end

    hit = _find_marker_in_page_texts(
        selected_pages, abs_start, number, True, kind_arg,
    )
    if hit:
        print(
            f"[VISION] regex hit marker={number} kind={kind_arg} "
            f"abs_page={hit[0]}",
            flush=True,
        )
    elif tagged.strip():
        from vision_extract import llm_locate_page
        locate = await llm_locate_page(
            tagged, number, kind_arg, abs_start, span_end,
        )
        if locate.found:
            idx = locate.abs_page - abs_start
            page_text = selected_pages[idx] if 0 <= idx < len(selected_pages) else ""
            clipped = _find_marker_in_page_texts(
                [page_text], locate.abs_page, number, True, kind_arg,
            )
            snippet = clipped[1] if clipped else page_text
            hit = (locate.abs_page, snippet)
            print(
                f"[VISION] Flash locate hit marker={number} kind={kind_arg} "
                f"abs_page={locate.abs_page}",
                flush=True,
            )

    if hit is None:
        return (
            f"'{marker}' Səhifə {printed_start_requested}-{printed_end_requested} aralığında BAŞLIQ kimi "
            "tapılmadı. Bu sual həmin fəsildə/aralıqda mövcud olmaya bilər, ya da fərqli səhifədədir."
        )

    found_abs, snippet = hit
    found_page = _absolute_to_printed_label(thread_key, found_abs)
    run_vision = kind_arg == "problems" or use_ocr
    if run_vision:
        return await _phase2_vision_and_figures(
            thread_key, file_path, found_abs, total_pages, number,
            found_page, snippet,
        )

    if not snippet:
        return (
            f"'{marker}' Səhifə {printed_start_requested}-{printed_end_requested} "
            "aralığında tapılmadı."
        )
    return f"[Tapıldı - Səhifə {found_page}]:\n{snippet}"


_EMBED_TIMEOUT_MESSAGE = (
    "Xəta: Sənədin indekslənməsi vaxt limitini keçdi (OCR, mətn çıxarışı və ya "
    "embedding mərhələsi cavab vermədi). Zəhmət olmasa sənədi yenidən yükləyin "
    "və ya daha kiçik bir fayl istifadə edin."
)


_EMBED_STALE_MESSAGE = (
    "Xəta: Sənəd indekslənərkən dəyişdirildi. Zəhmət olmasa sualı yenidən verin."
)


def _embed_job_is_current(thread_key: str, generation: int, user_id: str) -> bool:
    auth = _thread_auth.get(thread_key)
    return (
        _doc_generation.get(thread_key, 0) == generation
        and bool(auth)
        and auth[0] == user_id
    )


def _guarded_replace_chunks(
    thread_key: str,
    stored_path: str,
    chunks: list[str],
    vectors: list[list[float]],
    generation: int,
    user_id: str,
) -> bool:
    """Runs on a worker thread. Replaces this thread's pgvector rows only if the
    document and user are still the ones this job was started for. The check
    happens INSIDE the write lock, so a stale writer can never land after a
    newer upload's delete."""
    with _vector_write_lock(thread_key):
        if not _embed_job_is_current(thread_key, generation, user_id):
            return False
        insert_document_chunks(thread_key, stored_path, chunks, vectors)
        return True


def _get_embed_gate() -> asyncio.Semaphore:
    global _embed_gate
    if _embed_gate is None:
        _embed_gate = asyncio.Semaphore(1)
    return _embed_gate


async def _run_embedding(
    thread_key: str,
    chunks: list[str],
    stored_path: str,
    generation: int,
    user_id: str,
) -> str:
    """The single embedding code path. Returns "ok" or "stale". Embeds in small
    batches so a cancel or a document change takes effect within one batch, and
    writes to pgvector only after a final currency check (inside the write lock)."""
    async with _get_embed_gate():
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            if not _embed_job_is_current(thread_key, generation, user_id):
                print(f"[EMBED] stale before batch {start} thread={thread_key}", flush=True)
                return "stale"
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            vectors.extend(await asyncio.to_thread(embeddings.embed_documents, batch))
        insert_batches = max(1, (len(chunks) + 39) // 40)
        written = await asyncio.wait_for(
            asyncio.to_thread(
                _guarded_replace_chunks,
                thread_key, stored_path, chunks, vectors, generation, user_id,
            ),
            timeout=45 * insert_batches,
        )
        return "ok" if written else "stale"


def _embed_inputs(thread_key: str, config: RunnableConfig) -> Optional[tuple]:
    """(chunks, stored_path, generation, user_id) or None when there is nothing to embed."""
    if _embeddings_ready.get(thread_key) or _scan_mode.get(thread_key):
        return None
    chunks = _indexed_chunks.get(thread_key) or []
    auth = _thread_auth.get(thread_key)
    if not chunks or not auth:
        return None
    file_path = _indexed_file_paths.get(thread_key) or ""
    stored_path = str((config.get("configurable") or {}).get("document_path") or file_path)
    return chunks, stored_path, _doc_generation.get(thread_key, 0), auth[0]


async def _embed_job_main(
    thread_key: str,
    chunks: list[str],
    stored_path: str,
    generation: int,
    user_id: str,
) -> str:
    t0 = time.time()
    try:
        outcome = await asyncio.wait_for(
            _run_embedding(thread_key, chunks, stored_path, generation, user_id),
            timeout=EMBED_JOB_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        outcome = "timeout"
    if outcome == "ok" and _embed_job_is_current(thread_key, generation, user_id):
        _embeddings_ready[thread_key] = True
    print(
        f"[EMBED] warm-up {outcome} thread={thread_key} chunks={len(chunks)} "
        f"took={time.time() - t0:.1f}s",
        flush=True,
    )
    return outcome


def _on_embed_job_done(thread_key: str, job: _EmbedJob, task: "asyncio.Task") -> None:
    """Done-callback: drop the registry entry (only if it is still this job) and
    retrieve any exception so it is never reported as 'never retrieved'."""
    if _embed_jobs.get(thread_key) is job:
        _embed_jobs.pop(thread_key, None)
    if task.cancelled():
        print(f"[EMBED] warm-up cancelled thread={thread_key}", flush=True)
        return
    exc = task.exception()
    if exc is not None:
        print(f"[EMBED] warm-up failed thread={thread_key}: {exc!r}", flush=True)


def _cancel_embed_job(thread_key: str, owner: Optional[str] = None) -> None:
    """Cancel this thread's warm-up. Safe from any thread and from sync code.
    With `owner`, only cancels a job started by that request."""
    job = _embed_jobs.get(thread_key)
    if job is None or job.task.done():
        return
    if owner is not None and job.owner_request_id != owner:
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is job.loop:
        job.task.cancel()
    else:
        job.loop.call_soon_threadsafe(job.task.cancel)


def cancel_embed_warmup(thread_key: str, owner: str) -> None:
    """Called when a request ends abnormally (client disconnect)."""
    _cancel_embed_job(thread_key, owner=owner or None)


async def cancel_all_embed_jobs(timeout: float = 5.0) -> None:
    """Server shutdown: cancel every warm-up and wait briefly for them to finish."""
    tasks = [job.task for job in list(_embed_jobs.values()) if not job.task.done()]
    for thread_key in list(_embed_jobs):
        _cancel_embed_job(thread_key)
    if tasks:
        await asyncio.wait(tasks, timeout=timeout)


def start_embed_warmup(thread_key: str, config: RunnableConfig) -> bool:
    """Start (once per document) a background embed for a prepared digital book.
    Never touches the pipeline status, so it cannot show a banner on another turn."""
    inputs = _embed_inputs(thread_key, config)
    if inputs is None:
        return False
    chunks, stored_path, generation, user_id = inputs
    existing = _embed_jobs.get(thread_key)
    if existing and not existing.task.done() and existing.generation == generation:
        return True
    _cancel_embed_job(thread_key)
    loop = asyncio.get_running_loop()
    owner = str((config.get("configurable") or {}).get("request_id") or "")
    task = loop.create_task(
        _embed_job_main(thread_key, chunks, stored_path, generation, user_id),
        name=f"embed-warmup:{thread_key}",
    )
    job = _EmbedJob(task, loop, generation, owner, user_id)
    _embed_jobs[thread_key] = job
    task.add_done_callback(lambda t, key=thread_key, j=job: _on_embed_job_done(key, j, t))
    print(f"[EMBED] warm-up started thread={thread_key} chunks={len(chunks)}", flush=True)
    return True


async def _embed_stored_chunks(thread_key: str, config: RunnableConfig) -> Optional[str]:
    """Make sure vectors exist before a search. None means search can run. A string is the student-facing error."""
    inputs = _embed_inputs(thread_key, config)
    if inputs is None:
        return None
    set_pipeline_status(thread_key, PIPELINE_STATUS_INDEX)
    job = _embed_jobs.get(thread_key)
    if job is not None and not job.task.done():
        # `asyncio.wait` (not `await task`): if THIS request is cancelled, the
        # warm-up it joined is left running.
        await asyncio.wait({job.task}, timeout=EMBED_JOB_BUDGET_SECONDS)
        if _embeddings_ready.get(thread_key):
            return None
        inputs = _embed_inputs(thread_key, config)
        if inputs is None:
            return None
    chunks, stored_path, generation, user_id = inputs
    try:
        outcome = await asyncio.wait_for(
            _run_embedding(thread_key, chunks, stored_path, generation, user_id),
            timeout=EMBED_JOB_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        return _EMBED_TIMEOUT_MESSAGE
    if outcome != "ok":
        return _EMBED_STALE_MESSAGE
    if _embed_job_is_current(thread_key, generation, user_id):
        _embeddings_ready[thread_key] = True
    return None


@tool
async def search_in_document(query: str, config: RunnableConfig) -> str:
    """
    TIER 2 — SEMANTIC FALLBACK. Searches the current conversation thread's
    vector database for the most relevant chunks matching the query. Use
    this ONLY when Tier 1 (`resolve_chapter_target`/`get_document_outline`/
    `read_page_range`/`locate_marker_in_range`) cannot resolve a structural
    reference — no
    chapter map was found, or the question is genuinely topical/fuzzy with
    no chapter reference at all. NOTE: this only searches the first
    MAX_INDEXED_CHUNKS chunks of the document — later chapters may be
    missing here even though they remain reachable via Tier 1.
    """
    blocked = _short_doc_block_message(config, "search_in_document")
    if blocked:
        return blocked

    thread_key = _resolve_thread_key(config)
    configurable = config.get("configurable") or {}
    remember_thread_auth(
        thread_key,
        str(configurable.get("user_id") or ""),
        str(configurable.get("access_token") or ""),
    )
    if _scan_mode.get(thread_key):
        return (
            "Xəta: Skan edilmiş dərslikdə semantik indeks yoxdur. "
            "Nömrələnmiş məsələ üçün 'resolve_chapter_target' + 'locate_marker_in_range' çağır. "
            "search_in_document İSTİFADƏ ETMƏ."
        )
    print("[TIMING] Tool Doc Search Started...", flush=True)
    t0 = time.time()
    try:
        embed_error = await _embed_stored_chunks(thread_key, config)
        if embed_error:
            return embed_error
        chunks = await asyncio.to_thread(search_document_chunks, thread_key, query, limit=5)
        if not chunks:
            return "Xəta: Hələ heç bir fayl indekslənməyib. Əvvəlcə 'process_and_index_documents' tool-unu çağırın."
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        return f"Axtarış zamanı xəta baş verdi: {str(e)}"
    finally:
        print(f"[TIMING] Tool Doc Search Ended - Took {time.time() - t0:.2f}s", flush=True)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
async def web_search_tool(query: str) -> str:
    """
    Executes live web searches for academic literature, recent publications, and real-time news.
    Returns only the Title, Direct URL, and a trimmed Snippet for each result to prevent
    the calling model from hallucinating or constructing generic/fabricated URLs.

    CRITICAL FACTUALITY RULE: The caller MUST present ONLY the real, verified facts contained
    in this payload — never speculative or future-tense phrasing (e.g. "olması planlaşdırılan",
    "nəşr ediləcək", "gələcəkdə gözlənilir"). Use the exact article titles and Direct URLs
    returned here as-is; do not invent publishing schedules or claims not present in the results.
    """
    print("[TIMING] Tool Tavily Started...", flush=True)
    t0 = time.time()
    try:
        # The Tavily SDK's `search` call is blocking network I/O; offload it to
        # a worker thread so it never stalls the asyncio event loop.
        response = await asyncio.to_thread(
            tavily_client.search, query=query, max_results=3, search_depth="basic"
        )
        results = response.get("results", [])

        if not results:
            return "Axtarış üzrə heç bir nəticə tapılmadı."

        formatted_results = []
        for item in results:
            direct_url = item.get("url", "").strip()
            if not direct_url:
                continue

            snippet = (item.get("content") or "").strip()[:250]

            formatted_results.append(
                f"Title: {item.get('title')}\n"
                f"Direct URL: {direct_url}\n"
                f"Snippet: {snippet}\n"
            )

        if not formatted_results:
            return "Axtarış üzrə heç bir nəticə tapılmadı."

        return "\n---\n".join(formatted_results)
    except Exception as e:
        print(f"[TAVILY SEARCH ERROR]: {str(e)}")
        return f"Axtarış xətası baş verdi: {str(e)}"
    finally:
        print(f"[TIMING] Tool Tavily Ended - Took {time.time() - t0:.2f}s", flush=True)

_SANDBOX_MODULES = {
    "math": math,
    "statistics": statistics,
    "random": random,
    "json": json,
}

def _sandboxed_import(name, *args, **kwargs):
    if name not in _SANDBOX_MODULES:
        raise ImportError(f"'{name}' module-una sandbox daxilində icazə verilmir.")
    return _SANDBOX_MODULES[name]

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "len": len, "list": list, "map": map, "max": max, "min": min,
    "pow": pow, "print": print, "range": range, "reversed": reversed,
    "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip, "__import__": _sandboxed_import,
}

# ---------------------------------------------------------------------------
# KaTeX raw-LaTeX-leak safety net. `MATH_WORKER_SYSTEM_PROMPT` instructs the
# LLM to always wrap LaTeX in `$...$`/`$$...$$` delimiters, but prompt
# adherence is never 100% guaranteed — an occasional slip leaves a raw
# command like `\text{N}\cdot\text{m}^2/\text{C}` sitting undelimited in the
# `summary`/`steps`/`value` strings this tool hands back to `math_worker`,
# which KaTeX then can't touch, so it renders as literal backslash-and-brace
# text on the Desk. This is a last-resort backend catch, not the primary
# fix — it deliberately only sweeps in tokens directly adjacent (single
# space or none) to a real `\command`, so it never risks swallowing
# surrounding prose.
# ---------------------------------------------------------------------------
_DELIMITED_MATH_RE = re.compile(r"\$\$[\s\S]*?\$\$|\$[^$\n]*?\$")
_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")
_LATEX_GROUP_RE = re.compile(r"^(\{[^{}]*\}|\[[^[\]]*\])")
_LATEX_LEFT_TIGHT = frozenset("0123456789)}]^_=+-*/")
_LATEX_RIGHT_TIGHT = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789^_+-*/=(){}[]\\"
)
_SCI_E_RE = re.compile(r"(?<![A-Za-z\\])(\d+\.?\d*)[eE]([+-]?\d+)")
_SUB_IDENT_RE = re.compile(r"(?<![\\$A-Za-z0-9])([A-Za-z])_([A-Za-z][A-Za-z0-9]*)\b")
_LEADING_MATH_LABEL_RE = re.compile(
    r"^((?:Sual|Question|Q|Step)\s*\d+\s*:\s*)",
    re.IGNORECASE,
)


def _python_e_to_latex(text: str) -> str:
    def _repl(match: "re.Match") -> str:
        coeff, exp = match.group(1), match.group(2)
        try:
            exp_i = int(exp)
        except ValueError:
            return match.group(0)
        return f"{coeff} \\times 10^{{{exp_i}}}"
    return _SCI_E_RE.sub(_repl, text)


_PROSE_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def _looks_like_prose(text: str) -> bool:
    """True when a string is a sentence, not an isolated equation.

    Whole-string `$...$` wrapping of prose (especially Azerbaijani/Unicode
    words) is what MathJax reports as a yellow "Math input error".
    """
    return len(_PROSE_WORD_RE.findall(text)) >= 3


def _texify_plain_segment(segment: str) -> str:
    if not segment:
        return segment
    converted = _python_e_to_latex(segment)
    converted = re.sub(
        r"(?<!\$)(\d+\.?\d*)\s*\\times\s*10\^\{([^{}]+)\}",
        r"$\1 \\times 10^{\2}$",
        converted,
    )
    converted = _SUB_IDENT_RE.sub(r"$\1_{\2}$", converted)
    if "$" in converted:
        return converted
    if not re.search(r"[=]|\\times", converted) or not re.search(r"\d", converted):
        return converted
    if _looks_like_prose(converted):
        return converted
    label = _LEADING_MATH_LABEL_RE.match(converted)
    if label:
        rest = converted[label.end():].strip()
        if rest and _looks_like_prose(rest):
            return converted
        return f"{label.group(1)}${rest}$" if rest else converted
    stripped = converted.strip()
    prefix_len = converted.find(stripped)
    suffix = converted[prefix_len + len(stripped):]
    return f"{converted[:prefix_len]}${stripped}${suffix}"


def _texify_desk_math(text: Optional[str]) -> str:
    """Convert Python-ish desk math (`1.07e4`, bare `Q_h`) into `$...$` TeX.
    Idempotent when the string is already delimited."""
    if not text:
        return ""
    parts: list[str] = []
    cursor = 0
    for span in _DELIMITED_MATH_RE.finditer(text):
        parts.append(_texify_plain_segment(text[cursor:span.start()]))
        parts.append(_python_e_to_latex(span.group(0)))
        cursor = span.end()
    parts.append(_texify_plain_segment(text[cursor:]))
    return "".join(parts)


def _float_to_latex_math(value: float) -> str:
    if value == 0.0:
        return "$0$"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        exp = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10 ** exp)
        return f"${mantissa:.4g} \\times 10^{{{exp}}}$"
    return f"${value:g}$"


def _extend_latex_run(text: str, cmd_start: int, cmd_end: int) -> tuple[int, int]:
    """Keep `{...}` / `[...]` groups intact (so `\\text{ m/s}` keeps its space)
    and glue only immediately-adjacent numbers/symbols — never prose."""
    start = cmd_start
    while start > 0 and text[start - 1] in _LATEX_LEFT_TIGHT:
        start -= 1
    i = cmd_end
    n = len(text)
    while i < n:
        group = _LATEX_GROUP_RE.match(text[i:])
        if group:
            i += len(group.group(0))
            continue
        ch = text[i]
        if ch in ".," and i + 1 < n and text[i + 1].isdigit():
            i += 1
            continue
        if ch in _LATEX_RIGHT_TIGHT:
            i += 1
            continue
        break
    return start, i


def _wrap_stray_commands_in_plain_text(segment: str) -> str:
    """Finds real `\\command` tokens in an ALREADY-undelimited slice of text
    (never called on the inside of an existing `$...$` span — see
    `_auto_wrap_stray_latex` below) and wraps each one, together with any
    directly-adjacent math-symbol/number tokens, in `$...$`."""
    if "\\" not in segment:
        return segment

    out: list[str] = []
    cursor = 0
    for match in _LATEX_COMMAND_RE.finditer(segment):
        if match.start() < cursor:
            continue
        start, end = _extend_latex_run(segment, match.start(), match.end())
        out.append(segment[cursor:start])
        out.append(f"${segment[start:end]}$")
        cursor = end
    out.append(segment[cursor:])
    return "".join(out)


def _normalize_math_delimiters(text: str) -> str:
    text = text.replace(r"\(\)", "").replace(r"\[\]", "")
    text = re.sub(r"\\\[(.*?)\\\]", lambda match: f"$${match.group(1)}$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", lambda match: f"${match.group(1)}$", text, flags=re.DOTALL)
    return text


def _auto_wrap_stray_latex(text: Optional[str]) -> Optional[str]:
    """Public entry point: wraps stray, undelimited LaTeX commands anywhere
    in `text` in `$...$`, while leaving content already inside an existing
    `$...$`/`$$...$$` span completely untouched (so correctly-delimited math
    is never double-wrapped or mangled)."""
    if not text:
        return text
    text = _normalize_math_delimiters(text)
    if "\\" not in text:
        return text

    result_parts = []
    cursor = 0
    for span in _DELIMITED_MATH_RE.finditer(text):
        result_parts.append(_wrap_stray_commands_in_plain_text(text[cursor:span.start()]))
        result_parts.append(span.group(0))
        cursor = span.end()
    result_parts.append(_wrap_stray_commands_in_plain_text(text[cursor:]))
    return "".join(result_parts)


def _format_calculation_value(value) -> str:
    """Desk `value` MUST be a primitive string. Nested dicts stringify as
    `[object Object]` in React; tiny floats also sometimes arrive wrapped
    in an extra object. Flatten those here before JSON serialization."""
    if isinstance(value, dict):
        for key in ("value", "result", "text", "display", "formatted"):
            inner = value.get(key)
            if inner is not None and not isinstance(inner, (dict, list)):
                return _format_calculation_value(inner)
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"${value}$"
    if isinstance(value, float):
        return _float_to_latex_math(value)
    if value is None:
        return ""
    return str(value)


def _sanitize_calculation_latex(result: dict) -> dict:
    """Applies desk-math texify then `_auto_wrap_stray_latex` to every
    user-facing string field of a `python_code_executor` calculation payload."""
    result["value"] = _auto_wrap_stray_latex(
        _texify_desk_math(_format_calculation_value(result.get("value")))
    ) or ""
    if isinstance(result.get("summary"), str):
        result["summary"] = _auto_wrap_stray_latex(_texify_desk_math(result["summary"]))
    steps = result.get("steps")
    if isinstance(steps, list):
        result["steps"] = [
            _auto_wrap_stray_latex(_texify_desk_math(s)) if isinstance(s, str) else s
            for s in steps
        ]
    return result


# Physics/math line-art diagrams. The sandbox computes coordinates with
# `math`; the Next.js Desk (`DiagramCard`) draws SVG. Same split as charts:
# backend never emits SVG markup, only primitives.
_DIAGRAM_ELEMENT_TYPES = frozenset({"rect", "line", "vector", "circle", "arc", "text"})
_MAX_DIAGRAM_ELEMENTS = 64
_MAX_DIAGRAM_COORD = 1200.0


def _as_finite_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or abs(number) == float("inf"):
        return default
    return max(-_MAX_DIAGRAM_COORD, min(_MAX_DIAGRAM_COORD, number))


def _sanitize_diagram_element(raw) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    etype = str(raw.get("type") or "").strip().lower()
    if etype not in _DIAGRAM_ELEMENT_TYPES:
        return None
    element: dict = {"type": etype}
    for key in ("x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "width", "height", "start_deg", "end_deg"):
        if key in raw and raw[key] is not None:
            element[key] = _as_finite_float(raw[key])
    if etype == "text":
        element["text"] = str(raw.get("text") or raw.get("label") or "")[:80]
    elif raw.get("label") is not None:
        element["label"] = str(raw.get("label"))[:40]
    if raw.get("style") in ("solid", "dashed"):
        element["style"] = raw["style"]
    return element


def _sanitize_diagram_payload(result: dict) -> dict:
    """Clamps coordinates, drops unknown primitives, and bounds canvas size
    so a malformed `result` cannot blow up the Desk SVG renderer."""
    width = _as_finite_float(result.get("width"), 400.0) or 400.0
    height = _as_finite_float(result.get("height"), 320.0) or 320.0
    result["type"] = "diagram"
    result["width"] = max(160.0, min(800.0, width))
    result["height"] = max(120.0, min(800.0, height))
    result["title"] = str(result.get("title") or "Diagram")[:120]
    if isinstance(result.get("summary"), str):
        result["summary"] = _auto_wrap_stray_latex(result["summary"])
    raw_elements = result.get("elements") if isinstance(result.get("elements"), list) else []
    cleaned = []
    for item in raw_elements[:_MAX_DIAGRAM_ELEMENTS]:
        element = _sanitize_diagram_element(item)
        if element:
            cleaned.append(element)
    result["elements"] = cleaned
    return result


def _execute_sandboxed_code(code: str) -> str:
    """
    Synchronous sandbox worker for `python_code_executor`. Runs on a worker
    thread (via `asyncio.to_thread`) since `exec()` is a blocking, potentially
    CPU-bound call that must never run directly on the event loop.
    """
    stdout_capture = io.StringIO()
    exec_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
        "statistics": statistics,
        "random": random,
        "json": json,
    }
    exec_locals = {}

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, exec_globals, exec_locals)
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Kod icra edilərkən xəta baş verdi: {str(e)}"
        }, ensure_ascii=False)

    result = exec_locals.get("result", exec_globals.get("result"))

    if result is not None:
        try:
            if isinstance(result, dict) and result.get("type") in ("calculation", "explanation"):
                result = _sanitize_calculation_latex(result)
            elif isinstance(result, dict) and result.get("type") == "diagram":
                result = _sanitize_diagram_payload(result)
            return json.dumps(result, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            return json.dumps({
                "type": "error",
                "message": f"'result' dəyişəni JSON formatına çevrilə bilmədi: {str(e)}"
            }, ensure_ascii=False)

    printed_output = stdout_capture.getvalue().strip()
    if printed_output:
        return printed_output

    return json.dumps({
        "type": "error",
        "message": "Kod icra olundu, lakin 'result' dəyişəni təyin edilmədi və heç bir çıxış yazılmadı."
    }, ensure_ascii=False)

@tool
async def python_code_executor(code: str) -> str:
    """
    Safely executes short Python snippets for mathematical modeling, data
    calculations, statistical analysis, chart generation, and physics/math
    line-art diagrams for the central Desk workspace widgets (Next.js frontend).

    The executed code runs in a restricted sandbox (limited builtins, only
    `math`, `statistics`, `random`, and `json` importable). `sys.stdout` is
    captured, so `print()` output is returned if no `result` variable is set.

    PROTOCOL: The code MUST assign its final output to a local variable named
    `result`, following one of these JSON-serializable shapes:

    1) Chart rendering (bar/line/pie) for the Desk graph widget:
       result = {
           "type": "chart",
           "chart_type": "line" | "bar" | "pie",
           "title": "Torque vs. Angular Acceleration",
           "labels": ["0.1", "0.2", "0.3", "0.4"],
           "values": [12, 24, 36, 48]
       }

    2) Mathematical/statistical computations:
       Compute EVERY numeric result in Python first (`math.sqrt`, arithmetic,
       etc.). Then assign `result` with keys in this EXACT order — `steps`
       before `value` and `summary` — and build those two strings from the
       SAME Python variables (f-strings). Never type a second guessed literal
       that disagrees with the last step. ALGEBRA-CODE IDENTITY: the Python
       that produces the number MUST evaluate the same expression written in
       the `$$...$$` line (e.g. `math.sqrt(k * x_i**2 / m)` or
       `math.sqrt(2 * E_i / m)` for $v=\\sqrt{k x_i^2/m}$ — NEVER
       `math.sqrt(E_i / m)` for that claim).
       v_p = math.sqrt(2.586**2 + 3.800**2)
       result = {
           "type": "calculation",
           "steps": [
               "The passenger's **speed** is the magnitude of the resultant.\\n\\n$$v = \\sqrt{(2.586)^2 + (3.800)^2} = 4.60\\,\\mathrm{m/s}$$",
           ],
           "value": f"$v = {v_p:.2f}\\,\\mathrm{{m/s}}$",
           "summary": f"The passenger's speed relative to the shore is about {v_p:.2f} m/s, found by adding the three velocity vectors.",
       }
       CRITICAL: `value` MUST be a formatted LaTeX STRING wrapped in `$...$`
       (e.g. "$4.93 \\times 10^{-8}\\,\\mathrm{s}$" or "$e = 25.0\\%$"),
       NEVER a nested dict/object, NEVER a raw Python float, and NEVER Python
       scientific notation like "4.93e-8" or "1.07e4". Subscripts must be math
       mode (`$Q_h$`), not raw `Q_h`. Extreme magnitudes MUST use
       `\\times 10^{n}` inside `$...$`.
       NEVER wrap prose or full sentences in `$` delimiters in `summary` or
       `steps`. Use `$ ... $` strictly for equations and isolated variables.
       Keep text entirely outside the delimiters. If units or short words must
       be inside a formula, you MUST use `\\text{}` (e.g. `$v = 5.0 \\text{ m/s}$`).
       The optional `steps` list holds the full step-by-step derivation — each
       string becomes one `canvas_op: step` card. PEDAGOGICAL DEPTH: never skip
       intermediate algebra; every item is one JSON string with teaching prose
       then `$...$` / `$$...$$` math (a FULL pedagogical beat: what, why, then
       the algebra — not a one-line formula). One array item = one logical
       phase; do not split prose and related math (e.g. both x and y
       components) across multiple items. Why/How about a highlighted canvas
       step belongs in `steps`, not in chat. Call this tool ONCE with the
       complete `steps` list — never once per derivation step. Callers should
       put derivation detail there instead of repeating it in chat text.
       Anchoring to a question shape is owned by the frontend
       (`canvas_anchor_id`), not this tool.

    3) Socratic success — formalize EXACTLY ONE completed milestone.
       `steps` MUST be a one-element array. Call this ONLY after the student
       answered the current step correctly; never for hints.
       Intermediate (not the last unknown) — omit `value`; `summary` is
       invalid unless its last non-whitespace character is `?` (or `？`).
       Praise may come first; the last sentence MUST be the next
       operational question:
       result = {
           "type": "calculation",
           "summary": "Yes — $v_x = 50$. Which kinematic equation gives $v_y$ next?",
           "steps": [
               "The $x$-component of velocity is constant.\\n\\n$$v_x = v \\cos\\theta = 50\\,\\mathrm{m/s}$$",
           ],
       }
       Final unknown — you MUST still call this tool (not a chat-only
       wrap-up). Set `value` and open `summary` with Result found:
       result = {
           "type": "calculation",
           "value": "$v = 50\\,\\mathrm{m/s}$",
           "summary": "Result found: $v = 50\\,\\mathrm{m/s}$.",
           "steps": [
               "The speed is the magnitude of the resultant.\\n\\n$$v = \\sqrt{v_x^2 + v_y^2} = 50\\,\\mathrm{m/s}$$",
           ],
       }
       Call this ONCE per successful step. Never follow it with a second tool call.
       Goal bookkeeping (Socratic only, optional keys): `"complete_goal"` is
       true / false / an integer N (close the first N open goals).
       `"completed_goal_indices"` is a list of 0-based indices into the open
       goal list shown in [SOCRATIC GOAL STATE] (e.g. [1]) to close a goal the
       student answered out of order; indices refer to the list as shown this
       turn and may be combined with `complete_goal`.
       STEP CONTRACT (every Socratic `steps` string, whatever solution path the
       student took — substitution, subtraction, elimination — record THEIR
       equations in this same shape, never invent another path): one neutral
       sentence naming what is now established (never "the student", never
       ending in `:` or `,`), a blank line, then one `$$...$$` equation chain.
       Every number carries its unit inside the math with `\\mathrm{...}` or
       `\\text{...}` (units follow from the givens by dimensional analysis,
       e.g. $a = \\frac{92}{98} = 0.94\\,\\mathrm{m/s^2}$). A step that only
       repeats the previous card must use `"replace_last": true`.
       `"type": "explanation"` is for `[CANVAS EXPLAIN]` Why/How laterals only.

    4) Physics/math diagram (free-body, field vectors, geometry). Coordinates
       are SVG pixels (origin TOP-LEFT, +x right, +y down). Compute endpoints
       with `math` — never emit SVG markup. Allowed primitives: rect, line,
       vector, circle, arc, text.
       result = {
           "type": "diagram",
           "title": "Free-body diagram",
           "width": 400,
           "height": 300,
           "summary": "N upward, mg downward.",
           "elements": [
               {"type": "rect", "x": 170, "y": 130, "width": 60, "height": 40, "label": "m"},
               {"type": "line", "x1": 80, "y1": 170, "x2": 320, "y2": 170},
               {"type": "vector", "x1": 200, "y1": 130, "x2": 200, "y2": 50, "label": "N"},
               {"type": "circle", "cx": 80, "cy": 80, "r": 14, "label": "+q"},
               {"type": "arc", "cx": 200, "cy": 160, "r": 40, "start_deg": 0, "end_deg": 35},
               {"type": "text", "x": 210, "y": 44, "text": "N"},
           ]
       }
       Vector/arc angles: 0° along +x (right), counterclockwise in the
       mathematical sense. Convert to SVG with
       x2 = x1 + L*math.cos(math.radians(theta));
       y2 = y1 - L*math.sin(math.radians(theta)).

    Returns the JSON-serialized `result` dict. If `result` is not set, falls
    back to the raw captured stdout text. If neither is present, returns a
    JSON error payload.
    """
    print("[TIMING] Tool Code Executor Started...", flush=True)
    t0 = time.time()
    try:
        return await asyncio.to_thread(_execute_sandboxed_code, code)
    finally:
        print(f"[TIMING] Tool Code Executor Ended - Took {time.time() - t0:.2f}s", flush=True)


# ---------------------------------------------------------------------------
# Persistent pedagogical struggle memory — a SECOND on-disk Chroma collection,
# not `_vector_stores` (those are ephemeral per-thread document chunks).
# Lives next to the LangGraph checkpoint dir under `%LOCALAPPDATA%/LockNLearn`.
# ---------------------------------------------------------------------------
_STRUGGLE_COLLECTION = "struggle_memory"
_struggle_store: Optional[Chroma] = None


def _struggle_persist_dir() -> str:
    local_app = os.environ.get("LOCALAPPDATA") or os.environ.get("TMP") or os.path.join(
        os.path.expanduser("~"), ".locknlearn"
    )
    persist_dir = os.path.join(local_app, "LockNLearn", "struggle_memory")
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


def _get_struggle_store() -> Chroma:
    global _struggle_store
    if _struggle_store is None:
        persist_dir = _struggle_persist_dir()
        _struggle_store = Chroma(
            collection_name=_STRUGGLE_COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        print(f"[STRUGGLE] persist_directory={persist_dir}", flush=True)
    return _struggle_store


def index_struggle_memory(
    thread_id: str,
    question_id: str,
    topic: str,
    excerpt: str,
    formula: Optional[str] = None,
) -> None:
    """Writes one Needs Review record into the on-disk struggle collection."""
    store = _get_struggle_store()
    parts = [f"Topic: {topic}".strip(), (excerpt or "").strip()]
    if formula:
        parts.append(f"Formula: {formula}")
    text = "\n".join(part for part in parts if part)[:4000]
    if not text.strip():
        return
    uid = f"{thread_id}:{question_id}:{int(time.time() * 1000)}"
    store.add_texts(
        texts=[text],
        metadatas=[{
            "thread_id": thread_id or "",
            "question_id": question_id or "",
            "topic": (topic or "")[:200],
        }],
        ids=[uid],
    )
    print(f"[STRUGGLE] indexed question_id={question_id} topic={topic!r}", flush=True)


def recall_struggle_context(thread_id: str, user_prompt: str) -> str:
    """Returns a short bullet string of prior struggles related to this turn.

    Searches globally (not only this thread) so a later session still gets
    scaffolding on concepts the student marked Needs Review earlier.
    `thread_id` is kept for logging / future per-user filters.
    """
    query = (user_prompt or "").strip() or "study difficulty"
    try:
        store = _get_struggle_store()
        docs = store.similarity_search(query, k=4)
    except Exception as e:
        print(f"[STRUGGLE] recall failed thread={thread_id}: {e}", flush=True)
        return ""
    if not docs:
        return ""
    bullets = []
    for doc in docs:
        meta = doc.metadata or {}
        topic = meta.get("topic") or "concept"
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        bullets.append(f"- {topic}: {snippet}")
    return "The student previously struggled with:\n" + "\n".join(bullets)


# ---------------------------------------------------------------------------
# Per-worker toolsets for the Micro-Router & Specialized Workers architecture.
# Binding each future worker LLM to only its own narrow tool subset (instead
# of every tool) is what shrinks its tool-calling schema and lets its system
# prompt drop the rules for tools it will never call.
# ---------------------------------------------------------------------------
doc_worker_tools = [
    process_and_index_documents,
    get_document_outline,
    resolve_chapter_target,
    read_page_range,
    locate_marker_in_range,
    search_in_document,
]


math_worker_tools = [python_code_executor]
web_worker_tools = [web_search_tool]

# Kept for backward compatibility with the current flattened single-agent
# graph (`graph.py`), which still binds one model to every tool. Phase 2
# rewires `graph.py` onto the per-worker groups above and this alias can be
# retired at that point.
assistant_tools = doc_worker_tools + math_worker_tools + web_worker_tools