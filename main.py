import os
import re
import json
import time
import tempfile
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from graph import (
    workflow,
    CHECKPOINT_DB_PATH,
    WORKER_NODE_MAP,
    user_facing_llm_error,
    strip_protocol_tags,
    PARTIAL_OPENER_RE,
    _HIDDEN_BLOCK_NAME_PATTERNS,
)
from agents import DEEPSEEK_ROUTER_TIMEOUT_SECONDS, _deepseek_model
from locknlearn_schemas import StruggleRecord
from tools import (
    cancel_embed_warmup,
    cancel_all_embed_jobs,
    get_pipeline_status,
    index_struggle_memory,
    recall_struggle_context,
    remember_thread_auth,
)
from dotenv import find_dotenv, load_dotenv
import sentry_sdk

load_dotenv(find_dotenv(".env"))

sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    environment=os.environ.get("SENTRY_ENVIRONMENT", "development"),
    release=os.environ.get("SENTRY_RELEASE"),
    send_default_pii=False,
    traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
    enable_logs=True,
    max_request_body_size="never",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The graph's nodes are all coroutines now, so it must be compiled with an
    # async-native checkpointer. `AsyncSqliteSaver` must be constructed inside
    # a running event loop (it calls `asyncio.get_running_loop()`), which is
    # exactly what a FastAPI lifespan guarantees — hence compiling here rather
    # than at plain module-import time in graph.py.
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
        app.state.agent_app = workflow.compile(checkpointer=checkpointer)
        try:
            yield
        finally:
            await cancel_all_embed_jobs()


app = FastAPI(title="API", version="v2.0.0", lifespan=lifespan)

_DEV_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
_VERCEL_ORIGIN = "https://v0frontend4.vercel.app"


def _normalize_origin(origin: str) -> str:
    """Drop brackets, quotes, and a trailing slash so the browser Origin matches."""
    return origin.strip().strip("[]\"'").rstrip("/")


def _allowed_origins() -> list[str]:
    """Local Next.js, the Vercel app, and comma-separated CORS_ORIGINS."""
    configured = [
        _normalize_origin(origin)
        for origin in os.environ.get("CORS_ORIGINS", "").split(",")
    ]
    origins: list[str] = []
    for origin in (*_DEV_ORIGINS, _VERCEL_ORIGIN, *configured):
        if origin and origin not in origins:
            origins.append(origin)
    return origins


# Allows the Next.js frontend to call this API and read the streamed SSE
# response cross-origin. Localhost stays allowed for dev; set CORS_ORIGINS
# to the Vercel origin in production.
_cors_origins = _allowed_origins()
print(f"[CORS] allow_origins={_cors_origins}", flush=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True}


DEFAULT_STUDY_MODE = "detailed"
STUDY_MODE_INSTRUCTIONS = {
    "socratic": (
        "SOCRATIC TUTOR MODE: Chat is the draft (questions, struggle, hints). "
        "The canvas is the whiteboard (only formal math after a step is correct). "
        "Assume the student already read the problem — do NOT ask what the goal "
        "is. Open with one guiding question about the first mathematical step. "
        "Do not hint, give formulas, or call python_code_executor until they "
        "attempt the step. If they are wrong, stuck, or ask for help, hint in "
        "chat only. If they are correct (even a short informal answer), praise "
        "in chat then call python_code_executor ONCE with type \"calculation\". "
        "steps MUST be exactly one string (that milestone only — never the rest "
        "of the solution). Intermediate (V2S, components): omit value and "
        "complete_goal; summary must end with ? toward the stem-asked target. "
        "Never write the stem unknown's number until the student states it. "
        "Headings in the extract (due north / due east) must be copied verbatim. "
        "Final unknown only after the student states that number: call the tool "
        "once with complete_goal true; value and summary MUST start with "
        "Result found: [answer] (no follow-up question). Never rewrite the "
        "original question stem."
    ),
    "detailed": (
        "DETAILED EXPLANATION MODE: Provide a comprehensive academic explanation: "
        "define key concepts, then show full step-by-step mathematical reasoning. "
        "EXCEPTION: if this turn generates a chart or calculation on the Desk "
        "workspace, keep the chat reply to 2-3 sentences and put the step-by-step "
        "detail in the Desk card instead."
    ),
}


def _build_mode_prefixed_prompt(user_prompt: str, mode: str) -> str:
    """Prepends the study-mode system instruction ahead of the raw user prompt."""
    mode_key = (mode or DEFAULT_STUDY_MODE).strip().lower()
    mode_instruction = STUDY_MODE_INSTRUCTIONS.get(mode_key, STUDY_MODE_INSTRUCTIONS[DEFAULT_STUDY_MODE])
    return f"[STUDY MODE INSTRUCTION]: {mode_instruction}\n\n{user_prompt}"


_SUPPORTED_UI_LANGUAGES = frozenset({"en", "az", "tr", "ru"})


def _normalize_ui_language(value: str | None) -> str:
    code = (value or "az").strip().lower()
    if code in _SUPPORTED_UI_LANGUAGES:
        return code
    return "az"


def _with_ui_language(user_prompt: str, ui_language: str) -> str:
    return f"[UI LANGUAGE]: {ui_language}\n\n{user_prompt}"


def _extract_text_chunk(content) -> str:
    """Normalizes a streamed AIMessageChunk's content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return ""


_LEAK_PROBE_CAP_CHARS = 6000
_GOALS_BLOCK_CAP_CHARS = 600
_LEADING_ARRAY_RE = re.compile(r'\s*\[(?:"(?:\\.|[^"\\])*"|[^\[\]"])*\]')

# Full openers: (regex, closing regex). Matches only the literal protocol
# tokens, so `[[1, 2]]`, LaTeX, and Markdown pass straight through.
_STREAM_OPENERS = (
    (re.compile(r"\[{1,2}\s*SOCRATIC_GOALS\s*\]{1,2}", re.IGNORECASE),
     re.compile(r"\[{1,2}\s*/\s*SOCRATIC_GOALS\s*\]{1,2}", re.IGNORECASE),
     _GOALS_BLOCK_CAP_CHARS),
    # Hidden reasoning blocks: patterns come from graph.py so the stream filter
    # and the stateless sanitizer share one tolerant definition.
    *(
        (re.compile(rf"<\s*{pat}\b[^>]*>", re.IGNORECASE),
         re.compile(rf"<\s*/\s*{pat}\s*>", re.IGNORECASE),
         _LEAK_PROBE_CAP_CHARS)
        for pat in _HIDDEN_BLOCK_NAME_PATTERNS.values()
    ),
)
# Tail fragments that could still become an opener on the next chunk.
_STREAM_OPENER_PREFIXES = (
    "[[socratic_goals", "[socratic_goals", "<dependency_tree", "<scratchpad", "<think",
)
# Long enough for `<dependency_tree` plus a short attribute run (`PARTIAL_OPENER_RE`).
_STREAM_HOLDBACK_MAX = 90


def _opener_prefix_start(buf: str) -> Optional[int]:
    """Index where a trailing fragment of `buf` could still become a protocol opener:
    either a strict prefix of one (`<dep`) or a complete opener name that has not
    received its closing `>` / `]]` yet (`<dependency_tree`, `<think id="`)."""
    start = max(0, len(buf) - _STREAM_HOLDBACK_MAX)
    partial = PARTIAL_OPENER_RE.search(buf, start)
    cut = partial.start() if partial else None
    for index in range(max(start, len(buf) - 20), len(buf)):
        if buf[index] not in "[<":
            continue
        if cut is not None and index >= cut:
            break
        tail = buf[index:].lower()
        if any(p.startswith(tail) and len(tail) < len(p) for p in _STREAM_OPENER_PREFIXES):
            cut = index
            break
    if cut is not None:
        print(f"[SANITIZE] stream: holding partial opener {buf[cut:]!r}", flush=True)
    return cut


class _TokenLeakFilter:
    """Stateful guard for the raw token stream. Hides protocol blocks
    (`[[SOCRATIC_GOALS]]`, `<dependency_tree>`, `<scratchpad>`, `<think>`) even when a
    tag is split across chunks. It only ever holds back a short tail that could
    still become an opener (at most ~20 chars) or the body of a block being
    hidden, so ordinary prose, LaTeX, and Markdown stream with no added latency
    and are never rewritten. Call `flush()` when the stream ends."""

    def __init__(self):
        self._node: Optional[str] = None
        self._buffer = ""
        self._hiding = None  # (close_re, cap) while inside a hidden block

    def _reset(self, node_name: str) -> str:
        leftover = self._buffer if self._hiding is None else ""
        self._node = node_name
        self._buffer = ""
        self._hiding = None
        return leftover

    def feed(self, node_name: str, text_chunk: str) -> str:
        """Returns the text (possibly empty) that is now safe to emit."""
        prefix = ""
        if node_name != self._node:
            prefix = self._reset(node_name)
        self._buffer += text_chunk
        out = prefix
        while True:
            if self._hiding is not None:
                close_re, cap = self._hiding
                match = close_re.search(self._buffer)
                if match:
                    self._buffer = self._buffer[match.end():]
                    self._hiding = None
                    continue
                if len(self._buffer) > cap:
                    print(
                        f"[SANITIZE] stream: unclosed hidden block ({len(self._buffer)} chars)",
                        flush=True,
                    )
                    if cap == _GOALS_BLOCK_CAP_CHARS:
                        # Goals tag with no closer: drop only its array, keep the reply.
                        self._buffer = _LEADING_ARRAY_RE.sub("", self._buffer, count=1)
                    else:
                        # Malformed reasoning block: drop it rather than hold the reply back.
                        self._buffer = ""
                    self._hiding = None
                    continue
                return out
            earliest = None
            for opener_re, close_re, cap in _STREAM_OPENERS:
                match = opener_re.search(self._buffer)
                if match and (earliest is None or match.start() < earliest[0].start()):
                    earliest = (match, close_re, cap)
            if earliest is not None:
                match, close_re, cap = earliest
                print(f"[SANITIZE] stream: hiding {match.group(0)!r} in node={self._node}", flush=True)
                out += self._buffer[: match.start()]
                self._buffer = self._buffer[match.end():]
                self._hiding = (close_re, cap)
                continue
            cut = _opener_prefix_start(self._buffer)
            if cut is None:
                out += self._buffer
                self._buffer = ""
            else:
                out += self._buffer[:cut]
                self._buffer = self._buffer[cut:]
            return out

    def flush(self) -> str:
        """Release any held tail at end of stream (a hidden block stays dropped)."""
        leftover = self._buffer if self._hiding is None else ""
        self._buffer = ""
        self._hiding = None
        return strip_protocol_tags(leftover, "stream_flush")


def _message_has_tool_calls(msg) -> bool:
    """True when this chunk/message is a tool-calling turn — its prose is
    process narration ("I'll solve…") and must not reach the chat pane."""
    if msg is None:
        return False
    if getattr(msg, "tool_calls", None):
        return True
    if getattr(msg, "tool_call_chunks", None):
        return True
    extra = getattr(msg, "additional_kwargs", None) or {}
    return bool(extra.get("tool_calls") or extra.get("function_call"))


def _sse_event(
    chat_message: str = "",
    desk_update=None,
    active_problem_update=None,
    canvas_op=None,
    canvas_anchor_id=None,
    done: bool = False,
    status: str = "",
) -> str:
    # Final stateless net for every chat string (fallbacks and errors included).
    payload = {"chat_message": strip_protocol_tags(chat_message, "sse"), "desk_update": desk_update}
    if status:
        payload["status"] = status
    if active_problem_update is not None:
        payload["active_problem_update"] = active_problem_update
    if canvas_op is not None:
        payload["canvas_op"] = canvas_op
    if canvas_anchor_id:
        payload["canvas_anchor_id"] = canvas_anchor_id
    if done:
        payload["done"] = True
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# SSE comment frames keep the HTTP connection alive while `doc_worker` is
# silently indexing (no chat tokens). Without this, a long/stalled tool
# looks identical to a dead backend on the frontend.
_SSE_HEARTBEAT = ":\n\n"
_SSE_HEARTBEAT_SECONDS = 5.0
_ASTREAM_SENTINEL = object()


async def _anext_or_sentinel(iterator):
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return _ASTREAM_SENTINEL


_IMAGE_UPLOAD_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_thread_uploads: dict[str, str] = {}
_thread_images: dict[str, str] = {}


def _same_path(left: str, right: str) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _remove_temp_upload(path: str) -> None:
    if not path or not os.path.isfile(path):
        return
    try:
        os.remove(path)
        print(f"[DOC LOADER] Removed temp upload {path}", flush=True)
    except OSError as exc:
        print(f"[DOC LOADER] Temp remove failed: {exc}", flush=True)


def _keep_thread_upload(thread_id: str, new_path: str) -> None:
    previous = _thread_uploads.get(thread_id)
    _thread_uploads[thread_id] = new_path
    if previous and not _same_path(previous, new_path):
        _remove_temp_upload(previous)


def _keep_thread_image(thread_id: str, new_path: str) -> None:
    """Screenshot slot. A newer photo replaces only the previous photo."""
    previous = _thread_images.get(thread_id)
    _thread_images[thread_id] = new_path
    if previous and not _same_path(previous, new_path):
        _remove_temp_upload(previous)


def _is_thread_upload(thread_id: str, path: str) -> bool:
    registered = _thread_uploads.get(thread_id)
    return bool(path and registered and _same_path(registered, path))


def _is_kept_upload(thread_id: str, path: str) -> bool:
    if _is_thread_upload(thread_id, path):
        return True
    registered = _thread_images.get(thread_id)
    return bool(path and registered and _same_path(registered, path))


def _is_storage_object_key(value: str) -> bool:
    text = (value or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or re.match(r"^[A-Za-z]:/", text):
        return False
    return "/" in text and not os.path.isfile(text)


def _download_storage_object(object_key: str, access_token: str) -> str:
    """Pull a signed-in session PDF back from Storage when the local temp is gone."""
    import urllib.error
    import urllib.request

    url = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").rstrip("/")
    api_key = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or ""
    )
    if not url or not api_key or not access_token:
        return ""
    safe_key = object_key.replace("\\", "/").lstrip("/")
    request = urllib.request.Request(
        f"{url}/storage/v1/object/documents/{safe_key}",
        headers={
            "Authorization": f"Bearer {access_token}",
            "apikey": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[DOC LOADER] Storage restore failed: {exc}", flush=True)
        return ""
    suffix = os.path.splitext(safe_key)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(payload)
        print(f"[DOC LOADER] Restored storage object {safe_key} to {temp_file.name}", flush=True)
        return temp_file.name


@app.post("/submit_stream")
async def chat_stream_endpoint(
    request: Request,
    message: str = Form(...), 
    thread_id: str = Form("default_session"),
    user_name: str = Form("Guest"),
    mode: str = Form("detailed"),
    ui_language: str = Form("az"),
    document: UploadFile = File(None),
    canvas_anchor_id: str = Form(""),
    canvas_request_id: str = Form(""),
    canvas_branch_from_id: str = Form(""),
    explain_source: str = Form(""),
    user_id: str = Form(""),
    access_token: str = Form(""),
    document_path: str = Form(""),
):
    request_start = time.time()
    print(f"[TIMING] Request Started ('/submit_stream', thread_id={thread_id})...", flush=True)
    temp_path = ""
    try:
        agent_app = request.app.state.agent_app
        user_prompt = message

        if document and document.filename:
            print("[TIMING] Document Upload Read Started...", flush=True)
            t_doc = time.time()

            await document.seek(0)
            file_ext = os.path.splitext(document.filename)[1]
            is_image = file_ext.lower() in _IMAGE_UPLOAD_EXTS
            content = await document.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            if not is_image:
                _keep_thread_upload(thread_id, temp_path)
            else:
                _keep_thread_image(thread_id, temp_path)
            if is_image and _is_storage_object_key(document_path):
                live = _thread_uploads.get(thread_id) or ""
                if not (live and os.path.isfile(live)):
                    restored = _download_storage_object(document_path, access_token or "")
                    if restored:
                        _keep_thread_upload(thread_id, restored)
            clean_path = temp_path.replace("\\", "/")

            user_prompt = (
                f"{message} (File path: {clean_path}; "
                f"Original filename: {os.path.basename(document.filename)})"
            )
            print(f"[TIMING] Document Upload Read Ended - Took {time.time() - t_doc:.2f}s", flush=True)
        elif _is_storage_object_key(document_path):
            live = _thread_uploads.get(thread_id) or ""
            if not (live and os.path.isfile(live)):
                restored = _download_storage_object(document_path, access_token or "")
                if restored:
                    _keep_thread_upload(thread_id, restored)
                    clean_path = restored.replace("\\", "/")
                    user_prompt = (
                        f"{message} (File path: {clean_path}; "
                        f"Original filename: {os.path.basename(document_path)})"
                    )

        requested_mode = (mode or DEFAULT_STUDY_MODE).strip().lower()
        if requested_mode not in {"detailed", "socratic"}:
            requested_mode = DEFAULT_STUDY_MODE
        if requested_mode == "socratic":
            print("[FREEZE] socratic request forced onto detailed", flush=True)
            requested_mode = DEFAULT_STUDY_MODE
        source = (explain_source or "").strip().lower()
        if source in ("chat", "desk") or (canvas_branch_from_id or "").strip():
            requested_mode = DEFAULT_STUDY_MODE
        mode = requested_mode
        if source in ("chat", "desk"):
            user_prompt = f"[EXPLAIN SOURCE: {source}]\n{user_prompt}"
        user_prompt = _build_mode_prefixed_prompt(user_prompt, mode)
        user_prompt = _with_ui_language(user_prompt, _normalize_ui_language(ui_language))

        struggle_context = await asyncio.to_thread(
            recall_struggle_context, thread_id, message
        )
        if struggle_context:
            print(f"[STRUGGLE] recalled {len(struggle_context)} chars for thread={thread_id}", flush=True)

        remember_thread_auth(thread_id, user_id or "", access_token or "")
        inputs = {
            "messages": [HumanMessage(content=user_prompt)],
            "struggle_context": struggle_context or "",
            "canvas_anchor_id": canvas_anchor_id or None,
            "canvas_branch_from_id": canvas_branch_from_id or None,
            "study_mode": (mode or DEFAULT_STUDY_MODE).strip().lower(),
        }
        request_id = uuid.uuid4().hex
        config = {
            "configurable": {
                "thread_id": thread_id,
                "request_id": request_id,
                "user_name": user_name,
                "user_id": user_id or "",
                "access_token": access_token or "",
                "document_path": document_path or "",
                "explain_source": (explain_source or "").strip().lower(),
            },
            "metadata": {
                "session_type": "pdf_chat_session",
                "canvas_request_id": canvas_request_id or "",
            },
            # Hard defense-in-depth ceiling on top of the Doc Worker's
            # prompt-level anti-loop caps (DOC_WORKER_SYSTEM_PROMPT):
            # `process_and_index_documents` x1, `get_document_outline` x1,
            # `read_page_range`+`locate_marker_in_range` combined ~2-3,
            # `search_in_document` x3. LangGraph counts EVERY node execution
            # as one recursion step, including each worker<->tools round
            # trip — NOT just LLM calls — so this must cover the worst-case
            # LEGITIMATE turn: a first-time/freshly-referenced document
            # (indexing required) whose Tier 1 structural navigation is
            # attempted AND exhausted, THEN falls back to the full Tier 2
            # semantic search budget, chained into a downstream worker that
            # itself needs a self-correction retry round trip — NOT just a
            # single tier's tool loop with zero retries, which is the
            # scenario a previous version of this limit (20) was silently
            # undercounting (it omitted `process_and_index_documents`
            # entirely, so a doc_worker -> math_worker turn on a freshly
            # uploaded/referenced document could exhaust the ENTIRE budget
            # on doc_worker alone and hit the hard cutoff — appearing to the
            # user as doc_worker "silently" never handing off to
            # math_worker, when in fact `GraphRecursionError` fired):
            #   router(1)
            #   + doc_worker<->doc_tools: index(2) + outline(2)
            #     + read/locate x3 (6) + search x3 (6) = 16
            #   + doc_worker's final answer (1)
            #   + one chained downstream worker's own tool round trips,
            #     allowing for one self-correction retry (4)
            #   + that worker's final answer (1)
            #   + composer(1)
            #   = 24 steps.
            # 40 leaves generous margin above that legitimate ceiling (incl.
            # headroom for a 3-worker chain, e.g. web_worker + doc_worker +
            # math_worker) while still cutting off a genuinely runaway loop
            # in a small fraction of the time/cost it used to take to fail.
            "recursion_limit": 40,
        }

        async def _produce_events():
            # Micro-Router & Specialized Workers topology: every node that
            # can produce user-facing chat text is an LLM-calling reasoning
            # node — the four specialized workers (`chat_worker_node` for
            # chitchat/trivial/vision turns, plus doc/math/web). `router_node`
            # is deliberately NOT in this list (Phase 4): it now ONLY ever
            # produces a structured `RoutePlan` (routing decision), never
            # user-facing text, so including it here would risk leaking its
            # structured-output JSON into the chat stream for no benefit —
            # see the Phase 4 note above `router_llm` in agents.py. Every
            # worker's internal tool loop (`doc_tools_node`, `math_tools_node`,
            # `web_tools_node`) never invokes a chat model itself, so those
            # simply never appear in "messages"-mode events either — nothing
            # extra needs to be done to keep their raw tool JSON out of the
            # chat text stream.
            active_worker_nodes = [
                "chat_worker_node",
                "doc_worker_node",
                "math_worker_node",
                "web_worker_node",
            ]

            # Tracks whether ANY chat text was actually streamed token-by-token
            # this turn. As of Phase 4 this is effectively always true by the
            # time the turn finishes — every worker (including `chat_worker_node`
            # now) is a plain streaming chat completion, and `router_node`
            # itself never emits chat text at all. Kept as a defensive
            # fallback (rather than removed) so a turn that somehow produces
            # no token stream still gets `composer_node`'s `chat_reply`
            # delivered instead of silently dropping the reply.
            chat_text_streamed = False

            # Tracks which worker last streamed a chat-text chunk this turn,
            # so a `doc_worker` -> `math_worker` handoff (both producing
            # chat text in the SAME turn) never has its two replies
            # concatenated with no visual break — see the paragraph-break
            # injection right below.
            last_streaming_node: Optional[str] = None
            leak_filter = _TokenLeakFilter()
            last_status = ""

            def status_event() -> str:
                nonlocal last_status
                current = get_pipeline_status(thread_id)
                if not current or current == last_status:
                    return ""
                last_status = current
                return _sse_event(status=current)

            # QA on a 1-page PDF exposed `doc_worker`'s own raw extracted-text
            # reply leaking into the Chat pane BEFORE `math_worker` (chained
            # right after it) ever produced its own answer — confusing UX
            # for a purely internal handoff step. Only the LAST worker in
            # the Router's plan is "user-facing" for a given turn; every
            # earlier worker in a chain (e.g. `doc_worker` in
            # `['doc_worker', 'math_worker']`) should silently write to
            # state (`extracted_text`, `active_navigation`, ...) without its
            # own text ever reaching the SSE stream. Captured from
            # `router_node`'s own "updates" event below, the very first
            # thing the graph emits each turn. `None` until then; treated as
            # "stream everything" (the old behavior) as a safe fallback if
            # it's somehow never populated.
            terminal_worker_node: Optional[str] = None
            canvas_ops_emitted = 0
            anchor = canvas_anchor_id or None
            branch_from = canvas_branch_from_id or None
            done_sent = False
            is_socratic = (mode or DEFAULT_STUDY_MODE).strip().lower() == "socratic"
            socratic_tools_chat_reply = ""
            last_socratic_chat_reply = ""
            socratic_chat_yielded = False

            # "messages" mode streams token-level chat model chunks (the Chat
            # Sidebar text); "updates" mode streams full node outputs. We only
            # care about `composer_node`'s update there — it's the single
            # place (Phase 2's graph.py) that assembles the final
            # `chat_reply`/`desk_payload` for this turn, so intercepting only
            # that node's update is what keeps every other node's raw
            # ToolMessage/AIMessage content (e.g. `doc_tools_node`/
            # `math_tools_node`'s tool output) from ever leaking into the SSE
            # stream. `astream` (not `stream`) is required since every node in
            # the graph is a coroutine.
            try:
                stream = agent_app.astream(
                    inputs, config=config, stream_mode=["messages", "updates"]
                )
                aiter = stream.__aiter__()
                next_item = asyncio.create_task(_anext_or_sentinel(aiter))
                last_heartbeat_at = time.time()
                try:
                    while True:
                        done, _pending = await asyncio.wait(
                            {next_item}, timeout=0.4
                        )
                        stage = status_event()
                        if stage:
                            yield stage
                        if not done:
                            now = time.time()
                            if now - last_heartbeat_at >= _SSE_HEARTBEAT_SECONDS:
                                yield _SSE_HEARTBEAT
                                last_heartbeat_at = now
                            continue

                        item = next_item.result()
                        if item is _ASTREAM_SENTINEL:
                            break

                        stream_mode, data = item
                        if stream_mode == "messages":
                            msg, metadata = data
                            node_name = metadata.get("langgraph_node", "")

                            is_visible_worker = (
                                terminal_worker_node is None or node_name == terminal_worker_node
                            )
                            # Skip yield only — never `continue` the while-loop here.
                            # That would skip `__anext__` and spin on the same chunk.
                            skip_tokens = (
                                _message_has_tool_calls(msg)
                                or bool(branch_from)
                                or (is_socratic and node_name == "math_worker_node")
                            )
                            if (
                                not skip_tokens
                                and node_name in active_worker_nodes
                                and msg.content
                                and is_visible_worker
                            ):
                                text_chunk = _extract_text_chunk(msg.content)
                                if text_chunk:
                                    # Universal Pipeline Phase 2: buffer-then-strip
                                    # so a leaked <dependency_tree> block never
                                    # reaches SSE for ANY node/mode, not only the
                                    # Socratic math_worker path suppressed above.
                                    filtered_chunk = leak_filter.feed(node_name, text_chunk)
                                    if filtered_chunk:
                                        if last_streaming_node is not None and last_streaming_node != node_name:
                                            yield _sse_event(chat_message="\n\n", desk_update=None)
                                        last_streaming_node = node_name
                                        chat_text_streamed = True
                                        yield _sse_event(chat_message=filtered_chunk, desk_update=None)

                        elif stream_mode == "updates":
                            for node_name, node_update in (data or {}).items():
                                print(
                                    f"[SSE DEBUG] updates node={node_name} "
                                    f"has_update={bool(node_update)} "
                                    f"keys={list(node_update.keys()) if isinstance(node_update, dict) else type(node_update).__name__}",
                                    flush=True,
                                )
                                if node_update and node_name == "router_node":
                                    planned_workers = node_update.get("planned_workers") or []
                                    if planned_workers:
                                        terminal_worker_node = WORKER_NODE_MAP.get(planned_workers[-1])

                                stage = status_event()
                                if stage:
                                    yield stage

                                if (
                                    is_socratic
                                    and node_update
                                    and node_name in ("math_worker_node", "math_tools_node")
                                ):
                                    reply = node_update.get("chat_reply") or ""
                                    if isinstance(reply, str) and reply.strip():
                                        last_socratic_chat_reply = reply.strip()
                                        if node_name == "math_tools_node":
                                            socratic_tools_chat_reply = last_socratic_chat_reply

                                if not node_update or node_name != "composer_node":
                                    continue

                                held_tail = leak_filter.flush()
                                if held_tail:
                                    chat_text_streamed = True
                                    yield _sse_event(chat_message=held_tail, desk_update=None)

                                desk_payload = node_update.get("desk_payload")
                                desk_payloads = node_update.get("desk_payloads")
                                composer_chat_reply = (node_update.get("chat_reply") or "").strip()
                                if composer_chat_reply:
                                    last_socratic_chat_reply = composer_chat_reply
                                fallback_chat_message = "" if chat_text_streamed else composer_chat_reply
                                if is_socratic and not chat_text_streamed:
                                    fallback_chat_message = (
                                        composer_chat_reply
                                        or socratic_tools_chat_reply
                                        or last_socratic_chat_reply
                                        or "Let's break this down step-by-step."
                                    )
                                print(
                                    "[SSE DEBUG] composer fallback "
                                    f"chat_text_streamed={chat_text_streamed} "
                                    f"fallback={fallback_chat_message!r:.160} "
                                    f"done_sent={done_sent} "
                                    f"composer_chat_reply={composer_chat_reply!r:.120}",
                                    flush=True,
                                )
                                active_problem_update = node_update.get("active_problem")
                                ops = node_update.get("canvas_ops") or []

                                if fallback_chat_message:
                                    socratic_chat_yielded = True
                                    yield _sse_event(chat_message=fallback_chat_message, canvas_anchor_id=anchor)

                                if isinstance(ops, list) and len(ops) > canvas_ops_emitted:
                                    for op in ops[canvas_ops_emitted:]:
                                        yield _sse_event(canvas_op=op, canvas_anchor_id=anchor)
                                    canvas_ops_emitted = len(ops)

                                payloads = desk_payloads if desk_payloads else (
                                    [desk_payload] if desk_payload is not None else []
                                )
                                if not branch_from:
                                    for artifact in payloads:
                                        yield _sse_event(desk_update=artifact, canvas_anchor_id=anchor)

                                if active_problem_update:
                                    yield _sse_event(
                                        active_problem_update=active_problem_update,
                                        canvas_anchor_id=anchor,
                                    )

                                if not done_sent:
                                    done_sent = True
                                    yield _sse_event(done=True, canvas_anchor_id=anchor)

                        next_item = asyncio.create_task(_anext_or_sentinel(aiter))
                finally:
                    if not next_item.done():
                        next_item.cancel()
                print("[SSE DEBUG] astream loop exited", flush=True)
            except GraphRecursionError:
                # Hit the hard `recursion_limit` above — the agent was stuck
                # in a loop (e.g. repeated `search_in_document` calls) and
                # got cut off before ever reaching `composer_node`. Surface a
                # clean, user-facing message instead of leaking LangGraph's
                # raw internal exception text.
                elapsed = time.time() - request_start
                print(f"[TIMING] Request HIT RECURSION LIMIT (thread_id={thread_id}) - Took {elapsed:.2f}s", flush=True)
                yield _sse_event(
                    chat_message="\n\n⚠️ Sorğu həddindən artıq mürəkkəb oldu və dayandırıldı. "
                                 "Zəhmət olmasa sualınızı daha konkret şəkildə yenidən yazın.",
                    desk_update=None,
                    done=True,
                    canvas_anchor_id=anchor,
                )
                return
            except Exception as e:
                # The graph runs INSIDE this generator (streamed lazily by
                # StarletteStreamingResponse), so failures here (e.g. an
                # upstream 429/500 from the LLM provider) never reach the
                # outer try/except below — log + surface them here instead so
                # a failed turn is never silently swallowed.
                elapsed = time.time() - request_start
                print(f"[TIMING] Request FAILED mid-stream (thread_id={thread_id}) - Took {elapsed:.2f}s", flush=True)
                print(f"XƏTA BAŞ VERDİ (stream): {str(e)}", flush=True)
                yield _sse_event(
                    chat_message=f"\n\n{user_facing_llm_error(e)}",
                    desk_update=None,
                    done=True,
                    canvas_anchor_id=anchor,
                )
                return

            if is_socratic and not chat_text_streamed and not socratic_chat_yielded:
                yield _sse_event(
                    chat_message=last_socratic_chat_reply or "Let's break this down step-by-step.",
                    canvas_anchor_id=anchor,
                )
            if not done_sent:
                yield _sse_event(done=True, canvas_anchor_id=anchor)
            print(f"[TIMING] Request Ended (thread_id={thread_id}) - Took {time.time() - request_start:.2f}s", flush=True)

        async def event_generator():
            completed = False
            try:
                async for event in _produce_events():
                    yield event
                completed = True
            finally:
                if not completed:
                    # Client dropped / request cancelled: stop only the embed
                    # warm-up this request started. A finished turn keeps it.
                    cancel_embed_warmup(thread_id, request_id)
                if temp_path and not _is_kept_upload(thread_id, temp_path):
                    _remove_temp_upload(temp_path)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        if temp_path and not _is_kept_upload(thread_id, temp_path):
            _remove_temp_upload(temp_path)
        print(f"[TIMING] Request FAILED (thread_id={thread_id}) - Took {time.time() - request_start:.2f}s", flush=True)
        print(f"XƏTA BAŞ VERDİ: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Daxili xəta: {str(e)}")


class NameFormulaBody(BaseModel):
    formula: str
    formula_id: str
    user_id: str = ""
    access_token: str = ""


def _clip_formula_name(text: str) -> str:
    cleaned = re.sub(r"(?i)^chapter\b[:\s-]*", "", (text or "").strip().strip("\"'`"))
    words = cleaned.split()
    return " ".join(words[:6]).strip(" .")


def _formula_reply_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return " ".join(parts)
    return str(content or "")


def _patch_formula_context(formula_id: str, user_id: str, context: str, access_token: str) -> bool:
    import urllib.error
    import urllib.parse
    import urllib.request

    url = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").rstrip("/")
    api_key = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or ""
    )
    if not url or not api_key or not access_token or not formula_id or not user_id:
        return False
    query = urllib.parse.urlencode({"id": f"eq.{formula_id}", "user_id": f"eq.{user_id}"})
    request = urllib.request.Request(
        f"{url}/rest/v1/memory_formulas?{query}",
        data=json.dumps({"context": context}).encode("utf-8"),
        method="PATCH",
        headers={
            "Authorization": f"Bearer {access_token}",
            "apikey": api_key,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as error:
        print(f"[FORMULA NAME] context update failed: {error.code}", flush=True)
        return False
    except Exception as error:
        print(f"[FORMULA NAME] context update failed: {error}", flush=True)
        return False


@app.post("/name_formula")
async def name_formula(body: NameFormulaBody):
    """Names one saved formula from its TeX. The click does not wait on this."""
    formula = (body.formula or "").strip()
    if not formula:
        raise HTTPException(status_code=400, detail="Formula text is required.")
    model = _deepseek_model(0, DEEPSEEK_ROUTER_TIMEOUT_SECONDS)
    if model is None:
        raise HTTPException(status_code=503, detail="Formula naming is unavailable.")
    prompt = (
        f"Identify this physics/math formula: {formula}. "
        "Provide ONLY its scientific or conceptual name in 2 to 6 words. "
        "DO NOT include the word 'Chapter' or any generic text. "
        "Example output: 'Second Law of Thermodynamics' or 'Bulk Modulus'."
    )
    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
    except Exception as error:
        print(f"[FORMULA NAME] model failed: {error}", flush=True)
        raise HTTPException(status_code=502, detail="Formula naming failed.")
    name = _clip_formula_name(_formula_reply_text(response.content))
    if not name:
        raise HTTPException(status_code=502, detail="Formula naming failed.")
    if body.user_id and body.access_token:
        saved = await asyncio.to_thread(
            _patch_formula_context,
            body.formula_id,
            body.user_id,
            name,
            body.access_token,
        )
        if not saved:
            raise HTTPException(status_code=502, detail="Formula title was not saved.")
    return {"context": name}


@app.post("/memory/struggle")
async def memory_struggle(record: StruggleRecord):
    """Persists a Needs Review excerpt so later turns can inject pedagogical context."""
    try:
        await asyncio.to_thread(
            index_struggle_memory,
            record.thread_id,
            record.question_id,
            record.topic,
            record.excerpt,
            record.formula,
        )
    except Exception as e:
        print(f"[STRUGGLE] index failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Struggle memory write failed")
    return {"ok": True}