import base64
import difflib
import hashlib
import json
import os
import re
import time
import uuid
import asyncio
from typing import Annotated, TypedDict, Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

from tools import (
    doc_worker_tools,
    math_worker_tools,
    web_worker_tools,
    _auto_wrap_stray_latex,
    is_short_indexed_document,
    get_indexed_page_count,
    lookup_outline_chapter,
    read_page_range,
    consume_vision_figure_urls,
    store_vision_figure_urls,
    SHORT_DOC_BLOCKED_TOOLS,
    count_pdf_pages,
    EMPTY_PROMPT_AUTOSOLVE_MAX_PAGES,
    reset_thread_document_if_new_file,
    start_embed_warmup,
    PIPELINE_STATUS_MATH,
    PIPELINE_STATUS_RENDER,
    set_pipeline_status,
    _resolve_thread_key,
)
from vision_extract import (
    crop_image_diagrams_to_data_urls,
    vision_extract_image_file,
    vision_inventory_image_file,
)
from locknlearn_schemas import RoutePlan, SocraticGoalPlan
from agents import (
    router_llm,
    ROUTER_SYSTEM_PROMPT,
    socratic_planner_llm,
    SOCRATIC_PLANNER_SYSTEM_PROMPT,
    chat_worker_llm,
    chat_worker_vision_llm,
    CHAT_WORKER_SYSTEM_PROMPT,
    doc_worker_llm,
    DOC_WORKER_SYSTEM_PROMPT,
    math_worker_llm,
    MATH_WORKER_SYSTEM_PROMPT_SOCRATIC,
    MATH_WORKER_SYSTEM_PROMPT_DETAILED,
    web_worker_llm,
    WEB_WORKER_SYSTEM_PROMPT,
    ProviderUnavailableError,
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Micro-Router scratch state ---------------------------------------
    # Reset to fresh values by `router_node` at the START of every turn (see
    # its docstring) — these represent THIS turn's plan and THIS turn's
    # worker output, never accumulated/leaked across turns via the
    # checkpointer.
    # The FULL ordered plan `router_node` decided this turn — set ONCE and
    # never mutated afterward (unlike `worker_queue`, which is consumed/
    # popped by each worker as it finishes). Kept around purely so
    # `composer_node` can compare "what was planned" against
    # `completed_workers` ("what actually ran") and flag any silent
    # discrepancy — see `composer_node`'s `[HANDOFF WARNING]` check.
    planned_workers: list[str]
    # This turn's triggering `HumanMessage`, captured ONCE by `router_node`
    # (the graph's entry point, so `state["messages"][-1]` is guaranteed to
    # be it) before any worker appends its own tool-call noise. Every
    # worker's `build_worker_messages()` call re-injects this if the plain
    # last-N-message trim window (`get_trimmed_messages`/
    # `CONTEXT_WINDOW_SIZE`) ever slices it out — which happens easily in a
    # chained `doc_worker -> math_worker` turn once doc_worker's own tool
    # round trips alone exceed the window size, silently starving the
    # downstream worker of the actual instruction (e.g. "solve this") and
    # leaving it to reply with plain text instead of ever calling its tool.
    turn_user_message: Optional[BaseMessage]
    worker_queue: list[str]
    completed_workers: list[str]
    # The explicit handoff contract between `doc_worker_node` and
    # `math_worker_node`: whatever Doc Worker extracted/retrieved from the
    # document this turn, written once it produces its final (non-tool-call)
    # answer, and read by Math Worker (if queued next) via an injected
    # `[EXTRACTED DOCUMENT CONTEXT]` message — see `math_worker_node` below.
    # Named for what it actually IS (the extracted text), not a vague
    # "context" catch-all.
    extracted_text: Optional[str]
    desk_payload: Optional[dict]
    # All Desk artifacts produced THIS turn (diagram + calculation can both
    # land). `desk_payload` remains the last one for force-quit summaries.
    desk_payloads: Optional[list]
    # Echo of the frontend's board-question shape id for this turn (FormData).
    canvas_anchor_id: Optional[str]
    # Parent board-step id for a Why/How explanation branch (FormData).
    # When set, canvas `step` ops spawn as laterals instead of the main column.
    canvas_branch_from_id: Optional[str]
    # Incremental canvas ops derived from tool payloads, streamed as they land.
    canvas_ops: Optional[list]
    chat_reply: Optional[str]
    # Study mode for THIS turn (`socratic` / `detailed`),
    # set from FormData in `main.py`. Router must not clear it.
    study_mode: Optional[str]

    # Scratch, reset every turn by `router_node` — accumulates the resolved
    # page range / question marker / section_kind from THIS turn's
    # `resolve_chapter_target` / `read_page_range` / `locate_marker_in_range`
    # calls (populated by `doc_tools_node`) so `doc_worker_node` can commit
    # it into the PERSISTED `active_navigation` field below once it produces
    # its final answer. See `locknlearn_schemas.ActiveNavigation` for why
    # this two-step scratch->persisted handoff exists (mirrors
    # `desk_payload`'s pattern).
    navigation_update: Optional[dict]
    # Per-worker count of tools-node visits THIS turn, reset by `router_node`.
    # Caps runaway ReAct loops without the old per-tool-name blocks that
    # starved large-book `read_page_range` retries. See MAX_WORKER_TOOL_ROUND_TRIPS.
    tool_round_trips: dict
    # True when this turn is a new problem intake (new PDF path, new
    # screenshot, or a dissimilar typed stem). Math Worker then drops
    # prior-turn chat from its ephemeral window so the previous solution
    # cannot masquerade as this turn's task. Router sets it every turn.
    fresh_intake: bool
    # This turn's cached screenshot question is a different number from the
    # card already on the board. Router sets it false; doc_worker sets it
    # true. Composer stamps `freshCard` on the question op.
    fresh_question_card: bool

    # --- Session-level context --------------------------------------------
    # Deliberately NOT reset every turn — represents "what document/problem
    # is currently active" for this conversation thread (mirrors
    # `locknlearn_schemas.ActiveProblem`), so it survives turns that don't
    # touch the Doc Worker (e.g. a plain follow-up chitchat message). Only
    # overwritten when `doc_worker_node` actually processes a file.
    active_problem: Optional[dict]

    # Socratic dynamic todo — session-persisted, NOT cleared by router scratch.
    # Each item is {id, text, symbol, unit}. `pending_goals is None` =
    # uninitialized; non-empty = in progress; `[]` only after the last
    # explicit stem target is popped.
    pending_goals: Optional[list]
    is_solved: bool
    # Bookkeeping so a new canvas card / stem can reset the todo after SOLVED.
    # Never injected into the model prompt.
    socratic_scope_id: Optional[str]
    # Closed goal records (any order), so the final Result can name every
    # original target. Reset with the todo. Closing is idempotent per id.
    closed_goals: Optional[list]
    # Unit keyed by goal text, kept for older checkpoints. New plans store
    # the unit on the goal record. Reset with the todo.
    goal_units: Optional[dict]
    # Normalized equations of the last Socratic step, for repeat detection.
    last_socratic_step: Optional[str]

    # Deliberately NOT reset every turn either — "where in the book are we
    # right now" (mirrors `locknlearn_schemas.ActiveNavigation`). This is
    # what lets a RELATIVE follow-up ("Now move to Question 8") resolve
    # correctly even though `CONTEXT_WINDOW_SIZE` below only ever sends the
    # last 4 messages to a worker: the cursor lives here, in checkpointed
    # graph state, not in the trimmed chat window. Only overwritten when
    # `doc_worker_node` actually resolves a new page range this turn.
    active_navigation: Optional[dict]

    # Screenshot page for this thread. Separate from the PDF path in
    # `active_navigation` / `_thread_uploads`. Not cleared by router scratch.
    # A new photo replaces it; a new PDF does not.
    active_image: Optional[dict]

    # Recalled pedagogical struggle notes for THIS turn (set in main.py
    # `/submit_stream` from the on-disk struggle collection). The router
    # must not clear this mid-turn so every worker in a chain still sees it.
    struggle_context: Optional[str]


def _resolve_checkpoint_db_path() -> str:
    """LangGraph's SQLite checkpointer must NOT live inside a OneDrive/
    iCloud-synced project folder — file-sync locks on `checkpoint.db` are a
    prime suspect for the silent 3rd-request hang. Prefer a local
    unsynced directory (`%LOCALAPPDATA%\\LockNLearn` on Windows)."""
    local_app = os.environ.get("LOCALAPPDATA") or os.environ.get("TMP") or os.path.join(
        os.path.expanduser("~"), ".locknlearn"
    )
    checkpoint_dir = os.path.join(local_app, "LockNLearn")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "checkpoint.db")


CHECKPOINT_DB_PATH = _resolve_checkpoint_db_path()
print(f"[CHECKPOINT] SQLite path: {CHECKPOINT_DB_PATH}", flush=True)

# Strict context window — only the last N messages of the conversation are
# ever sent to the LLM. This is the single biggest lever against token-bloat
# latency on long-running chat threads.
CONTEXT_WINDOW_SIZE = 4

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
_SAME_PROBLEM_THRESHOLD = 0.8

DOC_TOOLS_MAP = {tool.name: tool for tool in doc_worker_tools}
MATH_TOOLS_MAP = {tool.name: tool for tool in math_worker_tools}
WEB_TOOLS_MAP = {tool.name: tool for tool in web_worker_tools}

# Maps a `RoutePlan.workers` entry (the router's vocabulary) to the actual
# graph node name that implements it.
WORKER_NODE_MAP = {
    "chat_worker": "chat_worker_node",
    "doc_worker": "doc_worker_node",
    "math_worker": "math_worker_node",
    "web_worker": "web_worker_node",
}

DESK_PAYLOAD_TOOL_NAME = "python_code_executor"
DESK_PAYLOAD_TYPES = {"chart", "calculation", "diagram", "explanation"}
EXPLANATION_ACK = "I've added the detailed explanation to the workspace."

PROVIDER_OVERLOAD_MESSAGE = (
    "AI servisləri müvəqqəti olaraq yüklənib, zəhmət olmasa bir az sonra yenidən cəhd edin."
)

# `main.py` annotates uploads as `(File path: <tmp>; Original filename: <name>)`.
_FILE_PATH_RE = re.compile(r"File path:\s*([^;)]+)")
_ORIG_FILENAME_RE = re.compile(r"Original filename:\s*([^)]+)")
_STUDY_MODE_BLOCK_RE = re.compile(
    r"\[STUDY MODE INSTRUCTION\]:.*?(?:\n\n|\r\n\r\n)",
    re.DOTALL,
)
_UI_LANGUAGE_BLOCK_RE = re.compile(
    r"\[UI LANGUAGE\]:\s*\S+\s*(?:\n\n|\r\n\r\n)",
    re.IGNORECASE,
)
_TYPED_PROBLEM_SOURCE = "Typed problem"
_SOCRATIC_MODE_RE = re.compile(
    r"\[STUDY MODE INSTRUCTION\]:\s*SOCRATIC",
    re.IGNORECASE,
)
_FILE_ANNOTATION_RE = re.compile(r"\s*\(File path:[^)]*\)\s*")
_TEMP_STEM_RE = re.compile(r"^(tmp|temp)[a-z0-9_\-]*$", re.IGNORECASE)
_CHAPTER_HEAD_RE = re.compile(
    r"^(?:(?P<kind>chapter|fəsil|ch\.?|section|bölmə)\s*)?(?P<num>\d+(?:\.\d+)?)"
    r"(?:\s*[:.\-–—]\s*|\s+)(?P<rest>.*)$",
    re.IGNORECASE,
)
_PROBLEM_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_SOLVEISH_RE = re.compile(
    r"\b(solve|həll|question|sual|problem|page|səhifə|chapter|fəsil|compute|find|hesabla)\b",
    re.IGNORECASE,
)
_GENERIC_QUERY_RE = re.compile(
    r"^\s*(?:(?:zəhmət\s+olmasa|please|pls)\s+)?"
    r"(?:(?:bunu(?:\s+da)?|bu\s+sualı|bu\s+məsələni|this|that|it|the\s+problem)\s+)?"
    r"(həll\s*et(?:məyi)?|solve(?:\s+(?:it|this|the\s+problem))?|hesabla|kömək(?:\s*et)?|help(?:\s+me)?)"
    r"(?:\s+(?:zəhmət\s+olmasa|please|pls|də|da))?"
    r"[.!?…]*\s*$",
    re.IGNORECASE,
)
_GENERIC_FILLER_RE = re.compile(
    r"\b(zəhmət\s+olmasa|please|pls|bunu|bu\s+sualı|bu\s+məsələni|"
    r"this|that|the\s+problem|also|too|da|də)\b",
    re.IGNORECASE,
)
_DEFAULT_PROBLEM_SUBTITLE = "Sənəd üzrə məsələnin həlli və təhlili"
_WEB_INTENT_RE = re.compile(
    r"\b(arxiv|paper|məqalə|citation|latest|son xəbər|web search|axtarış|google)\b",
    re.IGNORECASE,
)
_DOC_REF_RE = re.compile(
    r"\b(chapter|fəsil|page|səhifə|sual|məsələ|question|problem|section|bölmə)\b",
    re.IGNORECASE,
)
_FORMULA_EXTRACT_RE = re.compile(
    r"\b(?:formulas?|düstur(?:lar)?|formul)\b",
    re.IGNORECASE,
)
_READ_ONLY_RE = re.compile(
    r"\b(summarize|xülasə|extract|oxu|nə yazıb|what does it say|tapşırığı göstər)\b",
    re.IGNORECASE,
)
_ASSIGN_ONLY_RE = re.compile(
    r"əlav[əe]\s+et|add(?:\s+these)?\s+questions?",
    re.IGNORECASE,
)
_TASK_START_RE = re.compile(r"\[TASK START\]", re.IGNORECASE)
_CANVAS_EXPLAIN_RE = re.compile(r"\[CANVAS EXPLAIN\]", re.IGNORECASE)
_CHAT_ONLY_RE = re.compile(
    r"^\s*(salam|hey+|hi|hello|thanks|təşəkkür|sağ ol|ok|okay|nə var)[.!]?\s*$",
    re.IGNORECASE,
)
_OBJECTIVE_RE = re.compile(r"^\[OBJECTIVE\]:\s*(.+)\s*$", re.MULTILINE | re.IGNORECASE)
_FILENAME_PROBLEM_PAIR_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[-–]\s*(\d{1,3})(?!\d)")
_ROUTER_INPUT_MAX_CHARS = 800

_SOURCE_ALIASES = {
    "c&j": "Cutnell & Johnson",
    "cj": "Cutnell & Johnson",
    "cutnell": "Cutnell & Johnson",
    "serway": "Serway",
    "jewett": "Serway",
    "hrw": "Halliday & Resnick",
    "halliday": "Halliday & Resnick",
    "giancoli": "Giancoli",
    "knight": "Knight",
    "young": "Young & Freedman",
    "freedman": "Young & Freedman",
}

# Hard fail-safe: after this many worker <-> tools round trips in ONE turn,
# the worker is force-finished (no further LLM call). This is what stops a
# hallucinating `doc_worker` from burning the API budget until
# `recursion_limit` (the 363s / 40-step incident). It is a ROUND-TRIP cap,
# not a per-tool-name cap, so a large textbook can still do
# locate -> read -> outline -> read within the budget.
MAX_WORKER_TOOL_ROUND_TRIPS = 5


def get_trimmed_messages(messages, window_size: int = CONTEXT_WINDOW_SIZE):
    """Last-N trim that never splits an `(AIMessage tool_calls, ToolMessage*)`
    pair. If the slice would start on an orphan `ToolMessage`, earlier
    messages are prepended until its parent assistant turn is included."""
    if len(messages) <= window_size:
        return list(messages)
    start = len(messages) - window_size
    trimmed = list(messages[start:])
    while trimmed and isinstance(trimmed[0], ToolMessage) and start > 0:
        start -= 1
        trimmed = [messages[start]] + trimmed
    return trimmed


def _repair_openai_tool_pairs(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop orphan `ToolMessage`s and incomplete `AIMessage(tool_calls)`
    groups so DeepSeek/OpenAI never see `role=tool` without a matching
    parent `tool_calls` id (HTTP 400). Gemini is lenient; ChatOpenAI is not.
    """
    repaired: list[BaseMessage] = []
    open_ids: Optional[set[str]] = None
    open_start: Optional[int] = None

    def _drop_open_group():
        nonlocal open_ids, open_start
        if open_start is not None:
            del repaired[open_start:]
        open_ids = None
        open_start = None

    for message in messages:
        calls = list(getattr(message, "tool_calls", None) or [])
        if isinstance(message, AIMessage) and calls:
            if open_ids:
                _drop_open_group()
            repaired.append(message)
            open_ids = {str(tc.get("id") or "") for tc in calls}
            open_start = len(repaired) - 1
            continue

        if isinstance(message, ToolMessage):
            tid = str(getattr(message, "tool_call_id", None) or "")
            if not open_ids or tid not in open_ids:
                continue
            repaired.append(message)
            open_ids.discard(tid)
            if not open_ids:
                open_ids = None
                open_start = None
            continue

        if open_ids:
            _drop_open_group()
        repaired.append(message)

    if open_ids:
        _drop_open_group()
    return repaired


def _filter_foreign_tool_messages(messages: list[BaseMessage], own_tool_names: set[str]) -> list[BaseMessage]:
    """Drops tool-call `AIMessage`s/`ToolMessage`s that belong to a DIFFERENT
    worker's toolset (i.e. neither references a name in `own_tool_names`).

    Needed for same-turn worker chains (e.g. `doc_worker` -> `math_worker`):
    once `get_trimmed_messages` slices the raw message list down to the last
    N entries, those entries can be entirely an UPSTREAM worker's own
    tool-call round trips (which reference tools the CURRENT worker isn't
    even bound to). Feeding those orphaned/foreign tool messages into an LLM
    call is confusing at best; this keeps a worker's window free of any
    tool-call turn it didn't itself make.
    """
    filtered = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls and not any(tc.get("name") in own_tool_names for tc in tool_calls):
            continue
        if isinstance(message, ToolMessage) and message.name not in own_tool_names:
            continue
        filtered.append(message)
    return filtered


def build_worker_messages(state: "AgentState", own_tool_names: set[str]) -> list[BaseMessage]:
    """The single source of truth every worker node should use to build the
    conversational window for its LLM call (in place of a bare
    `get_trimmed_messages(state["messages"])` call). Layers THREE same-turn
    multi-worker-handoff safeguards on top of the plain last-N-message trim.
    All three only ever affect this EPHEMERAL, per-call list — never
    `state["messages"]` itself, which stays untouched for the checkpointer's
    permanent record and for `main.py`'s SSE "messages" stream (that's what
    actually delivered an upstream worker's text to the chat pane).

    1. Foreign tool noise stripped (`_filter_foreign_tool_messages` above):
       an upstream worker's own tool-call `AIMessage`/`ToolMessage` pairs
       (for tools THIS worker isn't bound to) are dropped.

    2. This-turn upstream `AIMessage`s stripped: any tool-call-free
       `AIMessage` that occurs AFTER `state["turn_user_message"]` in the
       FULL history can only be an already-finished upstream worker's own
       final answer from THIS turn (a worker never sees its own bare
       `AIMessage` when building a fresh call — finishing with one ends its
       turn entirely) — its content is already redundant with
       `extracted_text`/other injected context, so it is dropped rather
       than left dangling as the tail of the window. This is turn-scoped
       via identity, not blanket: legitimate `AIMessage`s from EARLIER
       conversation turns (needed for cross-turn continuity) sit BEFORE
       `turn_user_message` and are never touched.

    3. Original instruction always present: if `turn_user_message` itself
       got sliced out of the window by `get_trimmed_messages`, it is
       re-prepended.

    4. Structural safety net — never end on a bare assistant turn: chat-formatted
       models (Gemini included) are trained on strict human/assistant
       alternation. Generating "the next assistant turn" immediately after
       ANOTHER assistant turn strongly biases the model into treating that
       prior content as ITS OWN already-delivered final answer, producing a
       bare acknowledgment instead of doing the work — this was the exact
       root cause of `math_worker` finishing in ~1s with zero tool calls
       after a `doc_worker` handoff. If, after every filter above, the
       window still ends on a tool-call-free `AIMessage`, a synthetic
       trailing `HumanMessage` (restating the original request) is
       appended. `AIMessage`s that still have `tool_calls` are NEVER nudged
       (that would produce `assistant(tool_calls)` → `user` with no tool
       results — DeepSeek HTTP 400).

    5. OpenAI tool-pair repair (`_repair_openai_tool_pairs`): drop orphan
       `ToolMessage`s and incomplete `tool_calls` groups so the window is
       legal for ChatOpenAI/DeepSeek.
    """
    raw_messages = state["messages"]
    trimmed = _filter_foreign_tool_messages(get_trimmed_messages(raw_messages), own_tool_names)

    turn_user_message = state.get("turn_user_message")

    if turn_user_message is not None:
        this_turn_ids = set()
        seen_turn_start = False
        for message in raw_messages:
            if seen_turn_start:
                this_turn_ids.add(id(message))
            if message is turn_user_message:
                seen_turn_start = True

        trimmed = [
            message
            for message in trimmed
            if not (id(message) in this_turn_ids and isinstance(message, AIMessage) and not message.tool_calls)
        ]

    if turn_user_message is not None and not any(m is turn_user_message for m in trimmed):
        trimmed = [turn_user_message] + trimmed

    trimmed = _repair_openai_tool_pairs(trimmed)

    last = trimmed[-1] if trimmed else None
    if (
        last is not None
        and isinstance(last, AIMessage)
        and not getattr(last, "tool_calls", None)
    ):
        original_request = _extract_text(turn_user_message.content) if turn_user_message is not None else ""
        nudge_text = (
            "[SYSTEM NOTE]: The message above is reference context from another step this turn, "
            "NOT your own prior reply — you have not answered yet."
        )
        if original_request:
            nudge_text += f" Original request: {original_request}"
        nudge_text += " Act on it now and produce your complete response, including any required tool calls."
        trimmed = trimmed + [HumanMessage(content=nudge_text)]

    struggle = (state.get("struggle_context") or "").strip()
    if struggle:
        trimmed = [
            HumanMessage(content=f"[PEDAGOGICAL CONTEXT]:\n{struggle}")
        ] + trimmed

    return trimmed


def _read_image_as_base64(file_path: str) -> str:
    """Blocking disk read, meant to be run via `asyncio.to_thread`."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def _inline_image_if_present(message: BaseMessage) -> BaseMessage:
    """If this human turn references an image file, rewrite it into a
    multimodal message (text + image) so `chat_worker_llm` can see it
    directly and answer with a plain vision-capable chat reply — no
    dedicated vision worker/tool call required. Any other message
    (including document uploads, handled entirely by the Doc Worker's
    tools) passes through untouched.

    Phase 4: this now runs inside `chat_worker_node`, NOT `router_node`.
    The router itself never looks at message content beyond classifying it
    — it doesn't need the inlined image, and inlining it there would just
    make the router's own (small, tool-free) call heavier for no benefit."""
    msg_text = str(message.content)
    if "File path:" not in msg_text:
        return message

    file_path = _extract_file_path(msg_text)
    if not file_path or not any(file_path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        return message

    try:
        image_b64 = await asyncio.to_thread(_read_image_as_base64, file_path)
        return HumanMessage(
            content=[
                {"type": "text", "text": msg_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        )
    except Exception as e:
        return HumanMessage(content=f"{msg_text}\n\n[Şəklin oxunması zamanı xəta baş verdi: {str(e)}]")


_IMAGE_EXTRACTION_SYSTEM_PROMPT = (
    "You are an OCR/extraction assistant. Transcribe ONLY the visible "
    "problem statement in this image: every given number, unit, heading, "
    "direction, and condition, plus the exact question sentence, verbatim "
    "as plain text. Do NOT solve it. Do NOT add commentary, greetings, or "
    "your own reasoning. If the image is not a solvable math/physics "
    "problem, reply with exactly: NO_PROBLEM_FOUND."
)


async def _ocr_fallback_image_stem(file_path: str) -> str:
    """Plain OCR when structured VisionExtract fails. Never attaches a figure."""
    instruction = HumanMessage(
        content=f"Transcribe the problem statement in this image. (File path: {file_path})"
    )
    try:
        inlined = await _inline_image_if_present(instruction)
    except Exception:
        return ""
    sys_msg = HumanMessage(content=f"[SYSTEM INSTRUCTION]:\n{_IMAGE_EXTRACTION_SYSTEM_PROMPT}")
    try:
        response = await _ainvoke_llm(
            chat_worker_vision_llm, [sys_msg, inlined], "Image Extraction"
        )
    except Exception:
        return ""
    text = _peel_extract_wrapper(_extract_text(response.content).strip())
    if not text or text.strip().upper() == "NO_PROBLEM_FOUND":
        return ""
    return text


async def _vision_extract_image_stem(
    file_path: str,
    config: Optional[RunnableConfig] = None,
    marker: str = "",
) -> str:
    """Screenshot's extraction contract: same shape as PDF doc_worker output
    (a transcribed stem string) so it can feed `_doc_worker_finish_update`
    and `math_worker_node` unchanged, instead of chat_worker fusing read+solve.
    Diagram crops go to `_vision_extracts.image_urls` like PDF figures."""
    extract = await vision_extract_image_file(file_path, marker)
    stem = _peel_extract_wrapper((extract.stem or "").strip())
    if stem and stem.upper() != "NO_PROBLEM_FOUND":
        urls: list[str] = []
        if extract.diagrams:
            try:
                urls = await asyncio.to_thread(
                    crop_image_diagrams_to_data_urls, file_path, extract.diagrams
                )
            except Exception as e:
                print(f"[VISION] screenshot crop store failed: {e}", flush=True)
                urls = []
        if config is not None:
            store_vision_figure_urls(config, urls)
        print(
            f"[VISION] screenshot stem_len={len(stem)} n_figures={len(urls)}",
            flush=True,
        )
        return stem
    if config is not None:
        store_vision_figure_urls(config, [])
    print("[VISION] screenshot structured extract empty — OCR fallback, no figure", flush=True)
    return await _ocr_fallback_image_stem(file_path)


def _message_has_inline_image(message: BaseMessage) -> bool:
    """True when `_inline_image_if_present` attached an `image_url` block.
    Those turns must use Gemini (`chat_worker_vision_llm`), never DeepSeek."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict) and (item.get("type") == "image_url" or "image_url" in item):
            return True
    return False


def _extract_text(content) -> str:
    """Normalizes an AIMessage's `.content` (plain string, or a list of
    content-block dicts for multimodal responses) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _extract_file_path(text) -> Optional[str]:
    """Pulls the tempfile path main.py embeds after `File path:`."""
    if not isinstance(text, str) or "File path:" not in text:
        return None
    match = _FILE_PATH_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_original_filename(text) -> Optional[str]:
    """Display name of an uploaded document (`Original filename:`)."""
    if not isinstance(text, str):
        return None
    match = _ORIG_FILENAME_RE.search(text)
    return match.group(1).strip() if match else None


def _find_latest_uploaded_file_path(messages) -> Optional[str]:
    """Scans FULL message history (not just the trimmed window) backward
    for the most recent human turn that uploaded a document, so
    `doc_worker_node` can populate `active_problem` without an extra LLM
    call. Deliberately does NOT just check `messages[-1]`: by the time
    `doc_worker_node` produces its final answer, the most recent message is
    a `ToolMessage` (the `search_in_document` result), not the original
    upload turn."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            file_path = _extract_file_path(_extract_text(msg.content))
            if file_path and not _is_image_file_path(file_path):
                return file_path
    return None


def _find_latest_original_filename(messages) -> Optional[str]:
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        text = _extract_text(msg.content)
        file_path = _extract_file_path(text)
        if file_path and _is_image_file_path(file_path):
            continue
        name = _extract_original_filename(text)
        if name and not _is_image_file_path(name):
            return name
    return None


def _extract_desk_payload(raw_content: str) -> Optional[dict]:
    """Safely parses a `python_code_executor` ToolMessage payload into the
    structured Desk JSON object. Returns None (never raises) for anything
    that isn't a valid chart/calculation/error payload, so plain-text tool
    output cleanly leaves `desk_payload` untouched. `{"type": "error", ...}`
    is returned as-is (not swallowed) so callers can surface the execution
    failure back to the model/student instead of silently treating a failed
    tool call the same as "no tool call happened"."""
    if not raw_content:
        return None
    try:
        parsed = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("type") == "error":
        return parsed
    if parsed.get("type") not in DESK_PAYLOAD_TYPES:
        return None
    if parsed.get("type") == "calculation":
        parsed["value"] = _stringify_desk_value(parsed.get("value"))
    if parsed.get("type") == "diagram" and not isinstance(parsed.get("elements"), list):
        return None
    return parsed


def _merge_desk_payload(payloads: list, payload: dict) -> list:
    """Keep at most one `"type": "calculation"` card per turn. Math Worker
    often retries `python_code_executor` to fix formatting; appending both
    would render two identical Desk cards. Charts and diagrams still
    accumulate alongside the latest calculation."""
    if payload.get("type") == "calculation":
        payloads = [
            item
            for item in payloads
            if not (isinstance(item, dict) and item.get("type") == "calculation")
        ]
    payloads.append(payload)
    return payloads


def _is_socratic_mode(state: Optional[dict] = None, raw_text: str = "") -> bool:
    mode = ((state or {}).get("study_mode") or "").strip().lower()
    if mode == "socratic":
        return True
    return bool(_SOCRATIC_MODE_RE.search(raw_text or ""))


def _stamp_canvas_target(op: dict) -> dict:
    return {**op, "target": "canvas"}


def _payload_to_canvas_ops(
    payload: dict,
    branch_from_id: Optional[str] = None,
    socratic: bool = False,
) -> list:
    """Turn a python_code_executor artifact into canvas ops."""
    payload = _strip_dependency_tree_from_payload(payload)
    kind = payload.get("type")
    branch = (branch_from_id or "").strip() or None
    if kind in ("calculation", "explanation"):
        ops = []
        replace_last = (
            socratic
            and not branch
            and _truthy_replace_last(payload.get("replace_last"))
        )
        steps = payload.get("steps") or []
        if isinstance(steps, list):
            # Socratic whiteboard: never dump ahead — one milestone per print.
            if socratic and not branch:
                steps = steps[:1]
            for index, latex in enumerate(steps):
                if isinstance(latex, str) and latex.strip():
                    op = {"op": "step", "index": index, "latex": latex}
                    if branch:
                        op["branchFromId"] = branch
                    if replace_last:
                        op["replaceLast"] = True
                    ops.append(op)
        value = payload.get("value")
        has_value = isinstance(value, str) and bool(str(value).strip())
        summary = payload.get("summary") or ""
        if (
            not has_value
            and isinstance(summary, str)
            and _RESULT_FOUND_RE.match(summary.strip())
        ):
            extracted = _RESULT_FOUND_RE.sub("", summary.strip(), count=1).strip()
            if extracted:
                value = extracted
                has_value = True
        # Detailed mode always gets a RESULT. Socratic emits RESULT only on
        # the absolute final milestone (`value` present). Why/How laterals never.
        # Same-phase refinements (`replaceLast`) must not close the session
        # unless this payload already has a last-pop `value` (complete_goal
        # won over replace_last in `_apply_socratic_goal_payload`).
        if (
            kind == "calculation"
            and not branch
            and (not replace_last or has_value)
            and (not socratic or has_value)
        ):
            ops.append({
                "op": "result",
                "value": value if has_value else (payload.get("value") or ""),
                "summary": payload.get("summary") or "",
            })
        return [_stamp_canvas_target(op) for op in ops]
    if socratic:
        # Charts/diagrams in Socratic mode would leak a worked solution.
        return []
    if kind == "chart":
        return [_stamp_canvas_target({
            "op": "chart",
            "chart_type": payload.get("chart_type") or "line",
            "title": payload.get("title") or "",
            "labels": payload.get("labels") or [],
            "values": payload.get("values") or [],
        })]
    if kind == "diagram":
        return [_stamp_canvas_target({
            "op": "diagram",
            "title": payload.get("title") or "",
            "width": payload.get("width"),
            "height": payload.get("height"),
            "elements": payload.get("elements") or [],
        })]
    return []


def _split_explanation_steps(text: str) -> list[str]:
    """Fallback: turn a chat dump into lateral canvas cards when the tool was skipped."""
    body = (text or "").strip()
    if not body or body == EXPLANATION_ACK:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
    if len(parts) == 1:
        numbered = [p.strip() for p in re.split(r"\n(?=\d+[\.\)]\s)", parts[0]) if p.strip()]
        if len(numbered) > 1:
            parts = numbered
    return parts[:8]


def _collapse_desk_calculations(payloads: list) -> list:
    """Composer-side belt: replace earlier calculations in this turn's list
    with the latest one, preserving chart/diagram order."""
    last_calc_idx = None
    out: list = []
    for item in payloads:
        if isinstance(item, dict) and item.get("type") == "calculation":
            if last_calc_idx is None:
                last_calc_idx = len(out)
                out.append(item)
            else:
                out[last_calc_idx] = item
        else:
            out.append(item)
    return out


def _stringify_desk_value(value) -> str:
    """Mirrors tools._format_calculation_value so a nested dict never
    reaches the frontend as `[object Object]`."""
    if isinstance(value, dict):
        for key in ("value", "result", "text", "display", "formatted"):
            inner = value.get(key)
            if inner is not None and not isinstance(inner, (dict, list)):
                return _stringify_desk_value(inner)
        return json.dumps(value, ensure_ascii=False, default=str)
    if value is None:
        return ""
    return str(value)


def _next_worker_goto(state: AgentState, finished_worker: str) -> str:
    """Deterministically resolves the next hop from `worker_queue` — this,
    not another LLM call, is what decides "what happens next" after a
    worker finishes. `worker_queue` is only ever set by `router_node`
    (once) and consumed (never re-evaluated) by every worker after it.

    Logs every handoff decision explicitly (`[HANDOFF]`) — this is the
    single choke point every worker's final Command routes through, so if a
    multi-worker plan (e.g. `['doc_worker', 'math_worker']`) ever silently
    fails to chain into its next worker within the same turn, this log line
    is what makes that immediately diagnosable from server output instead
    of only being visible as a confusing frontend symptom (see the
    `recursion_limit` note in main.py for the concrete bug this caught:
    a compound doc_worker->math_worker turn on a freshly-indexed document
    could exhaust the OLD, too-tight recursion budget mid-handoff, ending
    the turn via `GraphRecursionError` right when math_worker should have
    taken over)."""
    queue = state.get("worker_queue") or []
    next_goto = WORKER_NODE_MAP[queue[0]] if queue else "composer_node"
    print(f"[HANDOFF] {finished_worker} done -> remaining_queue={queue} -> goto={next_goto}", flush=True)
    return next_goto


def _pop_worker_queue(state: AgentState) -> list[str]:
    queue = state.get("worker_queue") or []
    return queue[1:]


def _tool_round_trip_count(state: AgentState, worker: str) -> int:
    return (state.get("tool_round_trips") or {}).get(worker, 0)


def _next_round_trips(state: AgentState, worker: str) -> dict:
    trips = dict(state.get("tool_round_trips") or {})
    trips[worker] = trips.get(worker, 0) + 1
    return trips


def _latest_doc_extraction(state: AgentState) -> str:
    """Last successful page-read / marker-locate payload this turn — used
    when force-quitting `doc_worker` so `math_worker` still gets text."""
    for message in reversed(state.get("messages") or []):
        if not isinstance(message, ToolMessage):
            continue
        if message.name not in {"read_page_range", "locate_marker_in_range"}:
            continue
        content = str(message.content or "")
        if not content or content.startswith("Xəta:"):
            continue
        if "tapılmadı" in content[:240].lower():
            continue
        return content
    return state.get("extracted_text") or ""


def _looks_like_internal_prompt(text: str) -> bool:
    """True for JIT routing blobs (`[TASK START]`, tool-call instructions)
    that must never become a Desk title, subtitle, or chat-visible header."""
    blob = text or ""
    if _TASK_START_RE.search(blob):
        return True
    lowered = blob.lower()
    if "locate_marker_in_range" in lowered or "resolve_chapter_target" in lowered:
        return True
    if "[active navigation context]" in lowered:
        return True
    return False


_PROCESS_NARRATION_LINE_RE = re.compile(
    r"^\s*(?:"
    r"let me(?:\s+\w+){0,8}|"
    r"i(?:'ll| will| am going to|'m going to)\b|"
    r"wait[,.]?\b|"
    r"hold on\b|"
    r"i need to\b|"
    r"on second thought\b|"
    r"hmm[,.]?\b|"
    r"fixing (?:the )?latex\b|"
    r"calling (?:the )?(?:tool|python_code_executor)\b|"
    r"now (?:i'll|let me)\b|"
    r"okay,? let me\b"
    r").*$",
    re.IGNORECASE,
)

_MONOLOGUE_SENTENCE_RE = re.compile(
    r"(?i)(?:^|(?<=[.!?]\s))("
    r"(?:let me (?:reconsider|rethink|think|recalculate|double[- ]check|try again)|"
    r"wait,?\s+i (?:should|need)|on second thought|hmm,?\s+let me)"
    r"[^.!?]*[.!?]?\s*)"
)


def _is_process_narration_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    if _looks_like_internal_prompt(stripped):
        return True
    return bool(_PROCESS_NARRATION_LINE_RE.match(stripped))


def _strip_process_narration(text: str) -> str:
    """Drop leftover planning / LaTeX-fix play-by-play from chat_reply.
    Real solution sentences are kept."""
    if not text:
        return text
    kept = [ln for ln in text.splitlines() if not _is_process_narration_line(ln)]
    cleaned = "\n".join(kept).strip()
    cleaned = _MONOLOGUE_SENTENCE_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if cleaned and _is_process_narration_line(re.sub(r"\s+", " ", cleaned)):
        return ""
    return cleaned


_HIDDEN_QUESTION_CONTEXT_RE = re.compile(
    r"""^\[Context:\s*Question\b.*?"\]\s*""",
    re.IGNORECASE | re.DOTALL,
)


def _strip_turn_wrappers(text: str) -> str:
    """Drop UI-language, study-mode, and tempfile wrappers so classifiers
    and stem isolation see the student's actual body."""
    body = _UI_LANGUAGE_BLOCK_RE.sub("", text or "")
    body = _STUDY_MODE_BLOCK_RE.sub("", body)
    body = _FILE_ANNOTATION_RE.sub(" ", body)
    body = _HIDDEN_QUESTION_CONTEXT_RE.sub("", body)
    return re.sub(r"\s+", " ", body).strip()


def _maybe_language_only(body: str) -> bool:
    """A short follow-up with no new problem may only be asking for another reply language.
    The router model decides; this does not match phrases in any one language."""
    text = (body or "").strip()
    if not text or len(text) > 180 or len(text.split()) > 18:
        return False
    if _TASK_START_RE.search(text) or _FORMULA_EXTRACT_RE.search(text):
        return False
    if re.search(r"\d|[$=\\]", text):
        return False
    return True


def _user_facing_query(state: AgentState) -> str:
    """The student's actual question for this turn, with study-mode,
    file-path, and internal `[TASK START]` routing stripped — used as the
    Active Problem subtitle. Empty means "treat as generic" so the
    composer falls back to `[OBJECTIVE]` / topic instead of leaking
    the routing prompt onto the Desk."""
    message = state.get("turn_user_message")
    if message is None:
        return ""
    text = _extract_text(getattr(message, "content", message) or "")
    text = _strip_turn_wrappers(text)
    if _looks_like_internal_prompt(text):
        return ""
    return text


def _is_generic_query(query: str) -> bool:
    """True when the student gave no real methodological question — empty,
    'Bunu həll et', 'Bunu da həll et', 'solve this', or an internal
    `[TASK START]` routing prompt."""
    cleaned = re.sub(r"\s+", " ", (query or "").strip())
    if not cleaned:
        return True
    if _looks_like_internal_prompt(cleaned):
        return True
    if _GENERIC_QUERY_RE.match(cleaned):
        return True
    if _SOLVEISH_RE.search(cleaned) and len(cleaned.split()) <= 4:
        return True
    stripped = _GENERIC_FILLER_RE.sub(" ", cleaned)
    stripped = re.sub(r"[.!?…,]+", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if not stripped:
        return True
    if _GENERIC_QUERY_RE.match(stripped):
        return True
    remaining = stripped.split()
    return bool(_SOLVEISH_RE.search(stripped) and len(remaining) <= 3)


def _lookup_source_alias(stem: str) -> Optional[str]:
    lowered = stem.lower().strip()
    for key, name in sorted(_SOURCE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if lowered == key or lowered.startswith(f"{key} ") or lowered.startswith(f"{key}_"):
            return name
    first = re.split(r"[\s_\-]+", lowered)[0]
    return _SOURCE_ALIASES.get(first)


def _source_display_name(filename: Optional[str], fallback: Optional[str] = None) -> str:
    """Turns `C&J 2-23.pdf` into `Cutnell & Johnson`; ignores tempfile stems."""
    if not filename:
        return fallback or "Document"
    stem = os.path.splitext(os.path.basename(str(filename)))[0].strip()
    if not stem or _TEMP_STEM_RE.match(stem):
        return fallback or "Document"
    aliased = _lookup_source_alias(stem)
    if aliased:
        return aliased
    token = re.split(r"[\s_\-]+", stem)[0]
    if len(token) >= 3:
        return token
    return stem.replace("_", " ")


def _problem_from_filename(filename: Optional[str]) -> Optional[str]:
    """`C&J 2-23.pdf` → `Problem 2-23`."""
    if not filename:
        return None
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    pair = _FILENAME_PROBLEM_PAIR_RE.search(stem)
    if pair:
        return f"Problem {pair.group(1)}-{pair.group(2)}"
    numbered = re.search(
        r"(?:p(?:rob(?:lem)?)?|q(?:uestion)?|sual|məsələ)\s*(\d+(?:[-–.]\d+)?)",
        stem,
        re.IGNORECASE,
    )
    if numbered:
        return f"Problem {numbered.group(1).replace('.', '-')}"
    return None


def _split_chapter_title(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """('Chapter 24', "Gauss's Law") from a TOC heading, when possible."""
    if not raw:
        return None, None
    stripped = str(raw).strip()
    match = _CHAPTER_HEAD_RE.match(stripped)
    if not match:
        return stripped, None
    kind_raw = (match.group("kind") or "").lower()
    kind = "Section" if kind_raw in {"section", "bölmə"} else "Chapter"
    rest = (match.group("rest") or "").strip(" :-–—.")
    return f"{kind} {match.group('num')}", rest or None


def _problem_label(marker: Optional[str]) -> Optional[str]:
    if not marker:
        return None
    match = _PROBLEM_NUM_RE.search(str(marker))
    if match:
        return f"Problem {match.group(1)}"
    cleaned = str(marker).strip()
    return cleaned or None


def _compose_problem_title(
    source: str,
    chapter_label: Optional[str],
    problem: Optional[str],
) -> str:
    bits = [bit for bit in (chapter_label, problem) if bit]
    if bits:
        return f"{source}: {', '.join(bits)}"
    return source


_EXTRACT_FILLER_LINE_RE = re.compile(
    r"^\s*(?:"
    r"i (?:have|just )?(?:the )?(?:extracted |following )?(?:the )?(?:problem|question)(?: text)?"
    r"|here(?:'s| is) (?:the )?(?:extracted )?(?:problem|question)(?: text)?"
    r"|let me extract(?: it| the (?:problem|question))?"
    r"|səhifə\s+\S.{0,80}?sual"
    r"|page\s+\d+.{0,60}question"
    r"|tapıldı\b"
    r"|çıxar(?:dım|ılmış)"
    r"|aşağıdakı (?:sual|məsələ)"
    r"|işte (?:soru|problem)"
    r"|вот (?:извлечённ|извлеченн|задача|вопрос)"
    r"|i extracted\b"
    r")\s*[.!,]?\s*$",
    re.IGNORECASE,
)

_EXTRACT_FILLER_PREFIX_RE = re.compile(
    r"^(?:"
    r"I have the (?:extracted )?(?:problem|question)(?: text)?[.!,]?\s*"
    r"|(?:Now )?Let me extract(?: it| the (?:problem|question))?(?: (?:exactly|verbatim|faithfully|as[- ]is))?[.!,]?\s*"
    r")+",
    re.IGNORECASE,
)

_EXTRACT_META_SENTENCE_RE = re.compile(
    r"^(?:"
    r"i have (?:the )?(?:extracted )?(?:problem|question)(?: text)?(?: here)?"
    r"|let me extract"
    r"|i(?:'ll| will) extract"
    r"|here(?:'s| is) the extract(?:ed)?"
    r"|extracting (?:the )?(?:problem|question)"
    r"|now extracting"
    r"|i found the (?:problem|question)"
    r"|the (?:problem|question) (?:text )?is as follows"
    r"|mən (?:problemi|sualı) çıxar"
    r"|indi çıxararam"
    r"|izvleku"
    r"|izvlechen"
    r")\s*[.!,]?\s*$",
    re.IGNORECASE,
)

# OCR/doc_worker wrappers glued onto a real stem with ":" (sentence split
# does not break on colons). Strip these so isolate cannot treat the whole
# blob as one filler line.
_EXTRACT_WRAPPER_RE = re.compile(
    r"^(?:"
    r"(?:the )?(?:visible |transcribed )?(?:extracted )?(?:problem|question)"
    r"(?: statement| text)?"
    r"(?: shown(?: in(?: this| the)? image| above| below| here))?"
    r"\s*(?:is(?: as follows)?|:)|"
    r"here(?:'s| is) (?:the )?(?:extracted )?(?:problem|question)"
    r"(?: statement| text)?\s*(?:is(?: as follows)?|:)?|"
    r"(?:the )?(?:image|screenshot|photo) (?:shows|contains|depicts|reads)"
    r"(?: the following)?(?: problem| question)?:?|"
    r"transcribed(?: problem(?: statement)?)?\s*:"
    r")\s*",
    re.IGNORECASE,
)

# Do not split after a numbered stem like "65." — keep "65. A motorcycle..." together.
_SENTENCE_SPLIT_RE = re.compile(r"(?<!\d\.)(?<=[.!?])\s+|\n+")
_HR_SPLIT_RE = re.compile(r"\s*-{3,}\s*")
_COACH_SPEAK_LINE_RE = re.compile(
    r"^(?:you have read|you already(?: read)?|you(?:'ve| have) (?:read|seen|got)|"
    r"let(?:'s| us)\b|your turn\b|which quantity\b|guiding question\b|"
    r"i have extracted\b|assume you(?: have)? read|as you (?:have )?read|"
    r"now that you(?:'ve| have) read)",
    re.IGNORECASE,
)
_RESULT_FOUND_RE = re.compile(r"^result found:\s*", re.IGNORECASE)
_TRAILING_QUESTION_RE = re.compile(r"[?？]+\s*$")
_GOAL_ALREADY_QUESTION_RE = re.compile(
    r"^(who|what|which|how|when|where|why|whose|whom)\b|[?？]\s*$",
    re.IGNORECASE,
)
# Protocol tags the model may emit but the student must never see. Patterns
# match ONLY these literal tokens (the name SOCRATIC_GOALS / the XML tag name
# must be present), so LaTeX, Markdown, and Python-style lists like
# `[[1, 2], [3, 4]]` are never touched.
_SOCRATIC_GOALS_TAG_RE = re.compile(
    r"\[{1,2}\s*SOCRATIC_GOALS\s*\]{1,2}(.*?)\[{1,2}\s*/\s*SOCRATIC_GOALS\s*\]{1,2}",
    re.DOTALL | re.IGNORECASE,
)
_SOCRATIC_GOALS_OPEN_RE = re.compile(
    r"\[{1,2}\s*SOCRATIC_GOALS\s*\]{1,2}",
    re.IGNORECASE,
)
_SOCRATIC_GOALS_CLOSE_RE = re.compile(
    r"\[{1,2}\s*/\s*SOCRATIC_GOALS\s*\]{1,2}",
    re.IGNORECASE,
)
_SOCRATIC_GOALS_FENCE_RE = re.compile(
    r"```[a-z]*\s*(\[{1,2}\s*SOCRATIC_GOALS[\s\S]*?\[{1,2}\s*/\s*SOCRATIC_GOALS\s*\]{1,2})\s*```",
    re.IGNORECASE,
)
# A flat JSON array of strings right after an unclosed opener.
_GOALS_ARRAY_RE = re.compile(r'\s*\[(?:"(?:\\.|[^"\\])*"|[^\[\]"])*\]')
_HIDDEN_BLOCK_NAMES = ("dependency_tree", "scratchpad", "think")
# Tag-name regex per hidden block: tolerant of case, inner spaces, and
# `dependency tree` / `dependency-tree` spellings. Shared with main.py's
# streaming filter so the two can never drift apart.
_HIDDEN_BLOCK_NAME_PATTERNS = {
    "dependency_tree": r"dependency[_\- ]?tree",
    "scratchpad": r"scratchpad",
    "think": r"think",
}
_HIDDEN_BLOCK_RES = {
    name: (
        re.compile(rf"```[a-z]*\s*<\s*{pat}\b[\s\S]*?<\s*/\s*{pat}\s*>\s*```", re.IGNORECASE),
        re.compile(rf"<\s*{pat}\b[^>]*>.*?<\s*/\s*{pat}\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(rf"<\s*{pat}\b[^>]*>", re.IGNORECASE),
        re.compile(rf"<\s*/\s*{pat}\s*>", re.IGNORECASE),
    )
    for name, pat in _HIDDEN_BLOCK_NAME_PATTERNS.items()
}
# A trailing fragment that is an opener name WITHOUT its closing `>` / `]]` yet
# (e.g. `<dependency_tree`, `<think id="`, `[[SOCRATIC_GOALS`). The stream filter
# must hold it until the next chunk decides whether it is a real opener.
PARTIAL_OPENER_RE = re.compile(
    r"(?:<\s*(?:" + "|".join(_HIDDEN_BLOCK_NAME_PATTERNS.values()) + r")\b[^>]{0,60}"
    r"|\[{1,2}\s*SOCRATIC_GOALS\s*\]?)$",
    re.IGNORECASE,
)
_PROTOCOL_MARKERS = (
    "socratic_goals", "<dependency_tree", "<scratchpad", "<think", "</think",
    "</dependency_tree", "</scratchpad",
)
_PROTOCOL_TAG_ANY_RE = re.compile(
    r"<\s*/?\s*(?:" + "|".join(_HIDDEN_BLOCK_NAME_PATTERNS.values()) + r")\b",
    re.IGNORECASE,
)
_NEW_SOCRATIC_STEM_MIN_CHARS = 80
_SOCRATIC_EMPTY_CHAT_FALLBACK = "Let's break this down step-by-step."
_SOCRATIC_TOOL_ERROR_FALLBACK = (
    "That step could not be evaluated — please resend your last message or "
    "rephrase it and I will try again."
)
_SOCRATIC_SOLVED_CHAT_FALLBACK = (
    "The result is already on the board. Ask about any step and I will explain it."
)
_SOCRATIC_SOLVED_TOOL_STUB = (
    "The problem is already solved. Answer conceptual follow-ups in chat only. "
    "Do not call python_code_executor."
)
_SOCRATIC_REPLACE_LAST_RE = re.compile(
    r"(?i)(?:"
    r"\bsimplify\b|\bcalculate\b|\bexpand\b|"
    r"\bshow the algebra\b|\bshow (?:your|the) work\b|"
    r"\binstead of\b|\bshould be\b|\bcorrect (?:that|this|it)\b|"
    r"\bnot\s+[\d.]+\s*[a-zA-Z]*"
    r")"
)


def _sentence_is_coach_speak(sentence: str) -> bool:
    first = (sentence or "").strip()
    if not first:
        return False
    if _EXTRACT_META_SENTENCE_RE.match(first) or _EXTRACT_FILLER_LINE_RE.match(first):
        return True
    if _is_process_narration_line(first):
        return True
    return bool(_COACH_SPEAK_LINE_RE.match(first))


def _looks_like_coach_speak(text: str) -> bool:
    """True if ANY sentence is Socratic/meta coaching (not only the first)."""
    body = (text or "").strip()
    if not body:
        return False
    return any(
        _sentence_is_coach_speak(part)
        for part in _SENTENCE_SPLIT_RE.split(body)
        if part.strip()
    )


def _isolate_problem_stem(text: str) -> str:
    """Keep the PDF stem; drop RAG `---` coach segments and trailing coaching."""
    body = (text or "").strip()
    if not body:
        return ""
    kept_segments = []
    for segment in _HR_SPLIT_RE.split(body):
        segment = segment.strip()
        if not segment:
            continue
        sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(segment) if part.strip()]
        start = 0
        while start < len(sentences) and _sentence_is_coach_speak(sentences[start]):
            start += 1
        if start >= len(sentences):
            continue
        cut = []
        for sentence in sentences[start:]:
            if _sentence_is_coach_speak(sentence):
                break
            cut.append(sentence)
        if cut:
            kept_segments.append(" ".join(cut))
    return " ".join(kept_segments).strip()


def _summary_has_question(text: str) -> bool:
    return "?" in (text or "") or "？" in (text or "")


_MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_MATH_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[=+\-*/^]")
_UNIT_WRAP_RE = re.compile(r"\\(?:mathrm|text)\s*\{[^}]*\}")
_UNIT_MARK_RE = re.compile(r"\\(?:mathrm|text)\s*\{|%")


def _step_signature(step: str) -> str:
    """Unit-free token bag of a step's display equations, for repeat detection."""
    blocks = _MATH_BLOCK_RE.findall(step or "") or [step or ""]
    text = _UNIT_WRAP_RE.sub("", " ".join(blocks))
    return " ".join(_MATH_TOKEN_RE.findall(text))


def _first_step(payload: dict) -> str:
    steps = payload.get("steps")
    if isinstance(steps, list) and steps and isinstance(steps[0], str):
        return steps[0]
    return ""


def _step_repeats_last(payload: dict, state: dict) -> bool:
    """True when the new step's equations are (almost) all already on the last card."""
    sig = _step_signature(_first_step(payload))
    last = str(state.get("last_socratic_step") or "")
    new_tokens, old_tokens = set(sig.split()), set(last.split())
    if len(new_tokens) < 4 or not old_tokens:
        return False
    return len(new_tokens & old_tokens) / len(new_tokens) >= 0.8


_FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_MATH_SPAN_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.DOTALL)
_EQ_SPLIT_RE = re.compile(r"(?<![<>!])=(?!=)")
_NUMERAL_RE = re.compile(r"\d+(?:\.\d+)?")
_EXPONENT_RE = re.compile(r"\^\{?-?\d+\}?")
_SYMBOLIC_TAUTOLOGY_LEAD = (
    "That equation is true for every value of the unknown, so it does not determine it."
)
_SYMBOLIC_GIVEN_LEAD = "That step uses a number that is not in the problem."


def _latex_to_sympy_src(expr: str) -> str:
    """Best-effort LaTeX to a sympify string. Units and spacing are dropped."""
    s = _UNIT_WRAP_RE.sub("", expr or "")
    s = re.sub(r"\\(?:left|right|displaystyle|,|!|quad|qquad|;)", "", s)
    for _ in range(8):
        nxt = _FRAC_RE.sub(r"((\1)/(\2))", s)
        nxt = _SQRT_RE.sub(r"sqrt(\1)", nxt)
        if nxt == s:
            break
        s = nxt
    s = (
        s.replace(r"\cdot", "*")
        .replace(r"\times", "*")
        .replace(r"\div", "/")
        .replace(r"\pi", "pi")
    )
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
    s = s.replace("{", "").replace("}", "").replace("^", "**")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(\d)([A-Za-z(])", r"\1*\2", s)
    s = re.sub(r"(\))([A-Za-z(])", r")*\1", s)

    def _call(match: re.Match) -> str:
        name = match.group(1)
        if name in {"sqrt", "sin", "cos", "tan", "log", "exp"}:
            return match.group(0)
        return f"{name}*("

    return re.sub(r"([A-Za-z]+)\(", _call, s)


def _payload_math_text(payload: dict) -> str:
    parts = []
    steps = payload.get("steps")
    if isinstance(steps, list):
        parts.extend(step for step in steps if isinstance(step, str))
    value = payload.get("value")
    if isinstance(value, str):
        parts.append(value)
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary)
    return "\n".join(parts)


def _numerals_in(text: str) -> list[float]:
    """Plain numbers, ignoring exponents (`t^2`) and unit wrappers."""
    cleaned = _UNIT_WRAP_RE.sub(" ", text or "")
    cleaned = _EXPONENT_RE.sub(" ", cleaned)
    out = []
    for raw in _NUMERAL_RE.findall(cleaned):
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _near_number(value: float, other: float) -> bool:
    target, have = abs(value), abs(other)
    if abs(target - have) <= 0.02:
        return True
    return have > 1e-6 and abs(target - have) / have <= 0.03


def _numeral_is_derived(value: float, ledger: list[float]) -> bool:
    """True for a stem given, a small coefficient, or a short arithmetic
    combination of the givens (so 14*14/2 and 92/98 are allowed)."""
    if abs(value) <= 10 and abs(value - round(value)) < 1e-6:
        return True
    seeds = [n for n in ledger if n == n and abs(n) < 1e7]
    if any(_near_number(value, n) for n in seeds):
        return True
    pool = list(seeds)
    for _ in range(3):
        grown = list(pool)
        for left in pool:
            grown.extend((left * left, left / 2.0, left * 2.0))
            for right in seeds:
                grown.extend((left + right, left - right, right - left, left * right))
                if abs(right) > 1e-9:
                    grown.append(left / right)
        nxt = []
        for item in grown:
            if item != item or abs(item) >= 1e7:
                continue
            if any(abs(item - have) <= 1e-6 for have in nxt):
                continue
            nxt.append(item)
            if len(nxt) >= 400:
                break
        pool = nxt
        if any(_near_number(value, n) for n in pool):
            return True
    return False


def _sympy_module():
    try:
        import sympy
        return sympy
    except Exception as exc:
        print(f"[SYMBOLIC] sympy unavailable, gate skipped: {exc}", flush=True)
        return None


def _given_ledger(state: dict) -> tuple[list, str]:
    ledger = _numerals_in((state or {}).get("extracted_text") or "")
    if ledger:
        return ledger, "extract"
    cached = _numerals_in(_cached_image_stem(state or {}))
    if cached:
        return cached, "stem"
    return [], "empty"


def _iter_parsed_equations(text: str):
    """Parsed equation pairs. A parse miss is logged and skipped, never a close."""
    sympy = _sympy_module()
    if sympy is None:
        return
    for match in _MATH_SPAN_RE.finditer(text or ""):
        body = match.group(1) or match.group(2) or ""
        if "=" not in body:
            continue
        parts = [part.strip() for part in _EQ_SPLIT_RE.split(body) if part.strip()]
        if len(parts) < 2:
            continue
        for left, right in zip(parts, parts[1:]):
            try:
                lhs = sympy.sympify(_latex_to_sympy_src(left), locals={"sqrt": sympy.sqrt})
                rhs = sympy.sympify(_latex_to_sympy_src(right), locals={"sqrt": sympy.sqrt})
                simplified_left, simplified_right = sympy.simplify(lhs), sympy.simplify(rhs)
                difference = sympy.simplify(simplified_left - simplified_right)
            except Exception:
                print(f"[SYMBOLIC] unparsed equation {left[:40]!r} = {right[:40]!r}", flush=True)
                continue
            shared = simplified_left.free_symbols & simplified_right.free_symbols
            identity = bool(shared and difference.is_zero is True)
            yield {
                "left": left,
                "right": right,
                "lhs": simplified_left,
                "rhs": simplified_right,
                "identity": identity,
            }


def _lhs_is_exact_symbol(lhs, symbol_name: str) -> bool:
    """True only when the left side is that symbol. `v_{A0}` matches `v_A0`.
    `2*a` and `v_x` are not `v`."""
    cleaned = _clean_goal_symbol(symbol_name)
    if not cleaned:
        return False
    symbols = list(getattr(lhs, "free_symbols", ()) or ())
    if len(symbols) != 1 or _clean_goal_symbol(str(symbols[0])) != cleaned:
        return False
    try:
        import sympy
        difference = sympy.simplify(lhs - symbols[0])
    except Exception:
        return False
    return difference.is_zero is True


def _pure_rhs_number(rhs) -> Optional[float]:
    if getattr(rhs, "free_symbols", None):
        return None
    try:
        return float(rhs)
    except (TypeError, ValueError):
        return None


def _number_is_copied_given(value: float, ledger: list) -> bool:
    return any(_near_number(value, number) for number in ledger)


def _bound_rhs_ok(rhs, rhs_src: str, other_symbols: set, ledger: list) -> bool:
    """A bound goal may close on its own number. A copied stem given
    (`a = 2` when 2 is in the stem) still does not. An expression that
    still names another open goal does not."""
    number = _pure_rhs_number(rhs)
    if number is not None and _number_is_copied_given(number, ledger):
        return False
    if _rhs_certifies(rhs, rhs_src, other_symbols, ledger):
        return True
    return number is not None


def _rhs_certifies(rhs, rhs_src: str, other_symbols: set, ledger: list) -> bool:
    """A number made from the givens, or an expression that does not still
    contain another open goal's symbol. A small coefficient alone is not an
    answer, and a bare unknown numeral is not one either."""
    names = {str(symbol) for symbol in getattr(rhs, "free_symbols", ()) or ()}
    if names & other_symbols:
        return False
    if names:
        return True
    if not ledger:
        return False
    try:
        value = float(rhs)
    except (TypeError, ValueError):
        return False
    small_integer = abs(value) <= 10 and abs(value - round(value)) < 1e-6
    in_ledger = any(_near_number(value, number) for number in ledger)
    if _numeral_is_derived(value, ledger) and not (small_integer and not in_ledger):
        return True
    numerals = _numerals_in(rhs_src)
    if not numerals or not all(_numeral_is_derived(number, ledger) for number in numerals):
        return False
    only_coefficients = all(
        abs(number) <= 10 and abs(number - round(number)) < 1e-6 for number in numerals
    )
    return not (only_coefficients and not in_ledger)


def _equation_value_latex(left: str, right: str) -> str:
    body = f"{left.strip()} = {right.strip()}"
    if body.startswith("$"):
        return body if body.endswith("$") else body + "$"
    return f"${body}$"


def _confirm_goal_close(payload: dict, pending: list, state: dict) -> Optional[tuple]:
    """Index of the one open goal an accepted equation discharges, plus a
    value string. None when nothing matches or more than one goal matches.
    Identities are ignored. Chat prose never reaches this."""
    if not pending:
        return None
    ledger, _source = _given_ledger(state)
    matches: dict = {}
    saw_symbol_equation = False
    for equation in _iter_parsed_equations(_payload_math_text(payload)):
        if equation["identity"]:
            continue
        hit = [
            index
            for index, goal in enumerate(pending)
            if _lhs_is_exact_symbol(equation["lhs"], _goal_symbol(goal))
        ]
        if not hit:
            continue
        saw_symbol_equation = True
        if len(hit) != 1:
            print(
                f"[COMMIT] ambiguous lhs {equation['left'][:40]!r} goals={hit}",
                flush=True,
            )
            matches.clear()
            break
        index = hit[0]
        others = {
            _goal_symbol(goal)
            for other, goal in enumerate(pending)
            if other != index and _goal_symbol(goal)
        }
        if not _bound_rhs_ok(equation["rhs"], equation["right"], others, ledger):
            print(
                f"[COMMIT] rhs not certified {equation['left'][:24]!r} = {equation['right'][:24]!r}",
                flush=True,
            )
            continue
        matches[index] = _equation_value_latex(equation["left"], equation["right"])
    if len(matches) == 1:
        index, value = next(iter(matches.items()))
        symbol = _goal_symbol(pending[index])
        print(
            f"[COMMIT] bound number index={index} symbol={symbol!r} value={value[:60]!r}",
            flush=True,
        )
        return index, value
    if len(matches) > 1:
        print(f"[COMMIT] several goals match {sorted(matches)} — close nothing", flush=True)
    elif not saw_symbol_equation:
        print("[COMMIT] no bound-symbol equation — close nothing", flush=True)
    return None


def _certified_bound_values(text: str, pending: list, ledger: list) -> list:
    """Numeric results of bound-symbol equations the judge would accept.
    The unknown-given check must not reject those results."""
    if not pending:
        return []
    exempt = []
    for equation in _iter_parsed_equations(text):
        if equation["identity"]:
            continue
        if not any(_lhs_is_exact_symbol(equation["lhs"], _goal_symbol(goal)) for goal in pending):
            continue
        others = {
            _goal_symbol(goal)
            for goal in pending
            if not _lhs_is_exact_symbol(equation["lhs"], _goal_symbol(goal)) and _goal_symbol(goal)
        }
        if not _bound_rhs_ok(equation["rhs"], equation["right"], others, ledger):
            continue
        if getattr(equation["rhs"], "free_symbols", None):
            continue
        try:
            exempt.append(float(equation["rhs"]))
        except (TypeError, ValueError):
            continue
    return exempt


def _symbolic_gate_reason(payload: dict, state: dict) -> Optional[str]:
    """'tautology', 'unknown_given', or None. A parse miss, an unsure SymPy
    result, or an empty given list allows the step through. A bound-symbol
    equation is judged before the unknown-given reject so its own result
    (a ratio of givens) is not dropped."""
    if _sympy_module() is None:
        return None
    text = _payload_math_text(payload)
    parsed_pairs = 0
    identity_pairs = 0
    last_identity = ""
    for equation in _iter_parsed_equations(text):
        parsed_pairs += 1
        if equation["identity"]:
            identity_pairs += 1
            last_identity = f"{equation['left'][:40]} = {equation['right'][:40]}"
            print(
                f"[SYMBOLIC] identity span {equation['left'][:40]!r} = {equation['right'][:40]!r}",
                flush=True,
            )
    if parsed_pairs and identity_pairs == parsed_pairs:
        print(
            f"[SYMBOLIC] tautology reject span {last_identity!r} pairs={parsed_pairs}",
            flush=True,
        )
        return "tautology"
    ledger, source = _given_ledger(state)
    if not ledger:
        print("[SYMBOLIC] empty ledger — pass", flush=True)
        return None
    pending = _pending_goals_snapshot(state) or []
    exempt = _certified_bound_values(text, pending, ledger)
    for number in _numerals_in(text):
        if _numeral_is_derived(number, ledger):
            continue
        if any(_near_number(number, value) for value in exempt):
            continue
        print(
            f"[SYMBOLIC] unknown_given {number:g} ledger_size={len(ledger)} source={source} "
            f"sample={[round(n, 4) for n in ledger][:12]}",
            flush=True,
        )
        return "unknown_given"
    return None


def _symbolic_reject_reply(reason: str, pending: Optional[list]) -> str:
    lead = _SYMBOLIC_TAUTOLOGY_LEAD if reason == "tautology" else _SYMBOLIC_GIVEN_LEAD
    goal = pending[0] if pending else None
    return f"{lead} {_next_goal_question(goal)}"


def _lint_socratic_payload(payload: dict, units: dict, closed_now: list) -> dict:
    """Deterministic desk-quality pass: add the known unit to a single-target
    `value` that has none; log (never rewrite) truncated steps and unit-less values."""
    out = dict(payload)
    for step in out.get("steps") or []:
        if isinstance(step, str) and step.rstrip().endswith((":", ",")):
            print(f"[DESK LINT] step ends mid-sentence: {step.strip()[-60:]!r}", flush=True)
    value = out.get("value")
    if not (isinstance(value, str) and re.search(r"\d", value)) or _UNIT_MARK_RE.search(value):
        return out
    unit = ""
    if len(closed_now) == 1 and value.count("=") == 1:
        unit = _goal_unit(closed_now[0], units)
    if not unit or not value.rstrip().endswith("$"):
        print(f"[DESK LINT] missing unit in value {value[:60]!r} (no unambiguous goal unit)", flush=True)
        return out
    fixed = value.rstrip()[:-1].rstrip() + f"\\,\\mathrm{{{unit}}}$"
    print(f"[DESK LINT] unit appended: {fixed!r}", flush=True)
    summary = out.get("summary")
    if isinstance(summary, str) and value in summary:
        out["summary"] = summary.replace(value, fixed)
    out["value"] = fixed
    return out


def _normalize_socratic_calculation(payload: dict, allow_result: bool = False) -> dict:
    """Patch Socratic summary/value only — never add steps or extra tool calls.

    `allow_result` is true only after a real last pop (`pending_goals` empty /
    `is_solved`). Intermediate payloads must not be rewritten into Result found.
    """
    out = _strip_dependency_tree_from_payload(payload)
    summary = out.get("summary")
    summary = summary.strip() if isinstance(summary, str) else ""
    value = out.get("value")
    has_value = isinstance(value, str) and bool(value.strip())
    if not allow_result:
        out.pop("value", None)
        if _RESULT_FOUND_RE.match(summary):
            summary = _RESULT_FOUND_RE.sub("", summary, count=1).strip()
        out["summary"] = summary
        return out
    if not has_value and _RESULT_FOUND_RE.match(summary):
        rest = _RESULT_FOUND_RE.sub("", summary, count=1).strip()
        if rest:
            out["value"] = rest
            value = rest
            has_value = True
    if has_value:
        value_s = str(value).strip()
        summary = _TRAILING_QUESTION_RE.sub("", summary).strip()
        if not _RESULT_FOUND_RE.match(summary):
            summary = f"Result found: {value_s}." + (f" {summary}" if summary else "")
        out["summary"] = summary
        out["value"] = value_s
    elif summary:
        # Last pop with no `value` yet — never re-open the session with
        # "What is the next step?". Close from the remaining summary.
        summary = _TRAILING_QUESTION_RE.sub("", summary).strip()
        if not _RESULT_FOUND_RE.match(summary):
            summary = f"Result found: {summary}."
        out["summary"] = summary
        rest = _RESULT_FOUND_RE.sub("", summary, count=1).strip()
        if rest:
            out["value"] = rest
    else:
        out["summary"] = "Result found."
    return out


def _truthy_flag(value) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
        return True
    return False


def _truthy_replace_last(value) -> bool:
    return _truthy_flag(value)


def _socratic_user_wants_replace_last(state: AgentState) -> bool:
    """True for refine/correct commands on the current phase, not a new milestone answer."""
    query = _user_facing_query(state) or ""
    body = re.sub(r"\s+", " ", query).strip()
    if len(body) < 4:
        return False
    return bool(_SOCRATIC_REPLACE_LAST_RE.search(body))


def _coerce_goal_list(raw) -> Optional[list]:
    if not isinstance(raw, list):
        return None
    items = [str(item).strip() for item in raw if str(item).strip()]
    return items or None


_GOAL_SYMBOL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _clean_goal_symbol(raw) -> str:
    """Plain report symbol. `v_x` stays `v_x`; a bare `v` does not swallow it."""
    text = str(raw or "").strip().strip("$")
    text = text.replace("\\", "")
    text = re.sub(r"[{}\s]", "", text)
    match = _GOAL_SYMBOL_RE.fullmatch(text)
    return match.group(0) if match else ""


def _goal_text(goal) -> str:
    if isinstance(goal, dict):
        return str(goal.get("text") or "").strip()
    return str(goal or "").strip()


def _goal_symbol(goal) -> str:
    if isinstance(goal, dict):
        return _clean_goal_symbol(goal.get("symbol"))
    return ""


def _goal_id(goal) -> str:
    if isinstance(goal, dict):
        return str(goal.get("id") or "").strip()
    return ""


def _goal_unit(goal, units: Optional[dict] = None) -> str:
    if isinstance(goal, dict):
        unit = str(goal.get("unit") or "").strip()
        if unit:
            return unit
    text = _goal_text(goal)
    return str((units or {}).get(text) or "").strip()


def _goal_record(text: str, symbol: str = "", unit: str = "", index: int = 0) -> dict:
    return {
        "id": f"g{index}",
        "text": str(text or "").strip(),
        "symbol": _clean_goal_symbol(symbol),
        "unit": str(unit or "").strip(),
    }


def _as_goal_record(item, index: int, units: Optional[dict] = None) -> Optional[dict]:
    """Older checkpoints stored bare strings. A string has no symbol, so the
    judge will not close it; a new problem plans records."""
    units = units or {}
    if isinstance(item, dict):
        text = str(item.get("text") or "").strip()
        if not text:
            return None
        return {
            "id": str(item.get("id") or f"g{index}").strip() or f"g{index}",
            "text": text,
            "symbol": _clean_goal_symbol(item.get("symbol")),
            "unit": str(item.get("unit") or units.get(text) or "").strip(),
        }
    text = str(item or "").strip()
    if not text or text.startswith("{"):
        return None
    return _goal_record(text, "", str(units.get(text) or ""), index)


def _goal_records(raw, units: Optional[dict] = None) -> list:
    if not isinstance(raw, list):
        return []
    records = []
    for index, item in enumerate(raw):
        record = _as_goal_record(item, index, units)
        if record:
            records.append(record)
    return records


def _goal_sentence_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower().strip(" .?")


def _without_goal_sentences(stem: str, state: Optional[dict]) -> str:
    """Drop planner target sentences from the question-card stem. Does not
    write anything back into `extracted_text`."""
    keys = set()
    for goal in _pending_goals_snapshot(state) or []:
        key = _goal_sentence_key(_goal_text(goal))
        if key:
            keys.add(key)
    for goal in _goal_records((state or {}).get("closed_goals"), (state or {}).get("goal_units") or {}):
        key = _goal_sentence_key(_goal_text(goal))
        if key:
            keys.add(key)
    if not keys or not (stem or "").strip():
        return stem or ""
    kept = []
    for sentence in _SENTENCE_SPLIT_RE.split(stem):
        cleaned = sentence.strip()
        if cleaned and _goal_sentence_key(cleaned) not in keys:
            kept.append(cleaned)
    return " ".join(kept).strip()


def _pending_goals_snapshot(state: Optional[dict]) -> Optional[list]:
    """None = uninitialized; [] = solved; otherwise remaining goal records."""
    pending = (state or {}).get("pending_goals")
    if pending is None or not isinstance(pending, list):
        return None
    return _goal_records(pending, (state or {}).get("goal_units") or {})


def _socratic_turn_raw(state: AgentState) -> str:
    message = state.get("turn_user_message")
    if message is None:
        return ""
    return _extract_text(getattr(message, "content", message) or "")


def _socratic_scope_id(state: AgentState) -> Optional[str]:
    """Stable id for the current Socratic problem. None if this turn has no stem."""
    anchor = str(state.get("canvas_anchor_id") or "").strip()
    if anchor:
        return f"anchor:{anchor}"
    extract = _isolate_problem_stem(state.get("extracted_text") or "")
    if extract and len(extract) >= 24:
        digest = hashlib.sha256(extract.encode("utf-8")).hexdigest()[:16]
        return f"stem:{digest}"
    query = _user_facing_query(state) or ""
    if _looks_like_new_socratic_stem(query):
        stem = _isolate_problem_stem(query) or query
        digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:16]
        return f"stem:{digest}"
    return None


def _looks_like_new_socratic_stem(text: str) -> bool:
    body = re.sub(r"\s+", " ", (text or "").strip())
    if not body:
        return False
    if _CHAT_ONLY_RE.match(body) or _CANVAS_EXPLAIN_RE.search(body):
        return False
    if _is_generic_query(body):
        return False
    labeled = bool(re.search(r"\([ab]\)", body, re.IGNORECASE))
    if len(body) < _NEW_SOCRATIC_STEM_MIN_CHARS and not labeled:
        return False
    return _looks_like_problem_stem(body) or len(body) >= _NEW_SOCRATIC_STEM_MIN_CHARS


def _is_image_file_path(path: Optional[str]) -> bool:
    lowered = (path or "").lower()
    return bool(lowered) and any(lowered.endswith(ext) for ext in IMAGE_EXTENSIONS)


def _question_key(value: str) -> str:
    match = re.search(r"\d{1,3}", str(value or ""))
    if not match:
        return ""
    number = int(match.group(0))
    if number <= 0 or number >= 500:
        return ""
    return str(number)


def _numbers_in_text(body: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\b(\d{1,3})\b", body or ""):
        key = _question_key(match.group(1))
        if key and key not in found:
            found.append(key)
    return found


def _names_pdf_chapter(body: str) -> bool:
    """A textbook task start names a chapter. Those turns stay on the PDF."""
    text = body or ""
    if _TASK_START_RE.search(text) and "resolve_chapter_target" in text:
        return True
    return bool(re.search(
        r"\b(?:chapter|fəsil|bölmə|section|ch\.)\s*\d",
        text,
        re.IGNORECASE,
    ))


def _pdf_held(state: dict) -> bool:
    nav = state.get("active_navigation") or {}
    if nav.get("page_start") or nav.get("source"):
        return True
    src = str((state.get("active_problem") or {}).get("source_file") or "")
    return bool(src) and not _is_image_file_path(src)


def _image_questions(state: dict) -> dict:
    active = state.get("active_image") or {}
    raw = active.get("questions") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if str(value).strip()}


_QUESTION_REF_RE = re.compile(
    r"(?:\b(?:question|problem|exercise|sual|məsələ|soru|вопрос|задач\w*)\b|\bq\s*\.?\s*\d|#\s*\d)",
    re.IGNORECASE,
)


def _image_followup_kind(state: dict, body: str) -> Optional[str]:
    """'cache' or 'vision' when this text turn should read the screenshot."""
    if _FORMULA_EXTRACT_RE.search(body or "") or _names_pdf_chapter(body):
        return None
    active = state.get("active_image") or {}
    if not str(active.get("path") or ""):
        return None
    if (
        _is_socratic_mode(state)
        and _socratic_has_open_problem(state, body)
        and not _QUESTION_REF_RE.search(body or "")
    ):
        # A numeric Socratic answer ("v = 4 m/s", "92") is not a question
        # number — with or without a canvas anchor, an open problem exists.
        print(
            "[ROUTER] socratic numeric reply — not an image question reference "
            f"anchor={bool(state.get('canvas_anchor_id'))} stem_len={len(_cached_image_stem(state))}",
            flush=True,
        )
        return None
    questions = _image_questions(state)
    hits = [num for num in _numbers_in_text(body) if num in questions]
    if len(hits) == 1:
        return "cache"
    if _pdf_held(state):
        return None
    if len(body or "") <= 180 and len(_numbers_in_text(body)) == 1:
        return "vision"
    return None


def _select_image_stem(questions: dict, body: str) -> str:
    hits = [num for num in _numbers_in_text(body) if num in questions]
    if len(hits) == 1:
        return questions[hits[0]]
    if not questions:
        return ""

    def sort_key(key: str) -> int:
        try:
            return int(key)
        except ValueError:
            return 10**9

    return questions[sorted(questions, key=sort_key)[0]]


def _selected_image_number(questions: dict, body: str) -> str:
    hits = [num for num in _numbers_in_text(body) if num in questions]
    if len(hits) == 1:
        return hits[0]
    if questions:
        def sort_key(key: str) -> int:
            try:
                return int(key)
            except ValueError:
                return 10**9
        return sorted(questions, key=sort_key)[0]
    nums = _numbers_in_text(body)
    return nums[0] if len(nums) == 1 else ""


def _active_image_payload(
    path: str,
    original_name: str,
    questions: dict,
    active_number: str = "",
    stem: str = "",
    previous: Optional[dict] = None,
) -> dict:
    """Rebuild the screenshot record. `previous` (the stored dict) is the source
    of truth for anything the new values do not supply, so a rebuild can never
    drop the stem, the question map, or the file info."""
    previous = previous or {}
    payload = {
        "path": path or str(previous.get("path") or ""),
        "original_name": original_name or str(previous.get("original_name") or ""),
        "questions": questions or previous.get("questions") or {},
    }
    if active_number:
        payload["active_number"] = active_number
    elif previous.get("active_number"):
        payload["active_number"] = previous["active_number"]
    # Fallback copy for screenshots whose number was never typed or
    # inventoried — `questions` alone would be empty for those.
    kept_stem = (stem or "").strip() or str(previous.get("stem") or "").strip()
    if kept_stem:
        payload["stem"] = kept_stem
    return payload


async def _store_image_figures(file_path: str, diagrams, config: Optional[RunnableConfig]) -> None:
    urls: list[str] = []
    if diagrams and config is not None:
        try:
            urls = await asyncio.to_thread(
                crop_image_diagrams_to_data_urls, file_path, diagrams
            )
        except Exception as e:
            print(f"[VISION] screenshot crop store failed: {e}", flush=True)
            urls = []
    if config is not None:
        store_vision_figure_urls(config, urls)


async def _inventory_image_page(
    file_path: str,
    body: str,
    original_name: str,
    config: Optional[RunnableConfig],
) -> tuple[str, dict]:
    nums = _numbers_in_text(body)
    marker = nums[0] if len(nums) == 1 else ""
    inventory = await vision_inventory_image_file(file_path, marker)
    questions: dict[str, str] = {}
    for item in inventory.questions or []:
        key = _question_key(getattr(item, "number", ""))
        stem = _peel_extract_wrapper((getattr(item, "stem", "") or "").strip())
        if key and stem and stem.upper() != "NO_PROBLEM_FOUND":
            questions[key] = stem
    stem = _select_image_stem(questions, body)
    if not stem:
        stem = await _vision_extract_image_stem(file_path, config, marker)
        if marker and stem:
            questions[marker] = stem
        number = _selected_image_number(questions, body) or marker
        print(f"[VISION] screenshot inventory empty — single extract stem_len={len(stem)}", flush=True)
        return stem, _active_image_payload(file_path, original_name, questions, number, stem)
    await _store_image_figures(file_path, inventory.diagrams, config)
    number = _selected_image_number(questions, body)
    print(
        f"[VISION] screenshot inventory n={len(questions)} selected_len={len(stem)} q={number}",
        flush=True,
    )
    return stem, _active_image_payload(file_path, original_name, questions, number, stem)


async def _resolve_cached_image_turn(
    state: dict,
    body: str,
    config: Optional[RunnableConfig],
) -> Optional[tuple[str, dict, bool]]:
    kind = _image_followup_kind(state, body)
    if not kind:
        return None
    active = state.get("active_image") or {}
    path = str(active.get("path") or "")
    original = str(active.get("original_name") or "")
    questions = dict(_image_questions(state))
    previous = str(active.get("active_number") or "")
    if kind == "cache":
        hits = [num for num in _numbers_in_text(body) if num in questions]
        selected = hits[0]
        stem = questions[selected]
        fresh = not previous or previous != selected
        print(
            f"[VISION] screenshot cache hit q={selected} prev={previous or '-'} "
            f"fresh={fresh} stem_len={len(stem)}",
            flush=True,
        )
        if config is not None:
            store_vision_figure_urls(config, [])
        return (
            stem,
            _active_image_payload(path, original, questions, selected, stem, previous=active),
            fresh,
        )
    marker = _numbers_in_text(body)[0]
    if not path or not os.path.isfile(path):
        print("[VISION] screenshot cache miss but the image file is gone", flush=True)
        return None
    stem = await _vision_extract_image_stem(path, config, marker)
    kept = _cached_image_stem(state)
    if not stem and kept:
        # The re-extract found nothing for this number: keep the stored
        # problem untouched instead of replacing it with an empty record.
        print(
            f"[VISION] screenshot cache miss q={marker} extract empty — keeping stored "
            f"stem_len={len(kept)} keys={sorted(active.keys())}",
            flush=True,
        )
        return kept, dict(active), False
    if stem:
        questions[marker] = stem
    fresh = bool(stem) and (not previous or previous != marker)
    print(
        f"[VISION] screenshot cache miss q={marker} prev={previous or '-'} "
        f"fresh={fresh} stem_len={len(stem)} keys={sorted(active.keys())}",
        flush=True,
    )
    return (
        stem,
        _active_image_payload(path, original, questions, marker, stem, previous=active),
        fresh,
    )


def _drop_image_file_annotations(messages: list) -> list:
    """Copy of the worker window with image `File path:` notes removed.
    Checkpoint messages stay unchanged so the current image turn can still
    read its own path."""
    cleaned = []
    for message in messages:
        if not isinstance(message, HumanMessage):
            cleaned.append(message)
            continue
        text = _extract_text(message.content)
        path = _extract_file_path(text)
        if not _is_image_file_path(path):
            cleaned.append(message)
            continue
        stripped = _FILE_ANNOTATION_RE.sub(" ", text).strip()
        cleaned.append(HumanMessage(content=stripped or text))
    return cleaned


def _normalize_stem_for_match(text: str) -> str:
    body = _isolate_problem_stem(_strip_extract_filler(text or "")) or (text or "")
    body = body.lower()
    body = re.sub(r"[^\w\s]+", " ", body, flags=re.UNICODE)
    return re.sub(r"\s+", " ", body).strip()


def _stems_are_same_problem(left: str, right: str, threshold: float = _SAME_PROBLEM_THRESHOLD) -> bool:
    """True when two isolated stems are the same problem (spelling/OCR jitter)."""
    a = _normalize_stem_for_match(left)
    b = _normalize_stem_for_match(right)
    if not a or not b:
        return False
    if a == b:
        return True
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if tokens_a and tokens_b:
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        if jaccard >= threshold:
            return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter in longer and (len(shorter) / max(len(longer), 1)) >= threshold:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _should_reset_problem_state(state: AgentState, raw_text: str) -> bool:
    """Fresh active_problem / navigation for a new intake — not for follow-ups."""
    if _CANVAS_EXPLAIN_RE.search(raw_text or ""):
        return False
    body = _strip_turn_wrappers(raw_text or "")
    if _TASK_START_RE.search(body):
        return False
    file_path = _extract_file_path(raw_text) or ""
    prev = state.get("active_problem") or {}
    prev_extract = (state.get("extracted_text") or "").strip()
    baseline = prev_extract or (prev.get("description") or "")

    if file_path and not _is_image_file_path(file_path):
        prev_src = str(prev.get("source_file") or "")
        return prev_src != file_path

    if _is_image_file_path(file_path):
        # A screenshot must not wipe the textbook cursor or problem header.
        return False

    if _looks_like_new_socratic_stem(body):
        if not baseline:
            return False
        return not _stems_are_same_problem(body, baseline)
    return False


def _reset_socratic_goals() -> dict:
    return {
        "pending_goals": None,
        "is_solved": False,
        "socratic_scope_id": None,
        "closed_goals": None,
        "goal_units": None,
        "last_socratic_step": None,
    }


def _socratic_goals_uninitialized(state: Optional[dict]) -> bool:
    pending = (state or {}).get("pending_goals")
    return pending is None and not bool((state or {}).get("is_solved"))


def _maybe_reset_socratic_goals(state: AgentState) -> dict:
    """Reset session todo when this turn is clearly a new problem."""
    if _socratic_goals_uninitialized(state):
        return {}
    raw = _socratic_turn_raw(state)
    if "File path:" in (raw or ""):
        return _reset_socratic_goals()
    new_scope = _socratic_scope_id(state)
    stored = str(state.get("socratic_scope_id") or "").strip() or None
    if stored and new_scope and new_scope != stored:
        return _reset_socratic_goals()
    if bool(state.get("is_solved")) and _looks_like_new_socratic_stem(
        _user_facing_query(state)
    ):
        return _reset_socratic_goals()
    return {}


def _planner_stem(state: AgentState) -> str:
    """Problem text the goal planner should read, or "" when none is available."""
    extract = _isolate_problem_stem(state.get("extracted_text") or "")
    if len(extract) >= 8:
        return extract
    cached = _cached_image_stem(state)
    if cached:
        return _isolate_problem_stem(cached) or cached
    typed = _user_facing_query(state) or ""
    if _looks_like_new_socratic_stem(typed):
        return _isolate_problem_stem(typed) or typed
    # Short one-step problems fall under the routing length threshold; the
    # planner still needs them (a digit plus an explicit ask, not a bare answer).
    short = re.sub(r"\s+", " ", typed.strip())
    if (
        len(short) >= 15
        and re.search(r"\d", short)
        and _PLANNER_ASK_CUE_RE.search(short)
        and not _CHAT_ONLY_RE.match(short)
        and not _is_generic_query(short)
        and not _CANVAS_EXPLAIN_RE.search(short)
    ):
        return short
    return ""


_PLANNER_ASK_CUE_RE = re.compile(
    r"\b(?:find|calculate|determine|compute|solve|what|how\s+(?:much|many|far|fast|long|high))\b|[?？]",
    re.IGNORECASE,
)
_GENERIC_GOAL = "the final answer the problem asks for"


def _fallback_goal_from_stem(stem: str) -> str:
    """Deterministic backstop: the last sentence that asks for something, else a generic target."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(stem or "") if s and s.strip()]
    for sentence in reversed(sentences):
        if _PLANNER_ASK_CUE_RE.search(sentence):
            return sentence.strip(" .")[:160]
    return _GENERIC_GOAL


async def _plan_socratic_goals(stem: str) -> Optional[tuple[list, dict]]:
    """Returns at least one stem-asked target for any non-empty stem: the
    structured planner, then one nudged retry, then a deterministic backstop.
    None only when there is no stem at all."""
    if not (stem or "").strip():
        return None
    base = [
        HumanMessage(content=SOCRATIC_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"[PROBLEM]:\n{stem}"),
    ]
    nudge = HumanMessage(
        content=(
            "Your previous answer had an empty `goals` list. `goals` must contain at "
            "least one item: a one-step problem still has exactly one asked target."
        )
    )
    for source, messages in (("planner", base), ("retry", [*base, nudge])):
        try:
            plan = await _ainvoke_llm(socratic_planner_llm, messages, "Socratic Planner")
        except Exception as e:
            print(f"[SOCRATIC PLANNER] {source} failed: {e}", flush=True)
            continue
        goals = _coerce_goal_list(getattr(plan, "goals", None))
        print(
            f"[SOCRATIC PLANNER] source={source} goals={goals!r} "
            f"actor={getattr(plan, 'asked_actor', '')!r:.60}",
            flush=True,
        )
        if goals:
            raw_units = getattr(plan, "units", None) or []
            raw_symbols = getattr(plan, "symbols", None) or []
            records = [
                _goal_record(
                    goal,
                    raw_symbols[index] if index < len(raw_symbols) else "",
                    raw_units[index] if index < len(raw_units) else "",
                    index,
                )
                for index, goal in enumerate(goals)
            ]
            units = {record["text"]: record["unit"] for record in records if record["unit"]}
            print(
                f"[SOCRATIC PLANNER] units={units!r} "
                f"symbols={[record['symbol'] for record in records]!r}",
                flush=True,
            )
            return records, units
    record = _goal_record(_fallback_goal_from_stem(stem), "", "", 0)
    print(f"[SOCRATIC PLANNER] source=heuristic goals={[record]!r}", flush=True)
    return [record], {}


def _goal_state_block(state: AgentState) -> str:
    pending = _pending_goals_snapshot(state)
    solved = bool(state.get("is_solved")) or pending == []
    if solved:
        return (
            "[SOCRATIC GOAL STATE]: SOLVED (free-form tutor).\n"
            "All explicit goals are complete. Answer conceptual follow-ups in chat only. "
            "Do NOT call `python_code_executor`. Do NOT emit canvas steps or `Result found:`. "
            "A brand-new problem will reset this state."
        )
    if pending is None:
        # The planner normally guarantees goals; this is the safe baseline when
        # it could not run. No tag duty: the tutor must not guess or invent.
        baseline = (
            "Goals are unavailable this turn. Use ONLY the numbers and wording in "
            "[EXTRACTED DOCUMENT CONTEXT] (or the student's message); never invent "
            "or assume a value that is not written there. Ask exactly one short "
            "guiding question about the quantity the problem asks for.\n"
        )
        if state.get("canvas_anchor_id"):
            return (
                "[SOCRATIC GOAL STATE]: UNINITIALIZED (follow-up).\n"
                + baseline
                + "If this reply is only a step, ask one question and do NOT call "
                "`python_code_executor`.\n"
                "If you accept the student's final equation, call "
                "`python_code_executor` ONCE with type \"calculation\" and that "
                "equation in `steps`. Python writes the Result only when a planned "
                "goal symbol matches. Do not set `complete_goal` for a guess."
            )
        return (
            "[SOCRATIC GOAL STATE]: UNINITIALIZED.\n"
            + baseline
            + "Do NOT call `python_code_executor` this turn."
        )
    units = state.get("goal_units") or {}

    def _goal_line(index: int, goal) -> str:
        text = _goal_text(goal)
        symbol = _goal_symbol(goal)
        unit = _goal_unit(goal, units)
        line = f"  [{index}] {text}"
        if symbol:
            line += f"  (report as: ${symbol}$)"
        if unit:
            line += f"  (report in: {unit})"
        return line

    indexed = "\n".join(_goal_line(i, goal) for i, goal in enumerate(pending))
    closed = [
        _goal_text(goal)
        for goal in _goal_records(state.get("closed_goals"), units)
    ]
    closed_line = (
        f"closed_goals (already done, include in the final `value`) = "
        f"{json.dumps(closed, ensure_ascii=False)}\n"
        if closed else ""
    )
    focus = _goal_text(pending[0])
    return (
        "[SOCRATIC GOAL STATE]: ACTIVE.\n"
        f"pending_goals (0-based index):\n{indexed}\n"
        f"{closed_line}"
        "is_solved = false\n"
        f"Default focus: pending_goals[0] = {focus}\n"
        "Goals are independent: the student may solve them in any order. When you "
        "accept a step, call the tool once and put that goal's report symbol alone "
        "on the left of the equation ($a = ...$, not $v_x$ for a goal bound to $v$). "
        "Python closes the goal when that equation is its value or its full "
        "substitution, and only when exactly one open goal matches. A setup step "
        "that does not do that stays open. `complete_goal` is a hint, not the close.\n"
        "Do NOT set `value` or `Result found:` while any goal will remain open. "
        "When the equation closes the LAST open goal, Python writes the Result on "
        "that same call — the Result names EVERY original stem-asked target. "
        "GOAL SYNC: Chat/`summary` MUST contain EXACTLY ONE `?` aimed ONLY at the "
        "first goal still open after this turn. Do not ask about any other goal "
        "unless this same call's equation closes the ones before it."
    )


def _parse_goals_tag(text: str) -> tuple[Optional[list], str]:
    """Pull `[[SOCRATIC_GOALS]]...[[/SOCRATIC_GOALS]]` out of chat so students never see it."""
    body = _SOCRATIC_GOALS_FENCE_RE.sub(lambda m: m.group(1) or "", text or "")
    goals = None
    match = _SOCRATIC_GOALS_TAG_RE.search(body)
    if match:
        raw = (match.group(1) or "").strip()
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        goals = _coerce_goal_list(parsed)
    else:
        # Unclosed opener: still recover the array that follows it.
        open_at = _SOCRATIC_GOALS_OPEN_RE.search(body)
        array = _GOALS_ARRAY_RE.match(body, open_at.end()) if open_at else None
        if array:
            try:
                goals = _coerce_goal_list(json.loads(array.group(0)))
            except (TypeError, ValueError, json.JSONDecodeError):
                goals = None
    return goals, _strip_dependency_tree(strip_protocol_tags(body, "parse_goals_tag"))


def _has_protocol_marker(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _PROTOCOL_MARKERS) or bool(
        _PROTOCOL_TAG_ANY_RE.search(text or "")
    )


def strip_protocol_tags(text: str, where: str = "") -> str:
    """Remove model-internal protocol tags (`[[SOCRATIC_GOALS]]`, `<dependency_tree>`,
    `<scratchpad>`, `<think>`) from student-facing text. Returns `text` untouched
    when none is present, so LaTeX, Markdown, and chunk whitespace are never altered."""
    body = text or ""
    if not _has_protocol_marker(body):
        return body
    original = body
    body = _SOCRATIC_GOALS_FENCE_RE.sub(lambda m: m.group(1) or "", body)
    body = _SOCRATIC_GOALS_TAG_RE.sub("", body)
    open_at = _SOCRATIC_GOALS_OPEN_RE.search(body)
    while open_at:
        end = open_at.end()
        array = _GOALS_ARRAY_RE.match(body, end)
        if array:
            end = array.end()
        body = body[: open_at.start()] + body[end:]
        open_at = _SOCRATIC_GOALS_OPEN_RE.search(body)
    body = _SOCRATIC_GOALS_CLOSE_RE.sub("", body)
    for fence_re, closed_re, opener_re, closer_re in _HIDDEN_BLOCK_RES.values():
        body = fence_re.sub("", body)
        body = closed_re.sub("", body)
        open_at = opener_re.search(body)
        if open_at:
            body = body[: open_at.start()]
        body = closer_re.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    if body != original:
        print(
            f"[SANITIZE] {where or 'text'}: stripped protocol tags "
            f"({len(original)}->{len(body)}) preview={original[:40]!r}",
            flush=True,
        )
    return body


def _strip_dependency_tree(text: str) -> str:
    """Remove hidden protocol blocks from student-facing text and trim."""
    return re.sub(r"\n{3,}", "\n\n", strip_protocol_tags(text, "strip_tree")).strip()


def _clean_ai_message(message):
    """Copy of an AI message with protocol tags removed from its text content;
    tool_calls and every other field are kept."""
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not _has_protocol_marker(content):
        return message
    cleaned = _strip_dependency_tree(
        _SOCRATIC_GOALS_FENCE_RE.sub(lambda m: m.group(1) or "", content)
    )
    try:
        return message.model_copy(update={"content": cleaned})
    except Exception:
        return message


def _strip_dependency_tree_from_payload(payload: dict) -> dict:
    """Drop hidden trees from canvas `summary` / `steps` / `value`."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    for key in ("summary", "value"):
        if isinstance(out.get(key), str):
            out[key] = _strip_dependency_tree(out[key])
    steps = out.get("steps")
    if isinstance(steps, list):
        out["steps"] = [
            _strip_dependency_tree(step) if isinstance(step, str) else step
            for step in steps
        ]
    return out


def _collapse_repeated_chat(text: str) -> str:
    """If the same paragraph was concatenated 2–4 times, keep a single copy."""
    body = (text or "").strip()
    if not body:
        return ""
    blocks = [part.strip() for part in re.split(r"\n{2,}", body) if part.strip()]
    if len(blocks) >= 2 and all(block == blocks[0] for block in blocks):
        return blocks[0]
    flat = re.sub(r"\s+", " ", body).strip()
    for copies in (4, 3, 2):
        if len(flat) < copies * 48:
            continue
        unit_len = len(flat) // copies
        unit = flat[:unit_len].strip()
        if len(unit) < 48:
            continue
        if " ".join([unit] * copies) == flat:
            orig = blocks[0] if blocks and len(blocks[0]) >= 48 else unit
            return orig
    return body


def _apply_pending_goals_init(current: Optional[list], goals) -> Optional[list]:
    if current is not None:
        return current
    if goals and isinstance(goals[0], dict):
        return _goal_records(goals)
    texts = _coerce_goal_list(goals)
    if not texts:
        return None
    return [_goal_record(text, index=index) for index, text in enumerate(texts)]


def _complete_active_goal(pending: list, count: int = 1) -> tuple[list, bool]:
    """Pop `count` consecutive goals off the front. `count` > 1 is the
    already-stated-value scan's Parrot-Loop fix: the SAME student message
    validly answered [0] AND [1] (etc.) together, so both pop in one turn
    instead of re-asking for a number already given."""
    n = max(1, min(int(count or 1), len(pending))) if pending else 0
    rest = list(pending[n:]) if pending else []
    return rest, len(rest) == 0


def _goal_indices(raw, size: int) -> list:
    """Valid, unique 0-based indices from the tool's `completed_goal_indices`."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    seen: list = []
    for item in raw:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < size and idx not in seen:
            seen.append(idx)
    return seen


def _close_goals(pending: list, count: int, indices: list) -> tuple[list, list]:
    """Close the first `count` goals plus the goals at `indices` (indices refer
    to `pending` as it was BEFORE this call, in any order). Returns (rest, closed)."""
    n = max(0, min(int(count or 0), len(pending)))
    drop = set(range(n)) | set(indices)
    rest = [goal for i, goal in enumerate(pending) if i not in drop]
    closed = [goal for i, goal in enumerate(pending) if i in drop]
    return rest, closed


def _complete_goal_count(value) -> int:
    """Normalizes `complete_goal` — usually a bool (pop just [0]); may also
    be an int/numeral string when multiple consecutive goals were resolved
    in the same student message."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "yes"}:
            return 1
        if s in {"false", "no", ""}:
            return 0
        if s.lstrip("-").isdigit():
            return max(0, int(s))
    return 0


def _next_goal_question(goal) -> str:
    """Avoid 'What is Which motorcycle…?' when the remaining goal is already a question."""
    hint = _goal_text(goal)
    if not hint:
        return "What is the next step?"
    if _GOAL_ALREADY_QUESTION_RE.search(hint):
        if hint.endswith("?") or hint.endswith("？"):
            return hint
        return f"{hint}?"
    return f"What is {hint}?"


_SYNC_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_SYNC_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SYNC_STOPWORDS = frozenset({
    "the", "what", "which", "find", "how", "and", "for", "that", "with", "this",
    "its", "are", "was", "does", "value", "much", "many", "magnitude", "now",
    "next", "can", "you", "your", "from", "when", "where", "then", "will",
})


def _sync_numbers(text: str) -> set:
    out = set()
    for raw in _SYNC_NUM_RE.findall(text or ""):
        try:
            out.add(f"{float(raw.replace(',', '.')):g}")
        except ValueError:
            continue
    return out


def _sync_words(text: str) -> set:
    return {w.lower() for w in _SYNC_WORD_RE.findall(text or "")} - _SYNC_STOPWORDS


def _split_last_question(text: str) -> tuple[str, str]:
    """(text before the last question sentence, that sentence); ("", "") if none."""
    body = (text or "").rstrip()
    end = max(body.rfind("?"), body.rfind("？"))
    if end < 0:
        return "", ""
    head = body[:end]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "), head.rfind("？"), head.rfind("\n"))
    return body[: cut + 1].rstrip() if cut >= 0 else "", body[cut + 1: end + 1].strip()


def _reply_targets_other_goal(reply: str, pending: Optional[list]) -> Optional[int]:
    """Index of a LATER pending goal that the reply's question clearly asks about
    (and not goal [0]), else None. Deterministic word/number overlap, no LLM.
    Ambiguous replies return None so today's behaviour is unchanged."""
    if not pending or len(pending) < 2:
        return None
    _, question = _split_last_question(reply)
    if not question:
        return None
    q_words, q_nums = _sync_words(question), _sync_numbers(question)
    scores = []
    for goal in pending:
        text = _goal_text(goal)
        g_words, g_nums = _sync_words(text), _sync_numbers(text)
        word_part = len(q_words & g_words) / max(1, len(g_words))
        num_part = (len(q_nums & g_nums) / len(g_nums)) if g_nums else 0.0
        if g_nums and q_nums and not (q_nums & g_nums):
            num_part -= 0.5
        scores.append(word_part + 2.0 * num_part)
    best = max(range(len(scores)), key=lambda i: scores[i])
    if best > 0 and scores[best] >= 1.0 and scores[best] > scores[0] + 0.5:
        print(
            f"[SYNC] reply targets goal {best} not 0 scores={[round(s, 2) for s in scores]} "
            f"question={question[:80]!r}",
            flush=True,
        )
        return best
    return None


def _replace_question_sentence(text: str, goal: Optional[str]) -> str:
    """Keep the acknowledgement, swap the last question for the first open goal's."""
    prefix, _ = _split_last_question(text)
    return f"{prefix} {_next_goal_question(goal)}".strip()


def _fix_summary_goal_target(payload: dict, pending: list) -> dict:
    """Tool path: the card's guiding question must ask the first open goal."""
    summary = payload.get("summary")
    if not isinstance(summary, str) or _reply_targets_other_goal(summary, pending) is None:
        return payload
    print("[SYNC] tool summary rewritten to the first open goal", flush=True)
    return {**payload, "summary": _replace_question_sentence(summary, pending[0])}


def _strip_premature_close(payload: dict, remaining_goal: Optional[str] = None) -> dict:
    """Drop `value` / `Result found:` so `_normalize_socratic_calculation` cannot close."""
    out = dict(payload)
    out.pop("value", None)
    summary = out.get("summary")
    summary = summary.strip() if isinstance(summary, str) else ""
    if _RESULT_FOUND_RE.match(summary):
        summary = _RESULT_FOUND_RE.sub("", summary, count=1).strip()
    question = _next_goal_question(remaining_goal) if remaining_goal else ""
    if summary and question and not _summary_has_question(summary):
        summary = f"{summary} {question}"
    elif not summary and question:
        summary = question
    out["summary"] = summary
    out["complete_goal"] = False
    return out


def _socratic_visible_reply(
    prose: str,
    summary: str,
    *,
    solved: bool,
    pending: Optional[list],
) -> str:
    """Chat keeps the model's own sentence. A manufactured Result line and a
    bare "What is the next step?" never replace that sentence."""
    text = (prose or "").strip()
    card = (summary or "").strip()
    if _RESULT_FOUND_RE.match(card):
        card = _RESULT_FOUND_RE.sub("", card, count=1).strip()
    if not text:
        text = card
    if solved and re.fullmatch(r"\$[^$]+\$\.?", text.strip()):
        text = ""
    if solved:
        return text or _SOCRATIC_SOLVED_CHAT_FALLBACK
    if not text:
        return ""
    if _summary_has_question(text):
        return text
    goal = pending[0] if pending else None
    if not goal:
        return text
    return f"{text} {_next_goal_question(goal)}".strip()


def _init_socratic_goals_update(state: AgentState, goals: Optional[list]) -> dict:
    if not goals:
        return {}
    if _pending_goals_snapshot(state) is not None:
        return {}
    update = {
        "pending_goals": goals,
        "is_solved": False,
    }
    scope = _socratic_scope_id(state) or state.get("socratic_scope_id")
    if scope:
        update["socratic_scope_id"] = scope
    return update


def _payload_declares_result(payload: dict) -> bool:
    """True when the model itself closed the problem on this tool call."""
    value = payload.get("value")
    if isinstance(value, str) and value.strip():
        return True
    summary = payload.get("summary")
    return isinstance(summary, str) and bool(_RESULT_FOUND_RE.match(summary.strip()))


def _apply_socratic_goal_payload(
    state: AgentState,
    payload: dict,
    replace_last: bool,
) -> tuple[dict, dict]:
    """Confirm at most one bound goal from an accepted equation, then make
    the question or the Result match that commit. Flags do not close a goal."""
    out = dict(payload)
    updates: dict = {}
    pending = _pending_goals_snapshot(state)
    initialized = _apply_pending_goals_init(pending, out.get("goals"))
    if pending is None and initialized:
        pending = initialized
        updates.update(_init_socratic_goals_update(state, pending))

    is_calc = out.get("type") == "calculation" and bool(pending)
    hint = False
    if is_calc and pending:
        hint = _complete_goal_count(out.get("complete_goal")) > 0 or bool(
            _goal_indices(out.get("completed_goal_indices"), len(pending))
        )
    out.pop("completed_goal_indices", None)
    matched_value = ""
    if is_calc and pending:
        confirmed = _confirm_goal_close(payload, pending, state)
        if confirmed is None:
            if hint:
                print("[COMMIT] hint ignored — no unique symbol match", flush=True)
        else:
            index, matched_value = confirmed
            goal = pending[index]
            previous = _goal_records(
                state.get("closed_goals"), state.get("goal_units") or {}
            )
            already = {_goal_id(item) for item in previous if _goal_id(item)}
            goal_id = _goal_id(goal)
            if goal_id and goal_id in already:
                reason = "repeated card" if replace_last else "already closed"
                print(f"[COMMIT] goal {goal_id} {reason} — not popped again", flush=True)
                matched_value = ""
            else:
                before = list(pending)
                pending, closed = _close_goals(before, 0, [index])
                solved = not pending
                updates["pending_goals"] = pending
                updates["is_solved"] = solved
                updates["closed_goals"] = [*previous, *closed]
                print(
                    f"[SOCRATIC GOALS] closed={[_goal_text(item) for item in closed]!r} "
                    f"symbol={_goal_symbol(goal)!r} remaining={[_goal_text(item) for item in pending]!r}",
                    flush=True,
                )
                if not hint:
                    print("[COMMIT] closed without a complete_goal hint", flush=True)
                if solved:
                    out["value"] = matched_value or (
                        out.get("value") if isinstance(out.get("value"), str) else ""
                    )

    if pending:
        out = _fix_summary_goal_target(out, pending)
        out = _strip_premature_close(out, _goal_text(pending[0]))
    elif pending is None:
        out = _strip_premature_close(out)
    else:
        if "replace_last" in out:
            out = {key: value for key, value in out.items() if key != "replace_last"}
        current = out.get("value") if isinstance(out.get("value"), str) else ""
        if matched_value:
            out["value"] = matched_value
        elif "=" not in current:
            out.pop("value", None)
        updates["pending_goals"] = []
        updates["is_solved"] = True
    return out, updates


def _peel_extract_wrapper(text: str) -> str:
    """Strip OCR/doc_worker preambles glued onto a real stem (`The problem is:`)."""
    body = (text or "").strip()
    if not body:
        return ""
    for _ in range(3):
        peeled = _EXTRACT_WRAPPER_RE.sub("", body, count=1).strip()
        if peeled == body:
            break
        body = peeled
    return body


def _looks_like_problem_stem(sentence: str) -> bool:
    text = (sentence or "").strip()
    if not text:
        return False
    if (
        _EXTRACT_META_SENTENCE_RE.match(text)
        or _EXTRACT_FILLER_LINE_RE.match(text)
        or _is_process_narration_line(text)
    ):
        return False
    if re.match(r"^\d+[.)]\s+\S", text):
        return True
    if len(text) >= 24:
        return True
    return len(text) >= 8


def _strip_extract_filler(text: str) -> str:
    """Drop conversational preamble so the canvas shows only the problem stem.

    Sentence-level (not line-level) so a filler clause glued onto the stem
    cannot wipe the whole extract.
    """
    if not text:
        return ""
    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[OBJECTIVE]"):
            continue
        kept_lines.append(line)
    body = _peel_extract_wrapper("\n".join(kept_lines).strip())
    body = _EXTRACT_FILLER_PREFIX_RE.sub("", body).strip()
    body = _peel_extract_wrapper(body)
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(body) if part.strip()]
    start = next((i for i, sentence in enumerate(sentences) if _looks_like_problem_stem(sentence)), None)
    if start is not None:
        return " ".join(sentences[start:]).strip()
    return body


def _parse_objective_line(text: str) -> tuple[Optional[str], str]:
    """Splits a `[OBJECTIVE]: ...` prefix off doc_worker output."""
    if not text:
        return None, ""
    match = _OBJECTIVE_RE.search(text)
    objective = match.group(1).strip() if match else None
    cleaned = _OBJECTIVE_RE.sub("", text, count=1).strip() if match else text
    return objective or None, cleaned


def _infer_objective_from_extract(text: str) -> str:
    """Fallback 1-sentence objective when the student gave no real prompt."""
    if not text:
        return ""
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("[OBJECTIVE]"):
            continue
        if re.match(r"^(səhifə|page|tapıldı)\b", stripped, re.IGNORECASE):
            continue
        lines.append(stripped)
    blob = " ".join(lines)
    blob = re.sub(r"\s+", " ", blob).strip()
    if not blob:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", blob, maxsplit=1)[0]
    if len(sentence) > 120:
        return sentence[:117].rstrip() + "..."
    return sentence


def _subtitle_from_query(
    query: str,
    topic: Optional[str] = None,
    inferred: Optional[str] = None,
) -> str:
    """Prefer [OBJECTIVE] / extract summary / chapter topic when the user
    only said 'həll et'. Never copy a generic prompt or `[TASK START]`
    routing blob into the subtitle."""
    fallback = (inferred or topic or _DEFAULT_PROBLEM_SUBTITLE).strip()
    if _looks_like_internal_prompt(fallback):
        fallback = (topic or _DEFAULT_PROBLEM_SUBTITLE).strip()
        if _looks_like_internal_prompt(fallback):
            fallback = _DEFAULT_PROBLEM_SUBTITLE
    fallback = fallback[:200] if fallback else _DEFAULT_PROBLEM_SUBTITLE
    if _is_generic_query(query) or _looks_like_internal_prompt(query):
        return fallback
    cleaned = (query or "").strip()
    if topic and _SOLVEISH_RE.search(cleaned) and len(cleaned.split()) <= 14:
        return (topic or fallback)[:200]
    return cleaned[:200] if cleaned else fallback


def _doc_worker_finish_update(
    state: AgentState,
    reply_text: str,
    extra: Optional[dict] = None,
    config: Optional[RunnableConfig] = None,
    *,
    typed_input: bool = False,
    turn_file_path: Optional[str] = None,
) -> dict:
    update: dict = dict(extra or {})
    update["extracted_text"] = reply_text
    update["chat_reply"] = reply_text
    update["completed_workers"] = (state.get("completed_workers") or []) + ["doc_worker"]
    update["worker_queue"] = _pop_worker_queue(state)

    is_image_turn = _is_image_file_path(turn_file_path)
    if typed_input:
        file_path = None
        original_name = None
        previous: dict = {}
    elif is_image_turn:
        file_path = turn_file_path
        original_name = _find_latest_original_filename(state["messages"])
        previous = {}
    else:
        file_path = turn_file_path or _find_latest_uploaded_file_path(state["messages"])
        original_name = _find_latest_original_filename(state["messages"])
        previous = state.get("active_problem") or {}

    objective, cleaned_reply = _parse_objective_line(reply_text)
    if cleaned_reply:
        update["extracted_text"] = cleaned_reply
        update["chat_reply"] = cleaned_reply
        reply_text = cleaned_reply

    # Clean the STORED extract at the source, not only when composer_node
    # renders the canvas question card. `extracted_text` is what
    # `math_worker_node` reads back as `[EXTRACTED DOCUMENT CONTEXT]` — RAG
    # `---` coach segments and filler prose must not reach the solver.
    # `chat_reply` (read-only doc_worker turns) stays the original reply.
    isolated_stem = _isolate_problem_stem(_strip_extract_filler(update["extracted_text"]))
    if isolated_stem and not _looks_like_internal_prompt(isolated_stem):
        update["extracted_text"] = isolated_stem

    navigation_update = None if (typed_input or is_image_turn) else (
        update.get("navigation_update") or state.get("navigation_update")
    )
    if navigation_update:
        prev_nav = state.get("active_navigation") or {}
        update["active_navigation"] = {
            "source": file_path or prev_nav.get("source"),
            "page_start": navigation_update.get("page_start", prev_nav.get("page_start")),
            "page_end": navigation_update.get("page_end", prev_nav.get("page_end")),
            "last_question_marker": (
                navigation_update.get("last_question_marker")
                or prev_nav.get("last_question_marker")
            ),
            "section_kind": navigation_update.get("section_kind") or prev_nav.get("section_kind"),
            "section_title": navigation_update.get("section_title") or prev_nav.get("section_title"),
        }

    if typed_input:
        navigation: dict = {}
        update["active_navigation"] = None
        source = _TYPED_PROBLEM_SOURCE
        chapter_label, topic = None, None
    elif is_image_turn:
        navigation = {}
        source = _source_display_name(original_name or file_path, "Screenshot")
        chapter_label, topic = None, None
    else:
        navigation = (
            update.get("active_navigation")
            or navigation_update
            or state.get("active_navigation")
            or {}
        )
        page_start = navigation.get("page_start")
        chapter_raw = lookup_outline_chapter(config, page_start) if config else None
        chapter_label, topic = _split_chapter_title(chapter_raw)
        source = _source_display_name(
            original_name or previous.get("source_file") or file_path,
            previous.get("source"),
        )
    nav_problem = _problem_label(navigation.get("last_question_marker"))
    file_problem = _problem_from_filename(original_name or previous.get("source_file"))
    title = _compose_problem_title(source, chapter_label, nav_problem or file_problem)
    inferred = objective or _infer_objective_from_extract(reply_text)
    prior_description = previous.get("description") or ""
    if _looks_like_internal_prompt(prior_description):
        prior_description = ""
    description = _subtitle_from_query(
        _user_facing_query(state),
        topic or previous.get("topic"),
        inferred or prior_description,
    )
    if _looks_like_internal_prompt(title):
        title = _compose_problem_title(source, None, nav_problem or file_problem)
    if _looks_like_internal_prompt(description):
        description = (inferred or topic or _DEFAULT_PROBLEM_SUBTITLE)[:200]
        if _looks_like_internal_prompt(description):
            description = _DEFAULT_PROBLEM_SUBTITLE
    pdf_held = is_image_turn and _pdf_held(state)
    if not pdf_held and (typed_input or file_path or previous or navigation):
        update["active_problem"] = {
            "source": source,
            "source_file": None if (typed_input or is_image_turn) else (
                original_name or file_path or previous.get("source_file")
            ),
            "title": title,
            "description": description,
            "topic": topic or inferred or previous.get("topic"),
            "image_url": None,
        }

    return update


async def _force_extract_short_document(config: RunnableConfig) -> str:
    """Last-resort page dump when a short-doc turn is force-quit without a
    successful `read_page_range` — one backend call, no extra LLM round."""
    n = get_indexed_page_count(config)
    if n <= 0:
        return ""
    try:
        return str(await read_page_range.ainvoke(
            {"start_page": 1, "end_page": n},
            config=config,
        ))
    except Exception as e:
        print(f"[ANTI-LOOP] Auto read_page_range failed: {e}", flush=True)
        return ""


async def _invoke_worker_tool_call(tool_call: dict, tools_map: dict, config: RunnableConfig) -> ToolMessage:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id = tool_call.get("id")
    if not tool_id:
        tool_id = f"call_{tool_name}_{uuid.uuid4().hex[:12]}"
        tool_call["id"] = tool_id

    print(f"[TIMING] Tool '{tool_name}' Started...", flush=True)
    t0 = time.time()

    if tool_name in tools_map:
        # Forwarding `config` here is what makes the Phase 1 thread-scoped
        # vector store isolation in tools.py actually take effect: it's how
        # `process_and_index_documents`/`search_in_document` recover the
        # current run's `thread_id` at call time. Without this, both tools
        # would silently fall back to the shared default bucket regardless
        # of which conversation thread is running.
        result = await tools_map[tool_name].ainvoke(tool_args, config=config)
    else:
        result = f"Xəta: {tool_name} adında alət tapılmadı."

    print(f"[TIMING] Tool '{tool_name}' Ended - Took {time.time() - t0:.2f}s", flush=True)

    return ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)


def _router_user_payload(raw_text: str) -> str:
    """Compact router input: strip study-mode + tempfile, keep filename flag.
    The old last-4-messages window (including extracted textbook pages) is
    what ballooned the Router call to ~48s."""
    orig = _extract_original_filename(raw_text)
    has_file = "File path:" in (raw_text or "")
    body = _strip_turn_wrappers(raw_text or "")
    prefix = f"[uploaded_file={orig or 'document'}] " if has_file else ""
    payload = prefix + body
    if len(payload) > _ROUTER_INPUT_MAX_CHARS:
        return payload[:_ROUTER_INPUT_MAX_CHARS] + "…"
    return payload


def _heuristic_route(raw_text: str) -> Optional[list[str]]:
    """Skip the Router LLM entirely for obvious turns (PDF+həll et, greetings).
    Returns None when the cheap classifier should still run."""
    file_path = (_extract_file_path(raw_text) or "").lower()
    is_image = bool(file_path) and any(file_path.endswith(ext) for ext in IMAGE_EXTENSIONS)

    body = _strip_turn_wrappers(raw_text or "")
    has_file = "File path:" in (raw_text or "")

    if _FORMULA_EXTRACT_RE.search(body):
        return ["doc_worker"]

    if is_image:
        # Screenshot extraction contract: a math/physics screenshot gets the
        # SAME doc_worker -> math_worker chain as a PDF (doc_worker vision-
        # extracts the stem into `extracted_text` first). Casual/read-only
        # image turns ("what does this say", chit-chat) stay on chat_worker.
        if (_CHAT_ONLY_RE.match(body) and body) or _READ_ONLY_RE.search(body):
            return ["chat_worker"]
        return ["doc_worker", "math_worker"]

    if _CHAT_ONLY_RE.match(body) and not has_file:
        return ["chat_worker"]

    if _TASK_START_RE.search(body):
        return ["doc_worker", "math_worker"]

    if _ASSIGN_ONLY_RE.search(body) and not _TASK_START_RE.search(body):
        return ["chat_worker"]

    if _WEB_INTENT_RE.search(body) and not has_file and not _DOC_REF_RE.search(body):
        return ["web_worker"]

    if has_file:
        if _READ_ONLY_RE.search(body) and not _is_generic_query(body):
            return ["doc_worker"]
        if _is_generic_query(body):
            page_count = count_pdf_pages(_extract_file_path(raw_text) or "")
            if page_count == 0 or page_count > EMPTY_PROMPT_AUTOSOLVE_MAX_PAGES:
                print(
                    f"[ROUTER] empty/generic prompt + {page_count or 'unknown'}-page "
                    "document — clarification only (no auto-solve).",
                    flush=True,
                )
                return ["chat_worker"]
            return ["doc_worker", "math_worker"]
        return ["doc_worker", "math_worker"]

    if _DOC_REF_RE.search(body):
        if _READ_ONLY_RE.search(body) and not _is_generic_query(body):
            return ["doc_worker"]
        if _is_generic_query(body) or _SOLVEISH_RE.search(body):
            return ["doc_worker", "math_worker"]
        return ["doc_worker"]

    if _looks_like_new_socratic_stem(body):
        return ["doc_worker", "math_worker"]

    return None


def _socratic_has_open_problem(state: dict, body: str) -> bool:
    """A canvas anchor, or an unsolved screenshot stem when the anchor was lost."""
    if state.get("canvas_anchor_id"):
        return True
    return bool(
        _cached_image_stem(state)
        and not state.get("is_solved")
        and not _looks_like_new_socratic_stem(body)
    )


def _ensure_socratic_math_worker(state: dict, raw_text: str, workers: list[str]) -> list[str]:
    """Socratic problem turns must reach math_worker so a later *correct*
    step can be formalized onto the canvas. Opening/struggle turns still
    land here but must not call the tool. Greetings stay on chat_worker."""
    if not _is_socratic_mode(state, raw_text):
        return workers
    body = _strip_turn_wrappers(raw_text or "")
    if _maybe_language_only(body) and workers == ["chat_worker"]:
        return workers
    if _CHAT_ONLY_RE.match(body):
        return workers
    if "math_worker" in workers:
        return workers
    if (
        not _socratic_has_open_problem(state, body)
        and "File path:" not in (raw_text or "")
    ):
        return workers
    if "doc_worker" in workers:
        return [*workers, "math_worker"]
    return ["math_worker"]


def _is_provider_overload_error(err: BaseException) -> bool:
    if isinstance(err, ProviderUnavailableError):
        return True
    text = str(err).lower()
    markers = (
        "503", "429", "500", "unavailable", "high demand", "overloaded",
        "resource exhausted", "rate limit", "temporarily", "deadline exceeded",
        "timeout", "service unavailable", "internal error",
    )
    return any(marker in text for marker in markers)


def user_facing_llm_error(err: BaseException) -> str:
    """SSE-safe copy — never leak Gemini's raw 503 traceback to the student."""
    if _is_provider_overload_error(err):
        return PROVIDER_OVERLOAD_MESSAGE
    return "Xəta baş verdi. Zəhmət olmasa bir az sonra yenidən cəhd edin."


def _cached_image_stem(state: dict) -> str:
    """Stem already read from the screenshot. Prefer the active number, else the only one."""
    image = state.get("active_image") or {}
    questions = _image_questions(state)
    active = str(image.get("active_number") or "")
    if active and questions.get(active):
        return questions[active]
    if len(questions) == 1:
        return next(iter(questions.values()))
    return str(image.get("stem") or "").strip()


def _stem_for_turn(
    previous: str,
    replacement: Optional[str],
    *,
    new_document: bool,
) -> Optional[str]:
    """Follow-ups keep the stored problem. A new problem clears it. A non-empty
    replacement (the screenshot cache) wins over the stored stem."""
    fresh = (replacement or "").strip()
    if new_document:
        kept = fresh or None
    else:
        kept = fresh or (previous or "").strip() or None
    if kept:
        print(f"[ROUTER] stem kept len={len(kept)}", flush=True)
    else:
        print("[ROUTER] stem cleared", flush=True)
    return kept


def _router_dispatch_command(
    workers: list[str],
    turn_user_message,
    *,
    new_document: bool = False,
    previous_extract: str = "",
    extracted_text: Optional[str] = None,
) -> Command:
    requested = workers or ["chat_worker"]
    first_worker, remaining_queue = requested[0], requested[1:]
    update: dict = {
        "planned_workers": requested,
        "turn_user_message": turn_user_message,
        "worker_queue": remaining_queue,
        "completed_workers": [],
        "extracted_text": _stem_for_turn(
            previous_extract, extracted_text, new_document=new_document
        ),
        "desk_payload": None,
        "desk_payloads": None,
        "canvas_ops": [],
        "chat_reply": None,
        "navigation_update": None,
        "tool_round_trips": {},
        "fresh_intake": bool(new_document),
        "fresh_question_card": False,
    }
    if new_document:
        update["active_navigation"] = None
        update["active_problem"] = None
    return Command(
        update=update,
        goto=WORKER_NODE_MAP[first_worker],
    )


def _provider_fail_command(state: AgentState, worker: str, err: BaseException) -> Command:
    """Finish the current worker with a clean user-facing message so a 503
    after retries AND DeepSeek/Gemini failover still reaches `composer_node`
    instead of unwinding the SSE generator with a traceback."""
    return Command(
        update={
            "chat_reply": user_facing_llm_error(err),
            "completed_workers": (state.get("completed_workers") or []) + [worker],
            "worker_queue": _pop_worker_queue(state),
        },
        goto=_next_worker_goto(state, worker),
    )


async def _ainvoke_llm(llm, messages, label: str):
    """Single invoke path for every worker/router LLM.

    `llm` is already a retry + `with_fallbacks` chain built in agents.py
    (DeepSeek → Gemini for text/tools; Gemini-only for vision). LangGraph
    `stream_mode=['messages']` in main.py traces whichever provider actually
    served the call, so the frontend SSE stream does not change when a
    failover happens mid-turn.
    """
    print(f"[TIMING] LLM Call Started ({label})...", flush=True)
    t0 = time.time()
    try:
        response = await llm.ainvoke(messages)
        print(f"[TIMING] LLM Call Ended ({label}) - Took {time.time() - t0:.2f}s", flush=True)
        return response
    except Exception as e:
        print(
            f"[TIMING] LLM Call Ended ({label}) - Took {time.time() - t0:.2f}s (FAILED)",
            flush=True,
        )
        print(f"[{label}] provider error after retries/fallback: {e}", flush=True)
        raise


# =============================================================================
# ROUTER NODE — one-shot classifier. Heuristic first (PDF + "həll et" never
# waits on Gemini); otherwise a tiny JSON call with no chat history.
# =============================================================================
async def router_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["chat_worker_node", "doc_worker_node", "math_worker_node", "web_worker_node", "composer_node"]]:
    turn_user_message = state["messages"][-1] if state["messages"] else None
    raw_text = _extract_text(getattr(turn_user_message, "content", "") or "") if turn_user_message else ""
    file_path = _extract_file_path(raw_text)
    if file_path and not _is_image_file_path(file_path):
        reset_thread_document_if_new_file(config, file_path)
    reset_problem = _should_reset_problem_state(state, raw_text or "")

    if state.get("canvas_branch_from_id") or _CANVAS_EXPLAIN_RE.search(raw_text or ""):
        print("[ROUTER] canvas explanation branch → math_worker", flush=True)
        return _router_dispatch_command(
            ["math_worker"],
            turn_user_message,
            new_document=reset_problem,
            previous_extract=state.get("extracted_text") or "",
        )

    body = _strip_turn_wrappers(raw_text or "")
    language_only = _maybe_language_only(body)
    if _image_followup_kind(state, body):
        print("[ROUTER] image question from the screenshot cache → doc_worker", flush=True)
        return _router_dispatch_command(
            ["doc_worker", "math_worker"],
            turn_user_message,
            new_document=False,
            previous_extract=state.get("extracted_text") or "",
        )
    if (
        not language_only
        and _is_socratic_mode(state, raw_text or "")
        and _socratic_has_open_problem(state, body)
        and not _CHAT_ONLY_RE.match(body)
        and "File path:" not in (raw_text or "")
    ):
        print(
            "[ROUTER] socratic follow-up → math_worker "
            f"anchor={bool(state.get('canvas_anchor_id'))} "
            f"stem_len={len(_cached_image_stem(state))}",
            flush=True,
        )
        return _router_dispatch_command(
            ["math_worker"],
            turn_user_message,
            new_document=reset_problem,
            previous_extract=state.get("extracted_text") or "",
            extracted_text=_cached_image_stem(state) or None,
        )

    heuristic = _heuristic_route(raw_text)
    if heuristic:
        heuristic = _ensure_socratic_math_worker(state, raw_text or "", heuristic)
        print(f"[ROUTER] heuristic workers={heuristic}", flush=True)
        return _router_dispatch_command(
            heuristic,
            turn_user_message,
            new_document=reset_problem,
            previous_extract=state.get("extracted_text") or "",
        )

    payload = _router_user_payload(raw_text)
    messages_to_pass = [
        HumanMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=payload or "(empty)"),
    ]
    try:
        plan: RoutePlan = await _ainvoke_llm(router_llm, messages_to_pass, "Router")
    except Exception as e:
        fail_update = {
            "planned_workers": [],
            "turn_user_message": turn_user_message,
            "worker_queue": [],
            "completed_workers": [],
            "extracted_text": _stem_for_turn(
                state.get("extracted_text") or "",
                None,
                new_document=reset_problem,
            ),
            "desk_payload": None,
            "desk_payloads": None,
            "canvas_ops": [],
            "chat_reply": user_facing_llm_error(e),
            "navigation_update": None,
            "tool_round_trips": {},
            "fresh_intake": bool(reset_problem),
            "fresh_question_card": False,
        }
        if reset_problem:
            fail_update["active_navigation"] = None
            fail_update["active_problem"] = None
        return Command(
            update=fail_update,
            goto="composer_node",
        )
    requested_workers = list(getattr(plan, "workers", None) or []) or ["chat_worker"]
    requested_workers = _ensure_socratic_math_worker(state, raw_text or "", requested_workers)
    print(f"[ROUTER] workers={requested_workers}", flush=True)
    return _router_dispatch_command(
        requested_workers,
        turn_user_message,
        new_document=reset_problem,
        previous_extract=state.get("extracted_text") or "",
    )


# =============================================================================
# CHAT WORKER — chitchat / trivial general-knowledge / vision turns. A
# PLAIN chat completion (no tools, no structured output) so its tokens
# stream to the frontend exactly like a normal LLM response. Replaces the
# old router-internal `direct_response` fast path (see the Phase 4 note
# above `router_llm` in agents.py for why that leaked raw JSON into the
# chat stream and had to become a real worker instead).
# =============================================================================
async def chat_worker_node(
    state: AgentState,
) -> Command[Literal["doc_worker_node", "math_worker_node", "web_worker_node", "composer_node"]]:
    # Chat Worker has no tools of its own (no `.bind_tools()` call) — an
    # empty `own_tool_names` set means ANY tool-call noise from an upstream
    # worker in a chain gets stripped from its window entirely.
    trimmed_history = build_worker_messages(state, own_tool_names=set())

    if trimmed_history:
        # Image inlining happens ONLY here now — see `_inline_image_if_present`'s
        # docstring for why it moved out of `router_node`.
        trimmed_history = [*trimmed_history[:-1], await _inline_image_if_present(trimmed_history[-1])]

    sys_msg = HumanMessage(content=f"[SYSTEM INSTRUCTION]:\n{CHAT_WORKER_SYSTEM_PROMPT}")
    context_messages = []
    turn_msg = state.get("turn_user_message")
    raw_turn = _extract_text(getattr(turn_msg, "content", "") or "") if turn_msg is not None else ""
    turn_body = _strip_turn_wrappers(raw_turn)
    if "File path:" in (raw_turn or "") and _is_generic_query(turn_body):
        page_count = count_pdf_pages(_extract_file_path(raw_turn) or "")
        if page_count == 0 or page_count > EMPTY_PROMPT_AUTOSOLVE_MAX_PAGES:
            size_label = str(page_count) if page_count else "unknown"
            context_messages.append(
                HumanMessage(
                    content=(
                        f"[SYSTEM NOTE]: User attached a {size_label}-page document with no "
                        "problem/page. Do not solve. Ask which chapter/page/question to extract. "
                        "Reply in the user's language if the message is clearly in one language; "
                        "otherwise use the [UI LANGUAGE] locale."
                    )
                )
            )

    messages_to_pass = [sys_msg, *context_messages] + trimmed_history

    llm = chat_worker_llm
    label = "Chat Worker"
    if trimmed_history and _message_has_inline_image(trimmed_history[-1]):
        llm = chat_worker_vision_llm
        label = "Chat Worker (vision)"

    try:
        response = await _ainvoke_llm(llm, messages_to_pass, label)
    except Exception as e:
        return _provider_fail_command(state, "chat_worker", e)

    reply_text = _extract_text(response.content)
    return Command(
        update={
            "messages": [response],
            "chat_reply": reply_text,
            "completed_workers": (state.get("completed_workers") or []) + ["chat_worker"],
            "worker_queue": _pop_worker_queue(state),
        },
        goto=_next_worker_goto(state, "chat_worker"),
    )


# =============================================================================
# DOC WORKER — parses/searches uploaded documents. Writes its retrieved text
# into `extracted_text` so a downstream Math Worker (if queued) reads it
# directly from state instead of re-deriving it from chat history.
# =============================================================================
# Tool names whose args/results feed the persisted `active_navigation`
# cursor (see `locknlearn_schemas.ActiveNavigation`) — captured in
# `doc_tools_node` the moment the tool result exists, same rationale as
# `math_tools_node`'s `desk_payload` capture below.
NAVIGATION_TOOL_NAMES = {
    "read_page_range",
    "locate_marker_in_range",
    "resolve_chapter_target",
}
_BANK_SECTION_KINDS = frozenset({"problems", "conceptual", "objective"})

_RESOLVED_NAV_PATTERN = re.compile(
    r"\[RESOLVED\]\s+kind=(\S+)\s+pages=(\S+)-(\S+)\s+title=(.*)",
)


def _as_printed_int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


async def doc_worker_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["doc_tools_node", "math_worker_node", "web_worker_node", "chat_worker_node", "composer_node"]]:
    turn_msg = state.get("turn_user_message")
    turn_raw = _extract_text(getattr(turn_msg, "content", "") or "") if turn_msg is not None else ""
    turn_path = _extract_file_path(turn_raw) or ""
    turn_stem = _strip_turn_wrappers(turn_raw)

    formula_extract = bool(_FORMULA_EXTRACT_RE.search(turn_stem))
    if (
        turn_path
        and any(turn_path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        and not formula_extract
    ):
        # Screenshot extraction contract: none of DOC_TOOLS_MAP's tools
        # (read_page_range/locate_marker_in_range/search_in_document) work
        # on an image — vision-extract the stem directly instead of looping
        # the LLM through PDF-only tools. Bound to THIS turn's File path so
        # a prior screenshot cannot steal a later typed/PDF turn.
        original_name = _extract_original_filename(turn_raw) or ""
        extracted, active_image = await _inventory_image_page(
            turn_path, turn_stem, original_name, config
        )
        update = _doc_worker_finish_update(
            state, extracted, config=config, turn_file_path=turn_path
        )
        update["active_image"] = active_image
        update["fresh_intake"] = True
        return Command(update=update, goto=_next_worker_goto(state, "doc_worker"))

    if not turn_path and not formula_extract:
        cached = await _resolve_cached_image_turn(state, turn_stem, config)
        if cached:
            extracted, active_image, fresh_card = cached
            update = _doc_worker_finish_update(
                state,
                extracted,
                config=config,
                turn_file_path=str(active_image.get("path") or "") or None,
            )
            update["active_image"] = active_image
            update["fresh_intake"] = True
            update["fresh_question_card"] = fresh_card
            return Command(update=update, goto=_next_worker_goto(state, "doc_worker"))

    if (
        not turn_path
        and not formula_extract
        and _looks_like_new_socratic_stem(turn_stem)
        and not _TASK_START_RE.search(turn_stem)
    ):
        # Typed/pasted stem: same finish path as PDF extract. Do NOT call
        # document tools — that would read the previous thread PDF.
        print("[DOC WORKER] Typed problem stem — skipping PDF tools.", flush=True)
        return Command(
            update=_doc_worker_finish_update(
                state, turn_stem, config=config, typed_input=True
            ),
            goto=_next_worker_goto(state, "doc_worker"),
        )

    if reset_thread_document_if_new_file(config, turn_path or None):
        print("[DOC WORKER] Cleared stale document index before tools (new File path).", flush=True)

    trimmed_history = _drop_image_file_annotations(
        build_worker_messages(state, own_tool_names=set(DOC_TOOLS_MAP.keys()))
    )
    sys_msg = HumanMessage(content=f"[SYSTEM INSTRUCTION]:\n{DOC_WORKER_SYSTEM_PROMPT}")

    # Surfaces the PERSISTED navigation cursor (survives past the 4-message
    # trim window above) so a RELATIVE follow-up ("Now move to Question 8")
    # can be resolved by reusing the last page range instead of re-deriving
    # it from truncated chat history or re-calling `get_document_outline`.
    context_messages = []
    active_navigation = state.get("active_navigation")
    if (
        turn_path
        and not _is_image_file_path(turn_path)
        and active_navigation
        and active_navigation.get("source") not in (None, "", turn_path)
    ):
        active_navigation = None
    if active_navigation and active_navigation.get("page_start"):
        nav_line = (
            "[ACTIVE NAVIGATION CONTEXT]: The student's last resolved position in this "
            f"document was pages {active_navigation['page_start']}-{active_navigation['page_end']}"
        )
        kind = active_navigation.get("section_kind")
        title = active_navigation.get("section_title")
        if kind:
            nav_line += f" (kind={kind}"
            if title:
                nav_line += f", title={title}"
            nav_line += ")"
        if active_navigation.get("last_question_marker"):
            nav_line += f", last question referenced: '{active_navigation['last_question_marker']}'"
        chapter_heading = lookup_outline_chapter(config, active_navigation.get("page_start"))
        if chapter_heading:
            nav_line += f" Current chapter: {chapter_heading}"
        nav_line += (
            ". If the CURRENT message references a chapter/question only RELATIVELY "
            "(e.g. 'now question 8', 'next one', no new chapter named), reuse this EXACT page "
            "range via `locate_marker_in_range`/`read_page_range`. Do NOT re-call "
            "`get_document_outline` or `resolve_chapter_target` unless they named a NEW chapter."
        )
        context_messages.append(HumanMessage(content=nav_line))

    primary_path = _find_latest_uploaded_file_path(state["messages"])
    if primary_path:
        context_messages.append(HumanMessage(content=(
            f"[PRIMARY DOCUMENT]: File path: {primary_path}. "
            "Index this textbook path only. Do not call process_and_index_documents on an image."
        )))

    messages_to_pass = [sys_msg, *context_messages] + trimmed_history

    try:
        response = await _ainvoke_llm(doc_worker_llm, messages_to_pass, "Doc Worker")
    except Exception as e:
        return _provider_fail_command(state, "doc_worker", e)

    if response.tool_calls:
        # Already used the round-trip budget — do NOT dispatch another
        # tools node (that would spend another LLM call after the tools
        # return). Force-finish with the best extraction we already have.
        if _tool_round_trip_count(state, "doc_worker") >= MAX_WORKER_TOOL_ROUND_TRIPS:
            print(
                f"[ANTI-LOOP] Force-quit doc_worker after "
                f"{MAX_WORKER_TOOL_ROUND_TRIPS} tool round-trips (refusing further tool calls).",
                flush=True,
            )
            reply_text = _latest_doc_extraction(state) or _extract_text(response.content)
            return Command(
                update=_doc_worker_finish_update(
                    state, reply_text, extra={"messages": [response]}, config=config
                ),
                goto=_next_worker_goto(state, "doc_worker"),
            )
        return Command(update={"messages": [response]}, goto="doc_tools_node")

    reply_text = _extract_text(response.content)
    return Command(
        update=_doc_worker_finish_update(
            state, reply_text, extra={"messages": [response]}, config=config
        ),
        goto=_next_worker_goto(state, "doc_worker"),
    )


async def doc_tools_node(state: AgentState, config: RunnableConfig) -> Command[Literal["doc_worker_node", "math_worker_node", "web_worker_node", "chat_worker_node", "composer_node"]]:
    last_message = state["messages"][-1]
    round_trips = _next_round_trips(state, "doc_worker")
    trip_n = round_trips["doc_worker"]
    short_doc = is_short_indexed_document(config)

    async def _invoke_doc_tool(tool_call: dict) -> ToolMessage:
        tool_name = tool_call["name"]
        if short_doc and tool_name in SHORT_DOC_BLOCKED_TOOLS:
            n = get_indexed_page_count(config)
            print(f"[ANTI-LOOP] Short-doc blocked '{tool_name}' ({n} pages).", flush=True)
            tool_id = tool_call.get("id")
            if not tool_id:
                tool_id = f"call_{tool_name}_{uuid.uuid4().hex[:12]}"
                tool_call["id"] = tool_id
            return ToolMessage(
                content=(
                    f"Xəta: Bu sənəd QISADIR ({n} səhifə). '{tool_name}' BLOK EDİLİB. "
                    f"YALNIZ read_page_range(start_page=1, end_page={n}) çağır və dərhal yekun cavab ver."
                ),
                tool_call_id=tool_id,
                name=tool_name,
            )
        return await _invoke_worker_tool_call(tool_call, DOC_TOOLS_MAP, config)

    tool_messages = await asyncio.gather(
        *(_invoke_doc_tool(tc) for tc in last_message.tool_calls)
    )

    # Task Tracker turns: warm the vector index in the background so a later
    # Tier 2 search finds it ready. Idempotent; a no-op for scans, short docs,
    # and unsigned sessions. Never blocks this turn.
    _turn_msg = state.get("turn_user_message")
    _turn_text = (
        _extract_text(getattr(_turn_msg, "content", "") or "") if _turn_msg is not None else ""
    )
    if _TASK_START_RE.search(_turn_text):
        start_embed_warmup(_resolve_thread_key(config), config)

    update: dict = {"messages": list(tool_messages), "tool_round_trips": round_trips}

    tool_call_by_id = {tc.get("id"): tc for tc in last_message.tool_calls}
    for tool_message in tool_messages:
        if tool_message.name not in NAVIGATION_TOOL_NAMES:
            continue
        if str(tool_message.content).startswith("Xəta:"):
            continue

        call_args = (tool_call_by_id.get(tool_message.tool_call_id) or {}).get("args", {})
        navigation_update = dict(update.get("navigation_update") or state.get("navigation_update") or {})

        if tool_message.name == "resolve_chapter_target":
            parsed = _RESOLVED_NAV_PATTERN.search(str(tool_message.content))
            if parsed:
                kind, p_start, p_end, title = (
                    parsed.group(1), parsed.group(2), parsed.group(3), parsed.group(4),
                )
                start_i = _as_printed_int(p_start)
                end_i = _as_printed_int(p_end)
                if start_i is not None:
                    navigation_update["page_start"] = start_i
                if end_i is not None:
                    navigation_update["page_end"] = end_i
                navigation_update["section_kind"] = kind
                navigation_update["section_title"] = (title or "").strip()
            else:
                navigation_update["section_kind"] = call_args.get("section_kind") or "problems"
            update["navigation_update"] = navigation_update
            continue

        if call_args.get("start_page") is not None:
            navigation_update["page_start"] = call_args.get("start_page")
        if call_args.get("end_page") is not None:
            navigation_update["page_end"] = call_args.get("end_page")

        kind_from_args = str(call_args.get("section_kind") or "").strip().lower()
        if kind_from_args in _BANK_SECTION_KINDS:
            navigation_update["section_kind"] = kind_from_args

        if tool_message.name == "locate_marker_in_range":
            navigation_update["last_question_marker"] = call_args.get("marker")
            prev_nav = state.get("active_navigation") or {}
            if not navigation_update.get("section_kind"):
                navigation_update["section_kind"] = prev_nav.get("section_kind")
            if not navigation_update.get("section_title"):
                navigation_update["section_title"] = prev_nav.get("section_title")
            # Do NOT shrink page_start/page_end to the found page. A Problems
            # bank spans several pages; "now question 8" must keep that span.

        update["navigation_update"] = navigation_update

    if trip_n >= MAX_WORKER_TOOL_ROUND_TRIPS:
        print(
            f"[ANTI-LOOP] Force-quit doc_worker after {trip_n} tool round-trips "
            "(skipping further LLM calls).",
            flush=True,
        )
        history_plus_batch = {
            "messages": list(state.get("messages") or []) + list(tool_messages),
            "extracted_text": state.get("extracted_text"),
        }
        reply_text = _latest_doc_extraction(history_plus_batch)
        if not reply_text and short_doc:
            reply_text = await _force_extract_short_document(config)
        return Command(
            update=_doc_worker_finish_update(state, reply_text, extra=update, config=config),
            goto=_next_worker_goto(state, "doc_worker"),
        )

    return Command(update=update, goto="doc_worker_node")


# =============================================================================
# MATH / EXECUTION WORKER — DeepSeek-primary (see agents.math_worker_llm)
# for calculations, charts, and physics/math diagrams. Reads
# `extracted_text` (if the Doc Worker ran earlier this turn) so it never
# has to re-derive extracted numbers/formulas from raw chat history.
# =============================================================================
async def math_worker_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["math_tools_node", "doc_worker_node", "web_worker_node", "chat_worker_node", "composer_node"]]:
    set_pipeline_status(_resolve_thread_key(config), PIPELINE_STATUS_MATH)
    socratic = _is_socratic_mode(state)
    stem_checkpoint: dict = {}
    if (
        socratic
        and not (state.get("extracted_text") or "").strip()
        and not state.get("canvas_branch_from_id")
    ):
        # One choke point for every route into the tutor (shortcut, heuristic,
        # router LLM): the stored screenshot stem is the problem text.
        cached_stem = _cached_image_stem(state)
        if cached_stem:
            state = {**state, "extracted_text": cached_stem}
            stem_checkpoint = {"extracted_text": cached_stem}
            print(
                f"[MATH WORKER] extracted_text empty — filled from cached stem len={len(cached_stem)}",
                flush=True,
            )
    goal_updates: dict = {}
    view: dict = state
    if socratic:
        goal_updates = _maybe_reset_socratic_goals(state)
        if goal_updates:
            view = {**state, **goal_updates}
    solved = socratic and (
        bool(view.get("is_solved")) or _pending_goals_snapshot(view) == []
    )
    # Structural anti-loop: after the first tools visit in Socratic mode the
    # graph must not invoke the LLM again — prompt-only "call once" is not
    # enough; the model will retry and burn round-trips until the cap errors.
    if socratic and _tool_round_trip_count(state, "math_worker") >= 1:
        print(
            "[ANTI-LOOP] socratic math_worker already used its one tool slot — "
            "finishing without another LLM call.",
            flush=True,
        )
        return Command(
            update={
                "chat_reply": state.get("chat_reply") or "",
                "completed_workers": (state.get("completed_workers") or []) + ["math_worker"],
                "worker_queue": _pop_worker_queue(state),
                **goal_updates,
                **stem_checkpoint,
            },
            goto=_next_worker_goto(state, "math_worker"),
        )

    if (
        socratic
        and _pending_goals_snapshot(view) is None
        and not view.get("is_solved")
        and not state.get("canvas_branch_from_id")
    ):
        planner_stem = _planner_stem(state)
        if planner_stem:
            planned = await _plan_socratic_goals(planner_stem)
            if planned:
                planned_goals, planned_units = planned
                patch = _init_socratic_goals_update(view, planned_goals)
                if patch:
                    patch["goal_units"] = planned_units or None
                    patch["closed_goals"] = None
                goal_updates.update(patch)
                view = {**view, **patch}

    turn_msg = state.get("turn_user_message")
    turn_raw = _extract_text(getattr(turn_msg, "content", "") or "") if turn_msg is not None else ""
    fresh_intake = bool(state.get("fresh_intake")) or _is_image_file_path(
        _extract_file_path(turn_raw)
    )
    if fresh_intake:
        raw_messages = state.get("messages") or []
        start = 0
        if turn_msg is not None:
            for index, message in enumerate(raw_messages):
                if message is turn_msg:
                    start = index
                    break
            else:
                raw_messages = [turn_msg]
                start = 0
        scoped = {**state, "messages": list(raw_messages[start:])}
        trimmed_history = build_worker_messages(
            scoped, own_tool_names=set(MATH_TOOLS_MAP.keys())
        )
        print(
            f"[MATH WORKER] fresh_intake window n={len(trimmed_history)}",
            flush=True,
        )
    else:
        trimmed_history = build_worker_messages(
            state, own_tool_names=set(MATH_TOOLS_MAP.keys())
        )

    doc_worker_ran = "doc_worker" in (state.get("completed_workers") or [])
    ungrounded = doc_worker_ran and not (state.get("extracted_text") or "").strip()
    if (
        socratic
        and not ungrounded
        and not solved
        and not state.get("canvas_branch_from_id")
        and not (state.get("extracted_text") or "").strip()
        and _pending_goals_snapshot(view) is None
        and not _planner_stem(state)
    ):
        # No problem text from any source and no stored goals: the tutor would
        # invent values. Take the same gate as a failed extraction.
        ungrounded = True
        print("[MATH WORKER] socratic grounding gate: no problem text and no goals", flush=True)
    print(
        f"[MATH WORKER] extracted_text_len={len((state.get('extracted_text') or '').strip())} "
        f"socratic={socratic} doc_worker_ran={doc_worker_ran}",
        flush=True,
    )

    context_messages = []
    if fresh_intake:
        context_messages.append(
            HumanMessage(
                content=(
                    "[NEW PROBLEM THIS TURN]: Ignore prior conversation stems "
                    "and solutions. Use ONLY [EXTRACTED DOCUMENT CONTEXT] "
                    "below (or this turn's user message if no extract)."
                )
            )
        )
    if state.get("extracted_text"):
        context_messages.append(
            HumanMessage(content=f"[EXTRACTED DOCUMENT CONTEXT]:\n{state['extracted_text']}")
        )
    elif ungrounded:
        # GROUNDING GATE: doc_worker ran but produced no usable extract —
        # do not let the LLM fall back to pretrained/textbook memory.
        context_messages.append(
            HumanMessage(
                content=(
                    "[GROUNDING GATE]: Extraction failed — no verified problem "
                    "text was found for this document/page. Do NOT answer from "
                    "pretrained textbook memory and do NOT call "
                    "`python_code_executor`. Tell the student in chat, in one "
                    "short sentence, that the exact problem text could not be "
                    "located, and ask them for the page/chapter number or to "
                    "paste the problem statement directly."
                )
            )
        )
    if socratic and not ungrounded:
        context_messages.append(HumanMessage(content=_goal_state_block(view)))
    if solved and not ungrounded:
        context_messages.append(
            HumanMessage(
                content=(
                    "[SOCRATIC FREE-FORM TUTOR]: The problem is solved. Answer "
                    "conceptual Why/How questions in chat only. Do NOT call "
                    "`python_code_executor`. Do NOT emit canvas steps, a second "
                    "Result, or \"What is the next quantity to find?\"."
                )
            )
        )
    elif state.get("canvas_branch_from_id"):
        context_messages.append(
            HumanMessage(
                content=(
                    "[CANVAS EXPLANATION BRANCH]: The student asked Why/How about a "
                    "specific step already on the canvas. Call `python_code_executor` "
                    "ONCE with type \"calculation\" and put the full pedagogical "
                    "explanation in `steps` (prose then `$...$` / `$$...$$` in each "
                    "string). Do NOT write the explanation in chat. Chat text must be "
                    f"only: {EXPLANATION_ACK}"
                )
            )
        )
    elif socratic and not ungrounded:
        context_messages.append(
            HumanMessage(
                content=(
                    "[SOCRATIC WHITEBOARD]: Chat is the draft; the canvas is only "
                    "for verified formal milestones. Obey [SOCRATIC GOAL STATE] over "
                    "chat memory. DEFAULT: reply in chat with NO tool call (opening "
                    "question, hints, or struggle). After a premise-ok student "
                    "milestone THIS turn (stem numerals AND headings/directions "
                    "must match the extract — north stays north), "
                    "call `python_code_executor` ONCE with `type` \"calculation\" "
                    "and write THAT milestone now. Put the goal's report symbol "
                    "alone on the left of the equation. Python closes the goal "
                    "when that equation matches exactly one open goal; "
                    "`complete_goal` is only a hint. "
                    "Do not hoard setup equations. Never evaluate or write that "
                    "unknown's number until the student says it. If they say a "
                    "prior close was not the result, ask them to write the next "
                    "relation — do not compute it. `steps` MUST be EXACTLY one "
                    "string (that milestone only — never later goals). Intermediate: "
                    "omit `value`; `summary` is invalid unless it contains EXACTLY "
                    "ONE `?` (or `？`) aimed ONLY at pending_goals[0] — never a "
                    "later goal, never a second question in the same bubble. "
                    "If two or more goals remain after this step, no `value` / "
                    "`Result found:`. Python writes `value` and `Result found:` "
                    "when the equation closes the LAST open goal. No remaining-solution "
                    "dump, no second tool call, never rewrite the original question "
                    "stem. If the student asks to simplify/calculate/correct the "
                    "CURRENT card, call the tool once with replace_last true "
                    "(overwrite last step). New conceptual phase: omit replace_last "
                    "so a new step is appended."
                )
            )
        )

    math_prompt = (
        MATH_WORKER_SYSTEM_PROMPT_SOCRATIC if socratic else MATH_WORKER_SYSTEM_PROMPT_DETAILED
    )
    sys_msg = HumanMessage(content=f"[SYSTEM INSTRUCTION]:\n{math_prompt}")
    messages_to_pass = [sys_msg, *context_messages] + trimmed_history

    try:
        response = await _ainvoke_llm(math_worker_llm, messages_to_pass, "Math Worker")
    except Exception as e:
        return _provider_fail_command(state, "math_worker", e) if not stem_checkpoint else Command(
            update={
                "chat_reply": user_facing_llm_error(e),
                "completed_workers": (state.get("completed_workers") or []) + ["math_worker"],
                "worker_queue": _pop_worker_queue(state),
                **stem_checkpoint,
            },
            goto=_next_worker_goto(state, "math_worker"),
        )

    open_goals = _pending_goals_snapshot(view)
    if socratic and not solved and not response.tool_calls and open_goals and len(open_goals) >= 2:
        # Chat must not run ahead of state: a chat-only reply that asks about a
        # later goal (without a closing tool call) gets ONE corrective retry.
        first_try = _parse_goals_tag(_extract_text(response.content))[1]
        if _reply_targets_other_goal(first_try, open_goals) is not None:
            correction = HumanMessage(
                content=(
                    "[GOAL SYNC]: Your reply moved ahead, but this turn did not record "
                    f"an accepted step. The first open goal is: {_goal_text(open_goals[0])}. "
                    "Call `python_code_executor` ONCE with that goal's report symbol "
                    "on the left of the accepted equation, or stay on this goal and "
                    "ask its next micro-step. Do not open with \"Correct —\"."
                )
            )
            try:
                response = await _ainvoke_llm(
                    math_worker_llm,
                    [*messages_to_pass, AIMessage(content=first_try), correction],
                    "Math Worker (goal sync retry)",
                )
                print(
                    f"[SYNC] retry done tool_calls={bool(response.tool_calls)}", flush=True
                )
            except Exception as e:
                print(f"[SYNC] retry failed, keeping first reply: {e}", flush=True)

    def _finish_math_chat(reply: str, stored_message: BaseMessage) -> Command:
        print(
            "[SOCRATIC DEBUG] math_worker RETURNING "
            f"chat_reply={reply!r:.120} parsed_goals={parsed_goals!r}",
            flush=True,
        )
        return Command(
            update={
                "messages": [stored_message],
                "chat_reply": reply,
                "completed_workers": (state.get("completed_workers") or []) + ["math_worker"],
                "worker_queue": _pop_worker_queue(state),
                **goal_updates,
                **stem_checkpoint,
                **_init_socratic_goals_update(view, parsed_goals),
            },
            goto=_next_worker_goto(state, "math_worker"),
        )

    raw_reply = _extract_text(response.content)
    if _has_protocol_marker(raw_reply):
        print(
            f"[SANITIZE] math_worker raw model output carries protocol tags "
            f"socratic={socratic} tool_calls={bool(response.tool_calls)} "
            f"preview={raw_reply[:40]!r}",
            flush=True,
        )
    # Never persist tags to history: the next turn would copy them.
    response = _clean_ai_message(response)
    parsed_goals = None
    if socratic:
        parsed_goals, raw_reply = _parse_goals_tag(raw_reply)
    reply_text = _auto_wrap_stray_latex(raw_reply)
    if socratic and not (reply_text or "").strip():
        reply_text = _SOCRATIC_EMPTY_CHAT_FALLBACK
    print(
        "[SOCRATIC DEBUG] math_worker after parse "
        f"parsed_goals={parsed_goals!r} raw={raw_reply!r:.80} "
        f"reply_text={reply_text!r:.120} tool_calls={bool(response.tool_calls)}",
        flush=True,
    )

    if response.tool_calls:
        if solved:
            reply_text = reply_text or _SOCRATIC_SOLVED_CHAT_FALLBACK
            print(
                "[SOCRATIC GOALS] solved — dropping tool calls for free-form tutor.",
                flush=True,
            )
            return _finish_math_chat(reply_text, AIMessage(content=reply_text))
        if _tool_round_trip_count(state, "math_worker") >= MAX_WORKER_TOOL_ROUND_TRIPS:
            print(
                f"[ANTI-LOOP] Force-quit math_worker after "
                f"{MAX_WORKER_TOOL_ROUND_TRIPS} tool round-trips.",
                flush=True,
            )
            reply_text = reply_text or (
                (state.get("desk_payload") or {}).get("summary") or ""
            )
            return _finish_math_chat(reply_text, response)
        print(
            "[SOCRATIC DEBUG] math_worker RETURNING goto=math_tools_node "
            f"parsed_goals={parsed_goals!r}",
            flush=True,
        )
        return Command(
            update={
                "messages": [response],
                **goal_updates,
                **stem_checkpoint,
                **_init_socratic_goals_update(view, parsed_goals),
            },
            goto="math_tools_node",
        )

    rewritten = False
    if socratic and not solved and _reply_targets_other_goal(reply_text, open_goals) is not None:
        # Retry (or the first reply) still runs ahead of state: keep the
        # acknowledgement, ask the first open goal.
        reply_text = _replace_question_sentence(reply_text, open_goals[0])
        rewritten = True
        print("[SYNC] chat-only reply rewritten to the first open goal (fallback)", flush=True)
    stored = AIMessage(content=reply_text) if parsed_goals or rewritten or (
        socratic and reply_text == _SOCRATIC_EMPTY_CHAT_FALLBACK
    ) else response
    return _finish_math_chat(reply_text, stored)


async def math_tools_node(state: AgentState, config: RunnableConfig) -> Command[Literal["math_worker_node", "doc_worker_node", "web_worker_node", "chat_worker_node", "composer_node"]]:
    last_message = state["messages"][-1]
    round_trips = _next_round_trips(state, "math_worker")
    trip_n = round_trips["math_worker"]
    socratic = _is_socratic_mode(state)
    goal_updates: dict = {}
    view: dict = state
    if socratic:
        reset = _maybe_reset_socratic_goals(state)
        if reset:
            goal_updates.update(reset)
            view = {**state, **reset}
    already_solved = socratic and (
        bool(view.get("is_solved")) or _pending_goals_snapshot(view) == []
    )
    tool_calls = list(getattr(last_message, "tool_calls", None) or [])
    model_prose = ""
    if socratic:
        raw_model = _extract_text(getattr(last_message, "content", "") or "")
        _, model_prose = _parse_goals_tag(raw_model)
        model_prose = _auto_wrap_stray_latex(model_prose or "").strip()

    if already_solved:
        tool_messages = []
        for tool_call in tool_calls:
            name = tool_call.get("name")
            tool_id = tool_call.get("id") or f"call_{name}_{uuid.uuid4().hex[:12]}"
            if name == DESK_PAYLOAD_TOOL_NAME:
                tool_messages.append(
                    ToolMessage(
                        content=_SOCRATIC_SOLVED_TOOL_STUB,
                        tool_call_id=tool_id,
                        name=name,
                    )
                )
            else:
                tool_messages.append(
                    await _invoke_worker_tool_call(tool_call, MATH_TOOLS_MAP, config)
                )
        reply_text = _auto_wrap_stray_latex(
            _extract_text(getattr(last_message, "content", "") or "")
        )
        _, reply_text = _parse_goals_tag(reply_text or "")
        reply_text = reply_text or _SOCRATIC_SOLVED_CHAT_FALLBACK
        print(
            "[SOCRATIC GOALS] is_solved — skipped python_code_executor; no canvas ops.",
            flush=True,
        )
        return Command(
            update={
                "messages": list(tool_messages),
                "tool_round_trips": round_trips,
                "chat_reply": reply_text,
                "completed_workers": (state.get("completed_workers") or []) + ["math_worker"],
                "worker_queue": _pop_worker_queue(state),
                **goal_updates,
            },
            goto=_next_worker_goto(state, "math_worker"),
        )

    tool_messages = await asyncio.gather(
        *(_invoke_worker_tool_call(tc, MATH_TOOLS_MAP, config) for tc in tool_calls)
    )

    update: dict = {"messages": list(tool_messages), "tool_round_trips": round_trips}
    payloads = list(state.get("desk_payloads") or [])
    canvas_ops = list(state.get("canvas_ops") or [])
    ops_before = len(canvas_ops)
    branch_from_id = state.get("canvas_branch_from_id") or None
    desk_tool_ran = False
    socratic_payload: Optional[dict] = None
    tool_error_message: Optional[str] = None
    symbolic_reply: Optional[str] = None
    for tool_message in tool_messages:
        if tool_message.name != DESK_PAYLOAD_TOOL_NAME:
            continue
        payload = _extract_desk_payload(tool_message.content)
        if payload is None:
            continue
        if payload.get("type") == "error":
            # Surface the failure instead of silently dropping it — the
            # canvas gets no ops from this call, but the fallback reply
            # below must not pretend the step succeeded.
            tool_error_message = payload.get("message") or "Tool execution failed."
            continue
        # One canvas payload per Socratic turn — extra tool_calls in the same
        # AIMessage are executed (already gathered) but must not mint more ops
        # or bounce back to the worker.
        if socratic and desk_tool_ran:
            continue
        if (
            socratic
            and not branch_from_id
            and isinstance(payload, dict)
            and payload.get("type") == "calculation"
        ):
            reason = _symbolic_gate_reason(payload, state)
            if reason:
                # Refuse the write before any goal pop or canvas op. The tutor's
                # chat line becomes the fixed verdict plus the open-goal question.
                desk_tool_ran = True
                symbolic_reply = _symbolic_reject_reply(reason, _pending_goals_snapshot(view))
                print(f"[SYMBOLIC] rejected {reason} — no canvas write", flush=True)
                continue
        desk_tool_ran = True
        if (
            socratic
            and not branch_from_id
            and isinstance(payload, dict)
            and payload.get("type") == "calculation"
        ):
            replace_last = _truthy_flag(payload.get("replace_last")) or (
                _socratic_user_wants_replace_last(state)
            )
            if not replace_last and _step_repeats_last(payload, view):
                replace_last = True
                print("[DESK LINT] new step repeats the last card — forcing replace_last", flush=True)
            payload, goal_patch = _apply_socratic_goal_payload(view, payload, replace_last)
            closed_now = [
                g for g in (goal_patch.get("closed_goals") or [])
                if g not in (view.get("closed_goals") or [])
            ]
            if goal_patch:
                goal_updates.update(goal_patch)
                view = {**view, **goal_patch}
            allow_result = bool(view.get("is_solved")) or _pending_goals_snapshot(view) == []
            payload = _lint_socratic_payload(payload, view.get("goal_units") or {}, closed_now)
            payload = _normalize_socratic_calculation(payload, allow_result=allow_result)
            new_sig = _step_signature(_first_step(payload))
            if new_sig:
                goal_updates["last_socratic_step"] = new_sig
            if replace_last and not bool(view.get("is_solved")):
                payload = {**payload, "replace_last": True}
        if socratic:
            socratic_payload = payload
        canvas_ops.extend(
            _payload_to_canvas_ops(
                payload,
                branch_from_id=branch_from_id,
                socratic=socratic,
            )
        )
        # Why/How laterals stay canvas-only. Socratic success prints steps
        # but still must not mint a Desk calculation card / RESULT widget.
        if payload.get("type") == "explanation" or branch_from_id or socratic:
            continue
        update["desk_payload"] = payload
        payloads = _merge_desk_payload(payloads, payload)
    if (
        socratic
        and isinstance(socratic_payload, dict)
        and (bool(view.get("is_solved")) or _pending_goals_snapshot(view) == [])
    ):
        fresh = canvas_ops[ops_before:]
        has_result = any(isinstance(op, dict) and op.get("op") == "result" for op in fresh)
        value = socratic_payload.get("value") if isinstance(socratic_payload.get("value"), str) else ""
        if not has_result and value.strip() and "=" in value:
            summary = socratic_payload.get("summary") if isinstance(socratic_payload.get("summary"), str) else ""
            canvas_ops.append(_stamp_canvas_target({
                "op": "result",
                "value": value.strip(),
                "summary": summary,
            }))
    if payloads:
        update["desk_payloads"] = payloads
    if canvas_ops:
        update["canvas_ops"] = canvas_ops
    if goal_updates:
        update.update(goal_updates)

    new_ops = canvas_ops[ops_before:]
    step_generation_done = any(
        isinstance(op, dict) and op.get("op") in ("step", "result") for op in new_ops
    )
    if socratic:
        reply_text = ""
        if symbolic_reply:
            reply_text = symbolic_reply
        else:
            summary = ""
            if isinstance(socratic_payload, dict) and isinstance(socratic_payload.get("summary"), str):
                summary = socratic_payload.get("summary")
            pending_now = _pending_goals_snapshot(view)
            solved_now = bool(view.get("is_solved")) or pending_now == []
            reply_text = _socratic_visible_reply(
                model_prose,
                summary,
                solved=solved_now,
                pending=pending_now,
            )
        _, reply_text = _parse_goals_tag(reply_text)
        if not symbolic_reply and not (reply_text or "").strip():
            # Surface a real tool failure instead of the generic "let's break
            # this down" filler, which falsely implies the step succeeded.
            reply_text = (
                _SOCRATIC_TOOL_ERROR_FALLBACK if tool_error_message
                else _SOCRATIC_EMPTY_CHAT_FALLBACK
            )
        print(
            "[ANTI-LOOP] socratic math_tools consumed its one tool slot — "
            "finishing math_worker without a wrap-up LLM call. "
            f"tool_error={bool(tool_error_message)}",
            flush=True,
        )
        return Command(
            update={
                **update,
                "chat_reply": reply_text,
                "completed_workers": (state.get("completed_workers") or []) + ["math_worker"],
                "worker_queue": _pop_worker_queue(state),
            },
            goto=_next_worker_goto(state, "math_worker"),
        )
    if step_generation_done:
        payload = update.get("desk_payload") or state.get("desk_payload") or {}
        reply_text = payload.get("summary") or state.get("chat_reply") or ""
        print(
            "[HANDOFF] math_tools received calculation/explanation canvas ops — "
            "finishing math_worker without a wrap-up LLM call.",
            flush=True,
        )
        return Command(
            update={
                **update,
                "chat_reply": reply_text,
                "completed_workers": (state.get("completed_workers") or []) + ["math_worker"],
                "worker_queue": _pop_worker_queue(state),
            },
            goto=_next_worker_goto(state, "math_worker"),
        )

    if trip_n >= MAX_WORKER_TOOL_ROUND_TRIPS:
        print(
            f"[ANTI-LOOP] Force-quit math_worker after {trip_n} tool round-trips "
            "(skipping further LLM calls).",
            flush=True,
        )
        payload = update.get("desk_payload") or state.get("desk_payload") or {}
        reply_text = payload.get("summary") or ""
        if not reply_text.strip() and tool_error_message:
            reply_text = (
                "I ran into a repeated error trying to compute that "
                f"({tool_error_message}). Could you rephrase the calculation "
                "or double-check the numbers?"
            )
        return Command(
            update={
                **update,
                "chat_reply": reply_text,
                "completed_workers": (state.get("completed_workers") or []) + ["math_worker"],
                "worker_queue": _pop_worker_queue(state),
            },
            goto=_next_worker_goto(state, "math_worker"),
        )

    return Command(update=update, goto="math_worker_node")


# =============================================================================
# WEB RESEARCH WORKER — live web search for papers/citations/news.
# =============================================================================
async def web_worker_node(
    state: AgentState,
) -> Command[Literal["web_tools_node", "doc_worker_node", "math_worker_node", "chat_worker_node", "composer_node"]]:
    trimmed_history = build_worker_messages(state, own_tool_names=set(WEB_TOOLS_MAP.keys()))
    sys_msg = HumanMessage(content=f"[SYSTEM INSTRUCTION]:\n{WEB_WORKER_SYSTEM_PROMPT}")
    messages_to_pass = [sys_msg] + trimmed_history

    try:
        response = await _ainvoke_llm(web_worker_llm, messages_to_pass, "Web Worker")
    except Exception as e:
        return _provider_fail_command(state, "web_worker", e)

    if response.tool_calls:
        if _tool_round_trip_count(state, "web_worker") >= MAX_WORKER_TOOL_ROUND_TRIPS:
            print(
                f"[ANTI-LOOP] Force-quit web_worker after "
                f"{MAX_WORKER_TOOL_ROUND_TRIPS} tool round-trips.",
                flush=True,
            )
            return Command(
                update={
                    "messages": [response],
                    "chat_reply": _extract_text(response.content),
                    "completed_workers": (state.get("completed_workers") or []) + ["web_worker"],
                    "worker_queue": _pop_worker_queue(state),
                },
                goto=_next_worker_goto(state, "web_worker"),
            )
        return Command(update={"messages": [response]}, goto="web_tools_node")

    reply_text = _extract_text(response.content)
    return Command(
        update={
            "messages": [response],
            "chat_reply": reply_text,
            "completed_workers": (state.get("completed_workers") or []) + ["web_worker"],
            "worker_queue": _pop_worker_queue(state),
        },
        goto=_next_worker_goto(state, "web_worker"),
    )


async def web_tools_node(state: AgentState, config: RunnableConfig) -> Command[Literal["web_worker_node", "doc_worker_node", "math_worker_node", "chat_worker_node", "composer_node"]]:
    last_message = state["messages"][-1]
    round_trips = _next_round_trips(state, "web_worker")
    trip_n = round_trips["web_worker"]

    tool_messages = await asyncio.gather(
        *(_invoke_worker_tool_call(tc, WEB_TOOLS_MAP, config) for tc in last_message.tool_calls)
    )

    update: dict = {"messages": list(tool_messages), "tool_round_trips": round_trips}
    if trip_n >= MAX_WORKER_TOOL_ROUND_TRIPS:
        print(
            f"[ANTI-LOOP] Force-quit web_worker after {trip_n} tool round-trips "
            "(skipping further LLM calls).",
            flush=True,
        )
        last_content = str(tool_messages[-1].content) if tool_messages else ""
        return Command(
            update={
                **update,
                "chat_reply": last_content,
                "completed_workers": (state.get("completed_workers") or []) + ["web_worker"],
                "worker_queue": _pop_worker_queue(state),
            },
            goto=_next_worker_goto(state, "web_worker"),
        )

    return Command(update=update, goto="web_worker_node")


# =============================================================================
# COMPOSER — pure assembly, NO LLM call. By the time execution reaches here
# every worker that needed to run already has, and each one wrote its own
# `chat_reply`/`desk_payload` into state; this node's only job is to
# package those into the dual-payload shape `main.py` streams to the
# frontend (Phase 3 wires that streaming layer up to read this directly).
# =============================================================================
async def composer_node(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["__end__"]]:
    """
    Assembles the dual-payload SSE shape `main.py` streams: chat text,
    zero-or-more Desk artifacts (`chart`/`calculation`/`diagram`), incremental
    `canvas_ops`, and the persisted Active Problem header (`active_problem`).
    Desk payloads stay as the ad-hoc dicts `python_code_executor` emits —
    not `DeskState`.
    """
    print("[SOCRATIC DEBUG] composer ENTER", flush=True)
    set_pipeline_status(_resolve_thread_key(config), PIPELINE_STATUS_RENDER)
    chat_reply = state.get("chat_reply") or ""
    if _looks_like_internal_prompt(chat_reply):
        chat_reply = "\n".join(
            line for line in chat_reply.splitlines() if not _looks_like_internal_prompt(line)
        ).strip()
    _, chat_reply = _parse_goals_tag(chat_reply)
    chat_reply = _strip_process_narration(chat_reply)
    chat_reply = _collapse_repeated_chat(chat_reply)
    if _is_socratic_mode(state) and not (chat_reply or "").strip():
        chat_reply = _SOCRATIC_EMPTY_CHAT_FALLBACK
    desk_payload = state.get("desk_payload")
    desk_payloads = _collapse_desk_calculations(list(state.get("desk_payloads") or []))
    if not desk_payloads and desk_payload:
        desk_payloads = [desk_payload]
    if desk_payloads:
        desk_payload = desk_payloads[-1]

    active_problem = state.get("active_problem")
    planned = state.get("planned_workers") or []
    if active_problem and ("doc_worker" in planned or "math_worker" in planned):
        query = _user_facing_query(state)
        current = (active_problem.get("description") or "").strip()
        title = (active_problem.get("title") or "").strip()
        if _looks_like_internal_prompt(current):
            current = ""
        if _looks_like_internal_prompt(title):
            active_problem = {**active_problem, "title": active_problem.get("source") or "Problem"}
        if _is_generic_query(query) or not current:
            if not current or _is_generic_query(current):
                inferred = _infer_objective_from_extract(
                    state.get("extracted_text") or ""
                )
                description = _subtitle_from_query(
                    query,
                    active_problem.get("topic"),
                    inferred,
                )
                if description:
                    active_problem = {**active_problem, "description": description}
        elif query:
            refreshed = _subtitle_from_query(
                query,
                active_problem.get("topic"),
                current or active_problem.get("description"),
            )
            if refreshed and not _looks_like_internal_prompt(refreshed):
                active_problem = {**active_problem, "description": refreshed}

    # Defensive, log-only sanity check: compares the router's ORIGINAL full
    # plan (`planned_workers`, set once and never mutated) against
    # `completed_workers` (built up incrementally as each worker finishes).
    # In the current design these should always match by the time execution
    # legitimately reaches `composer_node` — every worker's own final
    # Command pops `worker_queue` and appends to `completed_workers` in the
    # SAME atomic update, and `_next_worker_goto` only routes here once the
    # queue is empty. This can't catch a hard `GraphRecursionError` (that
    # exception unwinds `.astream()` itself, so `composer_node` never runs
    # at all in that case — see main.py's `except GraphRecursionError`
    # handler for that failure mode instead), but it IS a cheap, permanent
    # tripwire against any FUTURE change that silently drops a queued
    # worker (e.g. a node that reaches `composer_node` through a new code
    # path without going through `_next_worker_goto`), which would
    # otherwise look identical to the doc_worker->math_worker handoff bug
    # this was added to guard against.
    completed = state.get("completed_workers") or []
    missing = [w for w in planned if w not in completed]
    if missing:
        print(
            f"[HANDOFF WARNING] planned_workers={planned} but completed_workers={completed} "
            f"-- missing={missing}. Turn ended without running every planned worker.",
            flush=True,
        )

    canvas_ops = list(state.get("canvas_ops") or [])
    branch_from_id = (state.get("canvas_branch_from_id") or "").strip() or None
    socratic = _is_socratic_mode(state)
    if not canvas_ops and desk_payloads:
        for payload in desk_payloads:
            if isinstance(payload, dict):
                canvas_ops.extend(
                    _payload_to_canvas_ops(
                        payload,
                        branch_from_id=branch_from_id,
                        socratic=socratic,
                    )
                )
    canvas_ops = [
        _stamp_canvas_target(op) if isinstance(op, dict) and op.get("target") != "canvas" else op
        for op in canvas_ops
        if isinstance(op, dict) and op.get("target") != "chat"
    ]
    explain_source = str(((config.get("configurable") or {}).get("explain_source") or "")).strip().lower()
    if explain_source == "chat":
        canvas_ops = []
    elif branch_from_id and explain_source == "desk":
        stamped = []
        for op in canvas_ops:
            if not isinstance(op, dict):
                continue
            if op.get("op") == "result":
                continue
            if op.get("op") == "step" and not op.get("branchFromId"):
                op = {**op, "branchFromId": branch_from_id}
            stamped.append(_stamp_canvas_target(op) if isinstance(op, dict) else op)
        canvas_ops = stamped
        has_solution = any(
            isinstance(op, dict) and op.get("op") in ("step", "result") for op in canvas_ops
        )
        if not has_solution:
            for index, latex in enumerate(_split_explanation_steps(chat_reply)):
                canvas_ops.append(_stamp_canvas_target({
                    "op": "step",
                    "index": index,
                    "latex": latex,
                    "branchFromId": branch_from_id,
                }))
        chat_reply = EXPLANATION_ACK

    if not branch_from_id:
        raw_extract = state.get("extracted_text") or ""
        _, extract_body = _parse_objective_line(raw_extract)
        peeled = _strip_extract_filler(extract_body)
        isolated = _isolate_problem_stem(peeled)
        extract_body = _without_goal_sentences(isolated, state)
        if not extract_body and peeled:
            if (
                not _looks_like_internal_prompt(peeled)
                and not _looks_like_coach_speak(peeled)
                and _looks_like_problem_stem(peeled)
            ):
                extract_body = _without_goal_sentences(peeled, state)
                print(
                    "[COMPOSER] question card fallback to peeled extract "
                    f"peeled_len={len(peeled)}",
                    flush=True,
                )
        print(
            "[SOCRATIC DEBUG] composer after stem isolate "
            f"extract_len={len(raw_extract)} "
            f"stem_len={len(extract_body or '')} "
            f"isolated_len={len(isolated or '')} "
            f"peeled_len={len(peeled or '')}",
            flush=True,
        )
        stem_ok = (
            bool(extract_body)
            and not _looks_like_internal_prompt(extract_body)
            and not _looks_like_coach_speak(extract_body)
            and _looks_like_problem_stem(extract_body)
        )
        print(
            f"[COMPOSER] stem_ok={stem_ok} stem_len={len(extract_body or '')}",
            flush=True,
        )
        # Always try to hydrate the parent card from a real PDF stem, even
        # in Socratic mode when canvas_anchor_id already points at a
        # placeholder. Why/How laterals stay step-only (this block is skipped).
        fresh_card = bool(state.get("fresh_question_card"))
        if stem_ok or (
            fresh_card
            and extract_body
            and not _looks_like_internal_prompt(extract_body)
        ):
            title = (active_problem or {}).get("title") or ""
            number = str((state.get("active_image") or {}).get("active_number") or "")
            source = f"Problem {number}" if fresh_card and number else title
            question_op = {
                "op": "question",
                "prompt": extract_body,
                "source": source,
            }
            if fresh_card:
                question_op["freshCard"] = True
            canvas_ops = [
                op for op in canvas_ops
                if not (isinstance(op, dict) and op.get("op") == "question")
            ]
            canvas_ops.insert(0, _stamp_canvas_target(question_op))
        if extract_body and not _looks_like_internal_prompt(extract_body):
            print("[SOCRATIC DEBUG] composer before vision consume", flush=True)
            figure_urls = consume_vision_figure_urls(config)
            print(
                "[SOCRATIC DEBUG] composer after vision consume "
                f"n_urls={len(figure_urls)}",
                flush=True,
            )
            for index, image_url in enumerate(figure_urls):
                canvas_ops.append(_stamp_canvas_target({
                    "op": "figure",
                    "image_url": image_url,
                    "index": index,
                }))
    else:
        print("[SOCRATIC DEBUG] composer stem/vision skipped (branch_from)", flush=True)

    # Never promote chat_reply / payload summary / coach-speak into a
    # question card. Question ops come only from a verified extract stem.
    canvas_ops = [
        op
        for op in canvas_ops
        if not (
            isinstance(op, dict)
            and op.get("op") == "question"
            and (
                _looks_like_coach_speak(str(op.get("prompt") or ""))
                or _looks_like_internal_prompt(str(op.get("prompt") or ""))
            )
        )
    ]

    has_solution = any(
        isinstance(op, dict) and op.get("op") in ("step", "result") for op in canvas_ops
    )
    if "math_worker" not in completed and not has_solution:
        canvas_ops = []

    print(
        "[SOCRATIC DEBUG] composer EXIT "
        f"chat_reply={chat_reply!r:.160} n_ops={len(canvas_ops)}",
        flush=True,
    )
    return Command(
        update={
            "chat_reply": chat_reply,
            "desk_payload": desk_payload,
            "desk_payloads": desk_payloads,
            "canvas_ops": canvas_ops,
            "canvas_anchor_id": state.get("canvas_anchor_id"),
            "canvas_branch_from_id": branch_from_id,
            "active_problem": active_problem,
        },
        goto=END,
    )


workflow = StateGraph(AgentState)

workflow.add_node("router_node", router_node)
workflow.add_node("chat_worker_node", chat_worker_node)
workflow.add_node("doc_worker_node", doc_worker_node)
workflow.add_node("doc_tools_node", doc_tools_node)
workflow.add_node("math_worker_node", math_worker_node)
workflow.add_node("math_tools_node", math_tools_node)
workflow.add_node("web_worker_node", web_worker_node)
workflow.add_node("web_tools_node", web_tools_node)
workflow.add_node("composer_node", composer_node)

workflow.add_edge(START, "router_node")

# MICRO-ROUTER & SPECIALIZED WORKERS — ONE-WAY, FAST ROUTING.
#
#                              ┌──────────────┐
#                    START ──▶ │ router_node  │  (1 structured LLM call;
#                              └──────┬───────┘   decides the FULL plan once,
#                                     │            NEVER produces a reply itself)
#                                     │ Command(goto=<first worker in plan>)
#                                     ▼
#      ┌─────────────────┐   ┌─────────────────┐   ┌────────────────┐   ┌─────────────────┐
#      │ chat_worker_node │  │ doc_worker_node │──▶│ math_worker_node │──▶│ web_worker_node │  (each
#      │  (plain chat,    │  │  ↕ doc_tools    │   │  ↕ math_tools     │   │  ↕ web_tools    │  optional;
#      │   no tools)      │  └────────┬────────┘   └─────────┬─────────┘   └────────┬────────┘  order = plan)
#      └────────┬─────────┘           └────────────────────────┴───────────────────────┘
#               └──────────────────────────────────┴───────────────────────────────────────┘
#                                                   ▼
#                                            ┌──────────────┐
#                                            │ composer_node│  (no LLM call; assembles
#                                            └──────┬───────┘   the dual payload)
#                                                   ▼
#                                                  END
#
# `worker_queue` (popped by each worker before its own `Command(goto=...)`)
# is what makes every hop after the router deterministic and LLM-free —
# no node ever routes back to `router_node`, and no node re-evaluates
# "what's next" via another LLM call. That is what keeps routing one-way
# and fast, instead of the old ping-pong evaluator-loop pattern. Each
# worker's own tool loop (worker <-> its local tools node) is a separate,
# bounded, single-purpose loop that never touches the router either.
# `chat_worker_node` (Phase 4) is the trivial-turn worker: it is what
# replaced the router's old in-line `direct_response` fast path, so
# chitchat/vision turns now stream clean prose exactly like any other
# worker's final answer, with no risk of raw structured-output JSON
# leaking into the token stream.
#
# Every node above is a coroutine (same reason as before): the graph must
# run via `.ainvoke()`/`.astream()`, and compilation (with `AsyncSqliteSaver`)
# happens in main.py's FastAPI `lifespan`, where an event loop is guaranteed
# to be running.
