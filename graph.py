from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from agents import (
    chat_worker_llm,
    doc_worker_llm,
    math_worker_llm,
    router_llm,
    web_worker_llm,
)
from locknlearn_schemas import RoutePlan
from tools import doc_worker_tools, math_worker_tools, web_worker_tools
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Micro-Router scratch state ---------------------------------------
    # Reset to fresh values by `router_node` at the START of every turn (see
    # its docstring) â€” these represent THIS turn's plan and THIS turn's
    # worker output, never accumulated/leaked across turns via the
    # checkpointer.
    # The FULL ordered plan `router_node` decided this turn â€” set ONCE and
    # never mutated afterward (unlike `worker_queue`, which is consumed/
    # popped by each worker as it finishes). Kept around purely so
    # `composer_node` can compare "what was planned" against
    # `completed_workers` ("what actually ran") and flag any silent
    # discrepancy â€” see `composer_node`'s `[HANDOFF WARNING]` check.
    planned_workers: list[str]
    # This turn's triggering `HumanMessage`, captured ONCE by `router_node`
    # (the graph's entry point, so `state["messages"][-1]` is guaranteed to
    # be it) before any worker appends its own tool-call noise. Every
    # worker's `build_worker_messages()` call re-injects this if the plain
    # last-N-message trim window (`get_trimmed_messages`/
    # `CONTEXT_WINDOW_SIZE`) ever slices it out â€” which happens easily in a
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
    # `[EXTRACTED DOCUMENT CONTEXT]` message â€” see `math_worker_node` below.
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

    # Scratch, reset every turn by `router_node` â€” accumulates the resolved
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
    # Deliberately NOT reset every turn â€” represents "what document/problem
    # is currently active" for this conversation thread (mirrors
    # `locknlearn_schemas.ActiveProblem`), so it survives turns that don't
    # touch the Doc Worker (e.g. a plain follow-up chitchat message). Only
    # overwritten when `doc_worker_node` actually processes a file.
    active_problem: Optional[dict]

    # Socratic dynamic todo â€” session-persisted, NOT cleared by router scratch.
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

    # Deliberately NOT reset every turn either â€” "where in the book are we
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


async def router_node(state: AgentState) -> Command[Literal["chat_worker_node", "doc_worker_node", "math_worker_node", "web_worker_node", "composer_node"]]:
    pass


async def chat_worker_node(state: AgentState) -> Command[Literal["doc_worker_node", "math_worker_node", "web_worker_node", "composer_node"]]:
    pass


async def doc_worker_node(state: AgentState) -> Command[Literal["doc_tools_node", "math_worker_node", "web_worker_node", "chat_worker_node", "composer_node"]]:
    pass


async def doc_tools_node(state: AgentState) -> Command[Literal["doc_worker_node", "math_worker_node", "web_worker_node", "chat_worker_node", "composer_node"]]:
    pass


async def math_worker_node(state: AgentState) -> Command[Literal["math_tools_node", "doc_worker_node", "web_worker_node", "chat_worker_node", "composer_node"]]:
    pass


async def math_tools_node(state: AgentState) -> Command[Literal["math_worker_node", "doc_worker_node", "web_worker_node", "chat_worker_node", "composer_node"]]:
    pass


async def web_worker_node(state: AgentState) -> Command[Literal["web_tools_node", "doc_worker_node", "math_worker_node", "chat_worker_node", "composer_node"]]:
    pass


async def web_tools_node(state: AgentState) -> Command[Literal["web_worker_node", "doc_worker_node", "math_worker_node", "chat_worker_node", "composer_node"]]:
    pass


async def composer_node(state: AgentState) -> Command[Literal["__end__"]]:
    pass


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
