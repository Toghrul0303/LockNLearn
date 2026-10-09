from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional

# ---------------------------------------------------------------------------
# Micro-Router contracts for the Micro-Router & Specialized Workers
# architecture — see graph.py for the node wiring that consumes these.
# ---------------------------------------------------------------------------

WorkerName = Literal["doc_worker", "math_worker", "web_worker", "chat_worker"]
"""Every specialized worker `router_node` is allowed to schedule. Kept as a
`Literal` (rather than a free-form string) so an invalid worker name is a
validation error at the router's structured-output boundary, not a silent
`KeyError`/no-op deep inside the graph.

`chat_worker` (Phase 4) replaces the old `direct_answer` sentinel — it is now
a REAL worker node (`chat_worker_node` in graph.py) rather than a branch
inside the router itself. See `RoutePlan`'s docstring for why."""


class RoutePlan(BaseModel):
    """Structured-output contract for the Micro-Router's single classification
    call (`router_node`).

    The router decides the FULL ordered sequence of workers needed for this
    turn exactly ONCE. Every downstream node then pops the next name off this
    list deterministically (via the `worker_queue` state field) — no node
    ever re-invokes the router mid-turn, which is what keeps routing one-way
    and fast instead of the old ping-pong evaluator-loop pattern.

    NOTE (Phase 4): this model USED to also carry a `direct_response: str`
    field so the router could answer trivial/chitchat turns inline, inside
    the same structured call, keeping the fast path to exactly one LLM call.
    That was removed: Gemini's structured-output API streams its JSON as
    partial text chunks, so `direct_response` was streaming raw, partially-
    formed JSON straight into the chat pane instead of clean prose — a
    fundamental conflict between "emit machine-parseable JSON" and "stream
    human-readable text" in a single call, not a fixable configuration
    tweak.     The router now ONLY ever produces `workers` — never user-facing text
    and never a `reasoning` essay (that field was removed after it inflated
    router latency to tens of seconds).
    """
    workers: List[WorkerName]


class SocraticGoalPlan(BaseModel):
    """Structured output of the once-per-problem Socratic goal planner.

    `entity_map` is private scratch (filled first so the model reasons before
    it commits); only `goals` is stored. Never shown to the student."""
    entity_map: str = Field(
        default="",
        description="Private scratch: every named actor, each given attached to its actor, then the question sentence.",
    )
    asked_actor: str = Field(
        default="",
        description="Which actor the question sentence asks about.",
    )
    goals: List[str] = Field(
        description="ONLY the quantities the question sentence asks the student to report; one item per asked target.",
    )
    units: List[str] = Field(
        default_factory=list,
        description="SI unit for each goal, same order as goals (e.g. m/s^2); empty string if dimensionless or unsure.",
    )
    symbols: List[str] = Field(
        default_factory=list,
        description="Report symbol for each goal, same order as goals (e.g. a, v, t). The symbol the answer is written as, not a component (v, not v_x). Empty string if unsure.",
    )


class DocumentContext(BaseModel):
    """Doc Worker's extracted/retrieved text, handed off through LangGraph
    state to downstream workers (e.g. Math Worker reads this instead of
    re-deriving it from chat history or re-querying the vector store)."""
    source_file: Optional[str] = None
    retrieved_text: str


class FormulaCard(BaseModel):
    title: str
    latex: str
    description: Optional[str] = None

class CalculationStep(BaseModel):
    step_number: int
    title: str
    formula: str
    result: str

class GraphData(BaseModel):
    x_label: str
    y_label: str
    data_points: List[dict]

class ActiveProblem(BaseModel):
    """Desk header: `title` is the top line (source + chapter + problem
    number, e.g. "Serway: Problem 1"); `description` is the subtitle
    (topic or a short objective). Internal routing prompts such as
    `[TASK START]` must never appear in either field.
    """
    source: str
    title: str = Field(
        ...,
        description=(
            'Clean user-facing header only, e.g. "Serway: Problem 1". '
            "Never include [TASK START], tool names, or routing instructions."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "Short topic or objective for the subtitle. Never copy "
            "[TASK START] routing text, locate_marker_in_range, or "
            "resolve_chapter_target instructions."
        ),
    )
    image_url: Optional[str] = None


class ActiveNavigation(BaseModel):
    """Persisted per-thread "where are we in the book" cursor — the
    solution to relative follow-ups like "Now move to Question 8" without
    a chapter name repeated.

    Written DETERMINISTICALLY by `doc_tools_node`/`doc_worker_node`
    (graph.py) from the actual arguments/results of `resolve_chapter_target`,
    `read_page_range`, or `locate_marker_in_range` this turn — NEVER inferred
    by an LLM guessing from chat history. This is what makes it survive
    `CONTEXT_WINDOW_SIZE` (graph.py only ever sends the LAST 4 messages to a
    worker): the field lives in LangGraph state, persisted by the
    checkpointer across turns, exactly like `ActiveProblem` above — not in
    the trimmed message window.

    For numbered textbook problems, `page_start`/`page_end` should stay the
    resolved bank span (Problems/Exercises), not shrink to a single found
    page — so "now question 8" still searches the same bank.
    `section_kind` is "problems" | "conceptual" | "objective" | "chapter".

    Kept as a plain `dict` in `AgentState` (see `active_problem`'s same
    treatment) rather than validated through this model at the state
    boundary; this class documents the shape both sides agree on.
    """
    source: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    last_question_marker: Optional[str] = None
    section_kind: Optional[str] = None
    section_title: Optional[str] = None


class DiagramBox(BaseModel):
    """Normalized crop box for a figure on a rendered PDF page.
    Coordinates are 0–1 of that page image, origin top-left.
    `page_offset`: 0 = page N, 1 = N+1, -1 = N-1.
    """
    page_offset: int = 0
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 1.0
    y_max: float = 1.0


class ImageQuestionStem(BaseModel):
    """One numbered problem transcribed from a screenshot page."""
    number: str = ""
    stem: str = ""

    @field_validator("number", mode="before")
    @classmethod
    def _number_text(cls, value):
        return "" if value is None else str(value)


class ImagePageInventory(BaseModel):
    """Every numbered question visible on one uploaded image.
    `diagrams` belong only to the question the student asked for, or the
    first numbered question when they did not name one."""
    questions: List[ImageQuestionStem] = []
    diagrams: List[DiagramBox] = []


class VisionExtract(BaseModel):
    """Gemini Flash structured extract of one textbook problem from page
    images (N and N+1, optionally N-1). `stem` is the verbatim question
    text for `extracted_text` / the canvas question card. Boxes are stored
    for a later canvas figure op — not invented SVG primitives.
    """
    found: bool = True
    stem: str = ""
    continues_on_next_page: bool = False
    figure_on_previous_page: bool = False
    diagrams: List[DiagramBox] = []


class LocatePageResult(BaseModel):
    """Fast-pass Flash text-mode page finder. Given tagged chapter OCR,
    returns the PDF page where the requested numbered item starts.
    `abs_page` is the 1-indexed PDF page (the `abs=` value in PAGE tags),
    not the printed folio. Never carries the OCR blob itself.
    """
    found: bool = False
    abs_page: int = 0
    printed_label: str = ""
    section_matched: str = "unknown"


class StruggleRecord(BaseModel):
    """Frontend-driven pedagogical memory: written when the student marks a
    solution as Needs Review, then recalled into worker prompts on later turns."""
    thread_id: str
    question_id: str
    topic: str
    excerpt: str
    formula: Optional[str] = None


class DeskState(BaseModel):
    active_problem: Optional[ActiveProblem] = None
    formulas: List[FormulaCard] = []
    steps: List[CalculationStep] = []
    graph: Optional[GraphData] = None

class LockNLearnPayload(BaseModel):
    """Wire-level contract streamed to the frontend over SSE (see main.py).

    `desk_update` is intentionally a plain `dict`, NOT `DeskState`. The
    `DeskState`/`FormulaCard`/`CalculationStep`/`GraphData` models above
    describe a richer, structured Desk shape that isn't what actually flows
    through today: `python_code_executor` (tools.py) emits an ad-hoc
    `{"type": "chart"|"calculation"|"diagram", ...}` payload. The frontend
    unpacks that into incremental `canvas_op` ticks (`question` / `step` /
    `result` / `chart` / `diagram`) against `canvas_anchor_id`. Forcing
    `desk_update` through `DeskState` validation would reject every real
    payload produced today. Reconciling the two schemas remains a separate
    future decision.
    """
    chat_message: str
    desk_update: Optional[dict] = None
    active_problem_update: Optional[dict] = None
    canvas_anchor_id: Optional[str] = None
    canvas_op: Optional[dict] = None