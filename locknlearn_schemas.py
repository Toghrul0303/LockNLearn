from typing import List, Literal

from pydantic import BaseModel


WorkerName = Literal["doc_worker", "math_worker", "web_worker", "chat_worker"]


class RoutePlan(BaseModel):
    """The router's one decision for this turn: which workers run, in order."""

    workers: List[WorkerName]
