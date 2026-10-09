from locknlearn_schemas import RoutePlan
from tools import doc_worker_tools, math_worker_tools, web_worker_tools

router_llm = None
chat_worker_llm = None
doc_worker_llm = None
math_worker_llm = None
web_worker_llm = None
chat_worker_vision_llm = None

_tool_groups = {
    "doc": doc_worker_tools,
    "math": math_worker_tools,
    "web": web_worker_tools,
}
_route_contract = RoutePlan
