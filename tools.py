from langchain_core.tools import tool


@tool
async def process_and_index_documents(file_path: str) -> str:
    """Read an uploaded file and prepare it for later retrieval."""
    pass


@tool
async def get_document_outline(file_path: str) -> str:
    """Return the chapter and section outline of the active document."""
    pass


@tool
async def resolve_chapter_target(marker: str) -> str:
    """Resolve a chapter, section, or question marker to a page range."""
    pass


@tool
async def read_page_range(start_page: int, end_page: int) -> str:
    """Read a page range from the active document."""
    pass


@tool
async def locate_marker_in_range(marker: str) -> str:
    """Find a question or heading inside the current page range."""
    pass


@tool
async def search_in_document(query: str) -> str:
    """Search the active document for a passage."""
    pass


@tool
async def python_code_executor(code: str) -> str:
    """Run a short Python calculation for the desk."""
    pass


@tool
async def web_search_tool(query: str) -> str:
    """Search the web for a citation or current fact."""
    pass


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
