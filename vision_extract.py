"""Lazy PDF page render + fast OCR + Gemini Flash extract / page routing.

Called from `locate_marker_in_range` / `read_page_range`. Does not import
tools.py (avoids a cycle with agents.py).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from typing import Optional

import pymupdf as fitz
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from locknlearn_schemas import (
    DiagramBox,
    ImagePageInventory,
    LocatePageResult,
    VisionExtract,
)

load_dotenv(find_dotenv(".env"))

FAST_OCR_DPI = 96
VISION_DPI = 144
FAST_OCR_MAX_PAGES = 36
VISION_TIMEOUT_SECONDS = 60
LOCATE_TIMEOUT_SECONDS = 20
VISION_MODEL = os.getenv("VISION_EXTRACT_MODEL") or os.getenv(
    "LLM_MODEL_NAME", "gemini-3.5-flash"
)
DEEPSEEK_ONLY = os.getenv("DEEPSEEK_ONLY", "").strip() == "1"
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_rapidocr_engine = None
_vision_llm = None
_locate_llm = None
_inventory_llm = None

_PAGE_TAG_RE = re.compile(
    r"=== PAGE printed=(\S+) abs=(\d+) ===",
)

_SECTION_INTENT = {
    "problems": (
        "Problems / Exercises / Məsələlər — numerical end-of-chapter "
        "problems. IGNORE Conceptual Questions, Objective / multiple-choice "
        "/ Quick Quiz banks, worked examples, and chapter theory."
    ),
    "conceptual": (
        "Conceptual Questions / Konseptual suallar. IGNORE Problems/"
        "Exercises and Objective / multiple-choice banks."
    ),
    "objective": (
        "Objective / multiple-choice / Quick Quiz. IGNORE Problems/"
        "Exercises and Conceptual Questions."
    ),
}

_LOCATE_PROMPT = """Here is the raw OCR / extracted text of a textbook chapter.
Each page is tagged exactly like:
=== PAGE printed=<folio printed on the page> abs=<1-indexed PDF page> ===

The user wants numbered item {marker} from this section:
{section_intent}

Navigate the heading hierarchy. A book often has BOTH a Questions/Conceptual
bank AND a later Problems/Exercises bank that reuse the same numbers.
Ignore irrelevant sections. Return ONLY the page where this EXACT item
starts in the requested bank.

Rules:
- found=true only if that number is a real heading in the requested bank.
- abs_page MUST be the integer after `abs=` in a PAGE tag (PDF page), NOT
  the printed folio, NOT a number from the problem itself.
- printed_label is the `printed=` value from that same tag.
- section_matched is one of: problems, conceptual, objective, unknown.
- If the number is missing from the requested bank, found=false and abs_page=0.
"""

_EXTRACT_PROMPT = """You are extracting ONE numbered textbook problem from page images.

The student asked for problem number {marker}.
Image 1 is page N (where the heading was found). Image 2 (if present) is page N+1.

Rules:
- Return the problem stem verbatim in the document's original language. Do not translate.
- Include every sentence, given value, and formula that belongs to THIS number only.
- If the problem continues at the bottom of image 1 onto image 2, include that continuation. Set continues_on_next_page true.
- Convert printed math to LaTeX ($...$ / $$...$$). Do not solve the problem.
- Do not include the next problem. Do not add preamble ("here is the question").
- If a figure/graph/photo belongs to this problem, set diagrams with 0-1 normalized boxes (origin top-left of THAT image). page_offset 0 = image 1, 1 = image 2.
- CRITICAL — DIAGRAM VALUES: a bounding box is not enough. You MUST also read every printed numeral, variable, unit, and short label written INSIDE or on that figure (examples: l = 0.350 m, I = 2.0 A, "Fig. 29-40", axis ticks, current arrows) and merge them into stem as given data so a downstream math agent can compute with them. Do not drop a value because it appears only in the art, not in the paragraph.
- If the figure for this problem is clearly on the page BEFORE image 1, set figure_on_previous_page true and do not guess a box on these images.
- If this number is not actually on these pages, found=false and stem empty.
"""

_PREV_PAGE_PROMPT = """The figure for problem {marker} may be on the previous page.

Image 1 is page N-1. Image 2 is page N (the page with the problem heading).

Return the full stem if any problem text starts on image 1.
If the figure is on image 1, put its 0-1 box in diagrams with page_offset 0 (image 1) or 1 (image 2).
CRITICAL — DIAGRAM VALUES: also read every printed numeral, variable, unit, and short label written inside that figure (e.g. l = 0.350 m) and merge them into stem as given data. A crop box alone is not enough.
Do not solve. Verbatim extract only.
"""

_SCREENSHOT_EXTRACT_PROMPT = """You are extracting ONE problem from a student screenshot or photo.

Rules:
- Return the problem stem verbatim in the document's original language. Do not translate.
- Include every sentence, given value, and formula that belongs to THIS problem.
- Convert printed math to LaTeX ($...$ / $$...$$). Do not solve the problem.
- Do not add preamble ("here is the question").
- If a figure/graph/photo/illustration belongs to this problem, set diagrams with 0-1 normalized boxes (origin top-left of THIS image). page_offset 0.
- Box ONLY the illustration, never the problem paragraph. Never return a box covering the whole image.
- If the image is text-only (problem statement with no diagram, graph, or illustration), leave diagrams empty.
- CRITICAL — DIAGRAM VALUES: a bounding box is not enough. You MUST also read every printed numeral, variable, unit, and short label written INSIDE or on that figure (examples: l = 0.350 m, I = 2.0 A, "Fig. 29-40", axis ticks, current arrows) and merge them into stem as given data so a downstream math agent can compute with them. Do not drop a value because it appears only in the art, not in the paragraph.
- If the image is not a solvable math/physics problem, found=false and stem empty.
"""

_SCREENSHOT_INVENTORY_PROMPT = """You are inventorying every numbered question on a student screenshot or photo.

Rules:
- Return one entry per numbered question visible on the image, in reading order.
- `number` is the printed question number as digits only (for example 20).
- `stem` is that question verbatim in the document's original language. Do not translate. Do not solve.
- Include every sentence, given value, and formula that belongs to that number.
- Convert printed math to LaTeX ($...$ / $$...$$).
- Do not merge sibling questions into one stem.
- Diagram boxes belong ONLY to question {marker}. If that is "the first", use the first numbered question.
- Box ONLY the illustration for that one question, never the paragraph and never the whole image. page_offset 0. Coordinates are 0-1, origin top-left.
- If that question's figure contains printed numerals, variables, units, or labels, merge them into THAT question's stem as given data.
- If the image has no numbered questions, return an empty questions list.
"""


def render_page_jpeg(file_path: str, abs_page: int, dpi: int) -> bytes:
    """1-indexed PDF page → JPEG bytes. Raises if the page does not exist."""
    doc = fitz.open(file_path)
    try:
        pix = _render_pixmap(doc, abs_page, dpi)
        return pix.tobytes("jpeg")
    finally:
        doc.close()


def render_page_rgb(file_path: str, abs_page: int, dpi: int):
    """1-indexed PDF page → RGB uint8 ndarray (copied, safe after close)."""
    import numpy as np

    doc = fitz.open(file_path)
    try:
        pix = _render_pixmap(doc, abs_page, dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        ).copy()
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        return img
    finally:
        doc.close()


def _render_pixmap(doc, abs_page: int, dpi: int):
    if abs_page < 1 or abs_page > doc.page_count:
        raise IndexError(f"PDF page {abs_page} out of range 1..{doc.page_count}")
    zoom = dpi / 72.0
    return doc.load_page(abs_page - 1).get_pixmap(
        matrix=fitz.Matrix(zoom, zoom), alpha=False,
    )


def _get_rapidocr():
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr import RapidOCR
        print("[VISION] RapidOCR engine loading...", flush=True)
        t0 = time.time()
        _rapidocr_engine = RapidOCR()
        print(f"[VISION] RapidOCR engine ready in {time.time() - t0:.2f}s", flush=True)
    return _rapidocr_engine


def ocr_jpeg_bytes(jpeg: bytes) -> str:
    """Low-fidelity page text for question-number localization only."""
    import numpy as np

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = None
    try:
        import cv2
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        from io import BytesIO
        from PIL import Image
        img = np.array(Image.open(BytesIO(jpeg)).convert("RGB"))
    return ocr_rgb_array(img)


def ocr_rgb_array(img) -> str:
    engine = _get_rapidocr()
    result = engine(img)
    return "\n".join(_rapidocr_lines(result))


def _rapidocr_lines(result) -> list[str]:
    txts = getattr(result, "txts", None)
    if txts is not None:
        try:
            lines = [str(t).strip() for t in list(txts) if str(t).strip()]
        except TypeError:
            lines = []
        if lines:
            return lines
    # Older RapidOCR: (det_results, elapse) where det_results is
    # [[box, text, score], ...]
    payload = result[0] if isinstance(result, (list, tuple)) and result else result
    if isinstance(payload, (list, tuple)):
        lines = []
        for item in payload:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text = str(item[1]).strip()
                if text:
                    lines.append(text)
        return lines
    return []


def _gemini_api_key() -> Optional[str]:
    return os.getenv("DOC_WORKER_API_KEY") or os.getenv(
        "GOOGLE_API_KEY", os.getenv("PRIMARY_ASSISTANT_API_KEY")
    )


def _extract_json_object(text: str) -> str:
    """Strip a Markdown fence and return the first {...} span of `text`."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    return body[start:end + 1] if 0 <= start < end else body


def _reply_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content or []
    )


def _append_to_text(content, extra: str):
    """Append `extra` to the first text part (or the string) of a message body."""
    if isinstance(content, str):
        return content + extra
    parts = [dict(p) if isinstance(p, dict) else p for p in content]
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            part["text"] = part.get("text", "") + extra
            return parts
    return [{"type": "text", "text": extra.strip()}, *parts]


class _DeepSeekJsonLLM:
    """DeepSeek vision rejects `response_format` and tools, so the schema is
    sent in the prompt and the reply is validated into the Pydantic model.
    `ainvoke` returns the model, or None when no valid JSON came back."""

    def __init__(self, schema: type[BaseModel], timeout: float):
        self._schema = schema
        self._footer = (
            "\n\nReply with ONLY one JSON object, no Markdown fences and no "
            "commentary, that matches this JSON Schema:\n"
            + json.dumps(schema.model_json_schema())
        )
        self._client = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_VISION_MODEL,
            temperature=0.0,
            max_retries=2,
            timeout=timeout,
        )

    async def ainvoke(self, messages):
        content = _append_to_text(messages[0].content, self._footer)
        for attempt in range(2):
            reply = await self._client.ainvoke([HumanMessage(content=content)])
            text = _reply_text(reply.content)
            try:
                return self._schema.model_validate_json(_extract_json_object(text))
            except ValueError as e:
                print(
                    f"[VISION] DeepSeek JSON invalid ({self._schema.__name__}, "
                    f"attempt {attempt + 1}): {str(e)[:200]}",
                    flush=True,
                )
                content = _append_to_text(
                    messages[0].content,
                    self._footer
                    + f"\n\nYour previous reply was invalid: {str(e)[:300]}\n"
                    "Return the corrected JSON object only.",
                )
        return None


def _vision_structured(schema: type[BaseModel], timeout: float):
    if DEEPSEEK_ONLY and os.getenv("DEEPSEEK_API_KEY"):
        return _DeepSeekJsonLLM(schema, timeout)
    return ChatGoogleGenerativeAI(
        model=VISION_MODEL,
        google_api_key=_gemini_api_key(),
        temperature=0.0,
        max_retries=2,
        timeout=timeout,
    ).with_structured_output(schema, method="json_schema")


def _get_vision_llm():
    global _vision_llm
    if _vision_llm is None:
        _vision_llm = _vision_structured(VisionExtract, VISION_TIMEOUT_SECONDS)
    return _vision_llm


def _get_inventory_llm():
    global _inventory_llm
    if _inventory_llm is None:
        _inventory_llm = _vision_structured(ImagePageInventory, VISION_TIMEOUT_SECONDS)
    return _inventory_llm


def _get_locate_llm():
    global _locate_llm
    if _locate_llm is None:
        _locate_llm = _vision_structured(LocatePageResult, LOCATE_TIMEOUT_SECONDS)
    return _locate_llm


def _tagged_abs_pages(tagged_text: str) -> dict[int, str]:
    """abs_page → printed_label from PAGE tags."""
    found: dict[int, str] = {}
    for match in _PAGE_TAG_RE.finditer(tagged_text or ""):
        found[int(match.group(2))] = match.group(1)
    return found


def _coerce_locate_result(
    result: LocatePageResult,
    tagged_text: str,
    abs_start: int,
    abs_end: int,
) -> LocatePageResult:
    tags = _tagged_abs_pages(tagged_text)
    if not result.found:
        return LocatePageResult(found=False)

    abs_page = int(result.abs_page or 0)
    if abs_page not in tags:
        label = str(result.printed_label or "").strip()
        if label:
            for tagged_abs, printed in tags.items():
                if printed == label:
                    abs_page = tagged_abs
                    break
    if abs_page not in tags or abs_page < abs_start or abs_page > abs_end:
        print(
            f"[VISION] Flash locate rejected abs_page={result.abs_page} "
            f"label={result.printed_label!r} span={abs_start}-{abs_end}",
            flush=True,
        )
        return LocatePageResult(found=False)

    matched = (result.section_matched or "").strip().lower()
    if matched not in {"problems", "conceptual", "objective", "unknown"}:
        matched = "unknown"
    return LocatePageResult(
        found=True,
        abs_page=abs_page,
        printed_label=tags.get(abs_page) or result.printed_label or str(abs_page),
        section_matched=matched,
    )


async def llm_locate_page(
    tagged_text: str,
    marker: str,
    section_kind: str,
    abs_start: int,
    abs_end: int,
) -> LocatePageResult:
    """Text-only Flash: return the PDF page of marker in section_kind.

    Timeout is independent of VISION_TIMEOUT_SECONDS so a hung locate
    does not block the later N/N+1 image call.
    """
    if not (tagged_text or "").strip():
        return LocatePageResult(found=False)

    kind = (section_kind or "problems").strip().lower()
    if kind not in _SECTION_INTENT:
        kind = "problems"
    prompt = _LOCATE_PROMPT.format(
        marker=marker,
        section_intent=_SECTION_INTENT[kind],
    )
    print(
        f"[VISION] Flash locate marker={marker} kind={kind} "
        f"span={abs_start}-{abs_end} chars={len(tagged_text)}",
        flush=True,
    )
    t0 = time.time()
    try:
        raw = await asyncio.wait_for(
            _get_locate_llm().ainvoke(
                [HumanMessage(content=prompt + "\n\n" + tagged_text)],
            ),
            timeout=LOCATE_TIMEOUT_SECONDS,
        )
    except Exception as e:
        print(f"[VISION] Flash locate failed: {e}", flush=True)
        return LocatePageResult(found=False)
    print(f"[VISION] Flash locate done in {time.time() - t0:.2f}s", flush=True)

    if not isinstance(raw, LocatePageResult):
        return LocatePageResult(found=False)
    return _coerce_locate_result(raw, tagged_text, abs_start, abs_end)


def _image_part(jpeg: bytes) -> dict:
    b64 = base64.b64encode(jpeg).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _clamp_boxes(extract: VisionExtract) -> VisionExtract:
    kept: list[DiagramBox] = []
    for box in extract.diagrams or []:
        x0, y0 = min(box.x_min, box.x_max), min(box.y_min, box.y_max)
        x1, y1 = max(box.x_min, box.x_max), max(box.y_min, box.y_max)
        x0, y0 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, y0))
        x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area < 0.01 or area > 0.80:
            continue
        kept.append(DiagramBox(
            page_offset=box.page_offset,
            x_min=x0, y_min=y0, x_max=x1, y_max=y1,
        ))
    extract.diagrams = kept
    return extract


async def vision_extract_pages(
    file_path: str,
    abs_pages: list[int],
    marker: str,
    previous_page: bool = False,
) -> VisionExtract:
    """Render `abs_pages` at VISION_DPI and ask Gemini Flash for a stem."""
    pages = [p for p in abs_pages if p >= 1]
    if not pages:
        return VisionExtract(found=False, stem="")

    jpegs: list[bytes] = []
    for page in pages:
        jpeg = await asyncio.to_thread(render_page_jpeg, file_path, page, VISION_DPI)
        jpegs.append(jpeg)

    prompt = (_PREV_PAGE_PROMPT if previous_page else _EXTRACT_PROMPT).format(
        marker=marker,
    )
    content: list = [{"type": "text", "text": prompt}]
    content.extend(_image_part(jpeg) for jpeg in jpegs)

    print(
        f"[VISION] LLM extract marker={marker} pages={pages} "
        f"prev={previous_page} bytes={sum(len(j) for j in jpegs)}",
        flush=True,
    )
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            _get_vision_llm().ainvoke([HumanMessage(content=content)]),
            timeout=VISION_TIMEOUT_SECONDS,
        )
    except Exception as e:
        print(f"[VISION] LLM extract failed: {e}", flush=True)
        return VisionExtract(found=False, stem="")
    print(f"[VISION] LLM extract done in {time.time() - t0:.2f}s", flush=True)

    if not isinstance(result, VisionExtract):
        return VisionExtract(found=False, stem="")
    return _clamp_boxes(result)


async def vision_inventory_image_file(file_path: str, marker: str = "") -> ImagePageInventory:
    """One Gemini pass over a screenshot: every numbered stem, plus diagram
    boxes for the requested question (or the first, when none was named)."""
    jpeg = await asyncio.to_thread(_load_image_jpeg_bytes, file_path)
    if not jpeg:
        return ImagePageInventory()
    which = marker.strip() or "the first"
    content: list = [
        {"type": "text", "text": _SCREENSHOT_INVENTORY_PROMPT.format(marker=which)},
        _image_part(jpeg),
    ]
    print(f"[VISION] LLM screenshot inventory bytes={len(jpeg)} marker={which}", flush=True)
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            _get_inventory_llm().ainvoke([HumanMessage(content=content)]),
            timeout=VISION_TIMEOUT_SECONDS,
        )
    except Exception as e:
        print(f"[VISION] LLM screenshot inventory failed: {e}", flush=True)
        return ImagePageInventory()
    print(f"[VISION] LLM screenshot inventory done in {time.time() - t0:.2f}s", flush=True)
    if not isinstance(result, ImagePageInventory):
        return ImagePageInventory()
    clamped = _clamp_boxes(VisionExtract(found=True, stem="", diagrams=result.diagrams))
    result.diagrams = clamped.diagrams
    return result


async def vision_extract_image_file(file_path: str, marker: str = "") -> VisionExtract:
    """Structured stem + optional diagram boxes for a screenshot, same schema as PDF pages."""
    jpeg = await asyncio.to_thread(_load_image_jpeg_bytes, file_path)
    if not jpeg:
        return VisionExtract(found=False, stem="")
    prompt = _SCREENSHOT_EXTRACT_PROMPT
    if marker.strip():
        prompt += f"\nExtract ONLY question number {marker.strip()}. Ignore the other questions on the page."
    content: list = [
        {"type": "text", "text": prompt},
        _image_part(jpeg),
    ]
    print(
        f"[VISION] LLM screenshot extract bytes={len(jpeg)}",
        flush=True,
    )
    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            _get_vision_llm().ainvoke([HumanMessage(content=content)]),
            timeout=VISION_TIMEOUT_SECONDS,
        )
    except Exception as e:
        print(f"[VISION] LLM screenshot extract failed: {e}", flush=True)
        return VisionExtract(found=False, stem="")
    print(f"[VISION] LLM screenshot extract done in {time.time() - t0:.2f}s", flush=True)
    if not isinstance(result, VisionExtract):
        return VisionExtract(found=False, stem="")
    return _clamp_boxes(result)


async def vision_extract_question(
    file_path: str,
    page_n: int,
    page_count: int,
    marker: str,
) -> VisionExtract:
    """Page N + N+1, then N-1 if the model says the figure is behind."""
    spread = [page_n]
    if page_n + 1 <= page_count:
        spread.append(page_n + 1)
    extract = await vision_extract_pages(file_path, spread, marker, previous_page=False)
    if not extract.figure_on_previous_page or page_n <= 1:
        return extract

    print(f"[VISION] N-1 fallback for marker={marker} page={page_n}", flush=True)
    prev = await vision_extract_pages(
        file_path, [page_n - 1, page_n], marker, previous_page=True,
    )
    if prev.stem and (not extract.stem or len(prev.stem) > len(extract.stem)):
        extract.stem = prev.stem
        extract.found = extract.found or prev.found
    shifted = []
    for box in prev.diagrams:
        # Prev-call images are [N-1, N] → offsets 0/-1 become -1, 1 becomes 0.
        new_offset = -1 if box.page_offset <= 0 else 0
        shifted.append(box.model_copy(update={"page_offset": new_offset}))
    extract.diagrams = shifted + list(extract.diagrams or [])
    extract.figure_on_previous_page = True
    return _clamp_boxes(extract)


def crop_page_box(file_path: str, abs_page: int, box: DiagramBox, dpi: int = VISION_DPI) -> bytes:
    """Clip a 0–1 box from a PDF page and return JPEG bytes."""
    doc = fitz.open(file_path)
    try:
        if abs_page < 1 or abs_page > doc.page_count:
            raise IndexError(f"PDF page {abs_page} out of range 1..{doc.page_count}")
        page = doc.load_page(abs_page - 1)
        rect = page.rect
        pad = 0.015
        x0 = max(0.0, min(1.0, min(box.x_min, box.x_max) - pad))
        y0 = max(0.0, min(1.0, min(box.y_min, box.y_max) - pad))
        x1 = max(0.0, min(1.0, max(box.x_min, box.x_max) + pad))
        y1 = max(0.0, min(1.0, max(box.y_min, box.y_max) + pad))
        clip = fitz.Rect(
            rect.x0 + x0 * rect.width,
            rect.y0 + y0 * rect.height,
            rect.x0 + x1 * rect.width,
            rect.y0 + y1 * rect.height,
        )
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        return pix.tobytes("jpeg")
    finally:
        doc.close()


def crop_diagrams_to_data_urls(
    file_path: str,
    page_n: int,
    diagrams: list,
    page_count: Optional[int] = None,
) -> list[str]:
    """Render up to 3 clamped boxes as `data:image/jpeg;base64,...` URLs."""
    urls: list[str] = []
    for box in (diagrams or [])[:3]:
        if not isinstance(box, DiagramBox):
            try:
                box = DiagramBox.model_validate(box)
            except Exception:
                continue
        abs_page = page_n + int(box.page_offset or 0)
        if page_count and (abs_page < 1 or abs_page > page_count):
            continue
        try:
            jpeg = crop_page_box(file_path, abs_page, box)
        except Exception as e:
            print(f"[VISION] crop failed page={abs_page}: {e}", flush=True)
            continue
        if not jpeg:
            continue
        urls.append("data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"))
    return urls


FIGURE_MAX_EDGE_PX = 1280
FIGURE_JPEG_QUALITY = 70
_CROP_PAD = 0.015


def _load_image_jpeg_bytes(file_path: str) -> bytes:
    """Downscale a raster screenshot for Gemini / SSE (long edge capped)."""
    if not file_path or not os.path.isfile(file_path):
        return b""
    from io import BytesIO
    from PIL import Image

    img = Image.open(file_path).convert("RGB")
    width, height = img.size
    longest = max(width, height)
    if longest > FIGURE_MAX_EDGE_PX:
        scale = FIGURE_MAX_EDGE_PX / float(longest)
        img = img.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=FIGURE_JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def image_file_to_jpeg_data_url(file_path: str) -> str:
    """Resize a screenshot/photo and return a JPEG data URL."""
    jpeg = _load_image_jpeg_bytes(file_path)
    if not jpeg:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def crop_image_box(file_path: str, box: DiagramBox) -> bytes:
    """Clip a 0–1 box from a raster screenshot and return JPEG bytes."""
    if not file_path or not os.path.isfile(file_path):
        return b""
    from io import BytesIO
    from PIL import Image

    img = Image.open(file_path).convert("RGB")
    width, height = img.size
    x0 = max(0.0, min(1.0, min(box.x_min, box.x_max) - _CROP_PAD))
    y0 = max(0.0, min(1.0, min(box.y_min, box.y_max) - _CROP_PAD))
    x1 = max(0.0, min(1.0, max(box.x_min, box.x_max) + _CROP_PAD))
    y1 = max(0.0, min(1.0, max(box.y_min, box.y_max) + _CROP_PAD))
    left = int(x0 * width)
    top = int(y0 * height)
    right = max(left + 1, int(x1 * width))
    bottom = max(top + 1, int(y1 * height))
    crop = img.crop((left, top, right, bottom))
    cw, ch = crop.size
    longest = max(cw, ch)
    if longest > FIGURE_MAX_EDGE_PX:
        scale = FIGURE_MAX_EDGE_PX / float(longest)
        crop = crop.resize(
            (max(1, int(cw * scale)), max(1, int(ch * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=FIGURE_JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def crop_image_diagrams_to_data_urls(file_path: str, diagrams: list) -> list[str]:
    """Render up to 3 clamped screenshot boxes as JPEG data URLs."""
    urls: list[str] = []
    for box in (diagrams or [])[:3]:
        if not isinstance(box, DiagramBox):
            try:
                box = DiagramBox.model_validate(box)
            except Exception:
                continue
        try:
            jpeg = crop_image_box(file_path, box)
        except Exception as e:
            print(f"[VISION] screenshot crop failed: {e}", flush=True)
            continue
        if not jpeg:
            continue
        urls.append("data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"))
    return urls


async def ocr_pdf_page(file_path: str, abs_page: int) -> str:
    def _run() -> str:
        img = render_page_rgb(file_path, abs_page, FAST_OCR_DPI)
        text = ocr_rgb_array(img)
        print(f"[VISION] OCR page {abs_page} chars={len(text)}", flush=True)
        return text

    return await asyncio.to_thread(_run)
