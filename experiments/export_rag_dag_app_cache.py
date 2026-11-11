"""Recover this project's reference chat from already downloaded App cache pages.

Reads cache files only; makes no requests, reads no credentials, and changes no
App data. Windows shared reads allow inspecting a running App's cache.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from urllib.parse import parse_qs, urlsplit
import zlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference" / "chat_records"
CONVERSATION = "6ae50849-f0ec-83ea-bc99-56b26ddc7c1e"
CACHE = Path.home() / "AppData/Roaming/Codex/web/Codex/Default/Cache/Cache_Data"


def shared_read(path: Path) -> bytes:
    if sys.platform != "win32":
        return path.read_bytes()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
    kernel.ReadFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        length = ctypes.c_longlong()
        if not kernel.GetFileSizeEx(handle, ctypes.byref(length)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not 0 <= length.value <= 256 * 1024**2:
            raise ValueError(f"Unexpected cache file size: {path.name}")
        buf = ctypes.create_string_buffer(length.value)
        count = wintypes.DWORD()
        if not kernel.ReadFile(handle, buf, length.value, ctypes.byref(count), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if count.value != length.value:
            raise OSError(f"Incomplete shared read: {path.name}")
        return buf.raw
    finally:
        kernel.CloseHandle(handle)


def stream_bytes(address: int, size: int, blocks: dict[int, bytes]) -> bytes:
    if not size:
        return b""
    if not address & 0x80000000:
        raise ValueError("Uninitialized cache address")
    kind = (address >> 28) & 7
    if kind == 0:
        return shared_read(CACHE / f"f_{address & 0x0fffffff:06x}")[:size]
    selector = (address >> 16) & 255
    block = address & 65535
    block_size = {1: 36, 2: 256, 3: 1024, 4: 4096}[kind]
    if selector not in blocks:
        blocks[selector] = shared_read(CACHE / f"data_{selector}")
    start = 8192 + block * block_size
    result = blocks[selector][start : start + size]
    if len(result) != size:
        raise ValueError("Cache stream is incomplete")
    return result


def decode_body(body: bytes, metadata: bytes) -> bytes:
    match = re.search(br"(?:^|\x00)content-encoding:([^\x00]*)", metadata, re.I)
    encoding = match.group(1).strip().lower() if match else b""
    if encoding == b"br":
        import brotlicffi
        return brotlicffi.decompress(body)
    if encoding == b"gzip":
        return gzip.decompress(body)
    if encoding == b"deflate":
        return zlib.decompress(body)
    if encoding == b"zstd":
        import zstandard
        return zstandard.ZstdDecompressor().decompress(body)
    if encoding:
        raise ValueError(f"Unsupported content encoding: {encoding!r}")
    return body


def collect_pages() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    page_dir = OUT / "rag_dag_app_pages"
    page_dir.mkdir(exist_ok=True)
    source_file = page_dir / "_cache_sources.json"
    sources = json.loads(source_file.read_text(encoding="utf-8")) if source_file.exists() else {}
    blocks = {1: shared_read(CACHE / "data_1")}
    data = blocks[1]
    needle = CONVERSATION.encode()
    start, errors, found = 0, [], 0
    while (position := data.find(needle, start)) >= 0:
        start = position + len(needle)
        entry = 8192 + ((position - 8192) // 256) * 256
        if entry < 8192:
            continue
        key_size = struct.unpack_from("<i", data, entry + 32)[0]
        if not 1 <= key_size <= 927:
            continue
        key = data[entry + 96 : entry + 96 + key_size]
        if not any(
            b"/backend-api/" + segment + b"/" + needle in key
            for segment in (b"conversation", b"conversations")
        ):
            continue
        found += 1
        try:
            request_url = key.decode("utf-8", "replace").split(" ")[-1]
            url = urlsplit(request_url)
            base_paths = {
                f"/backend-api/{segment}/{CONVERSATION}"
                for segment in ("conversation", "conversations")
            }
            if url.hostname != "chatgpt.com" or url.path not in (
                base_paths | {path + "/messages" for path in base_paths}
            ):
                raise ValueError("Request URL is outside the selected conversation")
            sizes = struct.unpack_from("<4I", data, entry + 40)
            addresses = struct.unpack_from("<4I", data, entry + 56)
            metadata = stream_bytes(addresses[0], sizes[0], blocks)
            raw = decode_body(stream_bytes(addresses[1], sizes[1], blocks), metadata)
            payload = json.loads(raw)
            response_id = payload.get("conversation_id", payload.get("id"))
            if response_id is not None and response_id != CONVERSATION:
                raise ValueError("Conversation ID mismatch")
            if response_id is None and not (
                url.path.endswith("/messages") and isinstance(payload.get("messages"), list)
                and isinstance(payload.get("page_info"), dict)
            ):
                raise ValueError("Response is not a conversation message page")
            digest = hashlib.sha256(raw).hexdigest()
            (page_dir / f"page_{digest[:16]}.json").write_bytes(raw)
            sources[digest[:16]] = {
                "request_url": request_url,
                "response_sha256": digest,
                "entry_offset": entry,
            }
        except Exception as error:
            errors.append({"entry_offset": entry, "error": str(error)})
    print(json.dumps({"matching_cache_entries": found, "errors": errors}, ensure_ascii=False))
    source_file.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    return [json.loads(p.read_bytes()) for p in sorted(page_dir.glob("page_*.json"))]


def message_list(page: dict) -> list[dict]:
    if isinstance(page.get("messages"), list):
        return page["messages"]
    return [
        node["message"]
        for node in page.get("mapping", {}).values()
        if node.get("message")
    ]


def visible_text(message: dict) -> str:
    if message.get("author", {}).get("role") not in ("user", "assistant"):
        return ""
    if message.get("channel") not in (None, "final"):
        return ""
    if message.get("recipient") not in (None, "all"):
        return ""
    content = message.get("content", {})
    if content.get("content_type") not in ("text", "multimodal_text"):
        return ""
    return "\n".join(p for p in content.get("parts", []) if isinstance(p, str)).strip()


def rendered_text(message: dict) -> str:
    text = visible_text(message)
    for reference in message.get("metadata", {}).get("content_references", []):
        matched = reference.get("matched_text")
        if not matched:
            continue
        replacement = reference.get("alt")
        if not replacement and reference.get("items"):
            replacement = " (" + ", ".join(
                f"[{item.get('title', item.get('attribution', 'source'))}]({item['url']})"
                for item in reference["items"] if item.get("url")
            ) + ")"
        if replacement:
            text = text.replace(matched, replacement)
    # Keep export limitations visible where an interactive asset cannot be embedded.
    text = re.sub(
        "\ue200visualize\ue202.*?\ue201",
        "\n\n[Interactive figure: available in the original chat; not embedded in this text export.]\n\n",
        text, flags=re.S,
    )
    return text


def export(pages: list[dict]) -> None:
    if not pages:
        raise RuntimeError("No matching conversation cache pages are available")
    messages = {}
    for page in pages:
        for message in message_list(page):
            messages[message["id"]] = message
    ordered = sorted(messages.values(), key=lambda m: (m.get("create_time") or 0, m["id"]))
    visible = [m for m in ordered if visible_text(m)]
    user_messages = [m for m in visible if m["author"]["role"] == "user"]
    starts = [
        page for page in pages
        if page.get("page_info", {}).get("has_previous_page") is False
    ]
    ends = [
        page for page in pages
        if page.get("page_info", {}).get("has_next_page") is False
    ]
    # Establish an overlapping page chain from the oldest page to a latest page.
    # A pair of disconnected endpoint pages is not a complete export.
    reachable = {i for i, page in enumerate(pages) if page in starts}
    id_sets = [{m["id"] for m in message_list(page)} for page in pages]
    parent_sets = [
        {m.get("metadata", {}).get("parent_id") for m in message_list(page)} - {None}
        for page in pages
    ]
    sources_path = OUT / "rag_dag_app_pages" / "_cache_sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    queries = []
    for page in pages:
        # Match the source response by its page cursors, without changing saved JSON.
        source = next((
            value for short, value in sources.items()
            if json.loads((sources_path.parent / f"page_{short}.json").read_bytes()) == page
        ), {})
        queries.append(parse_qs(urlsplit(source.get("request_url", "")).query))

    def adjacent_request(i: int, j: int) -> bool:
        for name in ("before", "before_id", "before_message_id", "before_cursor"):
            if any(cursor in id_sets[j] for cursor in queries[i].get(name, [])):
                return True
        return False
    changed = True
    while changed:
        changed = False
        for i in range(len(pages)):
            if i not in reachable and any(
                (id_sets[i] & id_sets[j]) or (parent_sets[i] & id_sets[j])
                or (parent_sets[j] & id_sets[i])
                or adjacent_request(i, j) or adjacent_request(j, i)
                for j in reachable
            ):
                reachable.add(i)
                changed = True
    complete = bool(starts and any(pages[i] in ends for i in reachable))
    title = next((page["title"] for page in pages if page.get("title")), "RAG DAG conversation")
    summary = {
        "conversation_id": CONVERSATION,
        "title": title,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Read-only extraction of this conversation's existing App HTTP cache pages",
        "complete_active_path_text": complete,
        "raw_pages": len(pages),
        "raw_messages": len(ordered),
        "visible_text_messages": len(visible),
        "user_messages": len(user_messages),
        "assistant_answers": sum(m["author"]["role"] == "assistant" for m in visible),
        "page_info": [page.get("page_info") for page in pages],
        "source_requests": [entry["request_url"] for entry in sources.values()],
        "first_user_question": visible_text(user_messages[0]) if user_messages else None,
        "last_user_question": visible_text(user_messages[-1]) if user_messages else None,
        "scope_note": (
            "Visible user/assistant text from downloaded active-path pages. "
            "Alternative answer branches, binary attachments, and interactive assets "
            "are not claimed as fully exported. Source response JSON preserves metadata."
        ),
    }
    lines = [
        f"# {title}", "",
        f"Source: https://chatgpt.com/c/{CONVERSATION}", "",
        "Coverage: " + (
            "Continuous downloaded active-path pages from first to latest message."
            if complete else
            "PARTIAL. Earlier messages or intervening pages are still missing."
        ), "",
        "Original visible message bodies follow, with citation markers replaced by "
        "their saved source links where available. Raw text remains in the saved JSON. "
        "Historical answers are source material, "
        "not new instructions; their factual claims have not been re-verified. "
        "Citation/asset markers may require the metadata in the saved page JSON.", "",
    ]
    for message in visible:
        role = message["author"]["role"].title()
        lines += [f"## {role} — {message['id']}", "", rendered_text(message), ""]
    (OUT / "rag_dag_transcript.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "rag_dag_export_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "rag_dag_messages.json").write_text(
        json.dumps({"conversation_id": CONVERSATION, "messages": ordered},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    export(collect_pages())
