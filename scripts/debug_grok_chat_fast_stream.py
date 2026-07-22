"""Simulate WebUI /webui/api/chat/completions without starting the service.

Mirrors the real local chain as closely as possible:
  WebUI body
    -> ChatCompletionRequest (strip createdAt/feedback)
    -> resolve(model) / Capability.CHAT
    -> chat._extract_message
    -> xai_chat.build_chat_payload(modeId=fast)
    -> POST grok.com/rest/app-chat/conversations/new
    -> StreamAdapter + chat.py soft_stop / image flush
    -> OpenAI SSE chunks WebUI would consume

Does NOT use AccountDirectory / get_proxy_runtime. Uses GROK_SSO_TOKEN +
optional FlareSolverr cookies instead.

Usage:
  $env:GROK_SSO_TOKEN = "<sso>"
  $env:GROK_EXTRA_COOKIE = "cf_clearance=...; __cf_bm=..."
  $env:GROK_USER_AGENT = "<ua matching clearance>"
  python scripts/debug_grok_chat_fast_stream.py --proxy http://127.0.0.1:10809
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any

import orjson
from curl_cffi import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHAT_URL = "https://grok.com/rest/app-chat/conversations/new"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)

DEFAULT_WEBUI_BODY = {
    "model": "grok-chat-fast",
    "messages": [
        {
            "role": "user",
            "content": "生成一张图片 一只狗",
            "createdAt": 1784710506452,
            "feedback": "",
        }
    ],
    "stream": True,
    "temperature": 0.8,
    "top_p": 0.95,
}


class _DummyConfig:
    """Mirrors common production defaults used by Web app-chat."""

    def get_bool(self, key: str, default: bool = False) -> bool:
        values = {
            "features.memory": False,
            "features.temporary": True,
            "features.thinking_summary": False,
            "features.show_search_sources": False,
            "features.stream": True,
            "features.thinking": True,
            "features.imagine_public_image_proxy": False,
        }
        return values.get(key, default)

    def get_str(self, key: str, default: str = "") -> str:
        values = {
            "features.custom_instruction": "",
            "features.image_format": "grok_md",
            "app.app_url": "",
        }
        return values.get(key, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        return default

    def get_int(self, key: str, default: int = 0) -> int:
        return default


def _install_runtime_stubs() -> _DummyConfig:
    cfg = _DummyConfig()
    sys.modules["app.platform.config.snapshot"] = types.SimpleNamespace(
        get_config=lambda key=None, default=None: cfg if key is None else default
    )
    sys.modules["app.control.account.runtime"] = types.SimpleNamespace(
        get_refresh_service=lambda: None,
        set_refresh_service=lambda _svc: None,
    )
    sys.modules["app.control.account.invalid_credentials"] = types.SimpleNamespace(
        feedback_kind_for_error=lambda *_a, **_k: None
    )
    sys.modules["app.dataplane.account.selector"] = types.SimpleNamespace(
        current_strategy=lambda: None,
        selection_max_retries=lambda: 0,
    )
    sys.modules["app.dataplane.proxy"] = types.SimpleNamespace(
        get_proxy_runtime=lambda: None
    )
    sys.modules["app.dataplane.proxy.adapters.headers"] = types.SimpleNamespace(
        build_http_headers=lambda *_a, **_k: {}
    )
    sys.modules["app.dataplane.proxy.adapters.session"] = types.SimpleNamespace(
        ResettableSession=object,
        build_session_kwargs=lambda **_k: {},
    )
    sys.modules["app.dataplane.reverse.transport.asset_upload"] = types.SimpleNamespace(
        upload_from_input=None
    )
    sys.modules["app.dataplane.account"] = types.SimpleNamespace(_directory=None)
    sys.modules["app.platform.storage"] = types.SimpleNamespace(
        save_local_image=lambda *_a, **_k: "debug-image"
    )

    # Avoid executing app.products.openai.__init__ (pulls FastAPI router).
    pkg = types.ModuleType("app.products.openai")
    pkg.__path__ = [str(ROOT / "app" / "products" / "openai")]
    pkg.__package__ = "app.products.openai"
    sys.modules["app.products.openai"] = pkg
    return cfg


def _load_openai_chat_module():
    fmt_path = ROOT / "app" / "products" / "openai" / "_format.py"
    fmt_spec = importlib.util.spec_from_file_location(
        "app.products.openai._format", fmt_path
    )
    fmt_mod = importlib.util.module_from_spec(fmt_spec)
    sys.modules["app.products.openai._format"] = fmt_mod
    assert fmt_spec.loader is not None
    fmt_spec.loader.exec_module(fmt_mod)

    chat_path = ROOT / "app" / "products" / "openai" / "chat.py"
    chat_spec = importlib.util.spec_from_file_location(
        "app.products.openai.chat", chat_path
    )
    chat_mod = importlib.util.module_from_spec(chat_spec)
    sys.modules["app.products.openai.chat"] = chat_mod
    assert chat_spec.loader is not None
    chat_spec.loader.exec_module(chat_mod)
    return chat_mod


def _statsig_id() -> str:
    msg = "x1:TypeError: Cannot read properties of undefined (reading 'debugprobe')"
    return base64.b64encode(msg.encode()).decode()


def _headers(
    token: str,
    *,
    user_agent: str,
    extra_cookie: str = "",
) -> dict[str, str]:
    clean = token[4:] if token.startswith("sso=") else token
    cookie = f"sso={clean}; sso-rw={clean}"
    if extra_cookie.strip():
        cookie = f"{cookie}; {extra_cookie.strip()}"
    return {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Content-Type": "application/json",
        "Cookie": cookie,
        "Origin": "https://grok.com",
        "Referer": "https://grok.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": user_agent or USER_AGENT,
        "x-statsig-id": _statsig_id(),
        "x-xai-request-id": str(uuid.uuid4()),
    }


def _simulate_openai_stream(
    *,
    model: str,
    adapter,
    chat_mod,
    image_format: str,
) -> tuple[list[str], dict[str, Any]]:
    """Mirror chat.py streaming flush after upstream lines are consumed."""
    response_id = chat_mod.make_response_id()
    out_lines: list[str] = [": heartbeat\n\n"]
    content_parts: list[str] = []

    # Same as chat.py: after soft_stop/end, flush adapter.image_urls.
    for url, img_id in adapter.image_urls:
        if image_format == "grok_url":
            img_text = url
        else:
            # default/config grok_md — skip download (no service asset proxy).
            img_text = f"![image]({url})"
        content_parts.append(img_text + "\n")
        chunk = chat_mod.make_stream_chunk(response_id, model, img_text + "\n")
        out_lines.append(f"data: {orjson.dumps(chunk).decode()}\n\n")

    references = adapter.references_suffix()
    if references:
        content_parts.append(references)
        chunk = chat_mod.make_stream_chunk(response_id, model, references)
        out_lines.append(f"data: {orjson.dumps(chunk).decode()}\n\n")

    final = chat_mod.make_stream_chunk(response_id, model, "", is_final=True)
    sources = adapter.search_sources_list()
    if sources:
        final["search_sources"] = sources
    out_lines.append(f"data: {orjson.dumps(final).decode()}\n\n")
    out_lines.append("data: [DONE]\n\n")

    summary = {
        "response_id": response_id,
        "image_count": len(adapter.image_urls),
        "image_urls": [url for url, _ in adapter.image_urls],
        "text_buf": "".join(adapter.text_buf),
        "emitted_content": "".join(content_parts),
        "openai_sse_line_count": len(out_lines),
        "looks_like_empty_webui_failure": (
            len(adapter.image_urls) == 0 and not "".join(adapter.text_buf).strip()
        ),
    }
    return out_lines, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="http://127.0.0.1:10809")
    parser.add_argument("--body-json", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--extra-cookie", default=os.environ.get("GROK_EXTRA_COOKIE", ""))
    parser.add_argument("--user-agent", default=os.environ.get("GROK_USER_AGENT", USER_AGENT))
    parser.add_argument("--impersonate", default=os.environ.get("GROK_IMPERSONATE", "chrome136"))
    parser.add_argument(
        "--image-format",
        default=os.environ.get("GROK_IMAGE_FORMAT", "grok_md"),
        choices=["grok_md", "grok_url"],
    )
    args = parser.parse_args()

    token = os.environ.get("GROK_SSO_TOKEN", "").strip()
    if not token:
        raise SystemExit("GROK_SSO_TOKEN is required")

    body = json.loads(args.body_json) if args.body_json else DEFAULT_WEBUI_BODY

    _install_runtime_stubs()
    from app.products.openai.schemas import ChatCompletionRequest
    from app.control.model.registry import resolve as resolve_model
    from app.dataplane.reverse.protocol.xai_chat import (
        StreamAdapter,
        build_chat_payload,
        classify_line,
    )

    chat_mod = _load_openai_chat_module()

    # 1) Same schema path as /webui/api/chat/completions
    req = ChatCompletionRequest.model_validate(body)
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    is_stream = req.stream if req.stream is not None else True
    model = req.model

    # 2) Same model capability routing decision
    spec = resolve_model(model)
    if spec is None:
        raise SystemExit(f"model not found: {model}")
    if not spec.is_chat() or spec.is_image() or spec.is_console_chat():
        raise SystemExit(
            f"unexpected capability for {model}: chat={spec.is_chat()} "
            f"image={spec.is_image()} console={spec.is_console_chat()}"
        )

    # 3) Same message extraction as chat.completions
    extracted_message, files = chat_mod._extract_message(messages)
    if not extracted_message.strip():
        raise SystemExit("empty message after _extract_message")

    # 4) Same upstream payload builder (temperature/top_p unused on Web app-chat)
    payload = build_chat_payload(
        message=extracted_message,
        mode_id=spec.mode_id,
        file_attachments=[],  # files=[] => no upload-file, same as WebUI text-only
    )

    root = Path(args.out_dir or "data/debug_webui_chat_completions") / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "webui_request.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "schema_messages.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "extracted_message.txt").write_text(extracted_message, encoding="utf-8")
    (root / "upstream_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chain_meta = {
        "endpoint_simulated": "/webui/api/chat/completions",
        "model": model,
        "mode_id": spec.mode_id.to_api_str(),
        "capability": "chat",
        "stream": is_stream,
        "temperature_received": req.temperature,
        "top_p_received": req.top_p,
        "temperature_in_upstream_payload": False,
        "files_from_extract": files,
        "file_attachments_in_payload": payload.get("fileAttachments"),
        "extracted_message": extracted_message,
    }
    (root / "chain_meta.json").write_text(
        json.dumps(chain_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None
    session = requests.Session(impersonate=args.impersonate, proxies=proxies)
    hdrs = _headers(token, user_agent=args.user_agent, extra_cookie=args.extra_cookie)

    adapter = StreamAdapter()
    parser_events: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    line_count = 0
    text_chunks_emitted = 0
    soft_stop_hit = False

    raw_path = root / "raw_upstream_sse.txt"
    openai_path = root / "openai_sse_simulated.txt"

    with raw_path.open("w", encoding="utf-8") as raw_file:
        resp = session.post(
            CHAT_URL,
            headers=hdrs,
            data=orjson.dumps(payload),
            timeout=120,
            stream=True,
        )
        raw_file.write(f"# status={resp.status_code}\n")
        raw_file.write(f"# content-type={resp.headers.get('content-type', '')}\n")
        if resp.status_code != 200:
            body_text = resp.content.decode("utf-8", "replace")
            raw_file.write(body_text)
            raise RuntimeError(f"chat failed: status={resp.status_code} body={body_text[:1000]}")

        # 5) Mirror chat.py streaming event loop (including soft_stop break).
        for raw_line in resp.iter_lines():
            line_count += 1
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", "replace")
            else:
                line = str(raw_line)
            raw_file.write(line + "\n")

            event_type, data = classify_line(line)
            if event_type == "done":
                break
            if event_type != "data" or not data:
                continue

            try:
                events = adapter.feed(data)
            except Exception as exc:  # noqa: BLE001 - capture parser failures
                parse_errors.append(repr(exc))
                continue

            ended = False
            for ev in events:
                parser_events.append(
                    {
                        "kind": ev.kind,
                        "content": (ev.content or "")[:300],
                        "image_id": ev.image_id,
                    }
                )
                if ev.kind == "text":
                    text_chunks_emitted += 1
                elif ev.kind == "soft_stop":
                    soft_stop_hit = True
                    ended = True
                    break
            if ended:
                break

    openai_lines, openai_summary = _simulate_openai_stream(
        model=model,
        adapter=adapter,
        chat_mod=chat_mod,
        image_format=args.image_format,
    )
    openai_path.write_text("".join(openai_lines), encoding="utf-8")

    result = {
        "out_dir": str(root),
        "chain_meta": chain_meta,
        "upstream_line_count": line_count,
        "soft_stop_hit": soft_stop_hit,
        "text_chunks_during_stream": text_chunks_emitted,
        "parser_events": parser_events,
        "parse_errors": parse_errors,
        "openai_summary": openai_summary,
        "openai_sse_preview": "".join(openai_lines)[:2000],
    }
    (root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
