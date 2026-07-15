import time
import typing as ty

from metrics import RequestMetrics

from .tokens import count_output_tokens

JsonDict = ty.Dict[str, ty.Any]


def get_usage_int(usage: ty.Any, key: str) -> ty.Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return int(value) if isinstance(value, int) else None


def get_prompt_cached_tokens(usage: ty.Any) -> ty.Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
    else:
        details = getattr(usage, "prompt_tokens_details", None)
    return get_usage_int(details, "cached_tokens")


def usage_to_json(usage: ty.Any) -> ty.Optional[JsonDict]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if hasattr(usage, "dict"):
        dumped = usage.dict()
        return dumped if isinstance(dumped, dict) else None
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    details = getattr(usage, "prompt_tokens_details", None)
    details_json = usage_to_json(details)
    if details_json:
        result["prompt_tokens_details"] = details_json
    return result or None


def chunk_to_raw_text(chunk: ty.Any) -> str:
    if hasattr(chunk, "model_dump_json"):
        return chunk.model_dump_json()
    if hasattr(chunk, "json"):
        return chunk.json()
    if isinstance(chunk, dict):
        import json

        return json.dumps(chunk, ensure_ascii=False)
    return repr(chunk)


async def collect_chat_completion_stream(
    req_idx: int,
    stream: ty.Any,
    stime: float,
) -> RequestMetrics:
    ttft: ty.Optional[float] = None
    last_chunk_at: ty.Optional[float] = None
    itls: ty.List[float] = []
    output_text_parts: ty.List[str] = []
    server_output_tokens = None
    server_input_tokens = None
    server_cached_tokens = None
    server_usage = None
    server_raw_chunks: ty.List[str] = []
    output_chars = 0
    output_chunks = 0

    async for chunk in stream:
        server_raw_chunks.append(chunk_to_raw_text(chunk))
        chunk_at = time.perf_counter()
        if ttft is None:
            ttft = chunk_at - stime
        elif last_chunk_at is not None:
            itls.append(chunk_at - last_chunk_at)
        last_chunk_at = chunk_at
        usage = getattr(chunk, "usage", None)
        usage_json = usage_to_json(usage)
        prompt_tokens = get_usage_int(usage, "prompt_tokens")
        completion_tokens = get_usage_int(usage, "completion_tokens")
        cached_tokens = get_prompt_cached_tokens(usage)
        if usage_json is not None:
            server_usage = usage_json
        if prompt_tokens is not None:
            server_input_tokens = prompt_tokens
        if completion_tokens is not None:
            server_output_tokens = completion_tokens
        if cached_tokens is not None:
            server_cached_tokens = cached_tokens
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta
        model_extra = getattr(delta, "model_extra", None) or {}
        text_parts = [
            getattr(delta, "reasoning_content", None)
            or model_extra.get("reasoning_content"),
            getattr(delta, "content", None),
        ]
        text_len = sum(len(x) for x in text_parts if x)
        if text_len:
            output_text_parts.extend(x for x in text_parts if x)
            output_chars += text_len
            output_chunks += 1

    etime = time.perf_counter()
    output_text = "".join(output_text_parts)
    output_tokens = count_output_tokens(output_text)
    return RequestMetrics(
        req_idx,
        True,
        ttft if ttft is not None else etime - stime,
        etime - stime,
        output_tokens,
        output_chars,
        output_chunks,
        tuple(itls),
        output_text=output_text,
        server_output_tokens=server_output_tokens,
        server_input_tokens=server_input_tokens,
        server_cached_tokens=server_cached_tokens,
        server_usage=server_usage,
        server_raw_chunks=tuple(server_raw_chunks),
    )
