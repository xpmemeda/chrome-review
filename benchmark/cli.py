import asyncio
import datetime
import mimetypes
import os
import sys
import time
import typing as ty
import urllib.parse
import uuid

import dataset as dataset_lib
from metrics import RequestMetrics

JsonDict = ty.Dict[str, ty.Any]
TokenCounter = ty.Callable[[str], int]

MODELAPI_BASE_URL = "https://device-intelligence.bytedance.net/api/v1"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ULTRAMAN_DEFAULT_HOST = "127.0.0.1"
ULTRAMAN_DEFAULT_PORT = 50050
ULTRAMAN_PROTO_PATH = os.path.dirname(__file__)
ULTRAMAN_RESERVED_OUTPUT_TOKENS = 8
_OUTPUT_TOKEN_COUNTER: ty.Optional[TokenCounter] = None
SDK_CHAT_COMPLETION_KEYS = {
    "audio",
    "extra_body",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "metadata",
    "modalities",
    "model",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
}


def split_http_url(url: str) -> ty.Tuple[str, int, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http":
        raise ValueError("Only http:// URLs are supported by the stdlib async client.")
    if not parsed.hostname:
        raise ValueError(f"Invalid URL: {url}")
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return parsed.hostname, port, path


class AsyncMultipartHttpClient:
    def __init__(self, url: str, timeout: float) -> None:
        self.host, self.port, self.path = split_http_url(url)
        self.timeout = timeout

    async def post_multipart(
        self,
        data: ty.Dict[str, str],
        files: ty.Dict[str, ty.Tuple[str, bytes, str]],
    ) -> ty.Tuple[float, bytes]:
        body, content_type = encode_multipart_form_data(data, files)
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), timeout=self.timeout
        )
        try:
            headers = [
                f"POST {self.path} HTTP/1.1",
                f"Host: {self.host}:{self.port}",
                f"Content-Type: {content_type}",
                f"Content-Length: {len(body)}",
                "Connection: close",
                "",
                "",
            ]
            writer.write("\r\n".join(headers).encode() + body)
            await asyncio.wait_for(writer.drain(), timeout=self.timeout)

            status_line = await asyncio.wait_for(
                reader.readline(), timeout=self.timeout
            )
            if not status_line:
                raise RuntimeError("empty HTTP response")
            parts = status_line.decode("latin1").strip().split(" ", 2)
            status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0

            resp_headers: JsonDict = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
                if line in (b"\r\n", b"\n", b""):
                    break
                name, _, value = line.decode("latin1").partition(":")
                resp_headers[name.lower()] = value.strip().lower()

            chunks: ty.List[bytes] = []
            if resp_headers.get("transfer-encoding") == "chunked":
                async for _, chunk in self._read_chunked(reader):
                    chunks.append(chunk)
            else:
                while True:
                    chunk = await asyncio.wait_for(
                        reader.read(65536), timeout=self.timeout
                    )
                    if not chunk:
                        break
                    chunks.append(chunk)
            data_bytes = b"".join(chunks)

            if status < 200 or status >= 300:
                err = data_bytes[:4096].decode("utf-8", "replace")
                raise RuntimeError(f"HTTP {status}: {err}")
            return time.perf_counter(), data_bytes
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_chunked(
        self, reader: asyncio.StreamReader
    ) -> ty.AsyncIterator[ty.Tuple[float, bytes]]:
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=self.timeout)
            if not line:
                break
            size_text = line.split(b";", 1)[0].strip()
            if not size_text:
                continue
            size = int(size_text, 16)
            if size == 0:
                await asyncio.wait_for(reader.readline(), timeout=self.timeout)
                break
            data = await asyncio.wait_for(
                reader.readexactly(size), timeout=self.timeout
            )
            await asyncio.wait_for(reader.readexactly(2), timeout=self.timeout)
            yield time.perf_counter(), data


def encode_multipart_form_data(
    data: ty.Dict[str, str],
    files: ty.Dict[str, ty.Tuple[str, bytes, str]],
) -> ty.Tuple[bytes, str]:
    boundary = f"----benchmark-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in data.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode())
        body.extend(b"\r\n")
    for name, (filename, content, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
            ).encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def request_to_openai_messages(request: dataset_lib.Request) -> ty.List[JsonDict]:
    return list(request["messages"])


def iter_message_text(messages: ty.Iterable[JsonDict]) -> ty.Iterator[str]:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    yield str(part.get("text", ""))


def extract_first_image_url(messages: ty.Iterable[JsonDict]) -> ty.Optional[str]:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                return image_url["url"]
    return None


def decode_data_url(data_url: str) -> ty.Tuple[str, bytes]:
    header, sep, encoded = data_url.partition(",")
    if sep != "," or not header.startswith("data:"):
        raise RuntimeError("image_url must be a base64 data URL.")
    mime = header[len("data:") :].split(";", 1)[0]
    if not mime:
        raise RuntimeError("image_url data URL must include a MIME type.")
    import base64

    return mime, base64.b64decode(encoded)


def count_output_tokens(text: str) -> int:
    if not text:
        return 0
    return get_output_token_counter()(text)


def get_output_token_counter() -> TokenCounter:
    global _OUTPUT_TOKEN_COUNTER
    if _OUTPUT_TOKEN_COUNTER is None:
        _OUTPUT_TOKEN_COUNTER = build_output_token_counter()
    return _OUTPUT_TOKEN_COUNTER


def build_output_token_counter() -> TokenCounter:
    try:
        import tiktoken
    except ImportError as e:
        raise RuntimeError(
            "Local output token counting requires the tiktoken Python package."
        ) from e

    encoding = tiktoken.get_encoding("o200k_base")
    return lambda text, encoding=encoding: len(encoding.encode(text))


def build_chat_payload(
    model: str,
    request: dataset_lib.Request,
    max_tokens: int,
    min_tokens: ty.Optional[int],
    temperature: float,
    top_p: ty.Optional[float],
    extra_body: JsonDict,
    sampling_params: ty.Optional[JsonDict] = None,
    include_usage: bool = True,
) -> JsonDict:
    payload, request_extra_body = split_sdk_payload_and_extra_body(request)
    payload.update(
        {
            "model": model,
            "messages": request_to_openai_messages(request),
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    )
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    if "tool_choice" in payload:
        payload["tool_choice"] = normalize_tool_choice(payload["tool_choice"])
    if top_p is not None:
        payload["top_p"] = top_p
    merged_extra_body = dict(request_extra_body)
    merged_extra_body.update(extra_body)
    if min_tokens is not None:
        merged_extra_body["min_tokens"] = min_tokens
    if sampling_params:
        merged_extra_body.update(sampling_params.get("extra_body", {}))
        payload.update({k: v for k, v in sampling_params.items() if k != "extra_body"})
    if merged_extra_body:
        payload["extra_body"] = merged_extra_body
    return payload


def normalize_tool_choice(tool_choice: ty.Any) -> ty.Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    string_choice = tool_choice.get("String")
    if isinstance(string_choice, str):
        return string_choice
    tool_function = tool_choice.get("ToolFunction")
    if isinstance(tool_function, dict):
        function_name = tool_function.get("name")
        if isinstance(function_name, str):
            return {"type": "function", "function": {"name": function_name}}
    return tool_choice


def split_sdk_payload_and_extra_body(request: JsonDict) -> ty.Tuple[JsonDict, JsonDict]:
    payload = {}
    extra_body = {}
    for key, value in request.items():
        if key == "extra_body":
            if isinstance(value, dict):
                extra_body.update(value)
            else:
                raise RuntimeError("extra_body in dataset request must be a JSON object")
        elif key in SDK_CHAT_COMPLETION_KEYS:
            payload[key] = value
        else:
            extra_body[key] = value
    return payload, extra_body


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


class SdkChatClient:
    def __init__(
        self,
        client: ty.Any,
        model: str,
        max_tokens: int,
        min_tokens: ty.Optional[int],
        temperature: float,
        top_p: ty.Optional[float],
        extra_body: ty.Optional[JsonDict],
        include_usage: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.extra_body = extra_body or {}
        self.include_usage = include_usage

    def build_payload(
        self,
        request: dataset_lib.Request,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> JsonDict:
        return build_chat_payload(
            model=self.model,
            request=request,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body=self.extra_body,
            sampling_params=sampling_params,
            include_usage=self.include_usage,
        )

    async def send_request(
        self,
        req_idx: int,
        request: dataset_lib.Request,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> RequestMetrics:
        stime = time.perf_counter()
        try:
            stream = await self.client.chat.completions.create(
                **self.build_payload(request, sampling_params)
            )
            return await collect_chat_completion_stream(req_idx, stream, stime)
        except Exception as e:
            etime = time.perf_counter()
            return RequestMetrics(
                req_idx,
                False,
                0.0,
                etime - stime,
                0,
                0,
                0,
                (),
                repr(e),
            )


class OpenAIClient(SdkChatClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_tokens: int,
        min_tokens: ty.Optional[int],
        temperature: float,
        top_p: ty.Optional[float],
        extra_body: ty.Optional[JsonDict],
    ) -> None:
        import openai

        super().__init__(
            client=openai.AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=0,
            ),
            model=model,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
        )


class ArkClient(SdkChatClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_tokens: int,
        min_tokens: ty.Optional[int],
        temperature: float,
        top_p: ty.Optional[float],
        extra_body: ty.Optional[JsonDict],
    ) -> None:
        if not api_key or api_key == "dummy" or not model:
            raise RuntimeError("--api-key and --model are required for --client ark")

        import httpx
        import openai

        super().__init__(
            client=openai.AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers={
                    "ark-thinking-summary": "skip-thinking-summary",
                },
                http_client=httpx.AsyncClient(trust_env=False, timeout=timeout),
                max_retries=0,
            ),
            model=model,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
            include_usage=False,
        )

    def build_payload(
        self,
        request: dataset_lib.Request,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> JsonDict:
        payload = super().build_payload(request, sampling_params)
        max_completion_tokens = payload.pop("max_tokens", self.max_tokens)
        extra_body = dict(payload.get("extra_body") or {})
        extra_body["max_completion_tokens"] = max_completion_tokens
        payload["extra_body"] = extra_body
        return payload


class ModelApiClient(SdkChatClient):
    def __init__(
        self,
        base_url: str,
        env: str,
        model: str,
        timeout: float,
        max_tokens: int,
        min_tokens: ty.Optional[int],
        temperature: float,
        top_p: ty.Optional[float],
        extra_body: ty.Optional[JsonDict],
    ) -> None:
        if not model:
            raise RuntimeError("model can't be empty if using modelapi.")

        import httpx
        import openai

        super().__init__(
            client=openai.AsyncOpenAI(
                base_url=base_url,
                api_key="empty",
                default_headers={
                    "x-tt-env": env,
                    "x-use-ppe": "1" if env else "0",
                },
                http_client=httpx.AsyncClient(trust_env=False, timeout=timeout),
                max_retries=0,
            ),
            model=model,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
        )
        self.user = f"benchmark-modelapi-{uuid.uuid4().hex}"

    def build_payload(
        self,
        request: dataset_lib.Request,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> JsonDict:
        payload = super().build_payload(request, sampling_params)
        payload["user"] = self.user
        return payload


class MockClient:
    async def send_request(
        self,
        req_idx: int,
        request: dataset_lib.Request,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> RequestMetrics:
        del request, sampling_params
        return RequestMetrics(req_idx, True, 0.0, 0.0, 0, 0, 0)


class DiffusionClient:
    def __init__(
        self,
        url: str,
        timeout: float,
        style: ty.Optional[str],
        seed: ty.Optional[int],
        steps: ty.Optional[int],
        extra_fields: ty.Optional[JsonDict],
    ) -> None:
        self.http_client = AsyncMultipartHttpClient(url, timeout)
        self.style = style
        self.seed = seed
        self.steps = steps
        self.extra_fields = extra_fields or {}

    async def send_request(
        self,
        req_idx: int,
        request: dataset_lib.Request,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> RequestMetrics:
        del sampling_params
        stime = time.perf_counter()
        try:
            messages = request["messages"]
            image_url = extract_first_image_url(messages)
            if image_url is None:
                raise RuntimeError("DiffusionClient requires an image request.")
            image_mime, image_bytes = decode_data_url(image_url)
            ext = mimetypes.guess_extension(image_mime) or ".png"
            data = {str(k): str(v) for k, v in self.extra_fields.items()}
            prompt = "\n".join(iter_message_text(messages))
            data["style"] = self.style or prompt or "摄影后期"
            if self.seed is not None:
                data["seed"] = str(self.seed + req_idx)
            if self.steps is not None:
                data["steps"] = str(self.steps)
            files = {
                "image": (
                    f"request-{req_idx}{ext}",
                    image_bytes,
                    image_mime,
                )
            }
            first_byte_ts, output = await self.http_client.post_multipart(data, files)
            etime = time.perf_counter()
            return RequestMetrics(
                req_idx,
                True,
                first_byte_ts - stime,
                etime - stime,
                0,
                len(output),
                1,
            )
        except Exception as e:
            etime = time.perf_counter()
            return RequestMetrics(
                req_idx,
                False,
                0.0,
                etime - stime,
                0,
                0,
                0,
                (),
                repr(e),
            )


def split_grpc_target(value: str) -> ty.Tuple[str, int]:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme:
        if parsed.scheme != "grpc":
            raise ValueError("Only grpc:// URLs are supported for ultraman.")
        if not parsed.hostname:
            raise ValueError(f"Invalid ultraman target: {value}")
        return parsed.hostname, parsed.port or ULTRAMAN_DEFAULT_PORT

    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        if not host or not rest.startswith(":"):
            raise ValueError(f"Invalid ultraman target: {value}")
        return host, int(rest[1:])

    host, sep, port_text = value.rpartition(":")
    if sep:
        return host, int(port_text)
    return value, ULTRAMAN_DEFAULT_PORT


def import_ultraman_proto(proto_path: str) -> ty.Tuple[ty.Any, ty.Any]:
    if proto_path and proto_path not in sys.path:
        sys.path.insert(0, proto_path)
    from llmserver.proto import ultraman_pb2, ultraman_pb2_grpc

    return ultraman_pb2, ultraman_pb2_grpc


class UltramanClient:
    def __init__(
        self,
        host: str,
        port: int,
        proto_path: str,
        model: str,
        timeout: float,
        max_tokens: int,
        min_tokens: ty.Optional[int],
        temperature: float,
        top_p: ty.Optional[float],
        top_k: int,
        repetition_penalty: float,
    ) -> None:
        if not host or not port:
            raise RuntimeError("ultraman requires host and port.")

        import grpc

        self.ultraman_pb2, ultraman_pb2_grpc = import_ultraman_proto(proto_path)
        target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        self.channel = grpc.insecure_channel(target)
        self.stub = ultraman_pb2_grpc.InferenceStub(self.channel)
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.session_id = f"benchmark-session-{uuid.uuid4().hex}"

    async def send_request(
        self,
        req_idx: int,
        request: dataset_lib.Request,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> RequestMetrics:
        del sampling_params
        stime = time.perf_counter()
        try:
            return await asyncio.to_thread(
                self._send_request_sync,
                req_idx,
                request,
                stime,
            )
        except Exception as e:
            etime = time.perf_counter()
            return RequestMetrics(
                req_idx,
                False,
                0.0,
                etime - stime,
                0,
                0,
                0,
                (),
                repr(e),
            )

    def _send_request_sync(
        self,
        req_idx: int,
        request: dataset_lib.Request,
        stime: float,
    ) -> RequestMetrics:
        grpc_request = self._build_request(req_idx, request)
        stream = self.stub.StreamingCall(grpc_request, timeout=self.timeout)
        return self._collect_stream(req_idx, stream, stime)

    def _build_request(self, req_idx: int, request: dataset_lib.Request) -> ty.Any:
        grpc_request = self.ultraman_pb2.InferenceRequest()
        now = datetime.datetime.now()
        grpc_request.req_id = (
            now.strftime("benchmark-%Y%m%d-%H%M%S-")
            + f"{now.microsecond // 1000:03d}-{req_idx}"
        )
        grpc_request.model_name = self.model
        grpc_request.task_id = "benchmark-task"
        grpc_request.inputs["session_id"].string_ = self.session_id
        grpc_request.inputs["_bot_id"].string_ = "benchmark"
        grpc_request.inputs["bot_id"].string_ = "benchmark"

        for message in request["messages"]:
            self._append_message(grpc_request, message)

        grpc_request.inputs["max_new_tokens"].int64_ = (
            self.max_tokens + ULTRAMAN_RESERVED_OUTPUT_TOKENS
        )
        if self.min_tokens is not None:
            compensated_min_tokens = self.min_tokens + ULTRAMAN_RESERVED_OUTPUT_TOKENS
            grpc_request.inputs["min_new_tokens"].int64_ = compensated_min_tokens
            grpc_request.inputs["min_tokens"].int64_ = compensated_min_tokens
        grpc_request.inputs["temperature"].float_ = self.temperature
        if self.top_p is not None:
            grpc_request.inputs["top_p"].float_ = self.top_p
        grpc_request.inputs["top_k"].int64_ = self.top_k
        grpc_request.inputs["repetition_penalty"].float_ = self.repetition_penalty
        return grpc_request

    def _append_message(self, grpc_request: ty.Any, message: JsonDict) -> None:
        msg_struct = self.ultraman_pb2.Struct()
        msg_struct.fields["role"].string_ = str(message["role"])
        content = message["content"]
        if isinstance(content, str):
            msg_struct.fields["content"].string_ = content
        else:
            msg_struct.fields["content"].value_list.values.extend(
                self._build_content_values(content)
            )
        grpc_request.inputs["messages"].value_list.values.append(
            self.ultraman_pb2.Value(struct_=msg_struct)
        )

    def _build_content_values(self, content: ty.Iterable[JsonDict]) -> ty.List[ty.Any]:
        values = []
        for part in content:
            part_type = part.get("type")
            part_struct = self.ultraman_pb2.Struct()
            part_struct.fields["type"].string_ = str(part_type)
            if part_type == "text":
                part_struct.fields["text"].string_ = str(part.get("text", ""))
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if not isinstance(image_url, dict):
                    raise RuntimeError("image_url content must be an object")
                url = image_url.get("url")
                if not isinstance(url, str):
                    raise RuntimeError("image_url.url must be a string")
                _, image_bytes = decode_data_url(url)
                part_struct.fields["type"].string_ = "image_binary"
                part_struct.fields["image_binary"].struct_.fields["binary"].bytes_ = (
                    image_bytes
                )
            elif part_type == "image_binary":
                part_struct.fields["image_binary"].struct_.fields["binary"].bytes_ = (
                    part["image_binary"]["binary"]
                )
            else:
                raise RuntimeError(f"unsupported ultraman content type: {part_type}")
            values.append(self.ultraman_pb2.Value(struct_=part_struct))
        return values

    def _collect_stream(
        self,
        req_idx: int,
        stream: ty.Iterable[ty.Any],
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

        for response in stream:
            server_raw_chunks.append(self._response_to_raw_text(response))
            response_usage = self._extract_server_usage(response)
            if response_usage["output_tokens"] is not None:
                server_output_tokens = response_usage["output_tokens"]
            if response_usage["input_tokens"] is not None:
                server_input_tokens = response_usage["input_tokens"]
            if response_usage["cached_tokens"] is not None:
                server_cached_tokens = response_usage["cached_tokens"]
            if response_usage["raw"]:
                server_usage = response_usage["raw"]
            text = self._extract_text(response)
            if not text:
                continue
            chunk_at = time.perf_counter()
            if ttft is None:
                ttft = chunk_at - stime
            elif last_chunk_at is not None:
                itls.append(chunk_at - last_chunk_at)
            last_chunk_at = chunk_at
            output_text_parts.append(text)
            output_chars += len(text)
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

    def _extract_text(self, response: ty.Any) -> str:
        content_node = (
            response.outputs["choice"]
            .struct_.fields["message"]
            .struct_.fields["content"]
        )
        return getattr(content_node, "string_", "")

    def _response_to_raw_text(self, response: ty.Any) -> str:
        import json

        raw = {
            "req_id": response.req_id,
            "model_name": response.model_name,
            "outputs": self._struct_fields_to_json(response.outputs),
            "events": [self._task_event_to_json(event) for event in response.events],
        }
        if response.HasField("task_id"):
            raw["task_id"] = response.task_id
        if response.HasField("forward_out_payload"):
            raw["forward_out_payload"] = (
                f"<bytes:{len(response.forward_out_payload)}>"
            )
        if response.HasField("kvcache_ref"):
            raw["kvcache_ref"] = response.kvcache_ref
        return json.dumps(raw, ensure_ascii=False)

    def _task_event_to_json(self, event: ty.Any) -> JsonDict:
        result = {
            "timestamp": event.timestamp,
            "role": int(event.role),
            "event_type": int(event.event_type),
            "task_id": event.task_id,
            "endpoint": event.endpoint,
        }
        for key in (
            "hit_len",
            "miss_len",
            "prompt_len",
            "new_token_len",
            "used_kv_blocks",
            "reason",
            "computed_len",
        ):
            if event.HasField(key):
                result[key] = getattr(event, key)
        return result

    def _extract_server_usage(self, response: ty.Any) -> JsonDict:
        result = {
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
            "raw": None,
        }
        if "usage" not in response.outputs:
            return result
        usage = response.outputs["usage"]
        if not usage.HasField("struct_"):
            return result
        fields = usage.struct_.fields
        result["raw"] = self._struct_fields_to_json(fields)
        result["input_tokens"] = self._extract_struct_int(fields, "prompt_tokens")
        result["output_tokens"] = self._extract_struct_int(fields, "completion_tokens")
        details = fields.get("prompt_tokens_details")
        if details is None or not details.HasField("struct_"):
            return result
        result["cached_tokens"] = self._extract_struct_int(
            details.struct_.fields,
            "cached_tokens",
        )
        return result

    def _struct_fields_to_json(self, fields: ty.Any) -> JsonDict:
        return {key: self._value_to_json(value) for key, value in fields.items()}

    def _value_to_json(self, value: ty.Any) -> ty.Any:
        kind = value.WhichOneof("kind")
        if kind == "struct_":
            return self._struct_fields_to_json(value.struct_.fields)
        if kind == "value_list":
            return [self._value_to_json(item) for item in value.value_list.values]
        if kind == "string_":
            return value.string_
        if kind == "int64_":
            return int(value.int64_)
        if kind == "float_":
            return float(value.float_)
        if kind == "bool_":
            return bool(value.bool_)
        if kind == "bytes_":
            return f"<bytes:{len(value.bytes_)}>"
        if kind == "float_list":
            return [float(item) for item in value.float_list.values]
        if kind == "int64_list":
            return [int(item) for item in value.int64_list.values]
        if kind == "string_list":
            return list(value.string_list.values)
        if kind == "bytes_list":
            return [f"<bytes:{len(item)}>" for item in value.bytes_list.values]
        return None

    def _extract_struct_int(self, fields: ty.Any, key: str) -> ty.Optional[int]:
        if key not in fields:
            return None
        return int(fields[key].int64_)
