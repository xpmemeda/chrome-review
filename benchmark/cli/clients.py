import asyncio
import datetime
import mimetypes
import sys
import time
import typing as ty
import urllib.parse
import uuid

import dataset as dataset_lib
from metrics import RequestMetrics

from .constants import (
    JsonDict,
    ULTRAMAN_DEFAULT_PORT,
    ULTRAMAN_RESERVED_OUTPUT_TOKENS,
)
from .http import AsyncMultipartHttpClient
from .messages import decode_data_url, extract_first_image_url, iter_message_text
from .payloads import build_chat_payload, collect_chat_completion_stream
from .tokens import count_output_tokens


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

    def build_extra_headers(self, req_idx: int) -> ty.Optional[JsonDict]:
        del req_idx
        return None

    def _attach_extra_headers(
        self,
        metric: RequestMetrics,
        extra_headers: ty.Optional[JsonDict],
    ) -> RequestMetrics:
        if not extra_headers:
            return metric
        x_tt_logid = extra_headers.get("x-tt-logid")
        if not isinstance(x_tt_logid, str):
            return metric
        return metric._replace(x_tt_logid=x_tt_logid)

    def _attach_client_send_timestamp(
        self,
        metric: RequestMetrics,
        client_send_timestamp: ty.Optional[str],
    ) -> RequestMetrics:
        if client_send_timestamp is None:
            return metric
        return metric._replace(client_send_timestamp=client_send_timestamp)

    async def send_request(
        self,
        req_idx: int,
        request: dataset_lib.Request,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> RequestMetrics:
        stime = time.perf_counter()
        extra_headers = None
        client_send_timestamp = None
        try:
            payload = self.build_payload(request, sampling_params)
            extra_headers = self.build_extra_headers(req_idx)
            if extra_headers:
                payload["extra_headers"] = extra_headers
            client_send_timestamp = datetime.datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )
            stream = await self.client.chat.completions.create(**payload)
            metric = await collect_chat_completion_stream(req_idx, stream, stime)
            metric = self._attach_extra_headers(metric, extra_headers)
            return self._attach_client_send_timestamp(metric, client_send_timestamp)
        except Exception as e:
            etime = time.perf_counter()
            metric = self._attach_extra_headers(
                RequestMetrics(
                    req_idx,
                    False,
                    0.0,
                    etime - stime,
                    0,
                    0,
                    0,
                    (),
                    repr(e),
                ),
                extra_headers,
            )
            return self._attach_client_send_timestamp(metric, client_send_timestamp)


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

    def build_extra_headers(self, req_idx: int) -> JsonDict:
        del req_idx
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return {"x-tt-logid": timestamp + uuid.uuid4().hex[:20].upper()}

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
                part_struct.fields["image_binary"].struct_.fields[
                    "binary"
                ].bytes_ = image_bytes
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
            raw["forward_out_payload"] = f"<bytes:{len(response.forward_out_payload)}>"
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
