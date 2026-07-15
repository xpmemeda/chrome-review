import datetime
import time
import typing as ty

import dataset as dataset_lib
from metrics import RequestMetrics

from .messages import request_to_openai_messages
from .payloads import collect_chat_completion_stream

JsonDict = ty.Dict[str, ty.Any]


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

    def _build_request_options(
        self,
        request: dataset_lib.StdChatApiRequest,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> JsonDict:
        request_options = dict(request)
        request_options.pop("messages", None)
        for key in ("model", "stream", "max_tokens", "temperature"):
            request_options.pop(key, None)

        request_extra_body = request_options.pop("extra_body", {})
        if not isinstance(request_extra_body, dict):
            raise RuntimeError("extra_body in dataset request must be a JSON object")

        if "tool_choice" in request_options:
            request_options["tool_choice"] = normalize_tool_choice(
                request_options["tool_choice"]
            )
        if self.include_usage:
            request_options["stream_options"] = {"include_usage": True}
        if self.top_p is not None:
            request_options["top_p"] = self.top_p

        merged_extra_body = dict(request_extra_body)
        merged_extra_body.update(self.extra_body)
        if self.min_tokens is not None:
            merged_extra_body["min_tokens"] = self.min_tokens
        if sampling_params:
            merged_extra_body.update(sampling_params.get("extra_body", {}))
            request_options.update(
                {k: v for k, v in sampling_params.items() if k != "extra_body"}
            )
        if merged_extra_body:
            request_options["extra_body"] = merged_extra_body
        return request_options

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
        request: dataset_lib.StdChatApiRequest,
        sampling_params: ty.Optional[JsonDict] = None,
    ) -> RequestMetrics:
        stime = time.perf_counter()
        extra_headers = None
        client_send_timestamp = None
        try:
            extra_headers = self.build_extra_headers(req_idx)
            client_send_timestamp = (
                datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
            )
            stream = await self._send_completion_stream(
                request,
                sampling_params,
                extra_headers,
            )
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

    async def _send_completion_stream(
        self,
        request: dataset_lib.StdChatApiRequest,
        sampling_params: ty.Optional[JsonDict],
        extra_headers: ty.Optional[JsonDict],
    ) -> ty.Any:
        request_options = self._build_request_options(request, sampling_params)
        if extra_headers:
            request_options["extra_headers"] = extra_headers
        model = request_options.pop("model", self.model)
        max_tokens = request_options.pop("max_tokens", self.max_tokens)
        temperature = request_options.pop("temperature", self.temperature)
        stream = request_options.pop("stream", True)
        return await self.client.chat.completions.create(
            model=model,
            messages=request_to_openai_messages(request),
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            **request_options,
        )


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
