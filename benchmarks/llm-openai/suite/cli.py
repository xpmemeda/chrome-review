import abc
import asyncio
import argparse
import json
import logging
import random
import time
import typing as ty

from .metrics import Metrics

Messages = ty.List[ty.Dict[str, str]]
SamplingParams = ty.Dict[str, ty.Any]
ClientType = ty.Type["BaseClient"]


class BaseClient(abc.ABC):
    client_type: str

    @classmethod
    @abc.abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args: argparse.Namespace) -> "BaseClient":
        raise NotImplementedError

    @abc.abstractmethod
    def send_request(
        self,
        req_idx: int,
        messages: Messages,
        sampling_params: ty.Optional[SamplingParams] = None,
    ) -> Metrics:
        raise NotImplementedError

    @abc.abstractmethod
    async def async_send_request(
        self,
        req_idx: int,
        messages: Messages,
        sampling_params: ty.Optional[SamplingParams] = None,
    ) -> Metrics:
        raise NotImplementedError


class MockClient(BaseClient):
    client_type = "mock"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        pass

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MockClient":
        return cls()

    def send_request(
        self,
        req_idx: int,
        messages: Messages,
        sampling_params: ty.Optional[SamplingParams] = None,
    ) -> Metrics:
        return Metrics(
            cid=f"mock-{req_idx}",
            ttft=0.0,
            itl_list=[],
            e2e=0.0,
            messages=messages,
            resp_text="",
        )

    async def async_send_request(
        self,
        req_idx: int,
        messages: Messages,
        sampling_params: ty.Optional[SamplingParams] = None,
    ) -> Metrics:
        return Metrics(
            cid=f"mock-{req_idx}",
            ttft=0.0,
            itl_list=[],
            e2e=0.0,
            messages=messages,
            resp_text="",
        )


class OpenAIClient(BaseClient):
    client_type = "openai"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--min-tokens",
            type=int,
            help="the min tokens to generate.",
            default=None,
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            help="the max tokens to generate.",
            default=None,
        )
        parser.add_argument(
            "--temperature",
            type=float,
            help="sampling params.",
            default=1.0,
        )
        parser.add_argument(
            "--cid",
            type=str,
            help="add client id to debug svr log.",
        )
        parser.add_argument(
            "--host", type=str, help="openai server host.", required=True
        )
        parser.add_argument(
            "--port", type=int, help="openai server port.", required=True
        )
        parser.add_argument(
            "--model",
            type=str,
            help="for example: deepseek-ai/DeepSeek-R1",
            required=True,
        )
        parser.add_argument("--api-key", type=str, default="dummy")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "OpenAIClient":
        return cls(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            apex_cid=args.cid,
            apex_cluster=args.apex_cluster,
        )

    def __init__(
        self,
        host: str,
        port: int,
        api_key: str,
        model: str,
        temperature: ty.Optional[float],
        min_tokens: ty.Optional[int],
        max_tokens: ty.Optional[int],
        apex_cid: ty.Optional[str],
        apex_cluster: ty.Optional[str],
    ) -> None:
        """
        NOTE: api_key can't be "", you can set it to any non-empty str.
        """
        import openai

        self.model = model
        self.rng = random.Random(1899)
        self.temperature = temperature
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.apex_cid = apex_cid
        self.apex_cluster = apex_cluster

        self.sync_client = openai.OpenAI(
            base_url=f"http://{host}:{port}/v1", api_key=api_key
        )
        self.async_client = openai.AsyncOpenAI(
            base_url=f"http://{host}:{port}/v1", api_key=api_key
        )

    def build_sampling_params(
        self, request_sampling_params: ty.Optional[SamplingParams]
    ) -> SamplingParams:
        sampling_params: SamplingParams = {"extra_body": {}}

        if self.temperature is not None:
            sampling_params["temperature"] = self.temperature

        # For fake datasets, pin both min_tokens and max_tokens for deterministic output.
        if self.min_tokens is not None and self.max_tokens is not None:
            n = self.rng.randint(self.min_tokens, self.max_tokens)
            sampling_params["extra_body"]["min_tokens"] = n
            sampling_params["max_tokens"] = n
        elif self.min_tokens is not None:
            sampling_params["extra_body"]["min_tokens"] = self.min_tokens
        elif self.max_tokens is not None:
            sampling_params["max_tokens"] = self.max_tokens

        if self.apex_cid:
            sampling_params["extra_body"]["client_id"] = self.apex_cid

        if self.apex_cluster:
            sampling_params["extra_body"]["internal_debug_app_id"] = self.apex_cluster

        if not request_sampling_params:
            return sampling_params

        merged_extra_body = dict(sampling_params.get("extra_body", {}))
        merged_extra_body.update(request_sampling_params.get("extra_body", {}))
        sampling_params.update(request_sampling_params)
        sampling_params["extra_body"] = merged_extra_body
        return sampling_params

    def build_cid(self, req_idx: int) -> str:
        prefix = self.apex_cid or "client"
        return f"{prefix}-{req_idx}"

    def send_request(
        self,
        req_idx: int,
        messages: Messages,
        sampling_params: ty.Optional[SamplingParams] = None,
    ) -> Metrics:
        stime = time.time()
        ttft_time = None
        itl_base = stime
        itl_list = []

        completion = self.sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **self.build_sampling_params(sampling_params),
        )

        resp_text = ""
        for chunk in completion:
            ttft_time = ttft_time or time.time()
            itl_time = time.time()
            itl_list.append(itl_time - itl_base)
            itl_base = itl_time

            resp = chunk.choices[0].delta.content
            if resp:
                resp_text += resp

        etime = time.time()
        ttft = (ttft_time or etime) - stime
        e2e = etime - stime
        itl_list = itl_list[1:]

        return Metrics(
            self.build_cid(req_idx), ttft, itl_list, e2e, messages, resp_text
        )

    async def async_send_request(
        self,
        req_idx: int,
        messages: Messages,
        sampling_params: ty.Optional[SamplingParams] = None,
    ) -> Metrics:
        stime = time.time()
        ttft_time = None
        itl_base = stime
        itl_list = []

        completion = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **self.build_sampling_params(sampling_params),
        )

        resp_text = ""
        async for chunk in completion:
            ttft_time = ttft_time or time.time()
            itl_time = time.time()
            itl_list.append(itl_time - itl_base)
            itl_base = itl_time

            resp = chunk.choices[0].delta.content
            if resp:
                resp_text += resp

        etime = time.time()
        ttft = (ttft_time or etime) - stime
        e2e = etime - stime
        itl_list = itl_list[1:]

        return Metrics(
            self.build_cid(req_idx), ttft, itl_list, e2e, messages, resp_text
        )


class SpamLocalClient(BaseClient):
    client_type = "spam"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--max-tokens",
            type=int,
            help="the max tokens to generate.",
            default=None,
        )
        parser.add_argument(
            "--temperature",
            type=float,
            help="sampling params.",
            default=1.0,
        )
        parser.add_argument(
            "--socket-timeout-ms",
            type=int,
            help="timeout for each local inference request.",
            default=5000,
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SpamLocalClient":
        return cls(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            socket_timeout_ms=args.socket_timeout_ms,
        )

    def __init__(
        self,
        temperature: ty.Optional[float],
        max_tokens: ty.Optional[int],
        socket_timeout_ms: int,
    ) -> None:
        from tfccitispykit.client import LLMSvrLocalClient as _LLMSvrLocalClient

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.socket_timeout_ms = socket_timeout_ms
        self.client = _LLMSvrLocalClient()

    @staticmethod
    def extract_messages(messages: Messages) -> ty.Tuple[str, str]:
        system_content = ""
        user_content = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
        return system_content, user_content

    @staticmethod
    def build_hyper_params(
        sampling_params: SamplingParams,
    ):
        from mmspamolvllmtemplatetext_pb2 import MMSpamOLVllmTemplateTextParams

        hyper_params = MMSpamOLVllmTemplateTextParams()
        if "temperature" in sampling_params:
            hyper_params.temperature = float(sampling_params["temperature"])
        if "top_p" in sampling_params:
            hyper_params.top_p = float(sampling_params["top_p"])
        if "top_k" in sampling_params:
            hyper_params.top_k = int(sampling_params["top_k"])
        if "max_tokens" in sampling_params:
            hyper_params.max_tokens = int(sampling_params["max_tokens"])
        return hyper_params

    def build_sampling_params(
        self, request_sampling_params: ty.Optional[SamplingParams]
    ) -> SamplingParams:
        sampling_params: SamplingParams = {}
        if self.temperature is not None:
            sampling_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            sampling_params["max_tokens"] = self.max_tokens
        if request_sampling_params:
            sampling_params.update(request_sampling_params)
        return sampling_params

    async def async_send_request(
        self,
        req_idx: int,
        messages: Messages,
        sampling_params: ty.Optional[SamplingParams] = None,
    ) -> Metrics:
        from mmspamolvllmtemplatetext_pb2 import (
            MMSpamOLVllmTemplateTextReq,
            MMSpamOLVllmTemplateTextResp,
        )

        merged_sampling_params = self.build_sampling_params(sampling_params)
        _, user_content = self.extract_messages(messages)
        if not user_content:
            raise RuntimeError("SpamLocalClient requires a user message.")

        req = MMSpamOLVllmTemplateTextReq()
        req.content = user_content

        resp = MMSpamOLVllmTemplateTextResp()
        hyper_params = self.build_hyper_params(merged_sampling_params)

        stime = time.time()
        ret = await self.client.run(
            req=req,
            resp=resp,
            hyper_params=hyper_params,
            socket_timeout_ms=self.socket_timeout_ms,
        )
        etime = time.time()

        if ret != 0:
            raise RuntimeError(f"SpamLocalClient run failed, ret={ret}")

        resp_text = resp.pred_text
        e2e = etime - stime
        return Metrics(
            f"spam-{req_idx}",
            e2e,
            [0.0],
            e2e,
            messages,
            resp_text,
        )

    def send_request(
        self,
        req_idx: int,
        messages: Messages,
        sampling_params: ty.Optional[SamplingParams] = None,
    ) -> Metrics:
        return asyncio.run(
            self.async_send_request(req_idx, messages, sampling_params=sampling_params)
        )


def get_cls(client_type: str) -> ClientType:
    if client_type == "mock":
        return MockClient
    elif client_type == "openai":
        return OpenAIClient
    elif client_type == "spam":
        return SpamLocalClient
    else:
        raise RuntimeError(f"unknown client_type={client_type}")
