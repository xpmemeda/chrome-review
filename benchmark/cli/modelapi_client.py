import datetime
import typing as ty
import uuid

import dataset as dataset_lib

from .messages import request_to_openai_messages
from .sdk_chat import SdkChatClient

JsonDict = ty.Dict[str, ty.Any]
MODELAPI_BASE_URL = "https://device-intelligence.bytedance.net/api/v1"


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
        extra_body = dict(request_options.get("extra_body") or {})
        extra_body["max_completion_tokens"] = max_tokens
        request_options["extra_body"] = extra_body
        request_options["user"] = self.user
        return await self.client.chat.completions.create(
            model=model,
            messages=request_to_openai_messages(request),
            stream=stream,
            temperature=temperature,
            **request_options,
        )
