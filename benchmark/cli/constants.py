import os
import typing as ty

JsonDict = ty.Dict[str, ty.Any]
TokenCounter = ty.Callable[[str], int]

MODELAPI_BASE_URL = "https://device-intelligence.bytedance.net/api/v1"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_BASE_URL = "https://ark-cn-beijing.bytedance.net/api/v3"
ULTRAMAN_DEFAULT_HOST = "127.0.0.1"
ULTRAMAN_DEFAULT_PORT = 50050
ULTRAMAN_PROTO_PATH = os.path.dirname(os.path.dirname(__file__))
ULTRAMAN_RESERVED_OUTPUT_TOKENS = 8

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
