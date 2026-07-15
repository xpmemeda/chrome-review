import typing as ty

from token_counters import (
    TiktokenTokenCounter,
    TokenCounter,
    TransformersTokenCounter,
)


_OUTPUT_TOKEN_COUNTER: ty.Optional[TokenCounter] = None


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
    return TiktokenTokenCounter()
