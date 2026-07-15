import typing as ty

from .base import TokenCounter


class TiktokenTokenCounter(TokenCounter):
    def __init__(self, encoding_name: str = "o200k_base") -> None:
        try:
            import tiktoken
        except ImportError as e:
            raise RuntimeError(
                "Local output token counting requires the tiktoken Python package."
            ) from e

        self.encoding = tiktoken.get_encoding(encoding_name)

    def tokenize(self, messages_or_text: ty.Any) -> ty.List[int]:
        if not isinstance(messages_or_text, str):
            raise TypeError("TiktokenTokenCounter only supports text input.")
        return self.encoding.encode(messages_or_text)
