import abc
import typing as ty


class TokenCounter(abc.ABC):
    def __call__(self, messages_or_text: ty.Any) -> int:
        return self.count(messages_or_text)

    @abc.abstractmethod
    def tokenize(self, messages_or_text: ty.Any) -> ty.List[int]:
        raise NotImplementedError

    def count(self, messages_or_text: ty.Any) -> int:
        return len(self.tokenize(messages_or_text))
