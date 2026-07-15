import typing as ty

from .sdk_chat import SdkChatClient

JsonDict = ty.Dict[str, ty.Any]


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
