import typing as ty

import dataset as dataset_lib

from .constants import JsonDict


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
