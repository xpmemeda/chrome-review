import base64
import json
import logging
import os
import random
import re
import typing as ty

from .base import Dataset, JsonDict, Messages, StdChatApiRequest

_APP_LIST_RE = re.compile(r"(The available app list is:\s*)\[(.*?)]", re.DOTALL)
DEFAULT_OMNI_IMAGE_CACHE_PATH = (
    "~/workspace/ocean/service_shell/benchmark/round1_image_base64_cache.json"
)
OMNI_IMAGE_NOISE_BYTES = 128


def _deepcopy_json(value: ty.Any) -> ty.Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _randbytes(rng: random.Random, size: int) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(size))


def _load_image_cache(cache_path: ty.Optional[str]) -> ty.Dict[str, str]:
    if not cache_path:
        return {}
    expanded_path = os.path.expanduser(cache_path)
    if not os.path.exists(expanded_path):
        logging.warning(
            "omni image cache not found, urls will be left unchanged: %s",
            expanded_path,
        )
        return {}

    with open(expanded_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return {}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]

    cache = {}
    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        encoded = item.get("base64")
        if isinstance(url, str) and isinstance(encoded, str):
            if encoded.startswith("data:"):
                cache[url] = encoded
            else:
                cache[url] = f"data:image/jpeg;base64,{encoded}"
    return cache


def _mutate_available_app_list(text: str, rng: random.Random) -> str:
    match = _APP_LIST_RE.search(text)
    if not match:
        return text

    apps = [app.strip() for app in match.group(2).split(",") if app.strip()]
    if len(apps) <= 1:
        return text

    rng.shuffle(apps)
    replacement = f"{match.group(1)}[{','.join(apps)}]"
    return text[: match.start()] + replacement + text[match.end() :]


def _add_noise_to_data_url(data_url: str, noise_bytes: int, rng: random.Random) -> str:
    prefix = "data:image/jpeg;base64,"
    if "," in data_url and data_url.startswith("data:"):
        prefix, encoded = data_url.split(",", 1)
        prefix = prefix + ","
    else:
        encoded = data_url

    raw = base64.b64decode(encoded)
    raw += _randbytes(rng, noise_bytes)
    return prefix + base64.b64encode(raw).decode("ascii")


def _iter_image_url_locs(messages: Messages) -> ty.Iterator[ty.Tuple[int, int]]:
    for message_idx, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_idx, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                yield message_idx, part_idx


def _extract_prompt_text(messages: Messages) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
    return "\n".join(parts)


class OmniMultiMessageDataset(Dataset):
    def __init__(
        self,
        num_requests: int,
        template_path: str,
        seed: int,
    ) -> None:
        if num_requests <= 0:
            raise RuntimeError(
                "num_requests must be positive for OmniMultiMessageDataset."
            )

        self.num_requests = num_requests
        self.template_path = os.path.expanduser(template_path)
        self.seed = seed
        self.body_template = self._load_template(self.template_path)
        self.image_cache = _load_image_cache(DEFAULT_OMNI_IMAGE_CACHE_PATH)
        logging.info(
            "loaded omni multi-message dataset, template=%s requests=%d image_cache=%d",
            self.template_path,
            self.num_requests,
            len(self.image_cache),
        )

    def _load_template(self, template_path: str) -> JsonDict:
        with open(template_path, "r", encoding="utf-8") as f:
            body = json.load(f)
        if not isinstance(body, dict):
            raise RuntimeError(f"omni template must be a JSON object: {template_path}")
        if not isinstance(body.get("messages"), list):
            raise RuntimeError(
                f"omni template must contain a messages list: {template_path}"
            )
        ignored_keys = sorted(key for key in body if key != "messages")
        if ignored_keys:
            logging.warning(
                "omni multi-message dataset only uses messages; ignored template fields: %s",
                ", ".join(ignored_keys),
            )
        return body

    def get(self, req_idx: int) -> StdChatApiRequest:
        return self._make_request(req_idx, warmup=False)

    def warmup(self, size: int) -> ty.List[StdChatApiRequest]:
        return [self._make_request(idx, warmup=True) for idx in range(size)]

    def size(self) -> int:
        return self.num_requests

    def _make_request(self, req_idx: int, warmup: bool) -> StdChatApiRequest:
        rng_seed = self.seed - 100000000 if warmup else self.seed
        rng = random.Random(rng_seed + req_idx)
        body = _deepcopy_json(self.body_template)
        messages = body["messages"]
        self._mutate_messages(messages, rng)
        return {"messages": messages}

    def _mutate_messages(self, messages: Messages, rng: random.Random) -> None:
        user_indices = [
            idx
            for idx, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        target_user_idx = None
        if len(user_indices) >= 2:
            target_user_idx = user_indices[-2]
        elif user_indices:
            target_user_idx = user_indices[0]

        image_url_locs = list(_iter_image_url_locs(messages))
        last_image_loc = image_url_locs[-1] if image_url_locs else None

        for message_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if message_idx == target_user_idx:
                if isinstance(content, str):
                    message["content"] = _mutate_available_app_list(content, rng)
                else:
                    self._mutate_message_text(content, rng)
            if isinstance(content, list):
                self._mutate_message_images(content, message_idx, last_image_loc, rng)

    def _mutate_message_text(self, content: ty.Any, rng: random.Random) -> None:
        if not isinstance(content, list):
            return
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                part["text"] = _mutate_available_app_list(part["text"], rng)

    def _mutate_message_images(
        self,
        content: ty.List[JsonDict],
        message_idx: int,
        last_image_loc: ty.Optional[ty.Tuple[int, int]],
        rng: random.Random,
    ) -> None:
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if not isinstance(url, str) or url not in self.image_cache:
                continue
            data_url = self.image_cache[url]
            if last_image_loc == (message_idx, part_idx):
                data_url = _add_noise_to_data_url(
                    data_url,
                    OMNI_IMAGE_NOISE_BYTES,
                    rng,
                )
            image_url["url"] = data_url
