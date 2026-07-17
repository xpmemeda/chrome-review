import heapq
import json
import sys
import time
import typing as ty
import logging
import tqdm

import dataset as dataset_lib


class RadixTreeNode:
    def __init__(self, parent: ty.Optional["RadixTreeNode"]) -> None:
        self.parent = parent
        self.ref = 0
        self.children: ty.Dict[int, "RadixTreeNode"] = {}
        self.access_time = time.time()
        self.k: ty.List[int] = []

    def size(self) -> int:
        return len(self.k)

    def index(self) -> int:
        return self.k[0]

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def update_access_time(self) -> None:
        self.access_time = time.time()

    def prefix(self, k: ty.List[int]) -> ty.List[int]:
        min_size = min(len(self.k), len(k))
        for i in range(min_size):
            if self.k[i] != k[i]:
                return k[:i]
        return self.k[:min_size]

    def child(self, k: ty.List[int]) -> ty.Optional["RadixTreeNode"]:
        if len(k) == 0:
            return None
        return self.children.get(k[0])

    def __lt__(self, other: "RadixTreeNode") -> bool:
        return self.access_time < other.access_time


class RadixTree:
    def __init__(self, capacity: int) -> None:
        self.root = RadixTreeNode(None)
        self.capacity = capacity
        self.free_size = capacity
        self.protected_size = 0
        self.evictable_size = 0

    def insert(self, k: ty.List[int]) -> ty.Optional[RadixTreeNode]:
        p = self.root
        n = self.root.child(k)
        free_size = self.free_size
        protected_size = self.protected_size
        evictable_size = self.evictable_size
        trace: ty.List[RadixTreeNode] = []

        while n:
            n.update_access_time()
            trace.append(n)
            prefix_size = len(n.prefix(k))
            if prefix_size == 0:
                break
            if prefix_size < n.size():
                n = self._split_node(n, prefix_size)[0]
            if n.ref == 0:
                protected_size += n.size()
                evictable_size -= n.size()
            n.ref += 1
            k = k[prefix_size:]
            p = n
            n = n.child(k)

        need = len(k)
        if need == 0:
            self.protected_size = protected_size
            self.evictable_size = evictable_size
            return None
        if need > self.free_size + evictable_size:
            for node in trace:
                node.ref -= 1
            return None
        if need > self.free_size:
            evict_size = self._evict(need - self.free_size)
            free_size += evict_size
            evictable_size -= evict_size

        free_size -= need
        protected_size += need
        self.free_size = free_size
        self.protected_size = protected_size
        self.evictable_size = evictable_size

        new_node = RadixTreeNode(p)
        new_node.k = k
        new_node.ref = 1
        p.children[new_node.index()] = new_node
        return new_node

    def match(self, k: ty.List[int]) -> ty.List[int]:
        n: ty.Optional[RadixTreeNode] = self.root
        result: ty.List[int] = []
        while n:
            prefix = n.prefix(k)
            k = k[len(prefix) :]
            result.extend(prefix)
            if len(prefix) < n.size():
                break
            n = n.child(k)
        return result

    def _split_node(
        self, n: RadixTreeNode, size: int
    ) -> ty.Tuple[RadixTreeNode, RadixTreeNode]:
        parent = n.parent
        if parent is None or n.size() <= size:
            raise RuntimeError("bad radix split")

        new_node = RadixTreeNode(parent)
        old_key = n.k
        n.parent = new_node
        n.k = old_key[size:]
        new_node.ref = n.ref
        new_node.access_time = n.access_time
        new_node.k = old_key[:size]
        new_node.children[n.index()] = n
        parent.children[new_node.index()] = new_node
        return new_node, n

    def _evict(self, need: int) -> int:
        leaves = self._collect_leaves()
        heapq.heapify(leaves)
        num_evicted = 0
        while num_evicted < need and leaves:
            node = heapq.heappop(leaves)
            if node == self.root or node.ref > 0:
                break
            num_evicted += node.size()
            parent = node.parent
            self._delete_leaf(node)
            if parent and parent.is_leaf():
                heapq.heappush(leaves, parent)
        return num_evicted

    def _collect_leaves(self) -> ty.List[RadixTreeNode]:
        result = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                result.append(node)
            else:
                stack.extend(node.children.values())
        return result

    def _delete_leaf(self, node: RadixTreeNode) -> None:
        parent = node.parent
        if parent is None or not node.is_leaf():
            raise RuntimeError("bad radix delete")
        parent.children.pop(node.index())


class PrefixHitProfiler:
    IMAGE_PLACEHOLDER = "<image>"

    def __init__(
        self, dataset: dataset_lib.Dataset, encoding_name: str = "o200k_base"
    ) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "Dataset prefix profiling requires tiktoken. "
                "Install it or disable profiling if you only want to run requests."
            ) from exc

        self.dataset = dataset
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.radix_tree = RadixTree(sys.maxsize)
        self._num_tokens: ty.List[int] = []
        self.prefix_hit_size: ty.List[int] = []

        logging.info("dataset profile ing.")

        for req_idx in tqdm.tqdm(range(dataset.size())):
            tokens = self._tokenize(dataset.get(req_idx))
            prefix = self.radix_tree.match(tokens)
            self._num_tokens.append(len(tokens))
            self.prefix_hit_size.append(len(prefix))
            self.radix_tree.insert(tokens)

        message = (
            "dataset profile: "
            f"requests={self.request_count} "
            f"num_tokens={self.num_tokens} "
            f"hit_tokens={self.hit_tokens} "
            f"avg_request_tokens={self.avg_request_tokens:.1f} "
            f"avg_hit_tokens={self.avg_hit_tokens:.1f} "
            f"token_hit_rate={self.token_hit_rate:.4f} "
            f"request_prefix_hit_rate={self.request_prefix_hit_rate:.4f}"
        )
        logging.info(message)

    def _tokenize(self, request: dataset_lib.StdChatApiRequest) -> ty.List[int]:
        text = json.dumps(
            self._profile_messages(request),
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return self.encoding.encode(text)

    def _profile_messages(
        self, request: dataset_lib.StdChatApiRequest
    ) -> dataset_lib.Messages:
        return [self._profile_message(message) for message in request["messages"]]

    def _profile_message(self, message: dataset_lib.JsonDict) -> dataset_lib.JsonDict:
        content = message.get("content")
        if not isinstance(content, list):
            return dict(message)
        profiled = dict(message)
        profiled["content"] = [
            self._profile_content_part(part) if isinstance(part, dict) else part
            for part in content
        ]
        return profiled

    def _profile_content_part(self, part: dataset_lib.JsonDict) -> dataset_lib.JsonDict:
        if part.get("type") != "image_url":
            return dict(part)
        profiled = dict(part)
        image_url = profiled.get("image_url")
        if isinstance(image_url, dict):
            profiled["image_url"] = dict(image_url)
            profiled["image_url"]["url"] = self.IMAGE_PLACEHOLDER
        return profiled

    @property
    def request_count(self) -> int:
        return len(self._num_tokens)

    @property
    def num_tokens(self) -> int:
        return sum(self._num_tokens)

    @property
    def hit_tokens(self) -> int:
        return sum(self.prefix_hit_size)

    @property
    def avg_request_tokens(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.num_tokens / self.request_count

    @property
    def avg_hit_tokens(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.hit_tokens / self.request_count

    @property
    def token_hit_rate(self) -> float:
        if self.num_tokens == 0:
            return 0.0
        return self.hit_tokens / self.num_tokens

    @property
    def request_prefix_hit_rate(self) -> float:
        if not self._num_tokens:
            return 0.0
        request_rates = [
            hit_tokens / num_tokens
            for hit_tokens, num_tokens in zip(self.prefix_hit_size, self._num_tokens)
            if num_tokens
        ]
        return sum(request_rates) / len(request_rates) if request_rates else 0.0
