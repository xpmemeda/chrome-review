import abc
import sys
import argparse
import json
import pathlib
import transformers
import logging
import tqdm
import numpy
import time
import heapq
import packaging.version as version
import typing as ty

Messages = ty.List[ty.Dict[str, str]]


def detect_dataset_type(dataset_path: str, max_lines: int = 1000) -> str:
    """
    返回:
    - "json"   : 普通 JSON 文件
    - "jsonl"  : JSON Lines 文件
    - "unknown": 无法判断 / 非法 JSON
    """

    path = pathlib.Path(dataset_path)

    # 先尝试按完整 JSON 解析
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return "json"
    except json.JSONDecodeError:
        pass

    # 再尝试按 JSONL 解析：每个非空行都必须是合法 JSON
    valid_lines = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError:
                    return "unknown"
        if valid_lines > 0:
            return "jsonl"

    except UnicodeDecodeError:
        return "unknown"

    return "unknown"


class Dataset(abc.ABC):
    @abc.abstractmethod
    def get(self) -> Messages:
        raise NotImplementedError()

    @abc.abstractmethod
    def size(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> int:
        raise NotImplementedError()


class TextDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset: ty.List[str] = []

        self._load_text(dataset_path)

        self.cursor = 0

    def get(self) -> str:
        if self.cursor >= len(self.dataset):
            return None

        text = self.dataset[self.cursor]
        self.cursor += 1

        return text

    def size(self) -> int:
        return len(self.dataset)

    def reset(self):
        self.cursor = 0

    def _load_text(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dataset.append([{"role": "user", "content": line}])


class StdOpenAIMessageJson(Dataset):
    def __init__(self, dataset_path):
        self.dataset: ty.List[Messages] = []

        self._load_json(dataset_path)

        self.cursor = 0

    def get(self) -> Messages:
        if self.cursor >= len(self.dataset):
            return None

        message = self.dataset[self.cursor]
        self.cursor += 1

        return message

    def size(self) -> int:
        return len(self.dataset)

    def reset(self):
        self.cursor = 0

    def _load_json(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "messages" in data:
                self.dataset.append(data["messages"])
            elif isinstance(data, list):
                self.dataset.append(data)
            else:
                raise RuntimeError(
                    f"Invalid JSON format in {dataset_path}: expected a list or a dict with 'messages' key."
                )

        for messages in self.dataset:
            logging.debug("Messages: %s", messages)


class StdOpenAIMessageJsonL(Dataset):
    def __init__(self, dataset_path):
        self.dataset: ty.List[Messages] = []

        self._load_jsonl(dataset_path)

        self.cursor = 0

    def get(self) -> Messages:
        if self.cursor >= len(self.dataset):
            return None

        message = self.dataset[self.cursor]
        self.cursor += 1

        return message

    def size(self) -> int:
        return len(self.dataset)

    def reset(self):
        self.cursor = 0

    def _load_json(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset.append(json.load(f))

    def _load_jsonl(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                if isinstance(data, dict) and "messages" in data:
                    self.dataset.append(data["messages"])
                elif isinstance(data, list):
                    self.dataset.append(data)
                else:
                    raise RuntimeError(
                        f"Invalid JSON format in {dataset_path}: expected a list or a dict with 'messages' key."
                    )


class Tokenizer:
    def __init__(self, tokenizer_path):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

    def tokenize(self, messages_or_text) -> ty.List[int]:
        if version.Version(transformers.__version__) >= version.Version("5.9.0"):
            if isinstance(messages_or_text, str):
                text = messages_or_text
            else:
                text = self._tokenizer.apply_chat_template(
                    messages_or_text, tokenize=False
                )
            logging.debug("Tokenizing messages: %s", text)
            tokens = self._tokenizer.encode(text)
            return tokens

        else:
            raise RuntimeError("Tokenizer requires transformers version >= 5.9.0")

    def count(self, messages_or_text) -> int:
        token_ids = self.tokenize(messages_or_text)
        return len(token_ids)

    def count_dataset(self, dataset: Dataset):
        num_tokens = 0

        while messages := dataset.get():
            num_tokens += self.count(messages)

        return num_tokens / dataset.size()


class RadixTreeNode:
    def __init__(self, tree: "RadixTree", parent: "RadixTreeNode"):
        self.tree = tree
        self.parent = parent
        self.ref = 0
        self.children = {}
        self.access_time = time.time()
        self.k = []

    def size(self) -> int:
        return len(self.k)

    def index(self) -> int:
        return self.k[0]

    def is_leaf(self):
        return len(self.children) == 0

    def update_access_time(self):
        self.access_time = time.time()

    def prefix(self, k) -> ty.List[int]:
        min_size = min(len(self.k), len(k))

        if min_size == 0:
            return []

        for i in range(min_size):
            if self.k[i] != k[i]:
                return k[:i]

        return self.k[:min_size]

    def child(self, k) -> ty.Optional["RadixTreeNode"]:
        if len(k) == 0:
            return None

        if k[0] in self.children:
            return self.children[k[0]]

        return None

    def __lt__(self, other: "RadixTreeNode"):
        return self.access_time < other.access_time


class RadixTree:
    def __init__(self, capacity: ty.Optional[int]):
        self.nodes = set()
        self.root = RadixTreeNode(self, None)
        self.nodes.add(self.root)

        self.capacity = capacity
        self.free_size = self.capacity
        self.protected_size = 0
        self.evictable_size = 0

    def insert(self, k) -> ty.Optional[RadixTreeNode]:
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
            return

        if need > self.free_size + evictable_size:
            for n in trace:
                n.ref -= 1
            return

        if need > self.free_size:
            evict_size = self._evict(need - self.free_size)
            free_size += evict_size
            evictable_size -= evict_size

        free_size -= need
        protected_size += need

        self.free_size = free_size
        self.protected_size = protected_size
        self.evictable_size = evictable_size

        new_n = RadixTreeNode(self, p)
        new_n.k = k
        new_n.ref = 1
        p.children[new_n.index()] = new_n

        self.nodes.add(new_n)

        return new_n

    def release(self, n: RadixTreeNode):

        while n:

            n.update_access_time()
            n.ref -= 1

            if n.ref == 0:
                self.protected_size -= n.size()
                self.evictable_size += n.size()

            n = n.parent

    def match(self, k: ty.List[int]) -> ty.List[int]:
        n = self.root
        r = []

        while n:
            prefix = n.prefix(k)

            k = k[len(prefix) :]
            r.extend(prefix)

            if len(prefix) < n.size():
                break

            n = n.child(k)

        return r

    def nnodes(self):
        count = 0
        stack = [self.root]

        while stack:
            x = stack.pop()
            count += 1
            stack.extend(x.children.values())

        return count

    def _split_node(self, n: RadixTreeNode, size: int):
        if n.size() <= size:
            raise RuntimeError()

        parent = n.parent
        k = n.k

        new_n = RadixTreeNode(self, parent)
        self.nodes.add(new_n)

        n.parent = new_n
        n.k = k[size:]

        new_n.ref = n.ref
        new_n.access_time = n.access_time
        new_n.k = k[:size]
        new_n.children[n.index()] = n

        parent.children[new_n.index()] = new_n

        return new_n, n

    def _evict(self, need) -> int:
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < need and len(leaves):
            x: RadixTreeNode = heapq.heappop(leaves)

            if x == self.root:
                break
            if x.ref > 0:
                break

            num_evicted += x.size()

            parent = x.parent
            self._delete_leaf(x)

            if parent.is_leaf():
                heapq.heappush(leaves, parent)

        return num_evicted

    def _collect_leaves(self) -> ty.List[RadixTreeNode]:
        ret = []
        stack: ty.List[RadixTreeNode] = [self.root]

        while stack:
            n = stack.pop()
            if n.is_leaf():
                ret.append(n)
            else:
                stack.extend(n.children.values())

        return ret

    def _delete_leaf(self, n: RadixTreeNode):
        if not n:
            raise RuntimeError()
        if not n.is_leaf():
            raise RuntimeError()

        p = n.parent
        p.children.pop(n.index())


class DatasetProfiler:
    def __init__(self, dataset: Dataset, tokenizer: Tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.radix_tree = RadixTree(sys.maxsize)

        self._num_tokens: ty.List[int] = []
        self.prefix_hit_size: ty.List[int] = []

        tqdm_iterater = tqdm.tqdm(range(dataset.size()))
        for _ in tqdm_iterater:
            messages = dataset.get()

            tokens = tokenizer.tokenize(messages_or_text=messages)
            prefix = self.radix_tree.match(tokens)

            self._num_tokens.append(len(tokens))
            self.prefix_hit_size.append(len(prefix))

            self.radix_tree.insert(tokens)

    @property
    def num_tokens(self):
        return sum(self._num_tokens)

    @property
    def hit_tokens(self):
        return sum(self.prefix_hit_size)

    @property
    def token_hit_rate(self):
        return self.hit_tokens / self.num_tokens

    @property
    def request_prefix_hit_rate(self):
        request_prefix_hit_rates = [
            hit_tokens / num_tokens
            for hit_tokens, num_tokens in zip(self.prefix_hit_size, self._num_tokens)
        ]
        return sum(request_prefix_hit_rates) / len(request_prefix_hit_rates)


if __name__ == "__main__":
    cmd_arguments = argparse.ArgumentParser()
    cmd_arguments.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    cmd_arguments.add_argument("--dataset", type=str, required=True)
    cmd_arguments.add_argument("--tokenizer", type=str, required=True)
    cmd_arguments = cmd_arguments.parse_args()

    logging.basicConfig(level=getattr(logging, cmd_arguments.log_level))

    # detect dataset type and load it accordingly
    dataset_type = detect_dataset_type(cmd_arguments.dataset)
    logging.info(f"Detected dataset type: {dataset_type}")
    if dataset_type == "json":
        dataset = StdOpenAIMessageJson(cmd_arguments.dataset)
    elif dataset_type == "jsonl":
        dataset = StdOpenAIMessageJsonL(cmd_arguments.dataset)
    else:
        logging.warning(
            f"Unknown dataset type: {dataset_type}, fallback to TextDataset"
        )
        dataset = TextDataset(cmd_arguments.dataset)

    tokenizer = Tokenizer(cmd_arguments.tokenizer)
    profiler = DatasetProfiler(dataset, tokenizer)

    num_tokens = profiler.num_tokens
    hit_tokens = profiler.hit_tokens
    token_hit_rate = profiler.token_hit_rate
    request_prefix_hit_rate = profiler.request_prefix_hit_rate
    logging.info(
        f"{num_tokens=}, {hit_tokens=}, {token_hit_rate=:.2f}, {request_prefix_hit_rate=:.2f}"
    )
