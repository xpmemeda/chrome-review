import abc
import time
import random
import asyncio
import argparse
import logging
import wonderwords
import platform
import subprocess
import re
import GPUtil

from packaging import version
from typing import Optional, Tuple
from transformers import AutoTokenizer


def init_logger(log_file):
    logger = logging.getLogger("benchmark_async_stream")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
    )
    logger.addHandler(file_handler)

    return logger


logger = None


class Device:
    def _get_cpu(self):
        def get_processor_model_name():
            if platform.system() != "Linux":
                raise NotImplementedError("Only support on Linux.")

            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1).strip()

            return ""

        return get_processor_model_name()

    def _get_gpu(self):
        return GPUtil.getGPUs()[0].name

    def __str__(self):
        return f"Device(CPU='{self._get_cpu()}', GPU='{self._get_gpu()}')"


class PerfMetrics:
    def __init__(self, num_prompt_tokens, num_output_tokens, ttft, e2e):
        self.num_prompt_tokens = num_prompt_tokens
        self.num_output_tokens = num_output_tokens
        self.ttft = ttft
        self.e2e = e2e

    def __repr__(self):
        return f"PerfMetrics(num_prompt_tokens={self.num_prompt_tokens}, num_output_tokens={self.num_output_tokens}, ttft={self.ttft:.2f}s, e2e={self.e2e:.2f}s)"


class MetricsManager:
    def __init__(self, trim_size=1):
        self.trim_size = trim_size

        self.num_prompts = 0

        self.arive_times = []
        self.num_prompt_tokens = []
        self.num_output_tokens = []
        self.ttfts = []
        self.e2es = []

    def update(self, metrics):
        self.num_prompts += 1

        self.arive_times.append(time.time())
        self.num_prompt_tokens.append(metrics.num_prompt_tokens)
        self.num_output_tokens.append(metrics.num_output_tokens)
        self.ttfts.append(metrics.ttft)
        self.e2es.append(metrics.e2e)

    def _num_tokens(self):
        num_prompt_tokens = self.num_prompt_tokens[
            self.trim_size : self.num_prompts - self.trim_size
        ]
        num_output_tokens = self.num_output_tokens[
            self.trim_size : self.num_prompts - self.trim_size
        ]
        num_tokens = sum(num_prompt_tokens) + sum(num_output_tokens)
        return num_tokens

    def _time(self):
        if self.trim_size == 0:
            t1 = self.arive_times[0] - self.e2es[0]
        else:
            t1 = self.arive_times[self.trim_size - 1]

        t2 = self.arive_times[self.num_prompts - self.trim_size - 1]

        return t2 - t1

    def _avg_ttft(self):
        ttfts = self.ttfts[self.trim_size : self.num_prompts - self.trim_size]
        return sum(ttfts) / len(ttfts)

    def _avg_e2e(self):
        e2es = self.e2es[self.trim_size : self.num_prompts - self.trim_size]
        return sum(e2es) / len(e2es)

    def _rps(self):
        n_requests = self.num_prompts - 2 * self.trim_size
        return n_requests / self._time()

    def _tps(self):
        n_tokens = self._num_tokens()
        return n_tokens / self._time()

    def __repr__(self):
        trim_size = self.trim_size
        if self.num_prompts <= 2 * self.trim_size:
            self.trim_size = 0
        repr = f"MetricsManager(Num prompts {self.num_prompts} Avg TTFT {self._avg_ttft():.3f} Avg E2E {self._avg_e2e():.3f} RPS {self._rps():.2f} TPS {self._tps():.2f})"
        self.trim_size = trim_size
        return repr

    @classmethod
    def merge_repr(cls, metrics_managers: list) -> str:
        def mean(x: list):
            return sum(x) / len(x)

        num_prompts = sum(
            metrics_manager.num_prompts for metrics_manager in metrics_managers
        )
        rps = sum([metrics_manager._rps() for metrics_manager in metrics_managers])
        tps = sum([metrics_manager._tps() for metrics_manager in metrics_managers])
        avg_ttft = mean(
            [metrics_manager._avg_ttft() for metrics_manager in metrics_managers]
        )
        avg_e2e = mean(
            [metrics_manager._avg_e2e() for metrics_manager in metrics_managers]
        )
        return f"MetricsManager(Num prompts {num_prompts} Avg TTFT {avg_ttft:.3f} Avg E2E {avg_e2e:.3f} RPS {rps:.2f} TPS {tps:.2f})"


class RequestManagerBase:
    class Request:
        def __init__(self, index: int, prompt: str, update_metrics: bool):
            self.index = index
            self.prompt = prompt
            self.update_metrics = update_metrics

    def __init__(
        self,
        tokenizer,
        num_prompts,
        task_num_prompt_tokens,
        task_num_prompt_hit_tokens,
        task_num_output_tokens,
    ):
        self.tokenizer = tokenizer

        self.num_prompts = num_prompts
        self.num_warmup_prompts = num_prompts // 4
        self.num_cooldown_prompts = num_prompts // 4
        self.task_num_prompt_tokens = task_num_prompt_tokens
        self.task_num_prompt_hit_tokens = task_num_prompt_hit_tokens
        self.task_num_output_tokens = task_num_output_tokens

        self.random_words = wonderwords.RandomWord().random_words(
            self.num_warmup_prompts + self.num_prompts + self.num_cooldown_prompts
        )
        self.random_warmup_words = self.random_words[: self.num_warmup_prompts]
        self.random_words = self.random_words[self.num_warmup_prompts :]
        self.random_cooldown_words = self.random_words[
            self.num_warmup_prompts + self.num_prompts :
        ]

        self.index = 0
        self.warmup_prompts = self.generate_warmup_prompts()
        self.prompts = self.generate_prompts()
        self.cooldown_prompts = self.generate_warmup_prompts()

    def _generate_prompts(
        self, random_words, num_prompt_tokens_list, task_num_prompt_hit_tokens
    ):
        prompts = []
        for num_prompt_tokens, random_word in zip(num_prompt_tokens_list, random_words):
            random_word = (
                "hi" * task_num_prompt_hit_tokens
                + " "
                + random_word
                + " "
                + "hi" * (num_prompt_tokens - task_num_prompt_hit_tokens)
            )
            tokens = self.tokenizer.tokenize(random_word)[:num_prompt_tokens]
            prompt = self.tokenizer.convert_tokens_to_string(tokens)

            x = len(self.tokenizer.encode(prompt))
            y = num_prompt_tokens
            assert x == y, "%d != %d" % (x, y)

            prompts.append(prompt)

        return prompts

    def generate_warmup_prompts(self):
        random.seed(0)
        random_num_prompt_tokens = [
            random.randint(
                self.task_num_prompt_tokens // 3, self.task_num_prompt_tokens * 2
            )
            for _ in range(self.num_warmup_prompts)
        ]
        return self._generate_prompts(
            self.random_warmup_words, random_num_prompt_tokens, 0
        )

    def generate_cooldown_prompts(self):
        cooldown_prompt_lens = [self.task_num_prompt_tokens] * self.num_cooldown_prompts
        return self._generate_prompts(
            self.random_cooldown_words, cooldown_prompt_lens, 0
        )

    def get_request(self) -> Optional[Request]:
        if self.warmup_prompts:
            return RequestManagerBase.Request(-1, self.warmup_prompts.pop(), False)

        if self.index < len(self.prompts):
            request = RequestManagerBase.Request(
                self.index, self.prompts[self.index], True
            )
            self.index += 1
            return request

        if self.cooldown_prompts:
            return RequestManagerBase.Request(-1, self.cooldown_prompts.pop(), False)

        return None

    @abc.abstractmethod
    def generate_prompts(self):
        pass

    @abc.abstractmethod
    def check_num_tokens(self, index: int, metrics: PerfMetrics):
        pass


class RequestManagerV1(RequestManagerBase):
    def generate_prompts(self):
        num_prompt_tokens_list = [
            self.task_num_prompt_tokens for _ in range(self.num_prompts)
        ]
        return self._generate_prompts(
            self.random_words, num_prompt_tokens_list, self.task_num_prompt_hit_tokens
        )

    def check_num_tokens(self, index: int, metrics: PerfMetrics):
        assert metrics.num_prompt_tokens == self.task_num_prompt_tokens
        assert metrics.num_output_tokens == self.task_num_output_tokens


class RequestManagerV2(RequestManagerBase):
    def __init__(
        self,
        tokenizer,
        num_prompts,
        task_num_prompt_tokens,
        task_num_prompt_hit_tokens,
        task_num_output_tokens,
    ):
        random.seed(0)
        assert num_prompts % 2 == 0
        self.random_num_prompt_tokens_list = []

        random_range = task_num_prompt_tokens // 3
        for _ in range(num_prompts // 2):
            random_size = random.randint(0, random_range)
            self.random_num_prompt_tokens_list += [
                task_num_prompt_tokens - random_size,
                task_num_prompt_tokens + random_size,
            ]
        random.shuffle(self.random_num_prompt_tokens_list)

        super().__init__(
            tokenizer,
            num_prompts,
            task_num_prompt_tokens,
            task_num_prompt_hit_tokens,
            task_num_output_tokens,
        )

    def generate_prompts(self):
        return self._generate_prompts(
            self.random_words,
            self.random_num_prompt_tokens_list,
            self.task_num_prompt_hit_tokens,
        )

    def check_num_tokens(self, index: int, metrics: PerfMetrics):
        assert metrics.num_prompt_tokens == self.random_num_prompt_tokens_list[index]
        assert metrics.num_output_tokens == self.task_num_output_tokens


class trtllmBackend:
    def __init__(
        self,
        model: str,
        tokenizer: str,
        batch_size: int,
        num_prompt_tokens: int,
        num_prompt_hit_tokens: int,
        num_output_tokens: int,
    ):
        self.batch_size = batch_size
        self.num_prompt_tokens = num_prompt_tokens
        self.num_prompt_hit_tokens = num_prompt_hit_tokens
        self.num_output_tokens = num_output_tokens

        import tensorrt_llm as trtllm
        from tensorrt_llm import BuildConfig
        from tensorrt_llm.bindings.executor import KvCacheConfig
        from tensorrt_llm.builder import PluginConfig

        plugin_config = PluginConfig.from_dict(
            {
                "dtype": "float16",
                "gemm_plugin": "float16",
                "_paged_kv_cache": True,
                "_use_paged_context_fmha": True,
            }
        )
        build_config = BuildConfig(
            max_batch_size=self.batch_size,
            opt_batch_size=self.batch_size,
            max_seq_len=4096,  # OOM if too large.
            plugin_config=plugin_config,
        )
        kv_cache_config = KvCacheConfig()
        kv_cache_config.enable_block_reuse = self.num_prompt_hit_tokens > 0

        assert version.parse(trtllm.__version__) >= version.parse("0.14.0")

        self._llm = trtllm.LLM(
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            dtype="float16",
            build_config=build_config,
            kv_cache_config=kv_cache_config,
        )
        self._sampling_params = trtllm.SamplingParams(
            temperature=0.0,
            min_tokens=num_output_tokens,
            max_tokens=num_output_tokens,
        )

    @classmethod
    def version(cls):
        import tensorrt_llm as trtllm

        return trtllm.__version__

    async def async_generate(self, prompt, num_output_tokens=0):
        if num_output_tokens:
            import tensorrt_llm as trtllm

            sampling_params = trtllm.SamplingParams(
                temperature=0.0,
                min_tokens=num_output_tokens,
                max_tokens=num_output_tokens,
            )
        else:
            sampling_params = self._sampling_params
        async for output in self._llm.generate_async(
            prompt, sampling_params, streaming=True
        ):
            yield output

    def get_num_tokens(self, output):
        num_prompt_tokens = len(output.prompt_token_ids)
        num_output_tokens = output.outputs[0].length
        return (num_prompt_tokens, num_output_tokens)


class vllmBackend:
    def __init__(
        self,
        model,
        tokenizer,
        batch_size,
        num_prompt_tokens,
        num_prompt_hit_tokens,
        num_output_tokens,
    ):
        self.batch_size = batch_size
        self.num_prompt_tokens = num_prompt_tokens
        self.num_prompt_hit_tokens = num_prompt_hit_tokens
        self.num_output_tokens = num_output_tokens

        import vllm

        assert version.parse(vllm.__version__) >= version.parse("0.6.0")

        engine_args = vllm.AsyncEngineArgs(
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            disable_log_requests=True,
            max_model_len=8192,
            enable_prefix_caching=self.num_prompt_hit_tokens > 0,
        )
        self._llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)

        self._sampling_params = vllm.SamplingParams(
            temperature=0.0,
            min_tokens=num_output_tokens,
            max_tokens=num_output_tokens,
        )

    @classmethod
    def version(cls):
        import vllm

        return vllm.__version__

    async def async_generate(self, prompt, num_output_tokens=0):
        request_id = str(random.randint(0, 10000000))
        if num_output_tokens:
            import vllm

            sampling_params = vllm.SamplingParams(
                temperature=0.0,
                min_tokens=num_output_tokens,
                max_tokens=num_output_tokens,
            )
        else:
            sampling_params = self._sampling_params
        async for output in self._llm.generate(prompt, sampling_params, request_id):
            yield output

    def get_num_tokens(self, output):
        num_prompt_tokens = len(output.prompt_token_ids)
        num_output_tokens = len(output.outputs[0].token_ids)
        return (num_prompt_tokens, num_output_tokens)


class sglangBackend:
    def __init__(
        self,
        model,
        tokenizer,
        batch_size,
        num_prompt_tokens,
        num_prompt_hit_tokens,
        num_output_tokens,
    ):
        self.batch_size = batch_size
        self.num_prompt_tokens = num_prompt_tokens
        self.num_prompt_hit_tokens = num_prompt_hit_tokens
        self.num_output_tokens = num_output_tokens

        import sglang

        # Not implemented or built, mostly likely because the current current device does not support this kernel (less likely TORCH_CUDA_ARCH_LIST was set incorrectly while building)
        # Possible solutions:
        # 1. disable cuda graph by --disable-cuda-graph
        # 2. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)
        # 3. disable torch compile by not using --enable-torch-compile
        # Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose
        self._llm = sglang.Engine(
            model_path=model, max_running_requests=batch_size, disable_cuda_graph=True
        )
        self._sampling_params = {
            "temperature": 0.0,
            "min_new_tokens": num_output_tokens,
            "max_new_tokens": num_output_tokens,
        }

    @classmethod
    def version(cls):
        import sglang

        return sglang.__version__

    async def async_generate(self, prompt, num_output_tokens=0):
        if num_output_tokens:
            sampling_params = {
                "temperature": 0.0,
                "min_new_tokens": num_output_tokens,
                "max_new_tokens": num_output_tokens,
            }
        else:
            sampling_params = self._sampling_params
        async for chunk in await self._llm.async_generate(
            prompt, sampling_params, stream=True
        ):
            yield chunk

    def get_num_tokens(self, output):
        metainfo = output["meta_info"]
        return metainfo["prompt_tokens"], metainfo["completion_tokens"]


class wnrBackend:
    def __init__(
        self,
        model,
        tokenizer,
        batch_size,
        num_prompt_tokens,
        num_prompt_hit_tokens,
        num_output_tokens,
    ):
        self.batch_size = batch_size
        self.num_prompt_tokens = num_prompt_tokens
        self.num_prompt_hit_tokens = num_prompt_hit_tokens
        self.num_output_tokens = num_output_tokens

        import wnr
        import wnr.vllm as vllm

        engine_args = vllm.AsyncEngineArgs(
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            disable_log_requests=True,
            max_model_len=8192,
        )
        self._llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)

        self._sampling_params = vllm.SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=num_output_tokens,
        )

    @classmethod
    def version(self):
        import wnr

        return wnr.__version__

    async def async_generate(self, prompt, num_output_tokens=0):
        if num_output_tokens:
            import wnr.vllm as vllm

            sampling_params = vllm.SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=num_output_tokens,
            )
        else:
            sampling_params = self._sampling_params
        request_id = str(random.randint(0, 10000000))

        async for output in self._llm.generate(prompt, sampling_params, request_id):
            yield output

    def get_num_tokens(self, output):
        num_prompt_tokens = len(output.prompt_token_ids)
        num_output_tokens = len(output.outputs[0].token_ids)
        return (num_prompt_tokens, num_output_tokens)


class Task:
    def __init__(
        self,
        coroutine_id,
        sync_steps,
        backend,
        requests_manager,
        metrics_manager,
    ):
        self.coroutine_id = coroutine_id

        self.sync_steps = sync_steps

        self.engine = backend
        self.requests_manager = requests_manager
        self.metrics_manager = metrics_manager

    async def run(self):
        await self.sync_coros()

        while request := self.requests_manager.get_request():
            logger.info("coroutine %d sent request.", self.coroutine_id)

            index = request.index
            prompt = request.prompt
            update_metrics = request.update_metrics

            perf_metrics = await self.send_request(prompt)
            if not update_metrics:
                continue

            self.requests_manager.check_num_tokens(index, perf_metrics)
            self.metrics_manager.update(perf_metrics)

            logger.info(
                "coroutine %d finish request: %s, %s",
                self.coroutine_id,
                perf_metrics,
                self.metrics_manager,
            )

    async def sync_coros(self):
        async for _ in self.engine.async_generate(
            "hi", self.coroutine_id * self.sync_steps + 1
        ):
            pass

    async def send_request(self, prompt):
        request_sent_time = time.time()

        first_token_time = None
        async for output in self.engine.async_generate(prompt):
            if first_token_time is None:
                first_token_time = time.time()

        elasped_first_token = first_token_time - request_sent_time
        elasped_request = time.time() - request_sent_time

        num_prompt_tokens, num_output_tokens = self.engine.get_num_tokens(output)

        return PerfMetrics(
            num_prompt_tokens, num_output_tokens, elasped_first_token, elasped_request
        )


async def main(args):
    global logger
    logger = init_logger(args.log)

    backends = {
        "wnr": wnrBackend,
        "trtllm": trtllmBackend,
        "vllm": vllmBackend,
        "sglang": sglangBackend,
    }
    backend_cls = backends[args.backend]

    logger.info(
        f"Benchmark start: "
        f"device={Device()}, "
        f"backend={args.backend}-{backend_cls.version()}, "
        f"model={args.model}, "
        f"num_coroutine={args.num_coroutine}, "
        f"num_prompt_tokens={args.num_prompt_tokens}, "
        f"num_prompt_hit_tokens={args.num_prompt_hit_tokens}, "
        f"num_output_tokens={args.num_output_tokens}"
    )

    backend = backend_cls(
        args.model,
        args.model,
        args.num_coroutine,
        args.num_prompt_tokens,
        args.num_prompt_hit_tokens,
        args.num_output_tokens,
    )
    requests_manager = RequestManagerV1(
        AutoTokenizer.from_pretrained(args.model, trust_remote_code=True),
        args.num_prompts,
        args.num_prompt_tokens,
        args.num_prompt_hit_tokens,
        args.num_output_tokens,
    )

    metrics_managers = []
    coroutines = []
    for i in range(args.num_coroutine):
        metrics_manager = MetricsManager()
        metrics_managers.append(metrics_manager)
        coroutines.append(
            asyncio.create_task(
                Task(
                    i,
                    3,
                    backend,
                    requests_manager,
                    metrics_manager,
                ).run()
            )
        )

    await asyncio.gather(*coroutines)

    # re pattern.
    logger.info(
        f"Benchmark done: "
        f"device={Device()}, "
        f"backend={args.backend}-{backend_cls.version()}, "
        f"model={args.model}, "
        f"num_coroutine={args.num_coroutine}, "
        f"num_prompt_tokens={args.num_prompt_tokens}, "
        f"num_prompt_hit_tokens={args.num_prompt_hit_tokens}, "
        f"num_output_tokens={args.num_output_tokens}, "
        f"Metrics={MetricsManager.merge_repr(metrics_managers)}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="benchmark-async.log")
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["wnr", "trtllm", "vllm", "sglang"],
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("-n", "--num-prompts", type=int, help="128", required=True)
    parser.add_argument("-b", "--num-coroutine", type=int, help="8", required=True)
    parser.add_argument("-i", "--num-prompt-tokens", type=int, required=True)
    parser.add_argument("-hh", "--num-prompt-hit-tokens", type=int, default=0)
    parser.add_argument("-o", "--num-output-tokens", type=int, required=True)
    args = parser.parse_args()

    asyncio.run(main(args))
