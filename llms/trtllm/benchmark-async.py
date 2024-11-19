import time
import random
import asyncio
import argparse
import logging
import wonderwords

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


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


logger = init_logger("benchmark-async.log")


class PromptsManager:
    def __init__(
        self,
        tokenizer,
        num_prompts,
        num_warmup_prompts,
        task_num_prompt_tokens,
        task_num_prompt_hit_tokens,
        task_num_output_tokens,
    ):
        self.tokenizer = tokenizer

        self.num_prompts = num_prompts
        self.num_warmup_prompts = num_warmup_prompts
        self.task_num_prompt_tokens = task_num_prompt_tokens
        self.task_num_prompt_hit_tokens = task_num_prompt_hit_tokens
        self.task_num_output_tokens = task_num_output_tokens

        self.index = 0
        self.warmup_prompts = self.generate_warmup_prompts()
        self.prompts = self.generate_prompts()

    def generate_warmup_prompts(self):
        random.seed(0)
        random_words = wonderwords.RandomWord().random_words(self.num_warmup_prompts)
        random_num_prompt_tokens = [
            random.randint(
                self.task_num_prompt_tokens // 3, self.task_num_prompt_tokens * 2
            )
            for _ in range(self.num_warmup_prompts)
        ]
        prompts = [
            random_word + "hi" * random_num_prompt_token
            for random_word, random_num_prompt_token in zip(
                random_words, random_num_prompt_tokens
            )
        ]
        return prompts

    def generate_prompts(self):
        random_words = wonderwords.RandomWord().random_words(self.num_prompts)
        prompts = []
        for random_word in random_words:
            random_word = (
                "hi" * self.task_num_prompt_hit_tokens + " " + random_word + " hihihi"
            )
            num_word_tokens = len(self.tokenizer.encode(random_word))
            prompt = random_word + "hi" * (
                self.task_num_prompt_tokens - num_word_tokens
            )
            x = len(self.tokenizer.encode(prompt))
            y = self.task_num_prompt_tokens
            assert x == y, "%d != %d" % (x, y)
            prompts.append(prompt)
        return prompts

    def get_prompt(self):
        if self.warmup_prompts:
            return self.warmup_prompts.pop()

        if self.index < len(self.prompts):
            prompt = self.prompts[self.index]
            self.index += 1
            return prompt
        else:
            return None


class MetricsManager:
    class PerfMetrics:
        def __init__(self, num_prompt_tokens, num_output_tokens, ttft, e2e):
            self.num_prompt_tokens = num_prompt_tokens
            self.num_output_tokens = num_output_tokens
            self.ttft = ttft
            self.e2e = e2e

        def __repr__(self):
            return f"PerfMetrics(num_prompt_tokens={self.num_prompt_tokens}, num_output_tokens={self.num_output_tokens}, ttft={self.ttft:.2f}ms, e2e={self.e2e:.2f}ms)"

    def __init__(
        self,
        num_prompts,
        num_warmup_prompts,
        task_num_prompt_tokens,
        task_num_output_tokens,
    ):
        self.num_prompts = num_prompts
        self.num_warmup_prompts = num_warmup_prompts
        self.task_num_prompt_tokens = task_num_prompt_tokens
        self.task_num_output_tokens = task_num_output_tokens

        self.finish_cnt = 0

        self.begin_time = None
        self.end_time = None

        self.ttfts = []
        self.e2es = []

    def update(self, metrics) -> bool:
        self.finish_cnt += 1
        if self.finish_cnt <= self.num_warmup_prompts:
            self.begin_time = time.time()
            return False

        assert (
            metrics.num_prompt_tokens == self.task_num_prompt_tokens
        ), f"{metrics.num_prompt_tokens} != {self.task_num_prompt_tokens}"
        assert (
            metrics.num_output_tokens == self.task_num_output_tokens
        ), f"{metrics.num_output_tokens} != {self.task_num_output_tokens}"

        self.ttfts.append(metrics.ttft)
        self.e2es.append(metrics.e2e)
        self.end_time = time.time()

        return True

    def _avg_ttft(self):
        return sum(self.ttfts) / len(self.ttfts)

    def _avg_e2e(self):
        return sum(self.e2es) / len(self.e2es)

    def _rps(self):
        return (self.finish_cnt - self.num_warmup_prompts) / (
            self.end_time - self.begin_time
        )

    def _tps(self):
        return self._rps() * (self.task_num_prompt_tokens + self.task_num_output_tokens)

    def __repr__(self):
        return f"MetricsManager(Num prompts {self.finish_cnt - self.num_warmup_prompts} Avg TTFT {self._avg_ttft():.2f} Avg E2E {self._avg_e2e():.2f} RPS {self._rps():.2f} TPS {self._tps():.2f})"


class Framework:
    def __init__(
        self,
        model: str,
        tokenizer: str,
        batch_size: int,
        num_prompt_tokens: int,
        num_output_tokens: int,
    ):
        self.batch_size = batch_size
        self.num_prompt_tokens = num_prompt_tokens
        self.num_output_tokens = num_output_tokens

        import tensorrt_llm as trtllm

        assert trtllm.__version__ == "0.14.0"

        self._llm = trtllm.LLM(
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            dtype="float16",
            build_config=trtllm.BuildConfig(
                max_batch_size=self.batch_size,
                opt_batch_size=self.batch_size,
                max_seq_len=2048,
                plugin_config=trtllm.builder.PluginConfig.from_dict(
                    {"dtype": "float16", "gemm_plugin": "float16"}
                ),
            ),
        )
        self._sampling_params = trtllm.SamplingParams(
            temperature=0.8,
            top_p=0.95,
            min_tokens=num_output_tokens,
            max_tokens=num_output_tokens,
        )

    async def async_generate(self, prompt):
        async for output in self._llm.generate_async(
            prompt, self._sampling_params, streaming=True
        ):
            yield output


class Task:
    def __init__(
        self,
        coroutine_id,
        framework,
        prompts_manager,
        metrics_manager,
    ):
        self.coroutine_id = coroutine_id

        self.engine = framework
        self.prompts_manager = prompts_manager
        self.metrics_manager = metrics_manager

    async def run(self):
        global warm_cnt, num_prompts, finished_request_cnt, benchmark_start_time

        while prompt := self.prompts_manager.get_prompt():
            logger.info("coroutine %d sent request.", self.coroutine_id)

            perf_metrics = await self.send_request(prompt)
            update = self.metrics_manager.update(perf_metrics)

            if not update:
                continue

            logger.info(
                "coroutine %d finish request: %s, %s",
                self.coroutine_id,
                perf_metrics,
                self.metrics_manager,
            )

    async def send_request(self, prompt):
        request_sent_time = time.time()

        first_token_time = None
        async for output in self.engine.async_generate(prompt):
            if first_token_time is None:
                first_token_time = time.time()

        elasped_first_token = first_token_time - request_sent_time
        elasped_request = time.time() - request_sent_time

        num_prompt_tokens = len(output.prompt_token_ids)
        num_output_tokens = output.outputs[0].length

        return MetricsManager.PerfMetrics(
            num_prompt_tokens, num_output_tokens, elasped_first_token, elasped_request
        )


async def main(args):
    framework = Framework(
        args.model,
        args.model,
        args.num_coroutine,
        args.num_prompt_tokens,
        args.num_output_tokens,
    )
    prompts_manager = PromptsManager(
        AutoTokenizer.from_pretrained(args.model, trust_remote_code=True),
        args.num_prompts,
        args.num_prompts // 4,
        args.num_prompt_tokens,
        args.num_prompt_hit_tokens,
        args.num_output_tokens,
    )
    metrics_manager = MetricsManager(
        args.num_prompts,
        args.num_prompts // 4,
        args.num_prompt_tokens,
        args.num_output_tokens,
    )

    coroutines = []
    for i in range(args.num_coroutine):
        coroutines.append(
            asyncio.create_task(
                Task(i, framework, prompts_manager, metrics_manager).run()
            )
        )
        await asyncio.sleep(0.1)

    await asyncio.gather(*coroutines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, help="128", required=True)
    parser.add_argument("--num-coroutine", type=int, help="8", required=True)
    parser.add_argument("--num-prompt-tokens", type=int, required=True)
    parser.add_argument("--num-prompt-hit-tokens", type=int, default=0)
    parser.add_argument("--num-output-tokens", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(main(args))
