import time
import asyncio
import argparse
import logging
import wonderwords

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from tensorrt_llm import LLM, SamplingParams

logger = None


def init_logger(log_file):
    global logger

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


class PerfMetrics:
    def __init__(
        self, num_prompt_tokens, num_output_tokens, elasped_first_token, elasped_request
    ):
        r"""The time unit is milliseconds"""

        self.num_prompt_tokens = num_prompt_tokens
        self.num_output_tokens = num_output_tokens
        self.elasped_first_token = elasped_first_token
        self.elasped_request = elasped_request

    def __repr__(self):
        return f"PerfMetrics(num_prompt_tokens={self.num_prompt_tokens}, num_output_tokens={self.num_output_tokens}, elasped_first_token={self.elasped_first_token:.2f}ms, elasped_request={self.elasped_request:.2f}ms)"


async def send_request_and_wait_for_finishing(
    engine: LLM, prompt: str, sampling_params: SamplingParams
):
    request_sent_time = time.time()

    first_token_time = None
    async for output in engine.generate_async(prompt, sampling_params, streaming=True):
        if first_token_time is None:
            first_token_time = time.time()

    elasped_first_token = first_token_time - request_sent_time
    elasped_request = time.time() - request_sent_time

    num_prompt_tokens = len(output.prompt_token_ids)
    num_output_tokens = output.outputs[0].length

    return PerfMetrics(
        num_prompt_tokens, num_output_tokens, elasped_first_token, elasped_request
    )


warm_cnt = 100
finished_request_cnt = 0
benchmark_start_time = time.time()


async def task_fn(
    coroutine_id: int, engine: LLM, prompt: str, sampling_params: SamplingParams
):
    while True:
        logger.info("coroutine %d sent request.", coroutine_id)
        perf_metrics = await send_request_and_wait_for_finishing(
            engine, prompt, sampling_params
        )
        logger.info("coroutine %d finish request: %s", coroutine_id, perf_metrics)

        global warm_cnt, finished_request_cnt, benchmark_start_time
        finished_request_cnt += 1
        if finished_request_cnt <= warm_cnt:
            benchmark_start_time = time.time()
            continue
        total_elasped_time = time.time() - benchmark_start_time

        logger.info("qps: %.2f", (finished_request_cnt - warm_cnt) / total_elasped_time)


def get_tokenizer(hfmodel_path: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(hfmodel_path, trust_remote_code=True)


def get_prompt(num_prompt_tokens):
    random_word = wonderwords.RandomWord().word()
    return random_word + "hi" * (num_prompt_tokens - 1)


async def main(args):
    init_logger(args.log_file)

    engine = LLM(model=args.engine_path, tokenizer=get_tokenizer(args.hfmodel_path))
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        min_length=args.output_len,
        max_new_tokens=args.output_len,
    )

    coroutines = []
    for i in range(args.num_coroutine):
        prompt = get_prompt(args.input_len)
        coroutines.append(
            asyncio.create_task(task_fn(i, engine, prompt, sampling_params))
        )

    await asyncio.gather(*coroutines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfmodel-path", type=str, required=True)
    parser.add_argument("--engine-path", type=str, required=True)
    parser.add_argument("--num-coroutine", type=int, default=8)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--log-file", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main(args))
