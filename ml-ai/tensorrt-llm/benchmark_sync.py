import argparse
import os
import sys
import torch
import time
import pdb
import wonderwords

from typing import List

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm import BuildConfig


def get_tokenizer(hfmodel_path: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(hfmodel_path, trust_remote_code=True)


def get_prompts(num_prompts, input_len, tokenizer):
    random_words = wonderwords.RandomWord().random_words(num_prompts)
    prompts = []
    for random_word in random_words:
        num_word_tokens = len(tokenizer.encode(random_word))
        prompts.append(random_word + " " + "hi" * (input_len - num_word_tokens))
    return prompts


def main(args):
    if args.engine_path is None:
        llm = LLM(
            model=args.hfmodel_path,
            tokenizer=get_tokenizer(args.hfmodel_path),
            dtype="float16",
            build_config=BuildConfig(
                max_batch_size=args.max_batch_size, opt_batch_size=args.max_batch_size
            ),
        )
    else:
        llm = LLM(
            model=args.engine_path,
            tokenizer=get_tokenizer(args.hfmodel_path),
            dtype="float16",
        )
    llm.generate("hello", sampling_params=SamplingParams(max_new_tokens=1))  # warm up.

    tokenizer = get_tokenizer(args.hfmodel_path)
    prompts = get_prompts(args.num_prompts, args.input_len, tokenizer)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        min_length=args.output_len,
        max_new_tokens=args.output_len,
    )

    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    t2 = time.time()

    assert len(outputs) == args.num_prompts
    for request_output in outputs:
        assert len(request_output.prompt_token_ids) == args.input_len
        assert len(request_output.outputs) == 1
        assert len(request_output.outputs[0].token_ids) == args.output_len

    elapsed_time = t2 - t1

    num_prompts = args.num_prompts
    num_tokens = args.num_prompts * (args.input_len + args.output_len)

    print(
        f"Throughput: {num_prompts / elapsed_time:.2f} requests/s, "
        f"{num_tokens / elapsed_time:.2f} tokens/s"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfmodel-path", type=str, required=True, help="load hf model and tokenizer.")
    parser.add_argument("--max-batch-size", type=int)

    parser.add_argument("--engine-path", type=str)

    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)

    args = parser.parse_args()

    main(args)
