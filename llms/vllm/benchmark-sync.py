import argparse
import time
import wonderwords
import vllm

from transformers import AutoTokenizer


def get_prompts(num_prompts, prefix_hit_len, input_len, tokenizer):
    random_words = wonderwords.RandomWord().random_words(num_prompts)
    prompts = []
    for random_word in random_words:
        random_word = "hi" * prefix_hit_len + " " + random_word + " hihihi"
        num_word_tokens = len(tokenizer.encode(random_word))
        prompt = random_word + "hi" * (input_len - num_word_tokens)
        x = len(tokenizer.encode(prompt))
        y = input_len
        assert x == y, "%d != %d" % (x, y)
        prompts.append(prompt)
    return prompts


def main(args: argparse.Namespace):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = get_prompts(
        args.num_prompts, args.prefix_hit_len, args.input_len, tokenizer
    )

    num_tokens_per_batch = args.batch_size * (args.input_len + args.output_len)

    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.model,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": max(8192, num_tokens_per_batch),
        "max_model_len": 8192,
        "enforce_eager": True,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    sampling_kwargs = {
        "temperature": 0.5,
        "n": 1,
        "ignore_eos": True,
        "min_tokens": args.output_len,
        "max_tokens": args.output_len,
    }

    llm_engine = vllm.LLM(**llm_kwargs)
    sampling_params = vllm.SamplingParams(**sampling_kwargs)

    # warmup.
    llm_engine.generate("hello world")

    t1 = time.time()
    request_outputs = llm_engine.generate(prompts, sampling_params, use_tqdm=True)
    t2 = time.time()

    assert len(request_outputs) == args.num_prompts
    for request_output in request_outputs:
        x = len(request_output.prompt_token_ids)
        y = args.input_len
        assert x == y, "%d != %d" % (x, y)
        x = len(request_output.outputs[0].token_ids)
        y = args.output_len
        assert x == y, "%d != %d" % (x, y)

    elapsed_time = t2 - t1

    num_prompts = args.num_prompts
    num_tokens = args.num_prompts * (args.input_len + args.output_len)

    print(
        f"Throughput: {num_prompts / elapsed_time:.2f} requests/s, "
        f"{num_tokens / elapsed_time:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--prefix-hit-len", type=int, default=0)
    args = parser.parse_args()

    main(args)
