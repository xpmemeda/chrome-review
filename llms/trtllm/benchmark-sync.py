import argparse
import time
import wonderwords

import tensorrt_llm as trtllm

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


def get_tokenizer(model: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


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


def build_config(arguments):
    return trtllm.BuildConfig(
        max_batch_size=arguments.batch_size,
        opt_batch_size=arguments.batch_size,
        max_seq_len=2048,
        plugin_config=trtllm.builder.PluginConfig.from_dict(
            {"dtype": "float16", "gemm_plugin": "float16"}
        ),
    )


def main(arguments):
    llm = trtllm.LLM(
        model=arguments.model,
        tokenizer=arguments.model,
        trust_remote_code=True,
        dtype="float16",
        build_config=build_config(arguments),
    )

    llm.generate("hello")  # warm up.

    tokenizer = get_tokenizer(arguments.model)
    prompts = get_prompts(
        arguments.num_prompts, arguments.prefix_hit_len, arguments.input_len, tokenizer
    )

    sampling_params = trtllm.SamplingParams(
        temperature=0.5,
        min_tokens=arguments.output_len,
        max_tokens=arguments.output_len,
    )

    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    t2 = time.time()

    assert len(outputs) == arguments.num_prompts
    for request_output in outputs:
        assert len(request_output.prompt_token_ids) == arguments.input_len
        assert len(request_output.outputs) == 1
        assert len(request_output.outputs[0].token_ids) == arguments.output_len

    elapsed_time = t2 - t1

    num_prompts = arguments.num_prompts
    num_tokens = arguments.num_prompts * (arguments.input_len + arguments.output_len)

    print(
        f"Throughput: {num_prompts / elapsed_time:.2f} requests/s, "
        f"{num_tokens / elapsed_time:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="load hf model and tokenizer."
    )
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--prefix-hit-len", type=int, default=0)

    arguments = parser.parse_args()

    main(arguments)
