import argparse
import time
import wonderwords
import sglang

from transformers import AutoTokenizer


def get_prompts(num_prompts, input_len, tokenizer):
    random_words = wonderwords.RandomWord().random_words(num_prompts)
    prompts = []
    for random_word in random_words:
        random_word += " hihihi"
        num_word_tokens = len(tokenizer.encode(random_word))
        prompt = random_word + "hi" * (input_len - num_word_tokens)
        x = len(tokenizer.encode(prompt))
        y = input_len
        assert x == y, "%d != %d" % (x, y)
        prompts.append(prompt)
    return prompts


def main(arguments: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, trust_remote_code=True)
    prompts = get_prompts(arguments.num_prompts, arguments.input_len, tokenizer)

    # server.py:Engine
    # server_args.py:ServerArgs
    llm = sglang.Engine(
        model_path=arguments.model,
        disable_cuda_graph=True,
        max_running_requests=arguments.batch_size,
        mem_fraction_static=0.9,
    )
    # https://sgl-project.github.io/references/sampling_params.html
    sampling_params = {
        "temperature": 0.5,
        "min_new_tokens": arguments.output_len,
        "max_new_tokens": arguments.output_len,
    }

    t1 = time.time()
    request_outputs = llm.generate(prompts, sampling_params)
    t2 = time.time()

    assert len(request_outputs) == arguments.num_prompts
    for request_output in request_outputs:
        x = request_output["meta_info"]["prompt_tokens"]
        y = arguments.input_len
        assert x == y, "%d != %d" % (x, y)
        x = request_output["meta_info"]["completion_tokens"]
        y = arguments.output_len
        assert x == y, "%d != %d" % (x, y)

    elapsed_time = t2 - t1

    num_prompts = arguments.num_prompts
    num_tokens = arguments.num_prompts * (arguments.input_len + arguments.output_len)

    print(
        f"Throughput: {num_prompts / elapsed_time:.2f} requests/s, "
        f"{num_tokens / elapsed_time:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()

    main(arguments)
