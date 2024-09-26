import argparse
import time
import wonderwords

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from tensorrt_llm import LLM, SamplingParams, BuildConfig


def get_tokenizer(model: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


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


def main(args):
    if args.engine_dir is None:
        build_config = BuildConfig(
            max_batch_size=args.batch_size,
            opt_batch_size=args.batch_size,
            max_seq_len=2048,
        )
        build_config = BuildConfig.from_json_file("default-buildconfig-0.14.0.json")
        build_config.update_from_dict({"max_batch_size": args.batch_size})
        llm = LLM(
            model=args.model_dir,
            tokenizer=get_tokenizer(args.model_dir),
            trust_remote_code=True,
            dtype="float16",
            build_config=build_config,
        )
        llm.save("b%d.engine" % args.batch_size)
    else:
        llm = LLM(
            model=args.engine_dir,
            tokenizer=get_tokenizer(args.model_dir),
            dtype="float16",
        )
    llm.generate("hello", sampling_params=SamplingParams(max_new_tokens=1))  # warm up.

    tokenizer = get_tokenizer(args.model_dir)
    prompts = get_prompts(args.num_prompts, args.input_len, tokenizer)

    # 0.14.0: hlapi/utils.py
    # 0.15.0: llmapi/utils.py:SamplingParams
    sampling_params = SamplingParams(
        temperature=0.5,
        min_tokens=args.output_len,
        max_tokens=args.output_len,
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
    parser.add_argument(
        "--model-dir", type=str, required=True, help="load hf model and tokenizer."
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--engine-dir", type=str)

    args = parser.parse_args()

    main(args)
