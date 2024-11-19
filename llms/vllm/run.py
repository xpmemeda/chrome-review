import sys
import vllm


def main():
    model_dir = sys.argv[1]
    llm = vllm.LLM(
        model=model_dir,
        tokenizer=model_dir,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=True,
    )

    # prompt = ["床前明月光，疑是地上霜。举头望明月，低头思故乡。请告诉我这首诗的作者，写作背景，它的大意是什么，要表达什么思想？"]
    prompt = ["hi" * 32]
    sampling_params = vllm.SamplingParams()
    output = llm.generate(prompt, sampling_params)
    print(output)
    output = llm.generate(prompt, sampling_params)
    print(output)


if __name__ == "__main__":
    main()
