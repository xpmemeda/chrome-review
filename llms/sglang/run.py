import sys
import sglang as sgl


def main(model_path):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = {"temperature": 0, "max_new_tokens": 16}

    llm = sgl.Engine(model_path=model_path, disable_cuda_graph=True)

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated text: ", end="", flush=True)

        for chunk in llm.generate(prompt, sampling_params, stream=True):
            # print(chunk["text"], end="", flush=True)
            print(chunk)
        print()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
if __name__ == "__main__":
    main(sys.argv[1])
