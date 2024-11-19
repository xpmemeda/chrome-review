import argparse
import pdb
import time
import torch
import wonderwords
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List


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


class transformersBackend:
    def __init__(
        self, model: str, tokenizer: str, bsz: int, num_prompt_tokens, num_output_tokens
    ):
        self.bsz = bsz
        self.num_prompt_tokens = num_prompt_tokens
        self.num_output_tokens = num_output_tokens

        # import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        self.model.to("cuda:0", torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=True
        )
        self.generation_config = GenerationConfig(
            do_sample=False,
            min_new_tokens=num_output_tokens,
            max_new_tokens=num_output_tokens,
        )

    @classmethod
    def version(cls):
        import transformers

        return transformers.__version__

    def generate(self, prompts: List[str]):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to("cuda:0")
        attention_mask = inputs["attention_mask"].to("cuda:0")
        outputs_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
        )

        assert outputs_ids.size(0) == self.bsz
        assert outputs_ids.size(1) == self.num_prompt_tokens + self.num_output_tokens

        outputs = self.tokenizer.batch_decode(outputs_ids)
        return outputs


class vllmBackend:
    def __init__(
        self, model: str, tokenizer: str, bsz: int, num_prompt_tokens, num_output_tokens
    ):
        self.bsz = bsz
        self.num_prompt_tokens = num_prompt_tokens
        self.num_output_tokens = num_output_tokens

        import vllm

        num_tokens_per_batch = bsz * (num_prompt_tokens + num_output_tokens)

        llm_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": bsz,
            "max_num_batched_tokens": max(8192, num_tokens_per_batch),
            "max_model_len": 8192,
            "enforce_eager": True,
        }
        sampling_kwargs = {
            "temperature": 0.5,
            "n": 1,
            "ignore_eos": True,
            "min_tokens": num_output_tokens,
            "max_tokens": num_output_tokens,
        }

        self.llm_engine = vllm.LLM(**llm_kwargs)
        self.sampling_params = vllm.SamplingParams(**sampling_kwargs)

    @classmethod
    def version(cls):
        import vllm

        return vllm.__version__

    def generate(self, prompts: List[str]):
        outputs = self.llm_engine.generate(
            prompts, self.sampling_params, use_tqdm=False
        )
        for output in outputs:
            num_prompt_tokens = len(output.prompt_token_ids)
            num_output_tokens = len(output.outputs[0].token_ids)
            assert num_prompt_tokens == self.num_prompt_tokens
            assert num_output_tokens == self.num_output_tokens
        return outputs


def main(arguments):
    backends = {"transformers": transformersBackend, "vllm": vllmBackend}
    backend_cls = backends[arguments.backend]
    backend = backend_cls(
        arguments.model,
        arguments.model,
        arguments.batch_size,
        arguments.num_prompt_tokens,
        arguments.num_output_tokens,
    )
    prompts = get_prompts(
        arguments.num_prompts,
        arguments.num_prompt_tokens,
        AutoTokenizer.from_pretrained(arguments.model, trust_remote_code=True),
    )
    batched_prompts = []
    for i in range(0, len(prompts) // arguments.batch_size):
        batched_prompts.append(
            prompts[i * arguments.batch_size : (i + 1) * arguments.batch_size]
        )
    t1 = time.time()
    pbar = tqdm.tqdm(total=arguments.num_prompts)
    for prompts in batched_prompts:
        backend.generate(prompts)
        pbar.update(arguments.batch_size)
        pbar.refresh()
    pbar.close()
    t2 = time.time()

    elapsed_time = t2 - t1
    num_prompts = arguments.num_prompts
    num_tokens = arguments.num_prompts * (
        arguments.num_prompt_tokens + arguments.num_output_tokens
    )

    print(
        f"Throughput: {num_prompts / elapsed_time:.2f} requests/s, "
        f"{num_tokens / elapsed_time:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", type=str, required=True, choices=["transformers", "vllm"]
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--num-prompt-tokens", type=int, required=True)
    parser.add_argument("--num-output-tokens", type=int, required=True)
    arguments = parser.parse_args()
    assert arguments.num_prompts % arguments.batch_size == 0
    main(arguments)
