import argparse
import pdb
import time
import torch
import wonderwords
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List

benchmark_device = "cuda:0"
benchmark_dtype = torch.half


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


def run_hf(model, tokenizer, batch_prompts: List[str], generation_config):
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(benchmark_device)
    attention_mask = inputs["attention_mask"].to(benchmark_device)
    outputs_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )
    outputs = tokenizer.batch_decode(outputs_ids)
    return outputs


def main(arguments):
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model, trust_remote_code=True
    ).to(benchmark_device, benchmark_dtype)

    prompts = get_prompts(arguments.num_prompts, arguments.input_len, tokenizer)
    batched_prompts = []
    for i in range(0, len(prompts) // arguments.batch_size):
        batched_prompts.append(
            prompts[i * arguments.batch_size : (i + 1) * arguments.batch_size]
        )

    generation_config = GenerationConfig(
        num_beams=1,
        min_new_tokens=arguments.output_len,
        max_new_tokens=arguments.output_len,
    )

    t1 = time.time()
    pbar = tqdm.tqdm(total=arguments.num_prompts)
    for prompts in batched_prompts:
        run_hf(model, tokenizer, prompts, generation_config)
        pbar.update(arguments.batch_size)
        pbar.refresh()
    pbar.close()
    t2 = time.time()

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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    arguments = parser.parse_args()
    assert arguments.num_prompts % arguments.batch_size == 0
    main(arguments)
