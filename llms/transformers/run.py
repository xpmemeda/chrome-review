import sys
import torch
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def main(model_dir, prompts):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # NOTE: If the padding_side is set to right, the response will be incorrect.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(
        "cuda:0", torch.float16
    )

    r"""
    do_sample (`bool`, *optional*, defaults to `False`):
        Whether or not to use sampling ; use greedy decoding otherwise.
    num_beams (`int`, *optional*, defaults to 1):
        Number of beams for beam search. 1 means no beam search.
    max_new_tokens (`int`, *optional*):
        The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    """
    generation_config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=16
    )

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to("cuda:0")
    attention_mask = inputs["attention_mask"].to("cuda:0")
    outputs_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )

    outputs = tokenizer.batch_decode(outputs_ids)
    print(outputs)


if __name__ == "__main__":
    model_dir = sys.argv[1]
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    main(model_dir, prompts)
