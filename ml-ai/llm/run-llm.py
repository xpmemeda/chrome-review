import sys
import torch
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def main(model_dir, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(
        "cuda:0", torch.float16
    )

    generation_config = GenerationConfig(num_beams=1, max_new_tokens=128)

    pdb.set_trace()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
    prompt = ["hello world", "I don't want work"]
    main(model_dir, prompt)
