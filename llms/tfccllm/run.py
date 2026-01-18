import logging
import sys
import vllm
import tfccllm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


model_dir = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
prompt = """你好"""
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
text = prompt


def main():
    model_path = sys.argv[1]

    model = tfccllm.create_model(
        model=model_path,
        engine="vllm-engine",
        trtllm_model_type="qwen",
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        skip_request_output_validity_check=True,
    )

    generation_config = tfccllm.TfccLLMGeneralGenerationConfig()
    generation_config.max_tokens = 512
    generation_config.top_p = 0.8
    generation_config.top_k = 20
    generation_config.temperature = 0.2

    prompts = [text]
    prompt_token_ids = []
    for prompt in prompts:
        model_inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].tolist()
        if len(input_ids) == 1:
            input_ids = input_ids[0]
        prompt_token_ids.append(input_ids)

    for i in range(10):
        outputs = model.generate(
            prompts=prompts,
            generation_config=generation_config,
            prompt_token_ids=prompt_token_ids,
        )
        for prompt, output in zip(prompts, outputs):
            logging.critical(f"inference ok inference result {output.result_text}")


if __name__ == "__main__":
    main()
