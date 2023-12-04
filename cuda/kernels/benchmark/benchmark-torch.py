import torch
from dobench import do_bench


def bmm(b, m, n, k):
    x = torch.rand(b, m, k, device="cuda:0", dtype=torch.half)
    y = torch.rand(k, n, device="cuda:0", dtype=torch.half)

    def fn():
        return torch.matmul(x, y)

    return fn


def bert_base_uncased(b, s, dt=torch.float16):
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained("/home/wnr/llms/bert-base-uncased")
    model = BertModel.from_pretrained("/home/wnr/llms/bert-base-uncased")
    model.to("cuda:0", dt)
    text = ["hi" * s] * b
    encoded_input = tokenizer(text, return_tensors="pt").to("cuda:0")

    def fn():
        return model(**encoded_input)

    return fn


def qwen2_7b(b, s, o):
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

    model_dir = "/home/wnr/llms/Qwen2-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to("cuda:0", torch.float16)
    text = ["hi" * s] * b
    ins = tokenizer(text, return_tensors="pt").to("cuda:0")
    generation_config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=o)

    def fn():
        return model.generate(**ins, generation_config=generation_config)

    return fn


print("bmm-1x8x4096x4096", do_bench(bmm(1, 8, 4096, 4096)))
print("bmm-8x4096x4096x4096", do_bench(bmm(8, 4096, 4096, 4096)))
print("bmm-256x4096x4096x4096", do_bench(bmm(256, 4096, 4096, 4096)))
print("bert-base-uncased", do_bench(bert_base_uncased(8, 256)))
print("qwen2_7b", do_bench(qwen2_7b(8, 1024, 32)))
