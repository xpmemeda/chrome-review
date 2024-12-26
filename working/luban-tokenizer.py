from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.models.bert import BertTokenizer

text = "i'm fine."

tokenizer = BertTokenizer.from_pretrained("/home/wnr/llms/chinese-roberta-wwm-ext")
print(type(tokenizer))
x = tokenizer.tokenize(text)
print(x)
x = tokenizer(text, padding=True, padding_side="left")
print(x)


tokenizer = BertTokenizer.from_pretrained("/home/wnr/llms/bert-base-chinese")
print(type(tokenizer))
x = tokenizer.tokenize(text)
print(x)
x = tokenizer(text, padding=True, padding_side="left")
print(x)