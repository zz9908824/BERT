# base
from transformers import AutoTokenizer

MODEL_NAME = "google-bert/bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sentence_a = "今天是晴天"
sentence_b = "小明准备去散步"

inputs = tokenizer(
    sentence_a,
    sentence_b,
    padding="max_length",
    truncation=True,
    max_length=20,
    return_tensors="pt"
)

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("tokens:")
print(tokens)

print("\ninput_ids:")
print(inputs["input_ids"])
print("shape:", inputs["input_ids"].shape)

print("\ntoken_type_ids:")
print(inputs["token_type_ids"])

print("\nattention_mask:")
print(inputs["attention_mask"])