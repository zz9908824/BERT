import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
# bert token classification
model_name = "google-bert/bert-base-chinese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token_model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=5
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = [
    "这个电影非常精彩",
    "这个产品非常好用",
    "这个电影十分无聊",
    "这个产品非常糟糕",
]
inputs = tokenizer(
    text,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt",
)
inputs = {
    name:tensor.to(device)
    for name,tensor in inputs.items()
}

token_model.eval()
with torch.no_grad():
    outputs = token_model(**inputs)

print("Token分类logits:",outputs.logits.shape)