import torch
from transformers import AutoModelForMaskedLM,AutoTokenizer

model_name = "google-bert/bert-base-chinese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

mlm_input = tokenizer("今天是[mask]天", return_tensors="pt")
mlm_input = {name:tensor.to(device) for name,tensor in mlm_input.items()}
mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

# 全量微调 print true
# for parameters in mlm_model.base_model.parameters():
#     print(parameters.requires_grad)
optimizer = torch.optim.AdamW(mlm_model.parameters(),lr = 2e-5)
# 冻结bert 只训练分类层
# for parameters in mlm_model.base_model.parameters():
#     parameters.requires_grad=False
# optimizer = torch.optim.AdamW(mlm_model.cls.parameters(),lr = 2e-5)

mlm_model.eval()
with torch.no_grad():
    outputs = mlm_model(**mlm_input)
print("mlm logits:", outputs.logits.shape)
