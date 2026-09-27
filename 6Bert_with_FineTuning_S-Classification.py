import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# bert sequence classification
model_name = "google-bert/bert-base-chinese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = [
    "这个电影非常精彩",
    "这个产品非常好用",
    "这个电影十分无聊",
    "这个产品非常糟糕",
]
# 0:负面 1：正面
labels = torch.tensor([1,1,0,0],dtype=torch.long)
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
labels = labels.to(device)

model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2)
model.to(device)
print("模型类型：",type(model))
print("分类层：",model.classifier)
print("BERT参数是否可训练：",all(param.requires_grad for param in model.bert.parameters()))

optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)

# 模型训练
model.train()
optimizer.zero_grad()

outputs = model(
    **inputs,
    labels = labels
)
loss = outputs.loss
logits = outputs.logits

print("input_ids形状：",inputs["input_ids"].shape)
print("labels形状：",labels.shape)
print("logits形状",logits.shape)
print("loss:",loss.item())

# 反向传播
loss.backward()

# 检查梯度
bert_parameter = next(    
model.base_model.parameters()
)
print(    
"BERT主体梯度范数：",    
bert_parameter.grad.norm().item()
)
print(    
"分类层梯度范数：",    
model.classifier.weight.grad.norm().item()
)

optimizer.step()
print("完成一次端到端参数更新")

# 加载bert模型预训练权重
# 构建样本 适用tokenizer对样本进行预处理
# automodelforsequenceclassification为bert模型添加分类层
# 适用样本对模型进行微调训练 反向传播更新模型主干和分类层参数