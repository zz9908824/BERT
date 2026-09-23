import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
# setup 使用的预训练bert模型
MODEL_NAME = "google-bert/bert-base-chinese"
# 加载模型和模型对应的分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
# 测试句子
text = "今天是[MASK]天，小明准备去散步。"
# 将句子进行分词和编码
inputs = tokenizer(
    text,
    return_tensors="pt"
)
# 打印分词结果
print("tokens:")
print(
    tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )
)
# 找到[MASK]的位置
mask_positions = (
    inputs["input_ids"] == tokenizer.mask_token_id
).nonzero(as_tuple=False)
mask_index = mask_positions[0, 1].item()
print("\n[MASK]位置:", mask_index)
# 推理时不计算梯度
model.eval()
# outputs.logits: [batch_size, sequence_length, vocab_size]
logits = outputs.logits
print("logits形状:", logits.shape)
# 取[MASK]位置对应的整个词表分数
mask_logits = logits[0, mask_index, :]
# 找到概率最高的5个Token
top5_ids = torch.topk(
    mask_logits,
    k=5
).indices.tolist()
top5_tokens = tokenizer.convert_ids_to_tokens(top5_ids)
print("\nTop-5预测:")
for token_id, token in zip(top5_ids, top5_tokens):
    print(f"{token:<8} token_id={token_id}")
# MLM LABELS
labels = torch.full_like(
    inputs["input_ids"],
    fill_value=-100
)
target_id = tokenizer.convert_tokens_to_ids("晴")
labels[0, mask_index] = target_id
with torch.no_grad():
    outputs = model(
        **inputs,
        labels=labels
    )
print("MLM labels:")
print(labels)
print("MLM loss:")
print(outputs.loss.item())