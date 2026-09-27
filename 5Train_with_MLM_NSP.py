import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForPreTraining

torch.manual_seed(42)
model_name = "google-bert/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForPreTraining.from_pretrained(model_name)

sentence_a = [
    "小明打开了课本。",
    "今天的天气非常晴朗",
]
sentence_b = [
    "小明准备开始学习数学",
    "小明即将参加毕业考试",
]
next_sentence_labels = torch.tensor([0, 1], dtype=torch.long)
encodings = tokenizer(
    sentence_a,
    sentence_b,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt",
    return_special_tokens_mask=True,
)
original_input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
token_type_ids = encodings["token_type_ids"]
special_tokens_mask = encodings["special_tokens_mask"]

def create_mlm_inputs(input_ids, attention_mask, special_tokens_mask, tokenizer, mlm_probability=0.15):
    masked_input_ids = input_ids.clone() # token_id and masked_token_id
    labels = input_ids.clone() # include -100 and mlm_masked_token_id
    eligible_mask_positions = ((special_tokens_mask == 0) & (attention_mask == 1)) # position that can be masked,excluding [CLS], [SEP], and [PAD]
    posibility_matrix = torch.full(labels.shape, mlm_probability, dtype=torch.float) # position that can be masked, posibility set to 0.15
    posibility_matrix.masked_fill_(~eligible_mask_positions, value=0.0) # positiong that can not be masked(such as [CLS], [SEP], and [PAD]), posibility set to 0.0
    selected_mask_positions = torch.bernoulli(posibility_matrix).bool() # according to posibility_matrix, select the position that can be masked, and return a boolean matrix
    for batch_index in range(input_ids.shape[0]):
        if not selected_mask_positions[batch_index].any():
            candidates = torch.nonzero(eligible_mask_positions[batch_index], as_tuple=False).squeeze(-1)
            selected_mask_positions[batch_index, candidates[0]] = True
    labels[~selected_mask_positions] = -100 # position that is not selected to be masked, set to -100, which will be ignored in loss calculation
    # 80% of the time, replace with [MASK]
    replace_with_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & selected_mask_positions # 80% of the selected positions will be replaced with [MASK]
    masked_input_ids[replace_with_mask] = tokenizer.mask_token_id
    # 10% of the time, replace with random token
    replace_with_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & selected_mask_positions & ~replace_with_mask # 10% of the selected positions will be replaced with random token
    random_tokens_id = torch.randint(low=0, high=tokenizer.vocab_size, size=labels.shape, dtype=torch.long)
    masked_input_ids[replace_with_random] = random_tokens_id[replace_with_random]
    # 10% of the time, keep the original token (do nothing)
    return masked_input_ids, labels
masked_input_token_ids, mlm_labels = create_mlm_inputs(original_input_ids, attention_mask, special_tokens_mask, tokenizer)
# result of create_mlm_inputs
for batch_index in range(original_input_ids.shape[0]):
    print(f"batch {batch_index}:")
    print("original_input_ids:", original_input_ids[batch_index])
    print("masked_input_token_ids:", masked_input_token_ids[batch_index])
    print("mlm_labels:", mlm_labels[batch_index])
    for position,label_id in enumerate(mlm_labels[batch_index]):
        if label_id != -100:
            original_token = tokenizer.convert_ids_to_tokens(original_input_ids[batch_index][position].item())
            masked_token = tokenizer.convert_ids_to_tokens(masked_input_token_ids[batch_index][position].item())
            label_token = tokenizer.convert_ids_to_tokens(label_id.item())
            print(f"position {position}: original_token: {original_token}, masked_token: {masked_token}, label_token: {label_token}")
    print("is next_sentence_labels:", next_sentence_labels[batch_index].item())

model.train()
outputs = model(
    input_ids=masked_input_token_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    labels=mlm_labels,
    next_sentence_label=next_sentence_labels
)
print("=======输出形状=======")
print("MLM logits shape:", outputs.prediction_logits.shape)
print("NSP logits shape:", outputs.seq_relationship_logits.shape)
print("总损失：", outputs.loss.item())
vocab_size = outputs.prediction_logits.shape[-1]
mlm_loss = F.cross_entropy(
    outputs.prediction_logits.view(-1, vocab_size),
    mlm_labels.view(-1),
    ignore_index=-100
)
nsp_loss = F.cross_entropy(
    outputs.seq_relationship_logits,
    next_sentence_labels
)
print("MLM loss:", mlm_loss.item())
print("NSP loss:", nsp_loss.item())
print("总损失:", mlm_loss.item() + nsp_loss.item())
# 两个任务更新参数
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
optimizer.zero_grad()
outputs.loss.backward()

bert_parameter = next(model.bert.parameters())
mim_parameter = next(model.cls.predictions.parameters())
nsp_parameter = next(model.cls.seq_relationship.parameters())
print("梯度验证")
print("bert主体梯度：", bert_parameter.grad.norm().item())
print("MLM头梯度：", mim_parameter.grad.norm().item())
print("NSP头梯度：", nsp_parameter.grad.norm().item())
optimizer.step()
print("参数更新完成")