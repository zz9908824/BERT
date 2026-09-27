import torch 
import torch.nn.functional as F
from transformers import AutoTokenizer,BertForPreTraining

torch.manual_seed(42)

Model_name = "google-bert/bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(Model_name)
model = BertForPreTraining.from_pretrained(Model_name)

# 构造NSP正样本和NSP负样本
sentence_a = [
    "小明打开了课本。",
    "今天的天气非常晴朗"
]
sentence_b = [
    "小明准备开始学习数学",
    "小明即将参加毕业考试"
]
# IsNext Sentence Prediction (NSP) 正样本标签为0，负样本标签为1
Next_sentence_labels = torch.tensor([0, 1], dtype=torch.long)
encodings = tokenizer(
    sentence_a,
    sentence_b,
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=64,
    return_special_tokens_mask=True
)
print("encodings type:\n", type(encodings))
print("encodings:", encodings)
original_input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
token_type_ids = encodings["token_type_ids"]
special_tokens_mask = encodings["special_tokens_mask"]
print("original_input_ids:", original_input_ids)
print("attention_mask:", attention_mask)
print("token_type_ids:", token_type_ids)
print("special_tokens_mask:", special_tokens_mask)
# 15%选词 15%选词中80%填[mask]10%填随机词10%填原词
def create_mlm_inputs(input_ids,attention_mask,special_tokens_mask,tokenizer,mlm_probability=0.15):
    # input_id 深拷贝
    masked_input_ids = input_ids.clone()
    labels = input_ids.clone()
    # 只有真实的token（非【pad】）和非特殊token([CLS]\[SEP]])才可以被选中作为[mask]
    eligible_mask_positions = ((special_tokens_mask==0) & (attention_mask==1))
    # 概率矩阵 每个位置被选中作为[mask]的概率为0.15
    posibility_matrix = torch.full(labels.shape,mlm_probability,dtype=torch.float)
    # 概率矩阵不满足eligibel_mask_position条件的位置被选中作为[mask]的概率为0
    posibility_matrix.masked_fill_(~eligible_mask_positions, value=0.0)
    # 伯努利概率分布采样，得到每个位置是否被选中作为[mask]的布尔矩阵
    selected_mask_positions = torch.bernoulli(posibility_matrix).bool()
    # 每个短句至少有一个token被选中作为[mask]，如果没有则随机选择一个token作为[mask]
    for batch_index in range(input_ids.shape[0]):
        if not selected_mask_positions[batch_index].any():
            candidates = torch.nonzero(eligible_mask_positions[batch_index],
                                       as_tuple=False).squeeze(-1)
            selected_mask_positions[batch_index,candidates[0]]=True
    # 没选中的位置不计算MLM损失
    labels[~selected_mask_positions]=-100
    # 选中的位置80%替换为[mask]
    replace_with_mask = torch.full(labels.shape,0.8,dtype=torch.float)
    replace_with_mask = (torch.bernoulli(replace_with_mask).bool() & selected_mask_positions)
    masked_input_ids[replace_with_mask]=(tokenizer.mask_token_id)
    # 选中的位置10%替换为随机token 剩余20%中的一半替换为随机token
    replace_with_random = torch.full(labels.shape,0.5)
    replace_with_random = (torch.bernoulli(replace_with_random).bool()&selected_mask_positions&~replace_with_mask)
    random_token_ids = torch.randint(low=0,high=tokenizer.vocab_size,size=labels.shape,dtype=torch.long)
    masked_input_ids[replace_with_random]=(random_token_ids[replace_with_random])
    # 选中位置10%token不变
    return masked_input_ids,labels

masked_input_ids,mlm_labels = create_mlm_inputs(original_input_ids,attention_mask,special_tokens_mask,tokenizer,mlm_probability=0.15)
# print("masked_input_ids:", masked_input_ids)
# print("mlm_labels:", mlm_labels)
# 查看构造结果
for batch_index in range(original_input_ids.shape[0]):
    original_tokens = tokenizer.convert_ids_to_tokens(original_input_ids[batch_index])
    masked_tokens = tokenizer.convert_ids_to_tokens(masked_input_ids[batch_index])
    print(f"batch_index:{batch_index}")
    print("original_tokens:", original_tokens)
    print("masked_tokens:", masked_tokens)
    print("MLM 预测位置：")
    for position,label_id in enumerate(mlm_labels[batch_index]):
        if label_id.item() != -100:
           input_token = tokenizer.convert_ids_to_tokens(masked_input_ids[batch_index][position].item())
           target_token = tokenizer.convert_ids_to_tokens(label_id.item())
           print(f"位置:{position}, 输入token:{input_token}, 目标token:{target_token}")
    print("NSP 预测标签:", Next_sentence_labels[batch_index].item())
            
