import torch
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
# bert resing comprehantion ,predict the begining of answer and the end of answer
questions = ["小明去了哪里？"]
context = ["今天天气很好，小明准备去公园散步或者和朋友去郊游。"]
model_name = "google-bert/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qa_inputs = tokenizer(questions,context,padding=True,truncation=True,return_tensors="pt")
qa_inputs = {name:tensor.to(device) for name,tensor in qa_inputs.items()}
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

qa_model.eval()

with torch.no_grad():
    outputs = qa_model(**qa_inputs)
print("start_logits:",outputs.start_logits.shape)
print("end_logits:",outputs.end_logits.shape)
