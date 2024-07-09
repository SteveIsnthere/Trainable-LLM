import json
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

model = BartForConditionalGeneration.from_pretrained("./trained_model")
tokenizer = BartTokenizer.from_pretrained("./trained_model")

def generate_follow_up(question, answer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = question + "<SEP>" + answer + "<QUS>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model.generate(**inputs, max_length=1024, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

responses = []

with open("test.json", 'r') as file:
    data = json.load(file)
    for d in data[:10]:
        q_id = d["id"]
        print(q_id)
        question = d["question"]
        answer = d["answer"]
        follow_up = generate_follow_up(question, answer)
        responses.append({"id": q_id, "follow_up": follow_up})
        
        
with open("res.json", 'w') as file:
    json.dump(responses, file, indent=4)