import evaluate
from typing import List
import re
import spacy
import json
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

model = BartForConditionalGeneration.from_pretrained("./trained_model")
tokenizer = BartTokenizer.from_pretrained("./trained_model")

nlp = spacy.load('en_core_web_md')

def bert_score(prediction, reference):
    return evaluate.load("bertscore").compute(predictions=[prediction], references=[reference], lang="en")['f1'][0]

def bleu(prediction, reference, geometric_mean=True):
    res = evaluate.load("bleu").compute(predictions=[prediction], references=[reference])
    if geometric_mean:
        return res['bleu']
    return res['precisions']

def rouge(prediction, reference):
    return evaluate.load("rouge").compute(predictions=[prediction], references=[reference])['rougeL']


def meteor(prediction, reference):
    return evaluate.load("meteor").compute(predictions=[prediction], references=[reference])['meteor']

def preprocess_spacy(
        doc,
        min_token_len=2,
        irrelevant_pos=["ADV", "CCONJ", "PUNCT", "PART", "DET", "ADP", "SPACE"],
):
    clean_text = []

    for token in doc:
        if token.like_email:  # Check if the token is an not like email
            clean_text.append("EMAIL")
        elif token.like_url:  # Check if the token is an not like email
            clean_text.append("URL")
        elif token.like_num:  # Check if the token is an not like email
            clean_text.append("NUM")
        elif (
                token.is_stop == False  # Check if it's not a stopword
                and len(token) > min_token_len  # Check if the word meets minimum threshold
                and token.pos_ not in irrelevant_pos
        ):  # Check if the POS is in the acceptable POS tags
            clean_text.append(token.lemma_.lower())
    return " ".join(clean_text)


def preprocess(text):
    # Replace a sequence of whitespaces by a single whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove other strange characters
    text = re.sub(r"""[\n\r]+""", "", text)

    # Remove other strange characters
    text = re.sub(r"""[\*\~]+""", "", text)

    # Replace slashes with spaces
    text = re.sub(r"""[\/]+""", " ", text)

    return text


def clean_text(text):
    text = preprocess(text)
    text = preprocess_spacy(nlp(text))
    return text


def nlp_similarity(text1, text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)    


def generate_follow_up(question, answer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = question + "<SEP>" + answer + "<QUS>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model.generate(**inputs, max_length=1024, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

res = []

# with open("test.json", 'r') as file:
#     data = json.load(file)
#     for d in data:
#         q_id = d["id"]
#         print(q_id)
#         question = d["question"]
#         answer = d["answer"]
#         o_follow_up = d["follow-up"]
#         follow_up = generate_follow_up(question, answer)
#         bleu_score = bleu(o_follow_up, follow_up, False)
#         bleu_mean = bleu(o_follow_up, follow_up, True)
        
#         res.append({"id": q_id, "follow_up": follow_up, "bert": bert_score(o_follow_up, follow_up), "nlp_similarity": nlp_similarity(o_follow_up, follow_up), "bleu_geo_mean": bleu_mean, "bleu1": bleu_score[0], "bleu2": bleu_score[1], "bleu3": bleu_score[2], "bleu4": bleu_score[3], "rouge": rouge(o_follow_up, follow_up), "meteor": meteor(o_follow_up, follow_up)})
#         with open("res2.json", 'w') as file:
#             json.dump(res, file, indent=4)


with open("test.json", 'r') as file:
    data = json.load(file)
    # for d in data:
    #     q_id = d["id"]
    #     print(q_id)
    #     question = d["question"]
    #     answer = d["answer"]
    #     o_follow_up = d["follow-up"]
    #     follow_up = generate_follow_up(question, answer)
        
    #     res.append({"id": q_id, "follow_up": follow_up})
    #     with open("out/test_q.json", 'w') as file:
    #         json.dump(res, file, indent=4)
    
    # res = []
                        
    # with open("out/test_q.json", 'r') as file:
    #     qs = json.load(file)
    #     for q in qs:
    #         q_id = q["id"]
    #         print(q_id)

    #         o_follow_up = data[q_id-3000]["follow-up"]
            
    #         res.append({"id": q_id, "follow_up": q['follow_up'], "bert": bert_score(o_follow_up, q['follow_up']), "nlp_similarity": nlp_similarity(o_follow_up, q['follow_up'])})
    #         with open("out/p1.json", 'w') as file:
    #             json.dump(res, file, indent=4)
    res = []
    
    with open("out/p1.json", 'r') as file:
        qs = json.load(file)
        for q in qs:
            q_id = q["id"]
            print(q_id)
            o_follow_up = data[q_id-3000]["follow-up"]
            bleu_score = bleu(o_follow_up, q['follow_up'], False)
            bleu_mean = bleu(o_follow_up, q['follow_up'], True)
            
            
            res.append({"id": q_id, "follow_up": q['follow_up'], "bert": q['bert'], "nlp_similarity": q['nlp_similarity'], "bleu_geo_mean": bleu_mean, "bleu1": bleu_score[0], "bleu2": bleu_score[1], "bleu3": bleu_score[2], "bleu4": bleu_score[3]})
            with open("out/p2.json", 'w') as file:
                json.dump(res, file, indent=4)
                
    res = []
                
    with open("out/p2.json", 'r') as file:
        qs = json.load(file)
        for q in qs:
            q_id = q["id"]
            print(q_id)
            o_follow_up = data[q_id-3000]["follow-up"]
            
            res.append({"id": q_id, "follow_up": q['follow_up'], "bert": q['bert'], "nlp_similarity": q['nlp_similarity'], "bleu_geo_mean": q['bleu_geo_mean'], "bleu1": q['bleu1'], "bleu2": q['bleu2'], "bleu3": q['bleu3'], "bleu4": q['bleu4'], "rouge": rouge(o_follow_up, q['follow_up']), "meteor": meteor(o_follow_up, q['follow_up'])})
            with open("out/res.json", 'w') as file:
                json.dump(res, file, indent=4)