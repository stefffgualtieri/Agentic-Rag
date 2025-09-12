from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from datasets import load_dataset
import re


"""
from huggingface_hub import login
login()
"""

def normalize_string(s):
    s = re.sub(r"[^\w\s]", "", s)
    return s.lower().strip()

embedding_model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
model_name = "google/gemma-3-270m-it"
dataset_name = "LLukas22/nq-simplified"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Caricamento del dataset
dataset = load_dataset(dataset_name, split="train")

# Creazione dei modelli di embedding e text generation
embedding_model = SentenceTransformer(embedding_model_name).to(device)
model = AutoModelForCausalLM.from_pretrained(
    model_name
).to(device)

# Creazione del tokenizer per il modello scelto
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Definizione del prompt per RAG
instruction = """You are a ChatBot that answers based on the given context and your prior knowledge. 
    Do not make things up. Do not include unnecessary information in your answer. 
    If the answer should be a single or few words then you should answer like that."""

# Per ogni riga del dataset impostiamo il contesto e lasciamo che il modello generi una risposta
# Successivamente la risposta viene formattata per eliminare i marker tipo <start of turn> ecc
# E viene normalizzata assieme alla risposta vera

model_answers = []
true_answers = []

for row in dataset.select(range(10)):
    context = row["context"]
    question = row["question"]
    answer = row["answers"]["text"][0]

    query = f"Context: {context} Question: {question}"

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{query}"},
    ]

    tokenized_chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    outputs = model.generate(input_ids=tokenized_chat, max_new_tokens=128)

    model_answer = tokenizer.decode(outputs[0])
    model_answer = model_answer.split("<start_of_turn>model", 1)[1]
    model_answer = model_answer.split("<end_of_turn>", 1)[0]


    model_answers.append(normalize_string(model_answer))
    true_answers.append(normalize_string(answer))

# Per ciascuna risposta del modello e risposta vera si effettua l'embedding semantico
# Poi calcoliamo la cosine similarity fra le due risposte per avere una metrica di valutazione
# del testo generato in relazione alla ground truth
semantic_sim = []
for m, t in zip(model_answers, true_answers):
    print(f"Model: {m} - True: {t}")
    m_emb = embedding_model.encode(m)
    t_emb = embedding_model.encode(t)
    semantic_sim.append(cos_sim(m_emb, t_emb).item())

print(f"Semantic similarities: {semantic_sim}")
print(f"Mean: {sum(semantic_sim)/len(semantic_sim)}")
