import json
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from heapq import heappop, heappush, heapify
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

model_name = "google/gemma-3-1b-it"

embedding_model = SentenceTransformer(
    "sentence-transformers/distiluse-base-multilingual-cased-v2"
)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


def normalize_string(s):
    s = re.sub(r"[^\w\s]", "", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Restituisce versione normalizzata del testo della pagina wikipedia
def get_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text


# Generazione di chunks di lunghezza fissata a l
def make_chunks(text, l):
    chunks = [text[i : i + l] for i in range(0, len(text), l)]
    return chunks


# Generazione di chunks di lunghezza variabile, che si interrompono
# alla fine della frase (.)
def make_sentence_chunks(unnormalized_text):
    chunks = unnormalized_text.split(".")
    return [normalize_string(chunk) for chunk in chunks]


# Trova i k chunks piu semanticamente simili a query e
# li ritorna in un vettore assieme al valore di similarità
def find_top_k(query, chunks, k):
    query_encoding = embedding_model.encode(query, device=device)
    h = []
    heapify(h)
    for chunk in chunks:
        encoding = embedding_model.encode(chunk, device=device)
        sim = cos_sim(query_encoding, encoding)
        element = (sim.item(), chunk)
        if len(h) < k:
            heappush(h, element)
        else:
            if sim > h[0][0]:
                heappop(h)
                heappush(h, element)
    return sorted(h, key=lambda x: x[0], reverse=True)

# Creazione del tokenizer per il modello scelto
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Definizione del prompt per RAG
instruction = """You are a ChatBot that answers based on the given context and your prior knowledge.
    The context will be given as a series of sentences preceded by the word \'Fact\' and a progressive number.
    Do not make things up. Do not include unnecessary information in your answer. 
    If the answer should be a single or few words then you should answer like that."""

#Per il momento sto considerando solo le istanze in cui esiste la short answer
#Come chunking strategy sentence_chunking

print(f'Loading dataset...')
with open("v1.0-simplified_nq-dev-all.jsonl", "r") as dataset:
    rows = [json.loads(line) for _, line in zip(range(100), dataset)]
    print(f'Dataset loaded!')
    result = []
    for row in rows:
        question = " ".join(row["question_tokens"])
        tokens = [t["token"] for t in row["document_tokens"]]
        document = get_text_from_html(" ".join(tokens))

        annotation = row["annotations"][0]

        if annotation["short_answers"]:
            short_answer = annotation["short_answers"][0]
            start, end = short_answer["start_token"], short_answer["end_token"]
            short_answer = " ".join(tokens[start:end])

            chunks = make_sentence_chunks(document)
            top_chunks = find_top_k(question, chunks, 10)

            context = [f'Fact{i}: {c[1]}' for i, c in zip(range(len(top_chunks)), top_chunks)]
            context = " ".join(context)
            #print(context)

            query = f"Context: {context} Question: {question}"

            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{query}"},
            ]

            tokenized_chat = tokenizer.apply_chat_template(
                messages, tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            outputs = model.generate(input_ids=tokenized_chat, max_new_tokens=128)

            model_answer = tokenizer.decode(outputs[0])
            model_answer = model_answer.split("<start_of_turn>model", 1)[1]
            model_answer = model_answer.split("<end_of_turn>", 1)[0]

            result.append({
                'question':normalize_string(question),
                'true_answer':normalize_string(short_answer),
                'model_answer': normalize_string(model_answer)
                })


for elem in result:
    print(f"Question: {elem['question']}")
    print(f"Model answer: {elem['model_answer']}")
    print(f"True answer: {elem['true_answer']}")
    print()

sim = []
for elem in result:
    t_emb = embedding_model.encode(elem["true_answer"])
    m_emb = embedding_model.encode(elem["model_answer"])
    sim.append(cos_sim(t_emb, m_emb).item())

print(sum(sim)/len(sim))