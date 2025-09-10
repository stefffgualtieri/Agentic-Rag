from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import sys
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device {device}')

model_name = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device)

messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": ""}
]

tokenized_chat = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
    ).to(device)

attention_mask = (tokenized_chat != tokenizer.pad_token_id).long()

print(attention_mask)
print(tokenizer.decode(tokenized_chat[0]))

outputs = model.generate(
    input_ids=tokenized_chat,
    attention_mask=attention_mask,
    max_new_tokens=128
)

print(tokenizer.decode(outputs[0]))