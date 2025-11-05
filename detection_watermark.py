# !pip install datasets transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 scipy

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello e il tokenizer
model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# Parametri della funzione di watermarking
SECRET_KEY = 250925
DELTA = 1.5  # Forza del watermark (valore tipico 0.5 - 2.0)
GAMMA = 0.20 # Frazione del vocabolario nella "green list"

def get_green_list(input_ids, vocab_size, secret_key, gamma):
    # Combina la chiave segreta con il contesto per creare un seed unico
    # Usa l'ultimo token come contesto per semplicità, ma potresti usare l'intera sequenza
    context_hash = hashlib.sha256(str(input_ids[-1].item()).encode('utf-8') + str(secret_key).encode('utf-8')).hexdigest()
    seed = int(context_hash, 16)

    # Inizializza il generatore di numeri casuali con il seed
    rng = random.Random(seed)

    # Crea una "green list" con una frazione del vocabolario
    green_list_size = int(vocab_size * gamma)

    green_list = rng.sample(range(vocab_size), green_list_size)

    return set(green_list)

#Definizione interfaccia di interazione col modello
system_prompt = "You are an AI assistant. Provide a direct answer to the user prompt."
user_prompt = "Tell me the main information about the sicilian city of Agrigento"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

#Codifica dell'input
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False
).to(model.device)

#Funzione del generatore
def generate_with_watermark_from_ids(model, tokenizer, input_ids, max_new_tokens=512, delta=DELTA, gamma=GAMMA, secret_key=SECRET_KEY):
    output_ids = input_ids.clone()
    model.eval()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(output_ids)
            logits = outputs.logits[:, -1, :]

        vocab_size = logits.shape[-1]
        green_list = get_green_list(output_ids[0], vocab_size, secret_key, gamma)

        for token_id in green_list:
            if token_id < vocab_size:
                logits[0, token_id] += delta

        next_token_id = torch.argmax(logits, dim=-1)
        output_ids = torch.cat([output_ids, next_token_id.unsqueeze(0)], dim=-1)

        if next_token_id == tokenizer.eos_token_id:
            break

    generated_ids = output_ids[0][len(input_ids[0]):]
    watermarked_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return watermarked_text

"""Test di generazione ed esempio di utilizzo del detector"""

# Test Generazione
watermarked_output = generate_with_watermark_from_ids(model, tokenizer, input_ids)
print("Testo watermarked:", watermarked_output)

unwatermarked_output = model.generate(input_ids, max_new_tokens=1024)

# Decodifica solo il testo generato
unwatermarked_text = tokenizer.decode(unwatermarked_output[0][len(input_ids[0]):], skip_special_tokens=True)
print("Testo non watermarked:", unwatermarked_text)

#Detector

#Definizione della funzione di detection
def detect_watermark(text, tokenizer, secret_key, gamma):
    # Codifica il testo in una sequenza di token
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    num_green_tokens = 0
    num_total_tokens = 0

    # Itera su ogni token, escluso il primo (il prompt)
    for i in range(1, input_ids.shape[1]):
        context = input_ids[0, :i]
        next_token = input_ids[0, i]

        # Ricrea la "green list" per il contesto attuale
        vocab_size = tokenizer.vocab_size
        green_list = get_green_list(context, vocab_size, secret_key, gamma)

        if next_token.item() in green_list:
            num_green_tokens += 1

        num_total_tokens += 1

    # Calcola il valore p
    # Ipotizziamo che la probabilità di un token "verde" sia gamma
    expected_mean = num_total_tokens * gamma
    expected_std = (num_total_tokens * gamma * (1 - gamma))**0.5

    z_score = (num_green_tokens - expected_mean) / expected_std
    p_value = norm.sf(z_score)

    return {
        "num_green_tokens": num_green_tokens,
        "num_total_tokens": num_total_tokens,
        "z_score": z_score,
        "p_value": p_value
    }

# Esempio di utilizzo del detector
detection_result = detect_watermark(watermarked_output, tokenizer, SECRET_KEY, GAMMA)
print("\nRisultati del rilevamento:")
print(f"Token verdi trovati: {detection_result['num_green_tokens']} su {detection_result['num_total_tokens']}")
print(f"Z-score: {detection_result['z_score']:.4f}")
print(f"P-value: {detection_result['p_value']:.4e}")

detection_result = detect_watermark(unwatermarked_text, tokenizer, SECRET_KEY, GAMMA)
print("\nRisultati del rilevamento non watermarked:")
print(f"Token verdi trovati: {detection_result['num_green_tokens']} su {detection_result['num_total_tokens']}")
print(f"Z-score: {detection_result['z_score']:.4f}")
print(f"P-value: {detection_result['p_value']:.4e}")

"""#Test su prompt multipli"""

# Lista di prompt da testare
prompts = [
    "Tell me the main information about the sicilian city of Agrigento",
    "What are the health benefits of eating apples?",
    "Explain the theory of relativity in simple terms.",
    "Who was Leonardo da Vinci and why is he important?",
    "Did dinosaurs have lips?",
    "Classify each of the following as a primary color or a secondary color",
    "Write a short story about a person who discovers a hidden room in their house. The story should include a plot twist and a clear resolution at the end.",
]

results = []

# tqdm per mostrare l’avanzamento sui prompt
for user_prompt in tqdm(prompts, desc="Generazione prompt"):
    print(f"\n--- Prompt: {user_prompt} ---")

    # Prepara i messaggi nel formato chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Codifica l'input
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False
    ).to(model.device)

    # --- Generazione watermarked ---
    watermarked_text = generate_with_watermark_from_ids(model, tokenizer, input_ids)
    detection_wm = detect_watermark(watermarked_text, tokenizer, SECRET_KEY, GAMMA)

    # --- Generazione non-watermarked ---
    unwatermarked_ids = model.generate(input_ids, max_new_tokens=512)
    unwatermarked_text = tokenizer.decode(
        unwatermarked_ids[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    detection_unwm = detect_watermark(unwatermarked_text, tokenizer, SECRET_KEY, GAMMA)

    # Salvataggio nella lista "results"
    results.append({
        "prompt": user_prompt,
        "wm_text": watermarked_text, # Save full watermarked text
        "wm_green_tokens": detection_wm["num_green_tokens"],
        "wm_total_tokens": detection_wm["num_total_tokens"],
        "wm_z_score": round(detection_wm["z_score"], 3),
        "wm_p_value": detection_wm["p_value"],
        "unwm_text": unwatermarked_text, # Save full unwatermarked text
        "unwm_green_tokens": detection_unwm["num_green_tokens"],
        "unwm_total_tokens": detection_unwm["num_total_tokens"],
        "unwm_z_score": round(detection_unwm["z_score"], 3),
        "unwm_p_value": detection_unwm["p_value"],
    })

# --- Tabella finale con i risultati ---
df = pd.DataFrame(results)
print("\n=== RISULTATI DEL DETECTOR ===")

print(df[["prompt","wm_text","wm_z_score", "wm_p_value","unwm_text","unwm_z_score", "unwm_p_value"]])


# (Opzionale) Salva i risultati in csv
df.to_csv("detection_multi_prompts.csv", index=False, encoding="utf-8")
