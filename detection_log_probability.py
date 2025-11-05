#!pip install transformers torch tqdm matplotlib scikit-learn scipy

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_ind

#Caricamento modello

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "Qwen/Qwen3-4B"

print("Caricamento del modello...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float16)

model.eval()
print("Modello caricato.")

#Caricamento e divisione del dataset
print("\nCaricamento del dataset OpenLLMText...")

BASE_PATH = './OpenTextLLM'
SOURCES_AI = ['ChatGPT', 'LLAMA', 'PaLM']
SOURCE_HUMAN = 'Human'

human_texts = []
ai_texts = []

# Carica i testi umani
human_folder_path = os.path.join(BASE_PATH, SOURCE_HUMAN)
human_file_path = os.path.join(human_folder_path, 'train-dirty.jsonl')
try:
    with open(human_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_point = json.loads(line)
            human_texts.append(data_point.get('text', ''))
except FileNotFoundError:
    print(f"ATTENZIONE: File non trovato: {human_file_path}")

# Carica i testi AI
for source_name in SOURCES_AI:
    ai_folder_path = os.path.join(BASE_PATH, source_name)
    ai_file_path = os.path.join(ai_folder_path, 'train-dirty.jsonl')
    try:
        with open(ai_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_point = json.loads(line)
                ai_texts.append(data_point.get('text', ''))
    except FileNotFoundError:
        print(f"ATTENZIONE: File non trovato: {ai_file_path}")

print("\nDataset caricato.")

#Definizione funzione di calcolo della Log-Probability
def calculate_avg_log_prob(text: str, model, tokenizer) -> float:
    safe_length = 2048

    # Filtra testi non validi
    if not text or not isinstance(text, str) or len(text.split()) < 5:
        return float('nan')

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=safe_length)
    input_ids = inputs["input_ids"].to(model.device)

    if input_ids.shape[1] <= 1:
        return float('nan')

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    avg_log_probability = -loss.item()
    return avg_log_probability

#Esecuzione della funzione sul campione
human_log_probs = []
machine_log_probs = []

num_samples = 5000

print(f"Inizio il calcolo della log probability media per un massimo di {num_samples} campioni (umani vs. tutti IA)...")

human_texts_sampled = human_texts[:num_samples]
ai_texts_sampled = ai_texts[:num_samples]

print(f"Campioni umani da analizzare: {len(human_texts_sampled)}")
print(f"Campioni IA da analizzare: {len(ai_texts_sampled)}")

print("\nCalcolo per i testi umani...")
for text in tqdm(human_texts_sampled):
    log_prob = calculate_avg_log_prob(text, model, tokenizer)
    if not np.isnan(log_prob):
        human_log_probs.append(log_prob)

print("\nCalcolo per i testi IA...")
for text in tqdm(ai_texts_sampled):
    log_prob = calculate_avg_log_prob(text, model, tokenizer)
    if not np.isnan(log_prob):
        machine_log_probs.append(log_prob)

#Analisi dei risultati medi
if human_log_probs and machine_log_probs:
    avg_log_prob_human = np.mean(human_log_probs)
    avg_log_prob_machine = np.mean(machine_log_probs)

    print(f"Log Probability Media (Testo Umano): {avg_log_prob_human:.4f}")
    print(f"Log Probability Media (Testo Macchina): {avg_log_prob_machine:.4f}")

# Calcola la deviazione standard per le probabilità logaritmiche umane e IA
    std_dev_human = np.std(human_log_probs)
    std_dev_machine = np.std(machine_log_probs)

    print(f"\nDeviazione Standard (Testo Umano): {std_dev_human:.4f}")
    print(f"Deviazione Standard (Testo Macchina): {std_dev_machine:.4f}")

#Valutazione

    # Prepara i dati bilanciati
    n_eval = min(len(human_log_probs), len(machine_log_probs))
    y_true = np.array([0] * n_eval + [1] * n_eval)
    scores = np.array(human_log_probs[:n_eval] + machine_log_probs[:n_eval])

    threshold = (np.mean(human_log_probs[:n_eval]) + np.mean(machine_log_probs[:n_eval])) / 2

    y_pred = np.array([1 if score > threshold else 0 for score in scores])

    # Calcolo metriche
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nSoglia scelta: {threshold:.4f}")
    print(f"\nAccuratezza: {accuracy:.4f}, Precisione: {precision:.4f}, Richiamo: {recall:.4f}, F1-Score: {f1:.4f}")

    # Matrice di Confusione
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\nMatrice di Confusione:")
    print(f"[[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

#Test statistico T-Test
def perform_t_test(data1, data2, data1_name="Data 1", data2_name="Data 2"):

    if not data1 or not data2:
        print(f"ATTENZIONE: Non ci sono abbastanza dati per eseguire il test t tra {data1_name} e {data2_name}.")
        return
    try:
        stat, p_value = ttest_ind(data1, data2)
        print(f"\n--- Test t di Student Indipendente tra {data1_name} e {data2_name} ---")
        print(f"\nStatistica del test: {stat:.4f}")
        print(f"Valore p: {p_value:.4f}")

        alpha = 0.05
        if p_value > alpha:
            print(f"Risultato: Non c'è una differenza statisticamente significativa tra le medie dei due campioni. Non possiamo rifiutare l'ipotesi nulla.")
        else:
            print(f"Risultato: C'è una differenza statisticamente significativa tra le medie dei due campioni. Rifiutiamo l'ipotesi nulla.")

    except Exception as e:
        print(f"Si è verificato un errore durante l'esecuzione del test t: {e}")

perform_t_test(human_log_probs, machine_log_probs, "Log-Probabilità Umani", "Log-Probabilità IA")
