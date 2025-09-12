#!pip install transformers torch tqdm numpy scikit-learn matplotlib scipy

#Importazioni e Caricamento Modello/Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import numpy as np

#Parametri principali

model_name = "Qwen/Qwen3-4B"
num_samples = 5000

# Carica modello e tokenizer

print("Caricamento del modello...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Modello caricato con successo.")

#Caricamento dataset e preprocessing OpenLLMText da file locali

print("\nCaricamento del dataset OpenLLMText...")

# Percorso della tua cartella OpenLLMText
BASE_PATH = './OpenTextLLM'
SOURCES_AI = ['ChatGPT', 'LLAMA', 'PaLM']
SOURCE_HUMAN = 'Human'

human_texts = []
llm_texts = []

# Carica i testi umani
human_file_path = os.path.join(BASE_PATH, SOURCE_HUMAN, 'test-dirty.jsonl')
try:
    with open(human_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            human_texts.append(json.loads(line).get('text', ''))
except FileNotFoundError:
    print(f"ATTENZIONE: File non trovato: {human_file_path}")

# Carica i testi AI da tutte le fonti
for source_name in SOURCES_AI:
    ai_file_path = os.path.join(BASE_PATH, source_name, 'test-dirty.jsonl')
    try:
        with open(ai_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                llm_texts.append(json.loads(line).get('text', ''))
    except FileNotFoundError:
        print(f"ATTENZIONE: File non trovato: {ai_file_path}")

print(f"Dataset caricato. Trovati {len(human_texts)} testi umani e {len(llm_texts)} testi AI.")

# Funzione di calcolo della perplessità

def calculate_perplexities_individual(texts, model, tokenizer):
    perplexities = []
    max_length = 512

    print(f"\nCalcolo della perplessità per {len(texts)} testi...")
    for text in tqdm(texts):
        if not text or not isinstance(text, str) or len(text.split()) < 5:
            perplexities.append(np.nan)
            continue
        try:
            encodings = tokenizer(text, max_length=max_length, truncation=True, return_tensors='pt')
            input_ids = encodings.input_ids.to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            perplexities.append(torch.exp(loss).item())
        except Exception as e:
            print(f"Errore su testo: '{text[:50]}...' -> {e}")
            perplexities.append(np.nan)

    valid_texts = [t for t, p in zip(texts, perplexities) if not np.isnan(p)]
    valid_perplexities = [p for p in perplexities if not np.isnan(p)]
    print(f"Completato: {len(valid_perplexities)}/{len(texts)} validi")
    return valid_perplexities, valid_texts

# Campiona e calcola perplessità

np.random.shuffle(human_texts)
np.random.shuffle(llm_texts)
human_texts_sampled = human_texts[:num_samples]
llm_texts_sampled = llm_texts[:num_samples]

human_perplexities, human_valid = calculate_perplexities_individual(human_texts_sampled, model, tokenizer)
llm_perplexities, llm_valid = calculate_perplexities_individual(llm_texts_sampled, model, tokenizer)

# Analisi medie e calcolo metriche di valutazione

if human_perplexities and llm_perplexities:
    print("\n--- Risultati analisi Perplessità ---")

    # Converti in array numpy per facilitare i calcoli
    human_perplexities_array = np.array(human_perplexities)
    llm_perplexities_array = np.array(llm_perplexities)

    # Calcola le medie
    mean_human = np.mean(human_perplexities_array)
    mean_llm = np.mean(llm_perplexities_array)
    print(f"\nPerplessità media testi umani: {mean_human:.2f}")
    print(f"Perplessità media testi LLM:   {mean_llm:.2f}")

    # Calcola le deviazioni standard
    std_dev_human = np.std(human_perplexities_array)
    std_dev_llm = np.std(llm_perplexities_array)
    print(f"\nDeviazione standard della perplessità per i testi umani: {std_dev_human:.4f}")
    print(f"Deviazione standard della perplessità per i testi generati da IA: {std_dev_llm:.4f}")

    print(f"\n--- Valutazione delle performance di classificazione ---")

    # Prepara i dati bilanciati per la valutazione
    n_eval = min(len(human_perplexities_array), len(llm_perplexities_array))
    y_true = np.array([0] * n_eval + [1] * n_eval) # 0: Umano, 1: AI
    scores = np.array(list(human_perplexities_array[:n_eval]) + list(llm_perplexities_array[:n_eval]))

    # Calcolo della soglia (punto medio tra le medie)
    threshold = (mean_human + mean_llm) / 2

    # Predizioni: la perplessità BASSA indica AI
    y_pred = np.array([1 if score < threshold else 0 for score in scores])

    # Calcolo delle metriche
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nSoglia scelta: {threshold:.2f}")
    print(f"\nAccuratezza: {accuracy:.4f}, Precisione: {precision:.4f}, Richiamo: {recall:.4f}, F1-Score: {f1:.4f}")

    # Matrice di Confusione
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\nMatrice di Confusione:")
    print(f"[[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

else:
    print("\nNon ci sono abbastanza dati validi per un'analisi.")

print("Esecuzione del test t di Welch per confrontare i dati di perplessità...")

# Assicurati che i dati siano disponibili e non vuoti
if 'human_perplexities' in locals() and human_perplexities and \
   'llm_perplexities' in locals() and llm_perplexities:

    # Converti in array numpy e rimuovi eventuali NaN se presenti
    human_perplexities_array = np.array(human_perplexities)
    llm_perplexities_array = np.array(llm_perplexities)

    # Rimuovi i NaN prima di eseguire il test
    human_valid_perplexities = human_perplexities_array[~np.isnan(human_perplexities_array)]
    llm_valid_perplexities = llm_perplexities_array[~np.isnan(llm_perplexities_array)]

    if len(human_valid_perplexities) > 1 and len(llm_valid_perplexities) > 1:
        t_statistic, p_value = ttest_ind(human_valid_perplexities, llm_valid_perplexities)

        print(f"\nRisultati del test t di Welch:")
        print(f"  Statistica t: {t_statistic:.4f}")
        print(f"  p-value: {p_value:.4f}")

        alpha = 0.05
        print(f"\nLivello di significatività (alpha): {alpha}")

        if p_value < alpha:
            print("\nIl p-value è inferiore ad alpha. Rifiutiamo l'ipotesi nulla.")
        else:
            print("\nIl p-value è maggiore o uguale ad alpha. Non rifiutiamo l'ipotesi nulla.")
    else:
        print("Non ci sono abbastanza dati validi (almeno 2 per gruppo) per eseguire il test t.")
else:
    print("I dati di perplessità non sono disponibili o sono vuoti. Assicurati di aver eseguito le celle precedenti.")
