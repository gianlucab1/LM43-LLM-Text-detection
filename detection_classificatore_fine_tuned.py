
#!pip install transformers datasets

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
from tqdm import tqdm
import numpy as np
import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizerFast
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset

#Pre-processing dei dati
BASE_PATH = './OpenTextLLM'
SOURCES_MAP = {
    'Human': 0, 'ChatGPT': 1, 'LLAMA': 1, 'PaLM': 1, 'GPT-2':1
}
SPLIT_FILENAME_MAP = {
    'train': 'train-dirty', 'validation': 'valid-dirty', 'test': 'test-dirty'
}

#Funzione di creazione del dataset
def load_and_prepare_data(base_path, sources_map, split_map, human_ratio=0.5):
    print("--- Inizio caricamento dati ---")
    all_data = {'train': [], 'validation': [], 'test': []}

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"La cartella base '{os.path.abspath(base_path)}' non esiste.")

    for source_name, label in sources_map.items():
        source_path = os.path.join(base_path, source_name)
        if not os.path.isdir(source_path):
            print(f"ATTENZIONE: Salto la fonte '{source_name}' (cartella non trovata).")
            continue

        for split_name, filename_prefix in split_map.items():
            file_path = os.path.join(source_path, f'{filename_prefix}.jsonl')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc=f"Lettura {split_name} ({source_name})"):
                        data_point = json.loads(line)
                        all_data[split_name].append({'text': data_point.get('text', ''), 'label': label})

    print("\n--- Caricamento completato. Bilanciamento e creazione dei Dataset... ---")

    balanced_data = {'train': [], 'validation': [], 'test': []}

    for split_name in all_data:
        df = pd.DataFrame(all_data[split_name])
        human_df = df[df['label'] == 0]
        ai_df = df[df['label'] == 1]

        # Determina il numero di campioni per classe per ottenere il rapporto desiderato
        if len(human_df) > 0 and len(ai_df) > 0:
            max_samples_per_class = min(len(human_df), len(ai_df))
            num_human_samples = int(max_samples_per_class * human_ratio * 2) # Aim for 50% of the total balanced dataset
            num_ai_samples = int(max_samples_per_class * (1 - human_ratio) * 2) # Aim for 50% of the total balanced dataset

            # Assicura di non richiedere più campioni di quelli disponibili
            num_human_samples = min(num_human_samples, len(human_df))
            num_ai_samples = min(num_ai_samples, len(ai_df))

            balanced_human_df = human_df.sample(n=num_human_samples, random_state=42)
            balanced_ai_df = ai_df.sample(n=num_ai_samples, random_state=42)

            balanced_df = pd.concat([balanced_human_df, balanced_ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)
            balanced_data[split_name] = balanced_df.to_dict('records')
        elif len(human_df) > 0:
             balanced_data[split_name] = human_df.to_dict('records')
        else:
             balanced_data[split_name] = ai_df.to_dict('records')


    train_ds = Dataset.from_pandas(pd.DataFrame(balanced_data['train']))
    val_ds = Dataset.from_pandas(pd.DataFrame(balanced_data['validation']))
    test_ds = Dataset.from_pandas(pd.DataFrame(balanced_data['test']))


    dataset_dict = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

    print("\n--- Riepilogo dei dati caricati ---")
    print(dataset_dict)
    if 'train' in dataset_dict and dataset_dict['train'].num_rows > 0:
        print("\nDistribuzione delle classi nel training set:")
        print(pd.DataFrame(balanced_data['train'])['label'].value_counts(normalize=True))

    print("\n--- Struttura dettagliata del Dataset ---")
    for split_name, dataset in dataset_dict.items():
        print(f"\nSplit: {split_name}")
        print(f"  Numero di esempi: {dataset.num_rows}")
        print(f"  Caratteristiche: {dataset.features}")
        if dataset.num_rows > 0:
            print(f"  Primo esempio: {dataset[0]}")


    return dataset_dict

#Tokenizzazione
print("\n--- Inizializzazione del Tokenizer ---")
MODEL_NAME = 'roberta-large'
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

#Funzione di tokenizzazione
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

print("\n--- Tokenizzazione dei dataset (potrebbe richiedere qualche minuto) ---")

raw_datasets = load_and_prepare_data(BASE_PATH, SOURCES_MAP, SPLIT_FILENAME_MAP)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

print("--- Tokenizzazione completata ---")

# Salvataggio del dataset tokenizzato
SAVE_PATH_TOKENIZED = '/content/drive/MyDrive/OpenTextLLM/tokenized_datasets_v2'
print(f"\n--- Salvataggio dei dataset tokenizzati in '{SAVE_PATH_TOKENIZED}' ---")
tokenized_datasets.save_to_disk(SAVE_PATH_TOKENIZED)
print("--- Salvataggio completato ---")

#Carica i dati tokenizzati (Usata per evitare di ripetere la tokenizzazione in sessioni differenti)
SAVE_PATH_TOKENIZED = './tokenized_dataset'

print(f"--- Caricamento dei dataset tokenizzati da '{SAVE_PATH_TOKENIZED}' ---")
try:
    tokenized_datasets = load_from_disk(SAVE_PATH_TOKENIZED)
    print("--- Caricamento completato ---")

    columns_to_keep = ['input_ids', 'attention_mask', 'labels']
    tokenized_datasets.set_format(type='torch', columns=columns_to_keep)

    print("\n--- Riepilogo dei dati caricati ---")
    print(tokenized_datasets)
    if 'train' in tokenized_datasets and tokenized_datasets['train'].num_rows > 0:
        print("\nDistribuzione delle classi nel training set:")
        train_df = tokenized_datasets['train'].to_pandas()
        print(train_df['labels'].value_counts(normalize=True))
        tokenized_datasets.set_format(type='torch', columns=columns_to_keep)

except Exception as e:
    print(f"Errore nel caricamento dei dataset tokenizzati: {e}")
    print("Assicurati che la cella precedente abbia salvato correttamente i dati in", SAVE_PATH_TOKENIZED)

"""#Caricamento dataset OpenLLMText tokenizzato"""

# Percorsi dei dataset tokenizzati
train_path = "./train"
valid_path = "./validation"
test_path  = "./test"

# Caricamento dei dataset tokenizzati
train_dataset = load_from_disk(train_path)
valid_dataset = load_from_disk(valid_path)
test_dataset  = load_from_disk(test_path)

# Imposta il formato PyTorch se necessario
columns_to_keep = ['input_ids', 'attention_mask', 'labels']
train_dataset.set_format(type='torch', columns=columns_to_keep)
valid_dataset.set_format(type='torch', columns=columns_to_keep)
test_dataset.set_format(type='torch', columns=columns_to_keep)

# Controllo dati
print("Train Dataset:")
print(train_dataset[:5])
if 'label' in train_dataset.features:
    print("\nTrain Label Distribution:")
    print(pd.Series(train_dataset['labels']).value_counts())

print("\nValid Dataset:")
print(valid_dataset[:5])
if 'label' in valid_dataset.features:
    print("\nValid Label Distribution:")
    print(pd.Series(valid_dataset['labels']).value_counts())

print("\nTest Dataset:")
print(test_dataset[:5])
if 'label' in test_dataset.features:
    print("\nTest Label Distribution:")
    print(pd.Series(test_dataset['labels']).value_counts())

# Inizializzazione del modello RoBERTa-base per classificazione binaria
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

#Funzione di calcolo delle metriche
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#Parametri di addestramento
training_args = TrainingArguments(
    output_dir="./roberta_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=50,
)

#Definizione del trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

#Avvio del training
trainer.train()

# Definizione directory di salvataggio e salvataggio stesso del modello fine-tunato
save_path = "./roberta_finetuned"
trainer.model.save_pretrained(save_path)

"""#Test del modello fine-tuned"""

#Carica modello
def load_model(model_path, device='cuda'):
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model

#Carica tokenizer standard di RoBERTa
def load_tokenizer(model_name="roberta-base"):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    return tokenizer

#Funzione di predizione
def predict_texts(texts, model, tokenizer, device='cuda'):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    return preds.cpu().numpy(), probs.cpu().numpy()

#Funzione di ricaricamento dell'ultimo checkpoint del modello
model_path = ".roberta_finetuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, device)
tokenizer = load_tokenizer("roberta-base")

#Caricamento del dataset di test tokenizzato
SAVE_PATH_TOKENIZED = './tokenized_datasets'
test_dataset_path = os.path.join(SAVE_PATH_TOKENIZED, 'test')

try:
    test_dataset = load_from_disk(test_dataset_path)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    print("Test dataset caricato con successo.")
except Exception as e:
    print(f"Errore nel caricamento del test dataset: {e}")
    exit()

#Campionamento casuale di 5000 esemoi dal dataset di test
num_test_examples = 5000
if num_test_examples > len(test_dataset):
    num_test_examples = len(test_dataset)
    print(f"Il numero di esempi richiesto ({num_test_examples}) è maggiore del dataset disponibile. Verrà usato l'intero dataset.")

subset_indices = list(range(num_test_examples))
test_subset = Subset(test_dataset, subset_indices)

#Preparazione dataloader del subset
test_batch_size = 64
test_dataloader = DataLoader(test_subset, batch_size=test_batch_size)

#Predizioni sul subset di test
print(f"Inizio previsioni su un sottoinsieme di {num_test_examples} esempi dal test dataset...")
all_preds = []
all_labels = []

for batch in tqdm(test_dataloader, desc="Predicting on test subset"):
    inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
    labels = batch['labels'].numpy()

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    all_preds.extend(preds)
    all_labels.extend(labels)

print("Previsioni completate.")

#Calcolo metriche
print("\nCalcolo metriche di valutazione sul sottoinsieme...")
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

print(f"\nRisultati sul Sottoinsieme di {num_test_examples} esempi:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

if 'all_labels' in locals() and 'all_preds' in locals():
    cm = confusion_matrix(all_labels, all_preds)

    # Stampa la confusion matrix come tabella
    cm_df = pd.DataFrame(cm, index=["Vero Umano", "Vero LLM"], columns=["Predetto Umano", "Predetto LLM"])
    print("\nMatrice di confusione:")
    print(cm_df)

else:
    print("Variabili all_labels o all_preds non trovate. Esegui la cella di predizione prima di calcolare la Confusion Matrix.")

# Puoi anche stampare un classification report dettagliato
from sklearn.metrics import classification_report
print("\nClassification Report sul sottoinsieme:")

target_names = ["Human", "LLM"]

if len(set(all_labels)) == 2:
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
else:
    print("Solo una classe nel subset di test. Impossibile generare.")

"""#Test sul dataset M4"""

# Test su M4
!git clone https://github.com/mbzuai-nlp/M4.git

# Percorso del dataset M4
m4_path = "./M4"

# Percorso del file JSONL
file_path = "./M4/data/reddit_bloomz.jsonl"

# Carica in pandas e stampa l'head del dataset
df = pd.read_json(file_path, lines=True)
print(df.head())

model_path = "./roberta_finetuned/checkpoint-12804"
output_csv = "./Opentext_processed/M4_reddit_predictions.csv"


# Carica modello e tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Funzione di predizione
def predict_texts(texts, batch_size=16):
    all_preds, all_probs = [], []
    non_empty_texts = [text for text in texts if text is not None and text.strip()]
    if not non_empty_texts:
        print("No non-empty texts to process.")
        return [], []
        
    for i in tqdm(range(0, len(non_empty_texts), batch_size), desc="Inferenza batch"):
        batch_texts = non_empty_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy().tolist())
    return all_preds, all_probs

# Itera sui file reddit_*.jsonl

texts, labels = [], []

for root, dirs, files in os.walk(m4_path):
    for file in files:
        if file.endswith(".jsonl") and "wikihow_" in file.lower():
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item is None:
                        continue
                    # Aggiungi testo umano
                    human_text = item.get("human_text","")
                    if human_text is not None:
                        texts.append(human_text)
                        labels.append(0)
                    # Aggiungi testo generato
                    machine_text = item.get("machine_text","")
                    if machine_text is not None:
                        texts.append(machine_text)
                        labels.append(1)


print(f"Totale esempi da predire: {len(texts)}")

# Predizione con il modello

preds, probs = predict_texts(texts, batch_size=16)

non_empty_indices = [i for i, text in enumerate(texts) if text is not None and text.strip()]
df_results = pd.DataFrame({
    "text": [texts[i] for i in non_empty_indices],
    "label": [labels[i] for i in non_empty_indices],
    "prediction": preds,
    "probabilities": probs
})

df_results.to_csv(output_csv, index=False)
print("Predizioni salvate in:", output_csv)


# Calcolo metriche di classificazione

print("\n--- Metriche di classificazione sul subset Reddit ---")
non_empty_labels = [labels[i] for i in non_empty_indices]
accuracy = accuracy_score(non_empty_labels, preds)
print(f"Accuracy: {accuracy:.4f}\n")
print(classification_report(non_empty_labels, preds, target_names=["Human", "LLM"]))

if 'non_empty_labels' in locals() and 'preds' in locals():
    print("Variables non_empty_labels and preds found. Calculating and printing Confusion Matrix...")

    cm = confusion_matrix(non_empty_labels, preds)

    cm_df = pd.DataFrame(cm, index=["True Human", "True LLM"], columns=["Predicted Human", "Predicted LLM"])
    print("\nConfusion Matrix:")
    print(cm_df)

else:
    print("Variables non_empty_labels or preds not found. Run the prediction cell before calculating the Confusion Matrix.")

# Caricamento dell'estensione tensorboard per la visualizzazione di informazioni sul training dai log
# %load_ext tensorboard

# Avvia TensorBoard puntando alla directory dei log
# %tensorboard --logdir /content/logs
