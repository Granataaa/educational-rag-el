from transformers import pipeline
import json
from tqdm import tqdm  # Libreria per mostrare una barra di progresso

with open("chunks_metadata/chunks_metadata300_embeddings_spacy_ner.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)
print(f"Caricamento di {len(chunk_data)} chunk dal file JSON...")

# Carica il modello NER con il tokenizer lento
ner_pipeline = pipeline("ner", model="Davlan/xlm-roberta-large-ner-hrl", tokenizer="Davlan/xlm-roberta-large-ner-hrl", use_fast=False)
print("Modello NER caricato.")

for i, chunk in enumerate(tqdm(chunk_data, desc="Processing chunks")):
    entities = ner_pipeline(chunk_data[i]['text'])
    chunk_data[i]['entities'] = [
        {"entity": ent["entity"], "score": float(ent["score"]), "word": ent["word"]} for ent in entities
    ]

with open("chunks_metadata/chunks_metadata300_embeddings_roberta_ner.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

