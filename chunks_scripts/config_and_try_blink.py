import BLINK.blink.main_dense as main_dense
import argparse
import json
from tqdm import tqdm
import torch

# 1. PREPARAZIONE DELLA CONFIGURAZIONE E DEL DEVICE
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

config = {
    "biencoder_model": "../BLINK/models/biencoder_wiki_large.bin",
    "biencoder_config": "../BLINK/models/biencoder_wiki_large.json",
    "crossencoder_model": "../BLINK/models/crossencoder_wiki_large.bin",
    "crossencoder_config": "../BLINK/models/crossencoder_wiki_large.json",
    "entity_catalogue": "../BLINK/models/entity.jsonl",
    
    # NON useremo più questo file, ma lo lasciamo per compatibilità
    "entity_encoding": "../BLINK/models/all_entities_large.t7",
    
    # ----- MODIFICA CHIAVE QUI -----
    "faiss_index": "flat", # o "hnsw", a seconda di quale indice hai scaricato
    "index_path": "../BLINK/models/faiss_flat_index.pkl", # Il percorso al file che hai scaricato
    # -----------------------------
    
    "top_k": 10,
    "fast": False,
    "output_path": "logs/",
    "interactive": False,
}

args = argparse.Namespace(**config)

# 2. CARICAMENTO DEI MODELLI (MODIFICATO PER LA GPU)
# ---------------------------------------------------
# Carica le configurazioni JSON per poterle modificare
with open(args.biencoder_config) as json_file:
    biencoder_params = json.load(json_file)
    biencoder_params["path_to_model"] = args.biencoder_model
    # LA MODIFICA CHIAVE: diciamo al modello su quale device lavorare
    biencoder_params["device"] = device

with open(args.crossencoder_config) as json_file:
    crossencoder_params = json.load(json_file)
    crossencoder_params["path_to_model"] = args.crossencoder_model
    crossencoder_params["device"] = device
    
# Carica i modelli usando le funzioni interne di BLINK con i parametri aggiornati
biencoder = main_dense.load_biencoder(biencoder_params)
crossencoder = main_dense.load_crossencoder(crossencoder_params)

# Carica il resto dei dati (catalogo, encodings)
(
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    faiss_indexer,
) = main_dense._load_candidates(
    args.entity_catalogue, 
    args.entity_encoding, 
    faiss_index=getattr(args, 'faiss_index', None), 
    index_path=getattr(args, 'index_path' , None),
)

# Riassembla la tupla 'models' come si aspetta la funzione run
models = (
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    faiss_indexer,
)

# 3. ELABORAZIONE IN BATCH
# -------------------------
with open("chunks_metadata/chunks_metadata300_embeddings_spacy_ner_positions.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)
print(f"Caricamento di {len(chunk_data)} chunk dal file JSON...")

# Step 1: Raccogli tutte le menzioni da tutti i chunk
all_mentions_to_link = []
mentions_metadata = [] # Per mappare i risultati indietro

for chunk_idx, chunk in enumerate(chunk_data):
    entities = chunk.get("entities", [])
    for entity in entities:
        mention_data = {
            "label": "unknown", 
            "label_id": -1, 
            "context_left": chunk["text"][:entity["start_char"]].lower(), 
            "mention": entity["text"].lower(), 
            "context_right": chunk["text"][entity["end_char"]:].lower()
        }
        all_mentions_to_link.append(mention_data)
        # Salva l'indice del chunk e l'entità originale per poterli ritrovare dopo
        mentions_metadata.append({
            "chunk_idx": chunk_idx,
            "original_entity": entity
        })

print(f"Trovate {len(all_mentions_to_link)} menzioni totali da linkare.")

# Step 2: Esegui BLINK una sola volta su tutte le menzioni
if all_mentions_to_link:
    _, _, _, _, _, all_predictions, all_scores = main_dense.run(
        args,
        None,
        *models,
        test_data=all_mentions_to_link
    )

    # Step 3: Mappa i risultati indietro nei chunk originali
    for i, metadata in enumerate(tqdm(mentions_metadata, desc="Mapping results")):
        chunk_idx = metadata["chunk_idx"]
        original_entity = metadata["original_entity"]
        
        # Inizializza la lista 'linked_entities' se non esiste
        if "linked_entities" not in chunk_data[chunk_idx]:
            chunk_data[chunk_idx]["linked_entities"] = []
            
        prediction = all_predictions[i]
        scores = all_scores[i]

        chunk_data[chunk_idx]["linked_entities"].append({
            "text": original_entity["text"],
            "type": original_entity["type"],
            "start_char": original_entity["start_char"],
            "end_char": original_entity["end_char"],
            # Prendi la predizione migliore (la prima)
            "predicted_entity_title": prediction[0],
            "predicted_entity_score": float(scores[0]),
            # Opzionale: salva tutte le top_k predizioni
            "top_k_predictions": list(zip(prediction, [float(s) for s in scores]))
        })

# 4. SALVATAGGIO FINALE
# -----------------------
with open("chunks_metadata/chunks_metadata300_blink_linked_spacy_batched.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

print("\nElaborazione completata con successo!")