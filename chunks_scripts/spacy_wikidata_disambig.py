import spacy
import requests
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time # Aggiunto per un piccolo ritardo tra le richieste API

# --- 1. Caricamento Modelli (una sola volta all'inizio) ---
print("Caricamento modelli in corso...")
st_model = SentenceTransformer('intfloat/multilingual-e5-large')
# Usiamo spaCy solo per NER e divisione in frasi
nlp = spacy.load("it_core_news_lg") 
print("Modelli caricati.")

# --- 2. Funzioni di Supporto ---
def cosine_similarity(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def wikidata_candidate_search(entity_text, limit=10):
    """Interroga l'API di Wikidata per ottenere una lista di candidati."""
    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbsearchentities", "format": "json", "language": "it", "search": entity_text, "limit": limit}
    headers = {"User-Agent": "MyThesisEntityLinker/1.0 (contatto@email.com)"} # Buona norma usare un User-Agent reale
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("search", [])
    except requests.exceptions.RequestException as e:
        print(f"Attenzione: Errore di rete per query '{entity_text}': {e}")
        return []

# --- 3. Caricamento Dati e Definizione Parametri ---
INPUT_FILE = "chunks_metadata/chunks_metadata300_embeddings_spacy_ner_positions.json"
OUTPUT_FILE = "chunks_metadata/chunks_linked_entities_emb_final.json"
ALPHA = 0.9 # 90% peso alla similarità semantica, 10% alla popolarità

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunk_data = json.load(f)
print(f"Caricati {len(chunk_data)} chunk dal file '{INPUT_FILE}'.")


# --- 4. Ciclo Principale di Elaborazione ---
# Usiamo tqdm per avere una barra di progresso sui chunk
for chunk in tqdm(chunk_data, desc="Linking entities in chunks"):
    # Aggiungiamo la nuova lista vuota al chunk corrente
    chunk['linked_entities'] = []
    
    # Processiamo il testo con spaCy solo se ci sono entità da linkare
    if not chunk.get("entities"):
        continue
        
    doc = nlp(chunk["text"])
    
    # Mappiamo le entità pre-calcolate alle entità trovate da spaCy nel doc attuale
    # per avere accesso facile all'oggetto 'ent.sent'
    ents_from_json = chunk["entities"]
    doc_ents = list(doc.ents)
    
    for ent_data in ents_from_json:
        # Trova l'oggetto Span di spaCy corrispondente per ottenere il contesto (ent.sent)
        current_ent_span = None
        for span in doc_ents:
            if span.start_char == ent_data["start_char"] and span.end_char == ent_data["end_char"]:
                current_ent_span = span
                break
        
        if not current_ent_span:
            continue

        candidates = wikidata_candidate_search(current_ent_span.text)
        time.sleep(0.1) # Piccolo ritardo per non sovraccaricare l'API di Wikidata

        if not candidates:
            continue
            
        context_sentence = current_ent_span.sent.text
        context_vector = st_model.encode(f"query: {context_sentence}")
        
        best_candidate_info = None
        highest_hybrid_score = -1
        
        all_candidates_info = []

        for rank, candidate in enumerate(candidates):
            cand_label = candidate.get("label", "")
            cand_desc = candidate.get("description", "")
            
            popularity_score = 1 / (rank + 1)
            
            candidate_text_for_embedding = f"{cand_label}. {cand_desc}"
            candidate_vector = st_model.encode(f"passage: {candidate_text_for_embedding}")
            
            similarity_score = cosine_similarity(context_vector, candidate_vector)
            
            hybrid_score = (ALPHA * similarity_score) + ((1 - ALPHA) * popularity_score)
            
            # Salviamo le info di ogni candidato per un'analisi più approfondita
            all_candidates_info.append({
                "id": candidate.get("id"),
                "label": cand_label,
                "description": cand_desc,
                "similarity_score": float(similarity_score),
                "popularity_score": float(popularity_score),
                "hybrid_score": float(hybrid_score)
            })

            if hybrid_score > highest_hybrid_score:
                highest_hybrid_score = hybrid_score
                best_candidate_info = candidate
        
        if best_candidate_info:
            # Aggiungiamo l'entità linkata al nostro chunk
            chunk['linked_entities'].append({
                "mention_text": current_ent_span.text,
                "start_char": current_ent_span.start_char,
                "end_char": current_ent_span.end_char,
                "ner_type": current_ent_span.label_,
                "wikidata_id": best_candidate_info.get("id"),
                "wikidata_label": best_candidate_info.get("label"),
                "wikidata_description": best_candidate_info.get("description"),
                "final_score": float(highest_hybrid_score),
                "all_candidates": all_candidates_info # Lista di tutti i candidati considerati
            })

# --- 5. Salvataggio Finale ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

print(f"\nElaborazione completata! I risultati sono stati salvati in '{OUTPUT_FILE}'.")