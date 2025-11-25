# import spacy
# import requests
# import json
# import numpy as np
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# import time # Aggiunto per un piccolo ritardo tra le richieste API

# # --- 1. Caricamento Modelli (una sola volta all'inizio) ---
# print("Caricamento modelli in corso...")
# st_model = SentenceTransformer('intfloat/multilingual-e5-large')
# # Usiamo spaCy solo per NER e divisione in frasi
# nlp = spacy.load("it_core_news_lg") 
# print("Modelli caricati.")

# # --- 2. Funzioni di Supporto ---
# def cosine_similarity(vec1, vec2):
#     if np.all(vec1 == 0) or np.all(vec2 == 0):
#         return 0.0
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def wikidata_candidate_search(entity_text, limit=10):
#     """Interroga l'API di Wikidata per ottenere una lista di candidati."""
#     url = "https://www.wikidata.org/w/api.php"
#     params = {"action": "wbsearchentities", "format": "json", "language": "it", "search": entity_text, "limit": limit}
#     headers = {"User-Agent": "MyThesisEntityLinker/1.0 (contatto@email.com)"} # Buona norma usare un User-Agent reale
#     try:
#         resp = requests.get(url, params=params, headers=headers, timeout=10)
#         resp.raise_for_status()
#         return resp.json().get("search", [])
#     except requests.exceptions.RequestException as e:
#         print(f"Attenzione: Errore di rete per query '{entity_text}': {e}")
#         return []

# # --- 3. Caricamento Dati e Definizione Parametri ---
# INPUT_FILE = "chunks_metadata_inginf.json"
# OUTPUT_FILE = "chunks_linked_entities_inginf.json"
# ALPHA = 0.9 # 90% peso alla similarità semantica, 10% alla popolarità

# with open(INPUT_FILE, "r", encoding="utf-8") as f:
#     chunk_data = json.load(f)
# print(f"Caricati {len(chunk_data)} chunk dal file '{INPUT_FILE}'.")


# # --- 4. Ciclo Principale di Elaborazione ---
# # Usiamo tqdm per avere una barra di progresso sui chunk
# for chunk in tqdm(chunk_data, desc="Linking entities in chunks"):
#     # Aggiungiamo la nuova lista vuota al chunk corrente
#     chunk['linked_entities'] = []
    
#     # Processiamo il testo con spaCy solo se ci sono entità da linkare
#     if not chunk.get("entities"):
#         continue
        
#     doc = nlp(chunk["text"])
    
#     # Mappiamo le entità pre-calcolate alle entità trovate da spaCy nel doc attuale
#     # per avere accesso facile all'oggetto 'ent.sent'
#     ents_from_json = chunk["entities"]
#     doc_ents = list(doc.ents)
    
#     for ent_data in ents_from_json:
#         # Trova l'oggetto Span di spaCy corrispondente per ottenere il contesto (ent.sent)
#         current_ent_span = None
#         for span in doc_ents:
#             if span.start_char == ent_data["start_char"] and span.end_char == ent_data["end_char"]:
#                 current_ent_span = span
#                 break
        
#         if not current_ent_span:
#             continue

#         candidates = wikidata_candidate_search(current_ent_span.text)
#         time.sleep(0.1) # Piccolo ritardo per non sovraccaricare l'API di Wikidata

#         if not candidates:
#             continue
            
#         context_sentence = current_ent_span.sent.text
#         context_vector = st_model.encode(f"query: {context_sentence}")
        
#         best_candidate_info = None
#         highest_hybrid_score = -1
        
#         all_candidates_info = []

#         for rank, candidate in enumerate(candidates):
#             cand_label = candidate.get("label", "")
#             cand_desc = candidate.get("description", "")
            
#             popularity_score = 1 / (rank + 1)
            
#             candidate_text_for_embedding = f"{cand_label}. {cand_desc}"
#             candidate_vector = st_model.encode(f"passage: {candidate_text_for_embedding}")
            
#             similarity_score = cosine_similarity(context_vector, candidate_vector)
            
#             hybrid_score = (ALPHA * similarity_score) + ((1 - ALPHA) * popularity_score)
            
#             # Salviamo le info di ogni candidato per un'analisi più approfondita
#             all_candidates_info.append({
#                 "id": candidate.get("id"),
#                 "label": cand_label,
#                 "description": cand_desc,
#                 "similarity_score": float(similarity_score),
#                 "popularity_score": float(popularity_score),
#                 "hybrid_score": float(hybrid_score)
#             })

#             if hybrid_score > highest_hybrid_score:
#                 highest_hybrid_score = hybrid_score
#                 best_candidate_info = candidate
        
#         if best_candidate_info:
#             # Aggiungiamo l'entità linkata al nostro chunk
#             chunk['linked_entities'].append({
#                 "mention_text": current_ent_span.text,
#                 "start_char": current_ent_span.start_char,
#                 "end_char": current_ent_span.end_char,
#                 "ner_type": current_ent_span.label_,
#                 "wikidata_id": best_candidate_info.get("id"),
#                 "wikidata_label": best_candidate_info.get("label"),
#                 "wikidata_description": best_candidate_info.get("description"),
#                 "final_score": float(highest_hybrid_score),
#                 "all_candidates": all_candidates_info # Lista di tutti i candidati considerati
#             })

# # --- 5. Salvataggio Finale ---
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(chunk_data, f, ensure_ascii=False, indent=2)

# print(f"\nElaborazione completata! I risultati sono stati salvati in '{OUTPUT_FILE}'.")


import spacy
import requests
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time

# --- 1. Caricamento Modelli ---
print("⏳ Caricamento modelli...")
# Carica sulla GPU se disponibile
st_model = SentenceTransformer('intfloat/multilingual-e5-large')

# Usa "lg" se hai RAM, altrimenti "sm". Se crasha per memoria, cambia in "it_core_news_sm"
try:
    nlp = spacy.load("it_core_news_lg")
except OSError:
    print("⚠️ Modello 'lg' non trovato/pesante. Uso 'sm'.")
    nlp = spacy.load("it_core_news_sm")

print("✅ Modelli caricati.")

# --- 2. Cache e Funzioni ---

# OTTIMIZZAZIONE 1: Cache per le chiamate API
# Memorizza i risultati di Wikidata per non richiederli due volte
wikidata_cache = {}

def cosine_similarity_batch(vec_a, matrix_b):
    """
    Calcola la similarità coseno tra un vettore (contesto) e una matrice di vettori (candidati).
    Molto più veloce di farlo uno per uno.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(matrix_b, axis=1)
    
    # Evita divisioni per zero
    if norm_a == 0: return np.zeros(len(matrix_b))
    norm_b[norm_b == 0] = 1e-10
    
    dot_products = np.dot(matrix_b, vec_a)
    return dot_products / (norm_a * norm_b)

def wikidata_candidate_search_cached(entity_text, limit=10):
    """Cerca su Wikidata usando la cache."""
    # Normalizza la chiave (minuscolo) per trovare più match
    cache_key = entity_text.lower().strip()
    
    if cache_key in wikidata_cache:
        return wikidata_cache[cache_key]

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities", 
        "format": "json", 
        "language": "it", 
        "search": entity_text, 
        "limit": limit
    }
    headers = {"User-Agent": "MyThesisLinker/1.0 (academic_research)"}
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 429:
            print("⚠️ Rate limit raggiunto. Attendo 5 secondi...")
            time.sleep(5)
            return []
            
        resp.raise_for_status()
        data = resp.json().get("search", [])
        
        # Salva in cache
        wikidata_cache[cache_key] = data
        return data
        
    except Exception as e:
        print(f"⚠️ Errore rete '{entity_text}': {e}")
        return []

# --- 3. Setup ---
INPUT_FILE = "../chunks_metadata_inginf.json"
OUTPUT_FILE = "../chunks_linked_entities_inginf.json"
ALPHA = 0.9 

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

print(f"📄 Caricati {len(chunk_data)} chunk.")

# --- 4. Loop Principale ---

# Variabili per statistiche
total_entities = 0
linked_entities = 0

for chunk in tqdm(chunk_data, desc="Linking Entities"):
    chunk['linked_entities'] = []
    
    # Se non ci sono entità nel JSON, saltiamo anche il parsing spaCy (risparmio tempo CPU)
    if not chunk.get("entities"):
        continue
        
    # Parsing spaCy necessario per il contesto (frasi)
    doc = nlp(chunk["text"])
    doc_ents = list(doc.ents)
    
    # Mappa per trovare velocemente gli span di spaCy
    ents_from_json = chunk["entities"]
    
    for ent_data in ents_from_json:
        total_entities += 1
        
        # Trova corrispondenza span
        current_ent_span = None
        for span in doc_ents:
            if span.start_char == ent_data["start_char"] and span.end_char == ent_data["end_char"]:
                current_ent_span = span
                break
        
        if not current_ent_span:
            continue

        # 1. Recupero Candidati (con Cache)
        candidates = wikidata_candidate_search_cached(current_ent_span.text)
        
        if not candidates:
            continue
            
        # OTTIMIZZAZIONE 2: Calcola embedding contesto UNA volta sola per entità
        context_sentence = current_ent_span.sent.text
        # Aggiungiamo 'query:' come richiesto dal modello E5
        context_vector = st_model.encode(f"query: {context_sentence}", normalize_embeddings=True)
        
        # OTTIMIZZAZIONE 3: Batch Encoding dei candidati
        # Prepariamo tutti i testi dei candidati in una lista
        candidate_texts = []
        popularity_scores = []
        
        for rank, cand in enumerate(candidates):
            label = cand.get("label", "")
            desc = cand.get("description", "")
            # Aggiungiamo 'passage:' come richiesto dal modello E5
            candidate_texts.append(f"passage: {label}. {desc}")
            popularity_scores.append(1 / (rank + 1))
            
        # Generiamo TUTTI gli embedding in un colpo solo sulla GPU (molto più veloce)
        candidate_vectors = st_model.encode(candidate_texts, normalize_embeddings=True)
        
        # Calcolo similarità vettoriale (Batch)
        # cosine_similarity_batch restituisce un array di score
        sim_scores = np.dot(candidate_vectors, context_vector)
        
        # Calcolo punteggi ibridi
        # Usiamo numpy per fare il calcolo su tutto l'array insieme
        pop_array = np.array(popularity_scores)
        hybrid_scores = (ALPHA * sim_scores) + ((1 - ALPHA) * pop_array)
        
        # Trova il migliore
        best_idx = np.argmax(hybrid_scores)
        best_score = hybrid_scores[best_idx]
        best_candidate = candidates[best_idx]
        
        # Salviamo i risultati completi (ricostruendo la struttura dati)
        all_candidates_info = []
        for i, cand in enumerate(candidates):
            all_candidates_info.append({
                "id": cand.get("id"),
                "label": cand.get("label"),
                "similarity_score": float(sim_scores[i]),
                "hybrid_score": float(hybrid_scores[i])
            })
            
        chunk['linked_entities'].append({
            "mention_text": current_ent_span.text,
            "start_char": current_ent_span.start_char,
            "end_char": current_ent_span.end_char,
            "ner_type": current_ent_span.label_,
            "wikidata_id": best_candidate.get("id"),
            "wikidata_label": best_candidate.get("label"),
            "wikidata_description": best_candidate.get("description"),
            "final_score": float(best_score),
            "all_candidates": all_candidates_info
        })
        linked_entities += 1
        
        # Piccolo sleep dinamico: se abbiamo usato la cache è 0, altrimenti un minimo
        # per non essere bannati, ma solo se abbiamo fatto una richiesta reale.
        # Dato che usiamo la cache, possiamo essere più aggressivi.
        
# --- 5. Salvataggio ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Completato! Linkate {linked_entities}/{total_entities} entità.")
print(f"📁 File salvato: {OUTPUT_FILE}")