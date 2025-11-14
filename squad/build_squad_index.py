import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import spacy
import requests
import time

# === CONFIGURAZIONE ===
INPUT_DOCS = "squad_it_docs.json"
OUTPUT_CHUNKS = "squad_metadata_all.json"
OUTPUT_INDEX = "squad_faiss_index_all.index"

# Modelli
st_model = SentenceTransformer("intfloat/multilingual-e5-large")
nlp = spacy.load("it_core_news_lg")


# --- ENTITY LINKING HELPER ---
def wikidata_candidate_search(entity_text, limit=5):
    url_wiki = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "it",
        "search": entity_text,
        "limit": limit,
    }
    try:
        resp = requests.get(url_wiki, params=params, headers={"User-Agent": "MyRAG/1.0"}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("search", [])
    except requests.exceptions.RequestException:
        return []


def link_entities_spacy(text, k_candidates=5, alpha=0.9):
    """
    Entity linking semplice: usa spaCy per NER + sim embedding con e5 + popolarità.
    Ritorna lista di {text, label, wikidata_id}.
    """
    doc = nlp(text)
    linked_entities = []
    for ent in doc.ents:
        candidates = wikidata_candidate_search(ent.text, limit=k_candidates)
        time.sleep(0.05)  # evita overload di richieste
        if not candidates:
            continue

        # vettore contesto
        context_vector = st_model.encode(f"query: {ent.sent.text}")

        best_id, best_score = None, -1
        for rank, cand in enumerate(candidates):
            popularity = 1 / (rank + 1)
            cand_text = f"{cand.get('label', '')}. {cand.get('description', '')}"
            cand_vec = st_model.encode(f"passage: {cand_text}")
            sim = np.dot(context_vector, cand_vec) / (np.linalg.norm(context_vector) * np.linalg.norm(cand_vec))
            hybrid_score = alpha * sim + (1 - alpha) * popularity
            if hybrid_score > best_score:
                best_id, best_score = cand.get("id"), hybrid_score

        if best_id:
            linked_entities.append({
                "mention": ent.text,
                "label": ent.label_,
                "wikidata_id": best_id
            })

    return linked_entities


# --- COSTRUZIONE ---
print(f"Carico documenti da {INPUT_DOCS} ...")
with open(INPUT_DOCS, "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"Totale documenti: {len(docs)}")

all_embeddings = []
all_chunks = []

for doc in tqdm(docs, desc="Processamento documenti"):
    text = doc["text"]

    # Entity linking
    linked_entities = link_entities_spacy(text)

    # Embedding
    emb = st_model.encode(f"passage: {text}", normalize_embeddings=True).astype("float32")

    # Salviamo il chunk arricchito
    new_chunk = {
        "chunk_id": doc["chunk_id"],
        "source": doc["source"],
        "text": text,
        "linked_entities": linked_entities
    }
    all_chunks.append(new_chunk)
    all_embeddings.append(emb)

# Convertiamo in matrice
embedding_matrix = np.vstack(all_embeddings)

# --- CREAZIONE INDICE FAISS ---
d = embedding_matrix.shape[1]
index = faiss.IndexFlatIP(d)  # IP = inner product (cosine sim con normalizzazione)
index.add(embedding_matrix)

# --- SALVATAGGIO ---
with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

faiss.write_index(index, OUTPUT_INDEX)

print(f"\n✅ Creati {OUTPUT_CHUNKS} e {OUTPUT_INDEX}")
