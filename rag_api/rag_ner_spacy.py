import faiss
import json
from sentence_transformers import SentenceTransformer
import requests
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json
import re
import spacy

primaVolta = True

def loading_spacy():
    global primaVolta, index, chunk_data, chunk_data_emb, model, nlp
    if primaVolta:
        # Carica tutto
        index = faiss.read_index("../faiss_index300.index")
        with open("../chunks_scripts/chunks_metadata/chunks_metadata300.json", "r", encoding="utf-8") as f:
            chunk_data = json.load(f)

        model = SentenceTransformer('intfloat/multilingual-e5-large')

        nlp = spacy.load("it_core_news_lg")

        # --- Preparazione (una volta sola) ---
        embeddings_list = [model.encode([chunk['text']], normalize_embeddings=True).astype("float32")[0] for chunk in chunk_data]
        index = faiss.IndexFlatIP(embeddings_list[0].shape[0])
        index.add(np.array(embeddings_list))

        chunk_data_emb = chunk_data.copy()

        # Salva embedding nel chunk per sicurezza
        for i, emb in enumerate(embeddings_list):
            chunk_data_emb[i]['embedding'] = emb

        print("Caricamento completato.")
        primaVolta = False
    else:
        print("Caricamento già effettuato.")

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
  organization=os.getenv("ORGANIZATION"),
  project=os.getenv("PROJECT"),
)

url = os.getenv("URL")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}"
}

def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def AIRequest(mess):

    payload = {
        "model": "gpt-4o",
        "messages": mess,
        "max_tokens": 3000,
        "temperature": 0.0
    }
    response = requests.post(url, json=payload, headers=headers)
    r = ""
    if response.status_code == 200:
        result = response.json()
        r = result['choices'][0]['message']['content']
    else:
        print(f"Errore: {response.status_code}, Dettagli: {response.text}")
    return r

def addLink(testo, x=10):
    # Funzione di sostituzione dinamica
    def replacer(match):
        numero = int(match.group(1))
        if 1 <= numero <= x:
            return f' <a href="#ris-{numero}">[{numero}]</a>'
        return match.group(0)  # Non modificare se fuori range

    pattern = r'\[(\d+)\]'  # Corrisponde a [numero]
    return re.sub(pattern, replacer, testo)

def LLMHelpFunc(query, results):
    # Prepara il messaggio per l'AI
    richiesta = f"""Ti sto inviando una query e i risultati più simili trovati da un sistema di Dense Retrieval.
    Tu devi analizzare i risultati e restituire una risposta compatta alla query utilizzando le informazioni dei risultati.
    La query è: {query}, i risultati sono: {results}
    I risultati sono in formato JSON e contengono i seguenti campi: chunk_id, text, source, start_time, end_time, entities.
    restituisci solo un testo unico con le note tipo [1] nelle parti di testo che hai scritto che sono prese da un file. solo questo testo e non altro, non specificare i riferimenti alla fine del testo.
    Se non hai trovato informazioni utili all'imterno dei risultati per la query data o la query sembra essere mal posta o fuori contesto, rispondi con "Nessun risultato trovato per questa domanda."
    """

    mess = [
        {"role": "user", "content": richiesta}
    ]

    # Invia la richiesta all'AI
    response = AIRequest(mess)

    # Aggiunge link html
    response = addLink(response)

    print("Risposta AI:", response)
    return response

def extract_entities(text):
    """ Estrae entità dalla query usando spaCy """
    return {ent.text.lower() for ent in nlp(text).ents}

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

# -------------------------
# 2. Hybrid search
# -------------------------
def hybrid_search_spacy(query, k_ric, LLMHelp, alpha, beta):
    """
    alpha = peso per similarità embeddings
    beta  = peso per overlap entità
    """
    # Embedding della query
    query_embedding = model.encode([query], normalize_embeddings=True).astype("float32")

    # Recupera top-K dal FAISS (veloce)
    # D, I = index.search(query_embedding, int(k_ric))

    # Entity linking della query
    query_entities = set([ent.text.lower() for ent in nlp(query).ents])

    scored_chunks = []

    # Calcola score embedding + entity su tutti i chunk
    for chunk in chunk_data_emb:
        chunk_emb = chunk.get('embedding')
        if chunk_emb is None:
            # fallback, calcola embedding al volo (lento)
            chunk_emb = model.encode([chunk['text']], normalize_embeddings=True).astype("float32")
        sim_dense = float(np.dot(query_embedding, chunk_emb.T))

        chunk_entities = set([ent['text'].lower() for ent in chunk.get('entities', [])])
        if query_entities or chunk_entities:
            sim_entity = len(query_entities & chunk_entities) / len(query_entities | chunk_entities)
        else:
            sim_entity = 0.0

        score = sim_dense + beta * sim_entity
        chunk_copy = chunk.copy()
        chunk_copy['similarity'] = score
        scored_chunks.append(chunk_copy)

    # Ordina tutti i chunk
    scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)

    # Prendi i top-k
    filtered_results = scored_chunks[:int(k_ric)]
    # for i, chunk in enumerate(filtered_results):
    #     print(f"Chunk {i+1} - Similarity: {chunk['similarity']}")

    if not filtered_results:
        return {"chunks": [], "testoRisp": "Nessun risultato trovato per questa domanda."}

    result_json = {"chunks": filtered_results}
    response_text = "Ecco i risultati!"

    if LLMHelp.lower() == "true":
        res = LLMHelpFunc(query, filtered_results)
        if res == "Nessun risultato trovato per questa domanda.":
            result_json["chunks"] = []
            response_text = res
        else:
            cleaned_chunks = []
            response_text = res
            for i, chunk in enumerate(result_json["chunks"]):
                x = f"[{i + 1}]"
                if x in res:
                    cleaned_chunks.append(chunk)
            result_json["chunks"] = cleaned_chunks

    result_json["testoRisp"] = response_text
    return result_json