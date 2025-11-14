import faiss
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import spacy
import re
import time

# --- 1. CARICAMENTO GLOBALE E CONFIGURAZIONE ---

# Variabili globali per contenere i modelli e i dati caricati
index = None
chunk_data_linked = None
st_model = None
nlp = None
primaVolta = True
cross_encoder_model = None

def loading_entity_linking():
    """
    Carica tutti i modelli e i dati necessari una sola volta.
    """
    global primaVolta, index, chunk_data_linked, st_model, nlp, cross_encoder_model
    if primaVolta:
        print("Inizio caricamento modelli e dati per Entity Linking RAG...")
        # Carica il modello per calcolare gli embedding
        st_model = SentenceTransformer('intfloat/multilingual-e5-large')

        print("Caricamento Cross-Encoder (potrebbe richiedere un po' di tempo)...")
        cross_encoder_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        # cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
        # nickprock/cross-encoder-italian-bert-stsb
        # Osiria/minilm-l6-h384-italian-cross-encoder
        print("Cross-Encoder caricato.")

        # Carica spaCy per il NER sulla query
        nlp = spacy.load("it_core_news_lg")

        # Carica l'indice Faiss (costruito sugli embedding del testo puro)
        index = faiss.read_index("../faiss_index300.index") 

        # Carica il NUOVO file JSON con le entità già linkate
        with open("../chunks_scripts/chunks_metadata/chunks_linked_entities_emb_final_noemb.json", "r", encoding="utf-8") as f:
            chunk_data_linked = json.load(f)
        
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


# --- 2. FUNZIONI HELPER (LLM, LINKING, ETC.) ---

def link_query_entities(query_text: str, k_candidates=10, alpha=0.9) -> set:
    """
    Funzione specializzata per eseguire l'entity linking solo su una query.
    Restituisce un set di Wikidata QID trovati.
    """
    doc = nlp(query_text)
    linked_qids = set()

    for ent in doc.ents:
        # La logica di linking è una versione compatta di quella che abbiamo già sviluppato
        candidates = wikidata_candidate_search(ent.text, limit=k_candidates)
        time.sleep(0.05)
        if not candidates: continue

        context_vector = st_model.encode(f"query: {ent.sent.text}")
        best_candidate_id = None
        highest_hybrid_score = -1

        for rank, candidate in enumerate(candidates):
            popularity_score = 1 / (rank + 1)
            cand_text = f"{candidate.get('label', '')}. {candidate.get('description', '')}"
            candidate_vector = st_model.encode(f"passage: {cand_text}")
            similarity_score = cosine_similarity(context_vector, candidate_vector)
            hybrid_score = (alpha * similarity_score) + ((1 - alpha) * popularity_score)

            if hybrid_score > highest_hybrid_score:
                highest_hybrid_score = hybrid_score
                best_candidate_id = candidate.get("id")
        
        if best_candidate_id:
            linked_qids.add(best_candidate_id)
            
    return linked_qids

def wikidata_candidate_search(entity_text, limit=10):
    # (Questa funzione e cosine_similarity sono le stesse dello script precedente)
    url_wiki = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbsearchentities", "format": "json", "language": "it", "search": entity_text, "limit": limit}
    try:
        resp = requests.get(url_wiki, params=params, headers={"User-Agent": "MyRAG/1.0"}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("search", [])
    except requests.exceptions.RequestException:
        return []

def cosine_similarity(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0): return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def AIRequest(mess):

    payload = {
        "model": "gpt-4o",
        "messages": mess,
        # "max_completion_tokens": 3000,
        "temperature": 0
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

    # print("Risposta AI:", response)
    return response

# --- 3. FUNZIONE PRINCIPALE DI RICERCA E RE-RANKING ---

def query_rag_with_cross_encoder(query: str, k_final_llm=10, k_rerank_rrf=20, k_initial_retrieval=50, LLMHelp="false"):
    """
    Esegue una pipeline RAG SOTA a 3 fasi:
    1. Retrieve con Faiss (veloce e ampio).
    2. Re-rank con RRF (efficiente, fonde segnali).
    3. Re-rank con Cross-Encoder (preciso, finale).
    """
    
    # --- FASE 1: RETRIEVAL DENSO (Faiss) ---
    query_embedding = st_model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_embedding, k_initial_retrieval)

    # --- FASE 2: PRIMO RE-RANKING (RRF) ---
    query_qids = link_query_entities(query)
    
    candidate_chunks_rrf = []
    # (La logica RRF rimane la stessa, la applichiamo ai k_initial_retrieval candidati)
    for i in range(len(indices[0])):
        chunk_index = indices[0][i]; dense_score = distances[0][i]
        chunk = chunk_data_linked[chunk_index]
        chunk_qids = {entity['wikidata_id'] for entity in chunk.get('linked_entities', [])}
        intersection = len(query_qids.intersection(chunk_qids))
        entity_score = intersection / len(query_qids) if query_qids else 0.0
        
        chunk_copy = chunk.copy()
        chunk_copy['dense_score'] = float(dense_score)
        chunk_copy['entity_score'] = float(entity_score)
        candidate_chunks_rrf.append(chunk_copy)
        
    entity_ranked_chunks = sorted(candidate_chunks_rrf, key=lambda x: x['entity_score'], reverse=True)
    rankings = {f"{chunk['source']}-{chunk['chunk_id']}": {} for chunk in candidate_chunks_rrf}
    for rank, chunk in enumerate(candidate_chunks_rrf): rankings[f"{chunk['source']}-{chunk['chunk_id']}"]["dense_rank"] = rank + 1
    for rank, chunk in enumerate(entity_ranked_chunks): rankings[f"{chunk['source']}-{chunk['chunk_id']}"]["entity_rank"] = rank + 1
    
    RRF_K = 60
    for chunk in candidate_chunks_rrf:
        chunk_id = f"{chunk['source']}-{chunk['chunk_id']}"
        rrf_score = (1 / (RRF_K + rankings[chunk_id]["dense_rank"])) + (1 / (RRF_K + rankings[chunk_id]["entity_rank"]))
        chunk['rrf_score'] = rrf_score
        
    rrf_reranked_results = sorted(candidate_chunks_rrf, key=lambda x: x['rrf_score'], reverse=True)
    
    # Prendiamo i migliori candidati da passare al Cross-Encoder
    top_candidates_for_cross_encoder = rrf_reranked_results[:k_rerank_rrf]

    # --- FASE 3: RE-RANKING DI PRECISIONE (Cross-Encoder) ---
    # print(f"Esecuzione Cross-Encoder su {len(top_candidates_for_cross_encoder)} candidati...")
    # Il Cross-Encoder ha bisogno di coppie (query, testo_chunk)
    cross_encoder_input = [[query, chunk['text']] for chunk in top_candidates_for_cross_encoder]
    
    # Calcola gli score di pertinenza
    cross_encoder_scores = cross_encoder_model.predict(cross_encoder_input, show_progress_bar=False)
    
    # Aggiungi i nuovi score ai chunk
    for i in range(len(cross_encoder_scores)):
        top_candidates_for_cross_encoder[i]['final_score'] = float(cross_encoder_scores[i])
        
    # Ordina i risultati finali in base allo score del Cross-Encoder
    final_results = sorted(top_candidates_for_cross_encoder, key=lambda x: x['final_score'], reverse=True)
    
    # Seleziona i top-k finali da passare all'LLM
    final_chunks_for_llm = final_results[:k_final_llm]
    
    # --- FASE 4: GENERAZIONE CON LLM ---
    if not final_chunks_for_llm:
        result_json = {"chunks": final_chunks_for_llm}
        result_json["testoRisp"] = "Nessun risultato trovato per questa domanda."
        #print(f"Risultati della query: {result_json}")
        return result_json

    result_json = {"chunks": final_chunks_for_llm}
    response_text = "Ecco i risultati!" # Testo di risposta predefinito

    campi_desiderati = ["chunk_id", "text", "source", "start_time", "end_time", "entities", "linked_entities"]

    chunks_filtrati = [{campo: chunk[campo] for campo in campi_desiderati} for chunk in final_chunks_for_llm]
    for chunk in chunks_filtrati:
        if "linked_entities" in chunk and "all_candidates" in chunk["linked_entities"]:
            # Check if 'all_candidates' exists before modifying it
            chunk["linked_entities"]["all_candidates"] = []

    response_data = {
        "chunks": chunks_filtrati,  # This is your list of dictionaries
        "testoRisp": response_text   # Add the LLM's response as a separate key
    }

    if LLMHelp.lower() == "true":
        res = LLMHelpFunc(query, chunks_filtrati)
        if res == "Nessun risultato trovato per questa domanda.":
            response_data["chunks"] = []
            response_data["testoRisp"] = res
        else:
            cleaned_chunks = []
            response_data["testoRisp"] = res
            for i, chunk in enumerate(response_data["chunks"]):
                x = f"[{i + 1}]"
                if x in res:
                    cleaned_chunks.append(chunk)
            response_data["chunks"] = cleaned_chunks

    return response_data

def query_entity_linking_rerank_RRF(query: str, k_final=10, k_initial_retrieval=30, LLMHelp="true"):
    """
    Esegue una ricerca RAG fondendo i ranking di Dense Retrieval e Entity Linking
    con la tecnica Reciprocal Rank Fusion (RRF).
    """
    
    # --- FASE 1: RETRIEVAL DENSO E VELOCE (come prima) ---
    query_embedding = st_model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_embedding, k_initial_retrieval)

    # --- FASE 2: ENTITY LINKING SULLA QUERY (come prima) ---
    # print(f"Linking entità per la query: '{query}'...")
    query_qids = link_query_entities(query)
    # print(f"QID trovati nella query: {query_qids}")

    # --- FASE 3: CALCOLO DEGLI SCORE E PREPARAZIONE PER RRF ---
    candidate_chunks = []
    for i in range(len(indices[0])):
        chunk_index = indices[0][i]
        dense_score = distances[0][i]
        
        chunk = chunk_data_linked[chunk_index]
        chunk_qids = {entity['wikidata_id'] for entity in chunk.get('linked_entities', [])}
        
        # Nuova metrica recall-oriented per l'entity_score
        intersection = len(query_qids.intersection(chunk_qids))
        entity_score = intersection / len(query_qids) if query_qids else 0.0
        
        chunk_copy = chunk.copy()
        chunk_copy['dense_score'] = float(dense_score)
        chunk_copy['entity_score'] = float(entity_score)
        candidate_chunks.append(chunk_copy)
        
    # --- FASE 4: RECIPROCAL RANK FUSION (RRF) ---
    # Creiamo due classifiche separate
    
    # La classifica densa è già data dall'ordine di `candidate_chunks`
    # La classifica delle entità la otteniamo ordinando per entity_score
    entity_ranked_chunks = sorted(candidate_chunks, key=lambda x: x['entity_score'], reverse=True)
    
    # Creiamo un dizionario per mappare l'ID del chunk alla sua posizione in ogni classifica
    rankings = {f"{chunk['source']}-{chunk['chunk_id']}": {} for chunk in candidate_chunks}
    
    for rank, chunk in enumerate(candidate_chunks):
        rankings[f"{chunk['source']}-{chunk['chunk_id']}"]["dense_rank"] = rank + 1
        
    for rank, chunk in enumerate(entity_ranked_chunks):
        rankings[f"{chunk['source']}-{chunk['chunk_id']}"]["entity_rank"] = rank + 1
        
    # Calcoliamo il punteggio RRF per ogni chunk
    RRF_K = 60 # Costante standard per RRF
    for chunk in candidate_chunks:
        chunk_id = f"{chunk['source']}-{chunk['chunk_id']}"
        dense_rank = rankings[chunk_id]["dense_rank"]
        entity_rank = rankings[chunk_id]["entity_rank"]
        
        rrf_score = (1 / (RRF_K + dense_rank)) + (1 / (RRF_K + entity_rank))
        chunk['final_score'] = rrf_score
        
    # Ordina i risultati in base al nuovo punteggio RRF
    reranked_results = sorted(candidate_chunks, key=lambda x: x['final_score'], reverse=True)
    
    # Prendi i migliori k_final risultati
    final_chunks = reranked_results[:k_final]
    
    # ... il resto della funzione (passaggio all'LLM) rimane identico ...
    if not final_chunks:
        result_json = {"chunks": final_chunks}
        result_json["testoRisp"] = "Nessun risultato trovato per questa domanda."
        #print(f"Risultati della query: {result_json}")
        return result_json

    result_json = {"chunks": final_chunks}
    response_text = "Ecco i risultati!" # Testo di risposta predefinito

    campi_desiderati = ["chunk_id", "text", "source", "start_time", "end_time", "entities"]#, "linked_entities"]

    chunks_filtrati = [{campo: chunk[campo] for campo in campi_desiderati} for chunk in final_chunks]
    for chunk in chunks_filtrati:
        if "linked_entities" in chunk and "all_candidates" in chunk["linked_entities"]:
            # Check if 'all_candidates' exists before modifying it
            chunk["linked_entities"]["all_candidates"] = []

    response_data = {
        "chunks": chunks_filtrati,  # This is your list of dictionaries
        "testoRisp": response_text   # Add the LLM's response as a separate key
    }

    if LLMHelp.lower() == "true":
        res = LLMHelpFunc(query, chunks_filtrati)
        if res == "Nessun risultato trovato per questa domanda.":
            response_data["chunks"] = []
            response_data["testoRisp"] = res
        else:
            cleaned_chunks = []
            response_data["testoRisp"] = res
            for i, chunk in enumerate(response_data["chunks"]):
                x = f"[{i + 1}]"
                if x in res:
                    cleaned_chunks.append(chunk)
            response_data["chunks"] = cleaned_chunks

    return response_data

def query_entity_linking_rerank(query: str, k_final=10, k_initial_retrieval=30, BETA=0.5, LLMHelp="true"):
    """
    Esegue una ricerca RAG con re-ranking basato su Entity Linking.
    - k_initial_retrieval: Numero di chunk da recuperare con Faiss per il re-ranking.
    - k_final: Numero di chunk da restituire dopo il re-ranking.
    - BETA: Peso dato allo score delle entità. 0.0 = solo dense, 1.0 = solo entità.
    """
    
    # --- FASE 1: RETRIEVAL DENSO E VELOCE ---
    query_embedding = st_model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
    # Cerca un numero maggiore di candidati per avere materiale su cui fare il re-ranking
    distances, indices = index.search(query_embedding, k_initial_retrieval)

    # --- FASE 2: ENTITY LINKING SULLA QUERY ---
    # print(f"Linking entità per la query: '{query}'...")
    query_qids = link_query_entities(query)
    # print(f"QID trovati nella query: {query_qids}")

    # --- FASE 3: RE-RANKING DEI RISULTATI ---
    reranked_results = []
    for i in range(len(indices[0])):
        chunk_index = indices[0][i]
        dense_score = distances[0][i]
        
        chunk = chunk_data_linked[chunk_index]
        
        # Calcola l'Entity Score
        chunk_qids = {entity['wikidata_id'] for entity in chunk.get('linked_entities', [])}
        
        # Usiamo Jaccard similarity per lo score delle entità
        # intersection = len(query_qids.intersection(chunk_qids))
        # union = len(query_qids.union(chunk_qids))
        # entity_score1 = intersection / union if union > 0 else 0.0

        # proviamo con una similarità recall-oriented
        intersection = len(query_qids.intersection(chunk_qids))
        entity_score = intersection / len(query_qids) if query_qids else 0.0
        # print(f"Chunk ID {chunk['source']}-{chunk['chunk_id']} | Dense Score: {dense_score:.4f} | Entity Score: {entity_score:.4f}")

        # f1 score
        # entity_score = (2 * entity_score1 * entity_score2) / (entity_score1 + 2 * entity_score2) if (entity_score1 + entity_score2) > 0 else 0.0

        # Calcola il punteggio finale ibrido
        final_score = dense_score + (BETA * entity_score)
        # print(final_score)

        chunk_copy = chunk.copy()
        chunk_copy['dense_similarity'] = float(dense_score)
        chunk_copy['entity_score'] = float(entity_score)
        chunk_copy['final_score'] = float(final_score)
        reranked_results.append(chunk_copy)
        
    # Ordina i risultati in base al nuovo punteggio finale
    reranked_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Prendi i migliori k_final risultati
    final_chunks = reranked_results[:k_final]

    if not final_chunks:
        result_json = {"chunks": final_chunks}
        result_json["testoRisp"] = "Nessun risultato trovato per questa domanda."
        #print(f"Risultati della query: {result_json}")
        return result_json

    result_json = {"chunks": final_chunks}
    response_text = "Ecco i risultati!" # Testo di risposta predefinito

    campi_desiderati = ["chunk_id", "text", "source", "start_time", "end_time", "entities", "linked_entities"]

    chunks_filtrati = [{campo: chunk[campo] for campo in campi_desiderati} for chunk in final_chunks]
    for chunk in chunks_filtrati:
        if "linked_entities" in chunk and "all_candidates" in chunk["linked_entities"]:
            # Check if 'all_candidates' exists before modifying it
            chunk["linked_entities"]["all_candidates"] = []

    response_data = {
        "chunks": chunks_filtrati,  # This is your list of dictionaries
        "testoRisp": response_text   # Add the LLM's response as a separate key
    }

    if LLMHelp.lower() == "true":
        res = LLMHelpFunc(query, chunks_filtrati)
        if res == "Nessun risultato trovato per questa domanda.":
            response_data["chunks"] = []
            response_data["testoRisp"] = res
        else:
            cleaned_chunks = []
            response_data["testoRisp"] = res
            for i, chunk in enumerate(response_data["chunks"]):
                x = f"[{i + 1}]"
                if x in res:
                    cleaned_chunks.append(chunk)
            response_data["chunks"] = cleaned_chunks

    return response_data

def query_rag(query, k_ric, LLMHelp):
    #loading()  # Assicura che tutto sia caricato (embeddings, indice, metadata)

    # Calcola e normalizza l'embedding della query
    query_embedding = st_model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")

    # Ricerca dei k più simili
    k = int(k_ric)
    D, I = index.search(query_embedding, k)

    # Applica soglia su similarità (cosine, dato che usi IndexFlatIP con normalizzazione)
    threshold = 0.0
    filtered_results = []
    for idx, score in zip(I[0], D[0]):
        #print(f"Similarità: {score}")
        if score >= threshold:
            data = chunk_data_linked[idx]
            data['similarity'] = float(score)
            filtered_results.append(data)

    if not filtered_results:
        result_json = {"chunks": filtered_results}
        result_json["testoRisp"] = "Nessun risultato trovato per questa domanda."
        #print(f"Risultati della query: {result_json}")
        return result_json

    result_json = {"chunks": filtered_results}
    response_text = "Ecco i risultati!" # Testo di risposta predefinito

    campi_desiderati = ["chunk_id", "text", "source", "start_time", "end_time"]#, "linked_entities"]

    chunks_filtrati = [{campo: chunk[campo] for campo in campi_desiderati} for chunk in filtered_results]
    for chunk in chunks_filtrati:
        if "linked_entities" in chunk and "all_candidates" in chunk["linked_entities"]:
            # Check if 'all_candidates' exists before modifying it
            chunk["linked_entities"]["all_candidates"] = []

    response_data = {
        "chunks": chunks_filtrati,  # This is your list of dictionaries
        "testoRisp": response_text   # Add the LLM's response as a separate key
    }

    if LLMHelp.lower() == "true":
        res = LLMHelpFunc(query, chunks_filtrati)
        if res == "Nessun risultato trovato per questa domanda.":
            response_data["chunks"] = []
            response_data["testoRisp"] = res
        else:
            cleaned_chunks = []
            response_data["testoRisp"] = res
            for i, chunk in enumerate(response_data["chunks"]):
                x = f"[{i + 1}]"
                if x in res:
                    cleaned_chunks.append(chunk)
            response_data["chunks"] = cleaned_chunks

    return response_data

# --- ESEMPIO DI ESECUZIONE ---
if __name__ == '__main__':
    loading_entity_linking()
    
    # Esempio di query
    mia_query = "Quali condizioni hanno portato Roger Sperry a ricevere il premio Nobel?"
    
    # Chiama la nuova funzione
    risultati = query_entity_linking_rerank(
        query=mia_query,
        k_final=10,
        k_initial_retrieval=30,
        BETA=0.5,
        LLMHelp="true"
    )
    
    print("\n--- RISULTATO FINALE ---")
    print(risultati['testoRisp'])
    print("\nChunk di riferimento:")
    for i, chunk in enumerate(risultati['chunks']):
        print(f"  {i+1}. [Score Finale: {chunk['final_score']:.4f}] [Score Entità: {chunk['entity_score']:.2f}] - {chunk['text'][:100]}...")