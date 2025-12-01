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

primaVolta = True

def loading():
    global primaVolta, index, chunk_data, model
    if primaVolta:
        # Carica tutto
        index = faiss.read_index("../faiss_index300.index")
        with open("../chunks_scripts/chunks_metadata/chunks_metadata300.json", "r", encoding="utf-8") as f:
            chunk_data = json.load(f)

        model = SentenceTransformer('intfloat/multilingual-e5-large')
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

    # print("Risposta AI:", response)
    return response

def query_rag(query, k_ric, LLMHelp):
    #loading()  # Assicura che tutto sia caricato (embeddings, indice, metadata)

    # Calcola e normalizza l'embedding della query
    query_embedding = model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")

    # Ricerca dei k più simili
    k = int(k_ric)
    D, I = index.search(query_embedding, k)

    # Applica soglia su similarità (cosine, dato che usi IndexFlatIP con normalizzazione)
    threshold = 0.0
    filtered_results = []
    for idx, score in zip(I[0], D[0]):
        #print(f"Similarità: {score}")
        if score >= threshold:
            data = chunk_data[idx]
            data['similarity'] = float(score)
            filtered_results.append(data)

    if not filtered_results:
        result_json = {"chunks": filtered_results}
        result_json["testoRisp"] = "Nessun risultato trovato per questa domanda."
        #print(f"Risultati della query: {result_json}")
        return result_json

    result_json = {"chunks": filtered_results}
    response_text = "Ecco i risultati!" # Testo di risposta predefinito

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

    #print(f"Risultati della query: {result_json}")
    
    return result_json