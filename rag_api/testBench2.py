from rag_service import loading, query_rag
from rag_el import loading_entity_linking, query_entity_linking_rerank, query_entity_linking_rerank_RRF, query_rag_with_cross_encoder
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import requests
import numpy as np
import time
from tqdm import tqdm

# Caricamento modello e benchmark
# loading()
loading_entity_linking()

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    organization=os.getenv("ORGANIZATION"),
    project=os.getenv("PROJECT"),
)

url = os.getenv("URL")
# print(f"[DEBUG] URL: {url}")

headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
}

def normalize(vecs):
        result = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return result

def AIRequest(mess):

    payload = {
            "model": "gpt-4o",
            "messages": mess,
            "temperature": 0.0
    }
    response = requests.post(url, json=payload, headers=headers)
    r = ""
    if response.status_code == 200:
            result = response.json()
            # print(f"[DEBUG] Response JSON: {result}")
            r = result['choices'][0]['message']['content']
    else:
            print(f"Errore: {response.status_code}, Dettagli: {response.text}")
    return r

def normRes(res):
        # Step 1: rimuovi blocco di codice markdown se presente
        res = res.strip("```").strip()

        # (opzionale) se ha un prefisso tipo "json\n", rimuovilo:
        if res.startswith("json"):
                res = res[4:].strip()

        clean_str = res.replace("\n", "").replace("\\", "").replace("chunk", "")

        final_data = json.loads(clean_str)

        return final_data

with open("../bench_scripts/benchmarks/benchmark_revisited_3_fixed_corrected.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Funzione per valutare una risposta con un LLM
def evaluate_with_llm(query, generated_answer):
    content = f"""
            Il tuo compito è valutare una risposta in base ad una domanda.
            la domanda è presa da un benchmark mentre la risposta è generata da un sistema di Dense-Retrieval, in particolare sono i top-3 chunk generati e messi insieme uno dopoml'altro.

            I criteri di valutazione sono i seguenti:

            1.  **Completezza (voto da 1 a 10):** La risposta generata copre tutti i punti chiave e le informazioni necessarie per rispondere in modo esaustivo alla domanda? Considera se mancano dettagli importanti.
            2.  **Rilevanza (voto da 1 a 10):** La risposta si attiene strettamente alla domanda o include informazioni non pertinenti? Valuta se il contenuto è focalizzato e non fuorviante.
            3.  **Chiarezza (voto da 1 a 10):** La risposta è ben formulata, facile da leggere e da capire? Valuta la grammatica, la sintassi e il flusso logico del testo.

            Domanda: {query}

            Risposta generata (3 chunks uniti): {generated_answer}

            Restituisci la tua valutazione in formato JSON con le seguenti chiavi:
            {{
                "completezza": <voto da 1 a 10>,
                "rilevanza": <voto da 1 a 10>,
                "chiarezza": <voto da 1 a 10>,
            }}

            """
    
    mess = [
            {"role": "system", "content": " Sei un assistente esperto nella valutazione di risposte generate da sistemi di Dense-Retrieval che deve rispondere in formato json e non aggiungere altro."},
            {"role": "user", "content": content},
        ]
        
    res = AIRequest(mess)
    res = normRes(res)
    
    return res

# Inizializzazione metriche
metrics = {
    "completezza": 0,
    "rilevanza": 0,
    "chiarezza": 0,
    "total_queries": 0
}

for i, item in enumerate(tqdm(benchmark, "Processing Benchmark")):

    # if metrics["total_queries"] >= 5:
    #     break
    
    q = item["query"]
    gold_answer = item["gold_answer"]
    if gold_answer is None:
        continue

    # results = query_rag(q, 3, "false")["chunks"]
    # results = query_entity_linking_rerank(query=q, k_final=3, k_initial_retrieval=30, BETA=0.5, LLMHelp="false")["chunks"]
    # results = query_entity_linking_rerank_RRF(query=q, k_final=3, k_initial_retrieval=30, LLMHelp="false")["chunks"]
    results = query_rag_with_cross_encoder(query=q, k_final_llm=3)["chunks"]
  

    # retrieved_docs = [f"{res['source']}-{res['chunk_id']}" for res in results]

    # Genera una risposta concatenando i contenuti dei documenti recuperati
    generated_answer = "\n".join(res["text"] for res in results)

    # Valuta la risposta con un LLM
    scores = evaluate_with_llm(q, generated_answer)

    # print(scores)

    # Aggiorna le metriche
    metrics["completezza"] += scores["completezza"]
    metrics["rilevanza"] += scores["rilevanza"]
    metrics["chiarezza"] += scores["chiarezza"]
    metrics["total_queries"] += 1

    time.sleep(3)  # Aggiungi un delay per evitare rate limit

# Normalizzazione delle metriche
for key in ["completezza", "rilevanza", "chiarezza"]:
    metrics[key] /= metrics["total_queries"]

# Salvataggio risultati
with open("./final_results/test2_EL_50_20_3_cross.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Stampa report
print("\n=== Risultati della Valutazione con LLM ===")
print(f"Query totali: {metrics['total_queries']}")
print(f"Completezza: {metrics['completezza']:.2f}")
print(f"Rilevanza: {metrics['rilevanza']:.2f}")
print(f"Chiarezza: {metrics['chiarezza']:.2f}")