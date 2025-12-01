from openai import OpenAI
from dotenv import load_dotenv
import requests
import os
import numpy as np
import json
import time

print("[DEBUG] Loading environment variables...")
load_dotenv()
key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    organization=os.getenv("ORGANIZATION"),
    project=os.getenv("PROJECT"),
)

url = os.getenv("URL")
print(f"[DEBUG] URL: {url}")

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
        "temperature": 0.2
    }
    response = requests.post(url, json=payload, headers=headers)
    r = ""
    if response.status_code == 200:
        result = response.json()
        print(f"[DEBUG] Response JSON: {result}")
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

def CreateBenchmark(text, chunks, entities):
    content = f"""
    Devo creare un benchmark per il mio dense retrieval. Ti dò:
    - un testo completo
    - la sua divisione in chunks
    - la lista di entità estratte dai chunks

    Genera **3 domande in totale**:
    1. Una domanda FATTUALE (es. dati, eventi, definizioni)
    2. Una domanda che richiede una SINTESI del contenuto (es. spiegazioni generali)
    3. Una domanda che richiede INFERENZA (es. relazioni implicite o deduzioni)

    ⚠️ Vincoli obbligatori:
    - Usa **solo le informazioni presenti nei chunk forniti**, senza aggiungere conoscenza esterna.
    - Ogni domanda deve menzionare almeno una delle ENTITÀ fornite.

    Per ogni domanda restituisci un oggetto con i seguenti campi:
    [
        {{
            "query": "...",  # la domanda generata
            "question_type": "fact" | "synthesis" | "inference",
            "gold_answer": "sourceTesto-chunkID",  # il chunk corretto
            "relevant_docs": ["sourceTesto-chunkID", ...]  # lista di chunk rilevanti
            "mentions": [   # entità usate nella query   ],
            "hops": 1 | 2 | 3   # numero di entità collegate necessarie per rispondere
        }}
    ]

    Esempio:
    [
        {{
            "query": "Quali sono le principali fasi del capitalismo secondo Schumpeter?",
            "question_type": "fact",
            "gold_answer": "economia_applicata_clean_13_Lez013.txt-4",
            "relevant_docs": [
                "economia_applicata_clean_13_Lez013.txt-6",
                "economia_applicata_clean_13_Lez013.txt-5"
            ],
            "mentions": [
                "Shumpeter",
            ],
            "hops": 1
        }}
    ]

    Il testo è: {text}
    I chunk forniti sono: {chunks}
    Le entità disponibili sono: {entities}
    """

    mess = [
        {"role": "system", "content": "Sei un assistente esperto di valutazione NLP e benchmark che deve rispondere in formato json e non aggiungere altro."},
        {"role": "user", "content": content},
    ]
    
    res = AIRequest(mess)
    res = normRes(res)

    with open("benchmarks/benchmark_revisited_2.json", "a", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        f.write("\n")  # Ensure each appended JSON object is on a new line
    print(f"[DEBUG] Finished writing to file.")

if __name__ == "__main__":
    print("[DEBUG] Starting main execution...")
    files = sorted([f for f in os.listdir("../onlyTextLessonsTurbo") if f.endswith(".txt")])
    selected_files = files[1::2]  # Prende un file sì e uno no
    print(f"[DEBUG] Selected files: {selected_files}, length: {len(selected_files)}")

    with open("../chunks_scripts/chunks_metadata/chunks_linked_entities_emb_final_noemb.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    print(f"[DEBUG] Loaded chunks metadata")  # Print first 3 chunks

    count = 0
    for file in selected_files:
        # count += 1
        # if count < 10: 
        #     continue
        print(f"[DEBUG] Processing file: {file}")
        with open(f"onlyTextLessonsTurbo/{file}", "r", encoding="utf-8") as f:
            text = f.read()

        filtered_chunks = [
            {"source": chunk["source"], "text": chunk["text"], "chunk_id": chunk["chunk_id"]}
            for chunk in chunks
            if chunk["source"] == file
        ]

        # solo entità non duplicate
        entities = list(set(
            ent["text"]
            for chunk in chunks if chunk["source"] == file
            for ent in chunk.get("entities", [])
        ))
        print(f"[DEBUG] Extracted {len(entities)} entities for {file}: {entities}")

        print(f"[DEBUG] Filtered chunks for {file}")

        CreateBenchmark(text, filtered_chunks, entities)
        time.sleep(31)