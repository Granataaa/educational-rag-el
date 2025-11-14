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

def AIRequest(mess):

    payload = {
        "model": "gpt-4o",
        "messages": mess,
        "temperature": 1.0
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

def sistema_query(benchmark, chunks_metadata):
    source = "linguaggio_e_comunicazione_clean_Lez008.txt"
    chunks = [
                {"source": chunk["source"], "text": chunk["text"], "chunk_id": chunk["chunk_id"]}
                for chunk in chunks_metadata    
                if chunk["source"] == source
            ]
    with open(f"onlyTextLessonsTurbo/{source}", "r", encoding="utf-8") as f:
        testo = f.read()
    query = 69

    content = f"""
    Sto creando un benchmark per testare un sistema di retrieval.
    Il benchmark è composto da domande e risposte corrette, ma alcune risposte sono mancanti.

    Ti fornisco:
    - la domanda (`query`),
    - il testo completo da cui è tratta,
    - la divisione del testo in `chunks`.

    Il tuo compito:
    1. Individua il chunk PIÙ rilevante che risponde alla query → questo diventa `gold_answer`.
    2. Individua TUTTI i chunk pertinenti (anche più di uno, se servono) → questi diventano `relevant_docs`.
    3. Se la domanda non è risolvibile o è ambigua, metti `gold_answer: null` e `relevant_docs: []`.

    IMPORTANTE:
    - `gold_answer` deve essere nel formato "source-chunk_id".
    - `relevant_docs` deve essere una lista di stringhe nello stesso formato.
    - Restituisci SOLO un JSON valido con questa struttura:
    {{
    "gold_answer": "source-chunk_id" oppure null,
    "relevant_docs": ["source-chunk_id1", "source-chunk_id2", ...]
    }}

    Ecco alcuni esempi dal benchmark:
    {json.dumps(benchmark[:3], ensure_ascii=False, indent=2)}

    Domanda da sistemare:
    {benchmark[query]["query"]}

    tipo della domanda da sistemare:
    {benchmark[query]["question_type"]}

    Testo della lezione:
    {testo}

    Divisione in chunks:
    {json.dumps(chunks, ensure_ascii=False, indent=2)}
    """

    mess = [
            {"role": "system", "content": "Sei un assistente esperto di valutazione NLP e benchmark che deve rispondere in formato json e non aggiungere altro."},
            {"role": "user", "content": content},
        ]
    
    res = AIRequest(mess)
    res = normRes(res)

    with open("benchmarks/query_revisited.json", "a", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        f.write("\n")  # Ensure each appended JSON object is on a new line

    
def validation(query, gold_answer, relevant_docs, question_type):
    content = f"""
    Sei un validatore di un benchmark per un sistema di retrieval.

    Ti fornisco:
    - la query (domanda dell’utente)
    - il gold_answer (chunk ritenuto corretto)
    - i relevant_docs (altri chunk che dovrebbero contenere informazioni pertinenti)
    - i testi effettivi di gold_answer e dei relevant_docs

    Il tuo compito è controllare e rispondere SOLO in JSON con la seguente struttura:

    {{
    "query": "testo della query",
    "gold_valid": "Yes" o "No",        // il gold risponde alla query?
    "relevant_docs_valid": [
        "Yes" o "No" per ogni relevant docs, // i relevant_docs sono pertinenti e aggiungono informazioni utili?
        ],
    "comments": "commento sintetico sugli errori, mancanze o doc irrilevanti"
    }}

    Domanda (query):
    {query}

    Tipo di domanda: {question_type}

    Gold answer (testo):
    {gold_answer}

    Relevant docs (testo):
    {json.dumps(relevant_docs, ensure_ascii=False, indent=2)}
    """

    mess = [
            {"role": "system", "content": "Sei un assistente esperto di valutazione NLP e benchmark che deve rispondere in formato json e non aggiungere altro."},
            {"role": "user", "content": content},
        ]
    
    res = AIRequest(mess)
    res = normRes(res)

    with open("benchmarks/validation_benchmark_2.json", "a", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        f.write("\n")  # Ensure each appended JSON object is on a new line


if __name__ == "__main__":
    print("[DEBUG] loading benchmark...")
    with open("benchmarks/benchmark_revisited_2.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    print("[DEBUG] loading the chunks metadata...")
    with open("../chunks_scripts/chunks_metadata/chunks_metadata300.json", "r", encoding="utf-8") as f:
        chunks_metadata = json.load(f)

    #sistema_query(benchmark, chunks_metadata)
    
    
    query_da_sistemare = []

    for i, item in enumerate(benchmark):
        q = item["query"]
        gold_answer = item["gold_answer"]
        relevant_docs = set(item["relevant_docs"])
        question_type = item.get("question_type")
        if question_type == "ambiguous":
            print(f"[DEBUG] Skipping ambiguous question at index {i}")
            continue
        if gold_answer is None:
            print(f"[DEBUG] Skipping item {i} with no gold_answer")
            query_da_sistemare.append(i)
            continue
        
        source, chunk_id = gold_answer.split("-", 1) if gold_answer else (None, None)
        if source is None or chunk_id is None:
            print(f"[DEBUG] Invalid gold_answer format in item {i}: {gold_answer}")
            continue
        chunk_id = int(chunk_id)
       
        # cerca tra i chunk
        for chunk in chunks_metadata:
            if chunk["source"] == source and chunk["chunk_id"] == chunk_id:
                gold_answer = chunk["text"]

        resolved_relevant_docs = []
        for doc in relevant_docs:
            doc_source, doc_chunk_id = doc.split("-", 1) if doc else (None, None)
            if doc_source is None or doc_chunk_id is None:
                print(f"[DEBUG] Invalid relevant_doc format in item {i}: {doc}")
                continue
            doc_chunk_id = int(doc_chunk_id)
            for chunk in chunks_metadata:
                if chunk["source"] == doc_source and chunk["chunk_id"] == doc_chunk_id:
                    resolved_relevant_docs.append(chunk["text"])

        validation(q, gold_answer, resolved_relevant_docs, question_type)

        time.sleep(20)