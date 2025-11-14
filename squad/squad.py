from datasets import load_dataset
import pandas as pd

# 1. Carica il dataset SQuAD-it
dataset = load_dataset("squad_it")

# Usiamo solo il train split per semplicità, ma puoi combinare anche con validation
train_data = dataset["train"]

# 2. Creiamo la lista di documenti unici (contexts)
docs = []
seen_contexts = {}
doc_id = 0

for item in train_data:
    context = item["context"]
    if context not in seen_contexts:
        seen_contexts[context] = doc_id
        docs.append({
            "chunk_id": doc_id,
            "text": context,
            "source": "squad_it"
        })
        doc_id += 1

# 3. Creiamo il benchmark di query → relevant_docs
queries = []
for item in train_data:
    context = item["context"]
    q = item["question"]
    rel_doc = seen_contexts[context]  # mappa al chunk_id corretto
    
    queries.append({
        "question": q,
        "relevant_docs": [rel_doc],  # lista, anche se uno solo
        "gold_answer": item["answers"]["text"][0]  # opzionale, per QA evaluation
    })

# 4. Converti in DataFrame o salva su file
df_docs = pd.DataFrame(docs)
df_queries = pd.DataFrame(queries)

# Salva su disco se vuoi
df_docs.to_json("squad_it_docs.json", orient="records", indent=2, force_ascii=False)
df_queries.to_json("squad_it_queries.json", orient="records", indent=2, force_ascii=False)

print("Numero di documenti:", len(df_docs))
print("Numero di query:", len(df_queries))
