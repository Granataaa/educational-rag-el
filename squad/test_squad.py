import json
from tqdm import tqdm
from rag_el_squad import (
    loading_entity_linking,
    query_entity_linking_rerank,
    query_entity_linking_rerank_RRF,
    query_rag_with_cross_encoder,
    query_rag
)

# --- INIZIO SCRIPT PRINCIPALE ---

loading_entity_linking()  # Carica index + chunk_data_linked su SQuAD-it

# Carica il benchmark generato da SQuAD-it
with open("squad_it_queries.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

metrics = {
    "exact_match": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0, "recall@10": 0,
    "precision@1": 0, "precision@3": 0, "precision@5": 0, "precision@10": 0,
    "mrr": 0, "total_queries": 0
}

for i, item in enumerate(tqdm(benchmark, "valutazione benchmark SQuAD-it")):
    q = item["question"]
    # se dentro relevant_docs c'è un numero >= 200, esce dal ciclo
    # if any(doc_id >= 200 for doc_id in item["relevant_docs"]):
    #     print("arrivati a 200, esco")
    #     break

    relevant_docs = set([f"squad_it-{doc_id}" for doc_id in item["relevant_docs"]])
    # gold_answer = item.get("gold_answer", None)

    metrics["total_queries"] += 1

    # --- ESEGUI LA QUERY ---
    # results = query_entity_linking_rerank(query=q, k_final=10, BETA=0.5, LLMHelp="false")["chunks"]
    # results = query_entity_linking_rerank_RRF(query=q, k_final=10, k_initial_retrieval=30, LLMHelp="false")["chunks"]
    results = query_rag_with_cross_encoder(query=q)["chunks"]
    # results = query_rag(q, 10, "false")["chunks"]

    retrieved_docs_ids = [f"{res['source']}-{res['chunk_id']}" for res in results]

    # --- CALCOLO METRICHE ---
    if retrieved_docs_ids and retrieved_docs_ids[0] in relevant_docs:
        metrics["exact_match"] += 1

    for k in [1, 3, 5, 10]:
        top_k_docs = retrieved_docs_ids[:k]
        relevant_found = sum(1 for doc in top_k_docs if doc in relevant_docs)
        metrics[f"recall@{k}"] += relevant_found / len(relevant_docs) if relevant_docs else 0
        metrics[f"precision@{k}"] += relevant_found / k

    for rank, doc in enumerate(retrieved_docs_ids, start=1):
        if doc in relevant_docs:  # MRR su relevant docs
            metrics["mrr"] += 1 / rank
            break

    # for rank, doc in enumerate(retrieved_docs_ids, start=1):
    #     if gold_answer and doc == gold_answer:
    #         metrics["mrr_gold"] += 1 / rank
    #         break

# Normalizzazione
for key in metrics:
    if key != "total_queries":
        metrics[key] /= metrics["total_queries"]

with open("squad_it_results_all_cross_50_20_10.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Stampa report finale
print("\n=== Risultati del Benchmark su SQuAD-it ===")
print(f"Query totali: {metrics['total_queries']}")
print(f"Exact Match (top-1): {metrics['exact_match']:.2%}")
for k in [1, 3, 5, 10]:
    print(f"Recall@{k}: {metrics[f'recall@{k}']:.2%}")
    print(f"Precision@{k}: {metrics[f'precision@{k}']:.2%}")
print(f"MRR: {metrics['mrr']:.4f}")