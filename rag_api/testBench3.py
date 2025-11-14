from rag_service import loading, query_rag
from rag_el import loading_entity_linking, query_entity_linking_rerank, query_entity_linking_rerank_RRF, query_rag_with_cross_encoder
import json
import time
from tqdm import tqdm

# Caricamento modello e benchmark
# loading()
loading_entity_linking()
with open("../bench_scripts/benchmarks/benchmark_revisited_3_fixed_corrected.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Inizializzazione metriche
metrics = {
    "exact_match": 0,
    "recall": 0,
    "precision": 0,
    "mrr_rev_docs": 0,
    "mrr_gold": 0,
    "total_queries": 0,  # Conta tutte le query
    "total_defined_queries": 0  # Conta solo le query con gold_answer definita
}

for i, item in enumerate(tqdm(benchmark, "Processing Benchmark")):
    
    q = item["query"]
    gold_answer = item["gold_answer"]
    relevant_docs = set(item["relevant_docs"])  # Usa un set per confronti più veloci

    # Incrementa il conteggio delle query totali
    metrics["total_queries"] += 1

    # Exact Match (considera tutte le query, incluse quelle ambigue)
    retrieved_docs = []
    if gold_answer is None:
        if not retrieved_docs:  # Nessuna risposta attesa e nessuna risposta recuperata
            metrics["exact_match"] += 1
        continue  # Salta il calcolo di Recall, Precision e MRR per le domande ambigue

    # Incrementa il conteggio delle query con risposte definite
    metrics["total_defined_queries"] += 1

    # Recupera i top-10 risultati con il sistema RAG completo (dense + LLM)
    # results = query_rag(q, 10, "true")["chunks"]
    # results = query_entity_linking_rerank(query=q, k_final=10, k_initial_retrieval=30, BETA=0.5, LLMHelp="true")["chunks"]
    # results = query_entity_linking_rerank_RRF(query=q, k_final=10, k_initial_retrieval=30, LLMHelp="true")["chunks"]
    results = query_rag_with_cross_encoder(query=q, k_final_llm=10, LLMHelp="true")["chunks"]

    # Ottieni i documenti filtrati dall'LLM
    retrieved_docs = [f"{res['source']}-{res['chunk_id']}" for res in results]

    # Exact Match (solo per query con gold_answer definita)
    if retrieved_docs and retrieved_docs[0] == gold_answer:
        metrics["exact_match"] += 1

    # Recall su tutti i documenti filtrati
    relevant_found = sum(1 for doc in retrieved_docs if doc in relevant_docs)
    metrics["recall"] += relevant_found / len(relevant_docs) if relevant_docs else 0

    # Precision su tutti i documenti filtrati
    metrics["precision"] += relevant_found / len(retrieved_docs) if retrieved_docs else 0

    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs: # MRR calcolato su tutti i relevant docs, non solo gold
            metrics["mrr_rev_docs"] += 1 / rank
            break

    # MRR (considera il primo documento rilevante tra tutti)
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc == gold_answer:
            metrics["mrr_gold"] += 1 / rank
            break

    time.sleep(30)

# Normalizzazione delle metriche
metrics["exact_match"] /= metrics["total_queries"]  # Dividi per tutte le query
for key in ["recall", "precision", "mrr_gold", "mrr_rev_docs"]:
    metrics[key] /= metrics["total_defined_queries"]  # Dividi solo per le query con risposte definite

# Salvataggio risultati
with open("./final_results/test3pt2_EL_50_20_10_cross.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Stampa report
print("\n=== Risultati del Benchmark con RAG Completo ===")
print(f"Query totali: {metrics['total_queries']}")
print(f"Query con risposte definite: {metrics['total_defined_queries']}")
print(f"Exact Match (top-1): {metrics['exact_match']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"MRR: {metrics['mrr_rev_docs']:.4f}")
print(f"MRR (solo gold): {metrics['mrr_gold']:.4f}")