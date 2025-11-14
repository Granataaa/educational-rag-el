import json
from rag_service import loading, query_rag # Assicurati che i tuoi import funzionino
from rag_el import loading_entity_linking, query_entity_linking_rerank, query_entity_linking_rerank_RRF, query_rag_with_cross_encoder

def print_debug_info(query_index, query, gold_docs, retrieved_results):
    """Stampa un report dettagliato per una singola query."""
    print("\n" + "="*25 + f" DEBUG QUERY #{query_index+1} " + "="*25)
    print(f"QUERY: {query}")
    print(f"DOCUMENTI RILEVANTI (GOLD): {gold_docs}")
    print("-"*70)
    
    # Stampa i QID trovati nella query (se la funzione li restituisce)
    # Questa parte va adattata se la tua funzione `query_entity_linking_rerank` non restituisce i QID della query
    # Per ora, la lasciamo come placeholder.
    # print(f"QID trovati nella query: {retrieved_results.get('query_qids', 'N/A')}")
    
    print("RISULTATI RECUPERATI E RIORDINATI:")
    if not retrieved_results:
        print("  >> Nessun risultato recuperato.")
    
    for i, res in enumerate(retrieved_results):
        doc_id = f"{res['source']}-{res['chunk_id']}"
        is_relevant = "SIUM" if doc_id in gold_docs else "NOPE"
        
        # Estrai gli score in modo sicuro
        final_score = res.get('final_score', 'N/A')
        dense_score = res.get('dense_similarity', res.get('similarity', 'N/A'))
        entity_score = res.get('entity_score', 'N/A')

        print(f"  {i+1}. {is_relevant} ID: {doc_id}")
        print(f"     Score Finale: {final_score:.4f} | Score Denso: {dense_score:.4f} | Score Entità: {entity_score:.4f}")
        print(f"     Testo: {res['text']}")
        
    print("="*70 + "\n")


# --- INIZIO SCRIPT PRINCIPALE ---

# Caricamento modello e benchmark
# loading() # Caricamento per il baseline
loading_entity_linking() # Caricamento per il sistema avanzato

with open("../benchmark_revisited_3_fixed_corrected.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Inizializzazione metriche
metrics = { "exact_match": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0, "recall@10": 0,
            "precision@1": 0, "precision@3": 0, "precision@5": 0, "precision@10": 0,
            "mrr_rev_docs": 0, "mrr_gold": 0, "total_queries": 0 }

# Ciclo sul benchmark
for i, item in enumerate(benchmark):
    q = item["query"]
    gold_answer = item["gold_answer"]
    if gold_answer is None:
        continue
    relevant_docs = set(item["relevant_docs"])

    metrics["total_queries"] += 1
    
    # --- ESEGUI LA QUERY ---
    # results = query_rag(q, 10, "false")["chunks"]
    # results = query_entity_linking_rerank(query=q, k_final=10, k_initial_retrieval=30, BETA=0.5, LLMHelp="false")["chunks"]
    # results = query_entity_linking_rerank_RRF(query=q, k_final=10, k_initial_retrieval=30, LLMHelp="false")["chunks"]
    results = query_rag_with_cross_encoder(query=q)["chunks"]
    
    retrieved_docs_ids = [f"{res['source']}-{res['chunk_id']}" for res in results]

    # Stampa il report di debug ogni 10 query
    # if (i + 1) % 10 == 0:
    # print_debug_info(i, q, relevant_docs, results)
    # entity_score = results[0]["entity_score"] if results else 0
    # print(f"Entity Score della prima risposta: {entity_score}")

    # --- CALCOLO METRICHE (invariato) ---
    if retrieved_docs_ids and retrieved_docs_ids[0] == gold_answer:
        metrics["exact_match"] += 1

    for k in [1, 3, 5, 10]:
        top_k_docs = retrieved_docs_ids[:k]
        relevant_found = sum(1 for doc in top_k_docs if doc in relevant_docs)
        metrics[f"recall@{k}"] += relevant_found / len(relevant_docs) if relevant_docs else 0
        metrics[f"precision@{k}"] += relevant_found / k

    for rank, doc in enumerate(retrieved_docs_ids, start=1):
        if doc in relevant_docs: # MRR calcolato su tutti i relevant docs, non solo gold
            metrics["mrr_rev_docs"] += 1 / rank
            break

    for rank, doc in enumerate(retrieved_docs_ids, start=1):
        if doc == gold_answer:
            metrics["mrr_gold"] += 1 / rank
            break

# Normalizzazione e salvataggio (invariato)
for key in metrics:
    if key != "total_queries":
        metrics[key] /= metrics["total_queries"]

with open("./final_results/EL_50_20_10_RRF_3.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Stampa report finale (invariato)
print("\n=== Risultati del Benchmark ===")
print(f"Query totali: {metrics['total_queries']}")
print(f"Exact Match (top-1): {metrics['exact_match']:.2%}")
for k in [1, 3, 5, 10]:
    print(f"Recall@{k}: {metrics[f'recall@{k}']:.2%}")
    print(f"Precision@{k}: {metrics[f'precision@{k}']:.2%}")
print(f"MRR (rev docs): {metrics['mrr_rev_docs']:.4f}")
print(f"MRR (solo gold): {metrics['mrr_gold']:.4f}")