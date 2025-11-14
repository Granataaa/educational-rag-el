import json

# Carica i file
with open("benchmarks/benchmark_revisited_1.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

with open("benchmarks/validation_benchmark.json", "r", encoding="utf-8") as f:
    validation = json.load(f)

# Aggiorna il benchmark
validation_index = 0  # Indice per scorrere il file di validazione
for i, benchmark_item in enumerate(benchmark):
    # Salta le domande ambigue
    if benchmark_item.get("question_type") == "ambiguous":
        print(f"[DEBUG] Skipping ambiguous question at index {i}")
        continue

    # Controlla se ci sono abbastanza elementi nella validazione
    if validation_index >= len(validation):
        print(f"[WARNING] Validation index {validation_index} exceeds validation size.")
        break

    validation_item = validation[validation_index]

    # Aggiorna gold_answer se non è valida
    if validation_item.get("gold_valid") == "No":
        print(f"[DEBUG] Invalid gold_answer for query {benchmark_item['query']}")
        benchmark_item["gold_answer"] = None

    # Filtra relevant_docs basandosi su relevant_docs_valid
    if "relevant_docs_valid" in validation_item:
        valid_docs = [
            doc for doc, is_valid in zip(
                benchmark_item["relevant_docs"], validation_item["relevant_docs_valid"]
            ) if is_valid == "Yes"
        ]
        benchmark_item["relevant_docs"] = valid_docs

    # Incrementa l'indice di validazione solo per domande non ambigue
    validation_index += 1

# Salva il benchmark aggiornato
with open("benchmarks/benchmark_revisited_1_fixed.json", "w", encoding="utf-8") as f:
    json.dump(benchmark, f, ensure_ascii=False, indent=2)

print("[INFO] Benchmark aggiornato salvato in benchmark_revisited_1_fixed.json")
