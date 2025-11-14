import json

# Carica il benchmark
with open("benchmarks/benchmark_revisited_3.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Aggiungi la gold_answer ai relevant_docs se non è già presente
for item in benchmark:
    gold_answer = item.get("gold_answer")
    if gold_answer and gold_answer not in item["relevant_docs"]:
        item["relevant_docs"].append(gold_answer)

# Salva il benchmark aggiornato
with open("benchmarks/benchmark_revisited_3_fixed_corrected.json", "w", encoding="utf-8") as f:
    json.dump(benchmark, f, ensure_ascii=False, indent=2)

print("[INFO] Benchmark aggiornato salvato in benchmark_revisited_3_fixed_corrected.json")