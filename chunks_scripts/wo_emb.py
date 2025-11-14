import json

with open("chunks_metadata/chunks_linked_entities_emb_final.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

for chunk in chunk_data:
    chunk["embedding"] = None

with open("chunks_metadata/chunks_linked_entities_emb_final_noemb.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)