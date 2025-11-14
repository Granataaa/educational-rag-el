import spacy
import json
import numpy as np

# Carica il modello spaCy
nlp = spacy.load("it_core_news_lg")

with open("chunks_metadata/chunks_metadata300_embeddings_spacy_ner_positions.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

count = 0
for chunk in chunk_data:
    chunk["noun_chunk"] = []
    if count >= 5 :
        break
    count += 1
    doc = nlp(chunk["text"])
    nouns = [token.text for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
    chunk["nouns/propn"] = nouns
    chunk["embedding"] = None
    for x in doc.noun_chunks:
        chunk["noun_chunk"].append(x.text)

with open("chunks_metadata/chunks_metadata300_nouns.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)