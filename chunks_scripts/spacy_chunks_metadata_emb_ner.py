import json
import spacy
from tqdm import tqdm

# Carica i chunk dal file JSON
# Assicurati che il file di input sia quello corretto
with open("chunks_metadata/chunks_metadata300_embeddings_spacy_ner.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)
print(f"Caricamento di {len(chunk_data)} chunk dal file JSON...")

# Carica il modello di spaCy
nlp = spacy.load("it_core_news_lg")
print("Modello spaCy caricato.")

# Esegui il NER su ogni chunk
for chunk in tqdm(chunk_data, desc="Processing chunks"):
    doc = nlp(chunk['text'])
    # Aggiungi start_char e end_char al dizionario di ogni entità
    chunk['entities'] = [
        {
            "text": ent.text,
            "type": ent.label_, # ent.label_ è l'attributo corretto in spaCy
            "start_char": ent.start_char,
            "end_char": ent.end_char
        } 
        for ent in doc.ents
    ]

# Salva i chunk aggiornati in un nuovo file JSON
with open("chunks_metadata/chunks_metadata300_embeddings_spacy_ner_positions.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

print("\nElaborazione completata. I risultati sono stati salvati in 'chunks_metadata300_embeddings_spacy_ner_positions.json'")