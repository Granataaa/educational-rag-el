import json
from tqdm import tqdm  # Libreria per mostrare una barra di progresso
import stanza

# Carica i chunk dal file JSON
with open("chunks_metadata/chunks_metadata300_embeddings_spacy_ner.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)
print(f"Caricamento di {len(chunk_data)} chunk dal file JSON...")

stanza.download("it")  # Scarica il modello per l'italiano
nlp = stanza.Pipeline(lang="it", processors="tokenize,ner", ner_batch_size=4, use_gpu=True)  # Pipeline per l'italiano
print("Modello NER caricato.")

# Esegui il NER su ogni chunk
for i, chunk in enumerate(tqdm(chunk_data, desc="Processing chunks")):
    doc = nlp(chunk['text'])  # Crea un oggetto Sentence per il testo del chunk
    chunk['entities'] = [{"text": ent.text, "type": ent.type} for ent in doc.entities]

# Salva i chunk aggiornati in un nuovo file JSON
with open("chunks_metadata/chunks_metadata300_embeddings_stanza_ner.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)