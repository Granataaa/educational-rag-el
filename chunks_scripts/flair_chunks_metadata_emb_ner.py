import json
from tqdm import tqdm  # Libreria per mostrare una barra di progresso
from flair.models import SequenceTagger
from flair.data import Sentence

# Carica i chunk dal file JSON
with open("chunks_metadata/chunks_metadata300_embeddings_spacy_ner.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)
print(f"Caricamento di {len(chunk_data)} chunk dal file JSON...")

# Carica il modello NER multilingue di Flair
tagger = SequenceTagger.load("ner-multi")  # Modello multilingue
print("Modello NER caricato.")

# Esegui il NER su ogni chunk
for i, chunk in enumerate(tqdm(chunk_data, desc="Processing chunks")):
    sentence = Sentence(chunk['text'])  # Crea un oggetto Sentence per il testo del chunk
    tagger.predict(sentence)  # Predici le entità
    # Estrai le entità
    chunk['entities'] = [{"text": entity.text, "type": entity.tag, "score": float(entity.score)} for entity in sentence.get_spans("ner")]

# Salva i chunk aggiornati in un nuovo file JSON
with open("chunks_metadata/chunks_metadata300_embeddings_flair_ner.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, ensure_ascii=False, indent=2)