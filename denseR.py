# # Importa le librerie necessarie per il progetto
# import os
# import json
# import re
# import faiss
# import numpy as np
# import spacy
# from sentence_transformers import SentenceTransformer
# from difflib import SequenceMatcher
# from tqdm import tqdm

# def chunk_text(text, max_tokens=300, min_tokens=20):
#     """
#     Suddivide un testo in chunk (blocchi) di dimensioni specificate, rispettando i confini delle frasi.
    
#     Args:
#         text (str): Il testo da suddividere.
#         max_tokens (int, optional): Numero massimo di token per chunk. Default: 300.
#         min_tokens (int, optional): Numero minimo di token per chunk. Default: 20.
    
#     Returns:
#         list: Una lista di chunk di testo.
#     """
#     doc = nlp(text)
#     chunks = []
#     current_chunk = []
#     current_len = 0

#     for sent in doc.sents:
#         token_len = len(sent.text.split())
        
#         if current_len + token_len <= max_tokens:
#             current_chunk.append(sent.text)
#             current_len += token_len
#         else:
#             chunk_text_str = " ".join(current_chunk)
#             if len(chunk_text_str.split()) >= min_tokens:
#                 chunks.append(chunk_text_str)
            
#             current_chunk = [sent.text]
#             current_len = token_len

#     if current_chunk:
#         chunk_text_str = " ".join(current_chunk)
#         if len(chunk_text_str.split()) >= min_tokens:
#             chunks.append(chunk_text_str)

#     return chunks

# def normalized(word):
#     """Normalizza una parola rimuovendo punteggiatura e convertendo in minuscolo."""
#     return word.strip(".,?!;:").lower()

# def similarity(a, b):
#     """Calcola la similarità tra due stringhe usando il rapporto di SequenceMatcher."""
#     return SequenceMatcher(None, a, b).ratio()

# def match_chunk_with_timestamps_fuzzy(chunk_text, word_list, prev, tolerance=0.7):
#     """
#     Trova il match approssimato tra un chunk e una lista di parole con timestamp.
    
#     Restituisce: (start_time, end_time) o (None, None) se nessun match trovato.
#     """
#     chunk_words = [normalized(w) for w in chunk_text.split()]
#     if not chunk_words:
#         return None, None
    
#     transcript_words = [normalized(w["text"]) for w in word_list]
#     max_score = 0
#     best_match_idx = None

#     for i in range(len(transcript_words) - len(chunk_words) + 1):
#         window = transcript_words[i:i + len(chunk_words)]
#         score = sum(similarity(w1, w2) for w1, w2 in zip(chunk_words, window)) / len(chunk_words)

#         if score > max_score:
#             max_score = score
#             best_match_idx = i

#     if max_score >= tolerance and best_match_idx is not None:
#         start_time = word_list[best_match_idx]["start"]
#         end_time_idx = best_match_idx + len(chunk_words) - 1
#         # Controlla che l'indice non superi la lunghezza della lista
#         if end_time_idx >= len(word_list):
#              end_time_idx = len(word_list) - 1
#         end_time = word_list[end_time_idx]["end"]
        
#         return start_time, end_time
    
#     # Se il matching fallisce, restituisce i timestamp in base al chunk precedente
#     return prev, None

# def match_chunk_exact(chunk_text, word_list, prev, lookahead=5):
#     """
#     Cerca un match esatto delle prime N parole del chunk nella trascrizione.
    
#     Restituisce: (start_time, end_time) o (None, None) se non trovato.
#     """
#     chunk_words = [normalized(w) for w in chunk_text.split()[:lookahead] if w.strip()]
    
#     if not chunk_words:
#         return None, None
    
#     transcript_words = [normalized(w["text"]) for w in word_list]
    
#     for i in range(len(transcript_words) - len(chunk_words) + 1):
#         if transcript_words[i:i+len(chunk_words)] == chunk_words:
#             start_idx = i
#             end_idx = i + len(chunk_text.split()) - 1
            
#             # Controlla che l'indice non superi la lunghezza della lista
#             if end_idx >= len(word_list):
#                  end_idx = len(word_list) - 1
            
#             start_time = word_list[start_idx]["start"]
#             end_time = word_list[end_idx]["end"]
            
#             # Se il timestamp di inizio è precedente a quello del chunk precedente, lo aggiusta
#             if start_time < prev:
#                 start_time = prev
            
#             return start_time, end_time

#     return prev, None

# def match_chunk_hybrid(chunk_text, word_list, prev, exact_lookahead=5, fuzzy_tolerance=0.7):
#     """
#     Combina matching esatto e fuzzy per trovare il miglior abbinamento di un chunk.
    
#     Restituisce: (start_time, end_time) o (None, None).
#     """
#     # Prova prima il matching esatto
#     start, end = match_chunk_exact(chunk_text, word_list, prev, lookahead=exact_lookahead)
#     if end is not None:
#         return start, end
    
#     # Se fallisce, prova il matching fuzzy
#     return match_chunk_with_timestamps_fuzzy(chunk_text, word_list, prev, tolerance=fuzzy_tolerance)

# def fill_missing_timestamps(all_chunk_data):
#     """
#     Riempie i timestamp mancanti nei metadati dei chunk usando i chunk vicini.
#     """
#     for i, chunk_data in enumerate(all_chunk_data):
#         if chunk_data["start_time"] is None:
#             if i > 0 and all_chunk_data[i - 1]["end_time"] is not None:
#                 chunk_data["start_time"] = all_chunk_data[i - 1]["end_time"]
#             else:
#                 chunk_data["start_time"] = 0.0 # Imposta a 0 se è il primo chunk
        
#         if chunk_data["end_time"] is None:
#             if i < len(all_chunk_data) - 1 and all_chunk_data[i + 1]["start_time"] is not None:
#                 chunk_data["end_time"] = all_chunk_data[i + 1]["start_time"]
#             else:
#                 # Se è l'ultimo chunk, usiamo il timestamp di inizio + 10s come stima
#                 if chunk_data["start_time"] is not None:
#                     chunk_data["end_time"] = chunk_data["start_time"] + 10.0


# def main():
#     """
#     Funzione principale per generare embeddings, chunk e indice FAISS.
#     """
#     # Inizializza i modelli necessari
#     print("⏳ Caricamento del modello Sentence-Transformer...")
#     model = SentenceTransformer('intfloat/multilingual-e5-large')
#     print("⏳ Caricamento del modello linguistico spaCy...")
#     global nlp
#     nlp = spacy.load("it_core_news_lg")
    
#     # Inizializzazione delle liste per i chunk e i metadati
#     all_chunks = []
#     all_chunk_data = []
    
#     # Directory di input per i file di testo e i file JSON con timestamp
#     input_text_dir = "lesson_text"
#     input_timestamps_dir = "lesson_timestamps"

#     print("🚀 Inizio dell'elaborazione dei file...")
#     print("-" * 50)
    
#     # Itera attraverso tutti i file nella directory di testo
#     for filename in tqdm(os.listdir(input_text_dir),"Processing text files"):
#         if filename.endswith(".txt"):
#             # print(f"🔄 Elaborazione del file: {filename}")
#             input_path = os.path.join(input_text_dir, filename)
            
#             # Leggi il testo dal file .txt
#             with open(input_path, "r", encoding="utf-8") as f:
#                 long_text = f.read()

#             # Suddividi il testo in chunk
#             chunks = chunk_text(long_text)
#             all_chunks.extend(chunks)

#             # Carica il file JSON con i timestamp corrispondente
#             timestamp_filename = filename.replace("_clean.txt", "_timestamps.json")
#             timestamp_path = os.path.join(input_timestamps_dir, timestamp_filename)
            
#             if not os.path.exists(timestamp_path):
#                 print(f"⚠️ File JSON non trovato per {filename}. Ignoro...")
#                 continue
                
#             with open(timestamp_path, "r", encoding="utf-8") as f:
#                 words = json.load(f)
            
#             prev = 0
#             for i, chunk_text_str in enumerate(chunks):
#                 # Trova i timestamp per il chunk usando la strategia ibrida
#                 start, end = match_chunk_hybrid(chunk_text_str, words, prev)
                
#                 # Aggiorna il timestamp precedente per il prossimo chunk
#                 if end is not None:
#                     prev = end
#                 else:
#                     prev = start

#                 # Estrai le entità nominate con spaCy
#                 # entities = [{"text": ent.text, "label": ent.label_} for ent in nlp(chunk_text_str).ents]

#                 entities = [
#                     {
#                         "text": ent.text,
#                         "type": ent.label_, 
#                         "start_char": ent.start_char,
#                         "end_char": ent.end_char
#                     } 
#                     for ent in nlp(chunk_text_str).ents
#                 ]
                
#                 # Aggiungi i metadati per il chunk
#                 all_chunk_data.append({
#                     "text": chunk_text_str,
#                     "source": filename,
#                     "chunk_id": i,
#                     "start_time": start,
#                     "end_time": end,
#                     "entities": entities
#                 })
#     print("-" * 50)
#     print("✅ Elaborazione dei file completata.")

#     print("⏳ Riempimento dei timestamp mancanti...")
#     fill_missing_timestamps(all_chunk_data)
#     print("✅ Timestamp mancanti riempiti.")

#     # Genera gli embeddings
#     print("⏳ Generazione degli embeddings...")
#     embeddings = model.encode(all_chunks, normalize_embeddings=True).astype("float32")
#     print(f"✅ Embeddings generati. Dimensione: {embeddings.shape}")
    
#     # Crea e salva l'indice FAISS
#     print("⏳ Creazione dell'indice FAISS...")
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dimension)
#     index.add(embeddings)
    
#     faiss_index_file = "faiss_index_inginf.index"
#     faiss.write_index(index, faiss_index_file)
#     print(f"✅ Indice FAISS salvato in: {faiss_index_file}")

#     # Salva i metadati
#     metadata_file = "chunks_metadata_inginf.json"
#     with open(metadata_file, "w", encoding="utf-8") as f:
#         json.dump(all_chunk_data, f, ensure_ascii=False, indent=2)
#     print(f"✅ Metadati dei chunk salvati in: {metadata_file}")

# if __name__ == "__main__":
#     main()


import os
import json
import re
import numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm
from multiprocessing import Pool

# --- NOTA BENE ---
# Non importiamo qui sentence_transformers, faiss o torch.
# Li importeremo solo dentro il main() per evitare che i processi figli
# cerchino di caricare la GPU o modelli pesanti inutilmente.

# --- Variabili Globali per i Worker ---
nlp = None

def init_worker():
    """Carica spaCy solo nei processi worker."""
    global nlp
    import spacy
    # Carichiamo il modello spaCy. 
    # Se hai problemi di memoria, usa "it_core_news_sm" invece di "lg"
    try:
        nlp = spacy.load("it_core_news_lg")
    except OSError:
        print("⚠️ Modello 'lg' non trovato o memoria insufficiente. Provo con 'sm'.")
        nlp = spacy.load("it_core_news_sm")

# --- Funzioni di Supporto ---

def chunk_text(text, max_tokens=300, min_tokens=20):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in doc.sents:
        token_len = len(sent)
        if current_len + token_len <= max_tokens:
            current_chunk.append(sent.text)
            current_len += token_len
        else:
            chunk_text_str = " ".join(current_chunk)
            if len(chunk_text_str.split()) >= min_tokens:
                chunks.append(chunk_text_str)
            current_chunk = [sent.text]
            current_len = token_len

    if current_chunk:
        chunk_text_str = " ".join(current_chunk)
        if len(chunk_text_str.split()) >= min_tokens:
            chunks.append(chunk_text_str)
    return chunks

def normalized(word):
    return word.strip(".,?!;:").lower()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def match_chunk_with_timestamps_fuzzy(chunk_text, word_list, prev, tolerance=0.7):
    chunk_words = [normalized(w) for w in chunk_text.split()]
    if not chunk_words:
        return None, None
    
    transcript_words = [normalized(w["text"]) for w in word_list]
    max_score = 0
    best_match_idx = None
    
    # Ottimizzazione: limitiamo la ricerca se la trascrizione è enorme
    search_limit = len(transcript_words) - len(chunk_words) + 1
    
    for i in range(search_limit):
        window = transcript_words[i:i + len(chunk_words)]
        if not window: continue
        
        # Piccola euristica: se la prima parola è completamente diversa, salta (velocizza molto)
        if chunk_words[0] != window[0] and similarity(chunk_words[0], window[0]) < 0.5:
             continue

        score = sum(similarity(w1, w2) for w1, w2 in zip(chunk_words, window)) / len(chunk_words)

        if score > max_score:
            max_score = score
            best_match_idx = i
        
        # Se troviamo un match quasi perfetto, ci fermiamo subito (early exit)
        if max_score > 0.95:
            break

    if max_score >= tolerance and best_match_idx is not None:
        start_time = word_list[best_match_idx]["start"]
        end_time_idx = min(best_match_idx + len(chunk_words) - 1, len(word_list) - 1)
        end_time = word_list[end_time_idx]["end"]
        return start_time, end_time
    
    return prev, None

def match_chunk_exact(chunk_text, word_list, prev, lookahead=5):
    chunk_words = [normalized(w) for w in chunk_text.split()[:lookahead] if w.strip()]
    if not chunk_words: return None, None
    
    transcript_words = [normalized(w["text"]) for w in word_list]
    
    for i in range(len(transcript_words) - len(chunk_words) + 1):
        if transcript_words[i:i+len(chunk_words)] == chunk_words:
            start_idx = i
            end_idx = min(i + len(chunk_text.split()) - 1, len(word_list) - 1)
            return word_list[start_idx]["start"], word_list[end_idx]["end"]

    return prev, None

def match_chunk_hybrid(chunk_text, word_list, prev, exact_lookahead=5, fuzzy_tolerance=0.7):
    start, end = match_chunk_exact(chunk_text, word_list, prev, lookahead=exact_lookahead)
    if end is not None: return start, end
    return match_chunk_with_timestamps_fuzzy(chunk_text, word_list, prev, tolerance=fuzzy_tolerance)

def fill_missing_timestamps(all_chunk_data):
    for i, chunk_data in enumerate(all_chunk_data):
        if chunk_data["start_time"] is None:
            chunk_data["start_time"] = all_chunk_data[i - 1]["end_time"] if i > 0 and all_chunk_data[i - 1]["end_time"] is not None else 0.0
        
        if chunk_data["end_time"] is None:
            if i < len(all_chunk_data) - 1 and all_chunk_data[i + 1]["start_time"] is not None:
                chunk_data["end_time"] = all_chunk_data[i + 1]["start_time"]
            else:
                chunk_data["end_time"] = (chunk_data["start_time"] + 10.0) if chunk_data["start_time"] is not None else 10.0

# --- Worker Function ---

def process_single_file(args):
    filename, input_text_dir, input_timestamps_dir = args
    input_path = os.path.join(input_text_dir, filename)
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            long_text = f.read()
    except Exception:
        return [], []

    chunks = chunk_text(long_text)
    
    timestamp_filename = filename.replace("_clean.txt", "_timestamps.json")
    timestamp_path = os.path.join(input_timestamps_dir, timestamp_filename)
    
    words = []
    if os.path.exists(timestamp_path):
        try:
            with open(timestamp_path, "r", encoding="utf-8") as f:
                words = json.load(f)
        except Exception: pass

    file_chunks = []
    file_chunk_data = []
    prev = 0.0
    
    for i, chunk_text_str in enumerate(chunks):
        start, end = match_chunk_hybrid(chunk_text_str, words, prev)
        if end is not None: prev = end
        elif start is not None: prev = start

        doc_ent = nlp(chunk_text_str)
        entities = [{"text": ent.text, "type": ent.label_, "start_char": ent.start_char, "end_char": ent.end_char} for ent in doc_ent.ents]
        
        file_chunks.append(chunk_text_str)
        file_chunk_data.append({
            "text": chunk_text_str, "source": filename, "chunk_id": i,
            "start_time": start, "end_time": end, "entities": entities
        })
        
    return file_chunks, file_chunk_data

# --- MAIN ---

def main():
    # Importiamo qui le librerie pesanti per non intasare i processi figli
    import faiss
    from sentence_transformers import SentenceTransformer

    input_text_dir = "lesson_text"
    input_timestamps_dir = "lesson_timestamps"
    
    files_to_process = [f for f in os.listdir(input_text_dir) if f.endswith(".txt")]
    worker_args = [(f, input_text_dir, input_timestamps_dir) for f in files_to_process]
    
    # --- PUNTO CRITICO: NUMERO DI PROCESSI ---
    # Impostiamo max 4 processi. Se hai 16GB di RAM o meno, prova anche 2.
    # Questo previene l'errore "paging file too small".
    NUM_WORKERS = 4 
    
    print(f"🚀 Inizio elaborazione parallela (max {NUM_WORKERS} processi)...")
    
    all_chunks = []
    all_chunk_data = []

    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, worker_args), 
            total=len(files_to_process),
            desc="Preprocessing (CPU)"
        ))

    print("⏳ Aggregazione risultati...")
    for chunks, metadata in results:
        all_chunks.extend(chunks)
        all_chunk_data.extend(metadata)

    print(f"✅ Preprocessing completato. Chunk totali: {len(all_chunks)}")
    fill_missing_timestamps(all_chunk_data)
    
    print("⏳ Caricamento modello GPU...")
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    print("⏳ Generazione embeddings...")
    embeddings = model.encode(all_chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=True).astype("float32")
    
    print("⏳ Creazione indice FAISS...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, "faiss_index_inginf.index")
    with open("chunks_metadata_inginf.json", "w", encoding="utf-8") as f:
        json.dump(all_chunk_data, f, ensure_ascii=False, indent=2)

    print("🎉 Finito!")

if __name__ == "__main__":
    main()