# Importa le librerie necessarie per il progetto
import whisper
import torch
import os
import json

# Carica il file di configurazione 'config.json'
with open("config.json", "r") as f:
    config = json.load(f)

# Controlla la disponibilità di una GPU e carica il modello Whisper
# Assicurati che una GPU sia disponibile, altrimenti il programma si interrompe
assert torch.cuda.is_available(), "GPU non rilevata!"
# Carica il modello 'turbo' di Whisper e lo sposta sulla GPU ('cuda')
model = whisper.load_model("turbo").to("cuda")

# Imposta i percorsi delle cartelle di input e output
# Percorso alla cartella che contiene le sottocartelle con i video
instance_folder = config['directoryVideo']['path']
# Percorso dove verranno salvate le trascrizioni
folder_transcription_base = "./transcription/"

# Ottiene i nomi di tutte le sottocartelle presenti nella cartella dei video
folders = [name for name in os.listdir(instance_folder) if os.path.isdir(os.path.join(instance_folder, name))]
print(f"Cartelle trovate: {folders}")

# Ciclo principale per elaborare ogni sottocartella
for folder in folders:
    # Crea il percorso di output per la sottocartella corrente
    folder_transcription = os.path.join(folder_transcription_base, folder)
    
    # Crea la cartella di trascrizione se non esiste già
    if not os.path.exists(folder_transcription):
        os.makedirs(folder_transcription)

    # Crea il percorso completo per la sottocartella dei video
    path = os.path.join(instance_folder, folder)
    # Trova tutti i file .mp4 all'interno della sottocartella
    instance_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".mp4")]
    print(f"File video da trascrivere nella cartella '{folder}': {instance_files}")

    # Ciclo per trascrivere ogni singolo video
    for video in instance_files:
        print(f"Inizio trascrizione per: {video}")
        
        # Effettua la trascrizione del video usando il modello Whisper
        result = model.transcribe(
            video,
            language="it",
            word_timestamps=True,
            verbose=False
        )

        # Prepara il nome del file di output per la trascrizione
        # Estrae il nome del file dal percorso completo
        nameFile = os.path.basename(video)
        # Rimuove l'estensione '.mp4'
        nameFile = os.path.splitext(nameFile)[0]
        # Unisce il percorso della cartella di output con il nome del file e l'estensione '.txt'
        nameFile = os.path.join(folder_transcription, nameFile + ".txt")
        print(f"Salvataggio trascrizione in: {nameFile}")

        # Scrive la trascrizione nel file di testo
        with open(nameFile, "w", encoding="utf-8") as f:
            # Scrive il testo completo della trascrizione
            f.write(result['text'] + "\n\n")
            # Scrive le intestazioni della tabella
            f.write("PAROLA".ljust(20) + "INIZIO".ljust(15) + "FINE".ljust(15) + "\n")
            f.write("-" * 50 + "\n")

            # Scrive i timestamp di ogni parola
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        line = (word['word'].ljust(20) +
                                f"{word['start']:.2f}s".ljust(15) +
                                f"{word['end']:.2f}s".ljust(15))
                        f.write(line + "\n")
        print("Trascrizione completata.")