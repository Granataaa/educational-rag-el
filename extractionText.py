import os
import re
import json

def process_transcription_file(input_file_path, output_text_dir, output_json_dir):
    """
    Elabora un singolo file di trascrizione, estraendo il testo e i timestamp.
    Salva il testo in un file .txt e i dati con i timestamp in un file .json.

    Args:
        input_file_path (str): Il percorso del file di trascrizione di input.
        output_text_dir (str): La directory dove salvare il file di solo testo.
        output_json_dir (str): La directory dove salvare il file JSON.
    """
    
    print(f"🔍 Elaborazione del file: {os.path.basename(input_file_path)}")

    # Apri il file di input e leggi tutte le righe
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    text_lines = []
    words_data = []
    in_text_section = True
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Se la riga contiene l'intestazione della tabella, entriamo nella sezione dei timestamp
        if line.startswith("PAROLA"):
            in_text_section = False
            continue
        
        # Se siamo nella sezione del testo, aggiungiamo la riga al testo estratto
        if in_text_section:
            text_lines.append(line)
        # Se siamo nella sezione dei timestamp, analizziamo la riga
        else:
            # Ignora la riga divisoria "---"
            if line.startswith("---"):
                continue

            parts = line.split()
            # Assicurati che la riga abbia il formato corretto (parola, start, end)
            if len(parts) == 3:
                word_text = parts[0]
                try:
                    # Rimuovi la 's' e converti i tempi in float
                    start_time = float(parts[1].replace("s", ""))
                    end_time = float(parts[2].replace("s", ""))
                    
                    # Aggiungi un dizionario per ogni parola alla lista
                    words_data.append({
                        "text": word_text,
                        "start": start_time,
                        "end": end_time
                    })
                except ValueError:
                    # Se la conversione fallisce, salta la riga
                    continue

    # Unisci le righe di testo in un unico paragrafo
    extracted_text = " ".join(text_lines).strip()

    # Estrai il nome del file senza estensione per usarlo nel nome dei file di output
    base_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    
    # Prepara i percorsi dei file di output
    output_text_path = os.path.join(output_text_dir, f"{base_filename}_clean.txt")
    output_json_path = os.path.join(output_json_dir, f"{base_filename}_timestamps.json")

    # Scrivi il testo estratto nel file .txt
    with open(output_text_path, 'w', encoding='utf-8') as out_file:
        out_file.write(extracted_text)
    print(f"✅ Testo salvato in: {output_text_path}")

    # Scrivi i dati con i timestamp nel file .json
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(words_data, json_file, ensure_ascii=False, indent=2)
    print(f"✅ JSON salvato in: {output_json_path}")
    print("-" * 30)


def main():
    """
    Funzione principale che esegue l'elaborazione su tutte le cartelle e i file.
    """
    # Definisci le directory di input e output
    input_dir = "transcription"
    output_text_dir = "aa_text"
    output_json_dir = "bb_time"

    # Crea le directory di output se non esistono
    os.makedirs(output_text_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    
    print("Inizio l'elaborazione dei file di trascrizione...")
    print("-" * 30)

    # Scorre le sottocartelle nella directory di input
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        
        # Assicurati che l'elemento sia una cartella
        if os.path.isdir(folder_path):
            # Scorre i file all'interno di ogni cartella
            for filename in os.listdir(folder_path):
                # Elabora solo i file .txt
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    process_transcription_file(file_path, output_text_dir, output_json_dir)

if __name__ == "__main__":
    main()