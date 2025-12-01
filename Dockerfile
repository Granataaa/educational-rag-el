# --- FASE 1: Preparazione dell'ambiente CUDA e Python con Conda ---
# Usiamo un'immagine PyTorch che include già CUDA e Conda.
# Scegliamo una versione che sia compatibile con Python 3.12 (se disponibile, altrimenti una 3.10 o 3.11).
# Controlla sempre le immagini disponibili su Docker Hub di PyTorch per la versione più adatta.
# Per esempio, potresti usare "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel" o simile.
# Cerchiamo di avvicinarci il più possibile a Python 3.12.7. Se non trovi una 3.12, una 3.10/3.11 andrà bene.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Imposta la directory di lavoro all'interno del container. Qui verranno copiati i tuoi file.
WORKDIR /app

# Copia il file environment.yml nel container.
COPY environment_crossplatform.yml .

# Crea l'ambiente Conda usando il file environment.yml
# '-f environment.yml' specifica il file da usare
# '--prefix /opt/conda/envs/myenv' specifica dove creare l'ambiente (così non interferisce con l'ambiente base)
# '--yes' per accettare automaticamente tutte le richieste.
# Questo passaggio può richiedere tempo a seconda delle dipendenze.
RUN conda env create -f environment_crossplatform.yml --prefix /opt/conda/envs/myenv && \
    conda clean --all -f -y

# Attiva l'ambiente Conda per le operazioni successive.
# Questo modifica la variabile PATH in modo che i comandi come 'python' usino l'interprete dell'ambiente 'myenv'.
ENV PATH="/opt/conda/envs/myenv/bin:$PATH"

RUN python -m spacy download it_core_news_lg-3.8.0 --direct --no-deps

# --- FASE 2: Preparazione dell'ambiente Node.js per React ---
# Installeremo Node.js usando un gestore pacchetti (apt-get)
# Aggiorna l'indice dei pacchetti e installa 'curl' e 'nodejs'
# Curl ci servirà per scaricare il setup di Node.js.
# Usiamo una versione LTS (Long Term Support) di Node.js, per esempio 18.x o 20.x
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Verifica che Node.js e npm siano installati e nel PATH subito dopo l'installazione.
# Se questo fallisce, ci darà un errore più specifico.
# L'output di "which npm" mostrerà il percorso di npm.
RUN which node && which npm

# Assicurati che il PATH dell'ambiente Conda sia attivo
ENV PATH="/opt/conda/envs/myenv/bin:$PATH"

# --- FASE 3: Copia del codice del progetto e installazione delle dipendenze React ---

# Copia tutto il resto del progetto nella directory di lavoro /app
# Assicurati che il tuo .dockerignore (se presente) escluda directory come node_modules o .git
COPY . .

# Vai nella directory del client React e installa le dipendenze
# Ho notato dall'output che la tua directory React è `/reactApi/react-client`.
# Assicurati che il percorso sia CORRETTO per la tua struttura.
WORKDIR /app/reactApi/react-client 
# CORREGGI QUI CON IL TUO PATH ESATTO

# Aggiungiamo un'istruzione RUN separata per debug.
# Questo ci dirà se npm è nel PATH in questo preciso punto.
RUN echo "Checking npm path before install..." && which npm && npm --version

# Ora esegui npm install
RUN npm install

# Build dell'applicazione React (opzionale ma consigliato per deployment di produzione)
RUN npm run build

# Torna alla directory radice del progetto
WORKDIR /app

# --- FASE 4: Configurazione e Comandi di Avvio ---

# Espone le porte che il server Python e il server React useranno.
# Assicurati che queste porte corrispondano a quelle usate dal tuo codice.
EXPOSE 5005
# Esempio per il server Python
EXPOSE 3000 
# Esempio per il server React (se usi 'npm start' direttamente nel container)

# Comando per avviare l'applicazione.
# Useremo 'CMD' per specificare il comando predefinito quando il container viene avviato.
# Per un'applicazione completa con server Python e client React, potresti voler usare 'supervisord'
# o un semplice script bash per avviarli entrambi.
# Per semplicità, inizialmente ti mostro come avviare il server Python.
# Avvia il server Python
CMD ["tail", "-f", "/dev/null"]
# CMD ["python", "server.py"]

# Se devi avviare sia il server Python che il client React, ti consiglio di creare un piccolo script bash:
# Esempio di script chiamato 'start_app.sh':
# #!/bin/bash
#
# # Avvia il server Python in background
# python server.py &
#
# # Avvia il client React (potrebbe essere necessario andare nella sua directory)
# cd client && npm start
#
# # Mantiene il container in esecuzione
# tail -f /dev/null
#
# Poi nel Dockerfile useresti:
# COPY start_app.sh .
# RUN chmod +x start_app.sh
# CMD ["./start_app.sh"]