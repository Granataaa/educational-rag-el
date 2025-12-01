# --- FASE 1: Preparazione dell'ambiente CUDA e Python con Conda ---
#FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
WORKDIR /app

# Copia environment.yml e crea l'ambiente Conda
COPY environment_crossplatform.yml .
RUN conda env create -f environment_crossplatform.yml --prefix /opt/conda/envs/myenv && \
    conda clean --all -f -y

# Attiva l'ambiente Conda nel PATH
ENV PATH="/opt/conda/envs/myenv/bin:$PATH"

# Installa le dipendenze per la nuova API FastAPI
COPY rag_api/requirements_fastapi.txt .
RUN pip install -r requirements_fastapi.txt

# Scarica il modello Spacy
RUN python -m spacy download it_core_news_lg-3.8.0 --direct --no-deps

# --- FASE 2: Preparazione dell'ambiente Node.js per React ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# --- FASE 3: Copia del codice e installazione dipendenze React ---
COPY . .

WORKDIR /app/reactApi/react-client 
RUN npm install
RUN npm run build

# Torna alla root
WORKDIR /app

# Espone le porte (documentazione)
EXPOSE 5005
EXPOSE 5006
EXPOSE 3000 

# Comando di default (verrà sovrascritto da docker-compose)
CMD ["tail", "-f", "/dev/null"]