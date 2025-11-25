<!-- # Istruzioni per l'Utilizzo del Progetto

Questo documento offre due set di istruzioni per l'avvio e l'utilizzo del progetto:

  * **Avvio con Conda (Consigliato per lo Sviluppo Locale):** Ideale per chi vuole eseguire i servizi direttamente sulla propria macchina, utilizzando un ambiente virtuale Conda. Offre maggiore flessibilità per il debug e lo sviluppo.
  * **Avvio con Docker (Consigliato per la Produzione e la Portabilità):** Utilizza un container Docker per creare un ambiente isolato e standardizzato. Questo approccio garantisce che l'applicazione funzioni allo stesso modo su qualsiasi sistema, eliminando problemi di dipendenze. ATTENZIONE: l'immagine docker pesa 73GB.

-----

## Prerequisiti

### Per l'Avvio con Conda

1.  **Conda** installato.
2.  **React** e **npm** installati.

### Per l'Avvio con Docker

1.  **Docker Desktop** installato e configurato (con WSL 2 su Windows).
2.  Per l'accelerazione GPU, assicurati di avere i **driver NVIDIA** e il **runtime Docker** configurati correttamente.

-----

## Configurazione Iniziale

### 1\. Configurare le Variabili d'Ambiente

Crea un file `.env` nella root del progetto (ad esempio, nella directory `rag_api`) con le seguenti variabili per l'API di OpenAI:

```ini
OPENAI_API_KEY=your_api_key_here
ORGANIZATION=your_organization_here
PROJECT=your_project_here
URL=your_api_url_here
```

### 2\. Configurare i File JSON

Modifica i seguenti file di configurazione con le specifiche del tuo ambiente.

  * **config.json** (nella directory principale):
      * **Per Conda:** `host` dovrebbe essere `"localhost"`.
      * **Per Docker:** `host` deve essere **"0.0.0.0"**.
  * **reactApi/react-client/public/config.json**:
      * **Sia per Conda che per Docker:** `api_base_url` deve essere `"http://localhost:5005"`.

### 3\. Preparare i File Video

Scarica i video e posizionali in due cartelle:

1.  `reactApi/react-client/public/video/nomeCorso`
2.  La directory specificata nel `path` di `config.json`.

-----

## Elaborazione dei Dati (Speech-to-Text, Embeddings)

Se devi eseguire nuovamente lo speech-to-text e generare gli embeddings, utilizza i seguenti script:

1.  `whisperXuniNet.ipynb`: Speech-to-Text
2.  `extractionText.ipynb`: Post-elaborazione testi (obbligatorio dopo Whisper)
3.  `denseR.ipynb`: Generazione embeddings

-----

## Avvio del Progetto

### Avvio con Conda

1.  **Configurare l'Ambiente Conda:**
    A seconda del tuo sistema operativo, importa l'ambiente virtuale corretto.

    ```bash
    # Su Windows
    conda env create -f environment.yml

    # Su altri sistemi
    conda env create -f environment_crossplatform.yml
    ```

    > **Nota:** Se necessario, modifica il file `.yml` per specificare la versione corretta di **CUDA Toolkit** compatibile con la tua GPU (`nvcc --version` o `nvidia-smi`).

2.  **Avviare il Server API:**

    ```bash
    cd rag_api
    python server.py
    ```

3.  **Avviare il Client React:**
    Apri un nuovo terminale:

    ```bash
    cd reactApi/react-client
    npm start
    ```

    Il progetto sarà accessibile su `http://localhost:3000`.

### Avvio con Docker

1.  **Costruire l'Immagine Docker:**
    Nella directory principale del progetto, esegui il seguente comando.

    ```bash
    docker build -t nome_immagine .
    ```

2.  **Avviare il Container:**
    Avvia il container con la mappatura delle porte e l'accesso alla GPU.

    ```bash
    docker run -it --rm \
    -p 5005:5005 \
    -p 3000:3000 \
    --gpus all \
    --env-file ./rag_api/.env \
    nome_immagine
    ```

3.  **Eseguire i Servizi nel Container:**
    Dopo aver avviato il container, apri **due terminali aggiuntivi** per connetterti al container. Usa `docker ps` per trovare il suo nome o ID.

      * **Terminale 1 (per il server API):**
        ```bash
        docker exec -it <CONTAINER_ID_O_NOME> bash
        cd rag_api
        python server.py
        ```
      * **Terminale 2 (per il client React):**
        ```bash
        docker exec -it <CONTAINER_ID_O_NOME> bash
        cd reactApi/react-client
        npm start
        ```
      * **Terminale 3 (per altri script):**
        Se devi avviare altri script (come quelli per l'elaborazione dei dati), apri un terzo terminale, connettiti al container e lanciali da lì.

Ora il progetto è in esecuzione e accessibile tramite `http://localhost:3000` nel tuo browser.

-----

## Documentazione API

La documentazione dell'API è disponibile all'indirizzo:
`http://localhost:5005/ui`

-----

## Note Finali

Se incontri problemi di connessione (come `ERR_CONNECTION_REFUSED` o `ERR_EMPTY_RESPONSE`), verifica che le mappature delle porte in `docker run` e la configurazione dell'host siano corrette. Inoltre, controlla che il tuo firewall non stia bloccando il traffico. -->



# Educational RAG with Entity Linking (EL)

This project implements an educational Question Answering system using **Retrieval-Augmented Generation (RAG)** enhanced with **Entity Linking (EL)**. It consists of a Python-based backend API (Flask/Connexion) and a React-based frontend.

The system is designed to ingest educational video content, transcribe it, index the information, and answer user queries by retrieving relevant context and linking entities to knowledge bases.

---

## 📂 Project Structure

* **`rag_api/`**: The backend logic.
    * **`server.py`**: Entry point for the Flask server.
    * **`rag_service.py`**: Core RAG logic.
    * **`rag_el.py`** & **`rag_ner_spacy.py`**: Modules handling Entity Linking and Named Entity Recognition.
    * **`controllers/`**: API endpoint controllers (e.g., `ask_controller.py`).
    * **`swagger/`**: OpenAPI specification (`openapi.yaml`).
* **`reactApi/react-client/`**: The frontend application built with React.
* **`bench_scripts/`**: Scripts for creating, validating, and fixing benchmarks to evaluate the system's performance.
* **`chunks_scripts/`**: Experimental scripts for text chunking, embedding generation, and NER using different models (Spacy, Flair, Roberta, Stanza).
* **`squad/`**: Utilities for building indices and testing the system against the SQuAD (Stanford Question Answering Dataset) format, specifically for Italian (`squad_it`).
* **Root Scripts**:
    * **`whisperXuniNet.py/.ipynb`**: Speech-to-Text transcription using Whisper.
    * **`extractionText.py`**: Post-processing of transcribed text.
    * **`denseR.py/.ipynb`**: Generation of dense embeddings for retrieval (using FAISS).

---

## 🚀 Getting Started

You can run this project in two ways:
1.  **Conda (Recommended for Local Development):** Best for debugging and flexibility.
2.  **Docker (Recommended for Production/Portability):** Ensures a standardized, isolated environment. **Note:** The Docker image size is approx. 73GB due to ML models.

### Prerequisites

#### For Conda
* **Conda** installed.
* **React** and **npm** installed.
* **CUDA Toolkit** (if using GPU acceleration).

#### For Docker
* **Docker Desktop** installed (with WSL 2 enabled on Windows).
* **NVIDIA Drivers** and **NVIDIA Container Toolkit** configured for GPU acceleration.

---

## ⚙️ Configuration

### 1. Environment Variables
Create a `.env` file in the project root (or inside `rag_api/` depending on your setup preferences) with the following OpenAI API credentials:

```ini
OPENAI_API_KEY=your_api_key_here
ORGANIZATION=your_organization_here
PROJECT=your_project_here
URL=your_api_url_here
````

### 2\. JSON Configuration

Update the configuration files to match your environment.

  * **`config.json`** (Root directory):

      * **For Conda:** Set `host` to `"localhost"`.
      * **For Docker:** Set `host` to `"0.0.0.0"`.
      * **`directoryVideo.path`**: Set the absolute path to your video course directory.

  * **`reactApi/react-client/public/config.json`**:

      * Set `api_base_url` to `"http://localhost:5005"`.

### 3\. Video Files Setup

Place your educational video files in the following two locations:

1.  `reactApi/react-client/public/video/CourseName` (for the frontend player).
2.  The directory specified in the `path` of your root `config.json` (for backend processing).

-----

## 🧠 Data Processing Pipeline

If starting from scratch with new videos, execute the following scripts in order to generate the necessary data (Text and Embeddings):

1.  **Speech-to-Text**: Run `whisperXuniNet.py` to transcribe video audio.
2.  **Text Post-Processing**: Run `extractionText.py` to clean and format the transcripts.
3.  **Embeddings Generation**: Run `denseR.py` to create vector embeddings for the content.

*Advanced users can check `chunks_scripts/` for alternative embedding/NER strategies (e.g., `roberta_chunks...`, `flair_chunks...`).*

-----

## 🏃‍♂️ Running the Project

### Option A: Using Conda

1.  **Create the Environment**:

    ```bash
    # Windows
    conda env create -f environment.yml

    # Other systems
    conda env create -f environment_crossplatform.yml
    ```

    *Tip: Check `environment.yml` to ensure the `cudatoolkit` version matches your GPU (`nvcc --version`).*

2.  **Start the API Server**:

    ```bash
    cd rag_api
    python server.py
    ```

3.  **Start the React Client**:
    Open a new terminal:

    ```bash
    cd reactApi/react-client
    npm start
    ```

    The app will be available at `http://localhost:3000`.

### Option B: Using Docker

1.  **Build the Image**:

    ```bash
    docker build -t educational-rag .
    ```

2.  **Run the Container**:

    ```bash
    docker run -it --rm \
    -p 5005:5005 \
    -p 3000:3000 \
    --gpus all \
    --env-file ./rag_api/.env \
    educational-rag
    ```

3.  **Start Services Inside Container**:
    You need to open separate terminal sessions connected to the running container (`docker exec`).

      * **Terminal 1 (API)**:
        ```bash
        docker exec -it <CONTAINER_ID> bash
        cd rag_api
        python server.py
        ```
      * **Terminal 2 (Frontend)**:
        ```bash
        docker exec -it <CONTAINER_ID> bash
        cd reactApi/react-client
        npm start
        ```

The application is now accessible at `http://localhost:3000`.

-----

## 📖 API Documentation

Once the server is running, the interactive Swagger API documentation is available at:
`http://localhost:5005/ui`

-----

## 🧪 Benchmarking & Evaluation

The repository includes tools to evaluate the RAG and Entity Linking performance.

  * **Benchmarking**: Use scripts in `bench_scripts/` (e.g., `createBenchmark.py`, `validateBenchmark.py`) to generate test sets.
  * **Results**: View evaluation outcomes in `rag_api/final_results/` and `rag_api/previous_results/`.
  * **SQuAD**: Use `squad/` scripts to build indices and run tests against the SQuAD Italian dataset structure.

-----

## 🛠 Troubleshooting

  * **Connection Refused**: If you see `ERR_CONNECTION_REFUSED`, check that `config.json` hosts are set correctly ("0.0.0.0" for Docker) and that ports 3000/5005 are mapped correctly.
  * **GPU Issues**: Run `test_gpu.py` to verify that your Python environment can see the CUDA device.
  * **Docker Size**: Ensure you have sufficient disk space (approx. 73GB required).

<!-- end list -->
