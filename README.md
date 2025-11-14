# Istruzioni per l'Utilizzo del Progetto

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

Se incontri problemi di connessione (come `ERR_CONNECTION_REFUSED` o `ERR_EMPTY_RESPONSE`), verifica che le mappature delle porte in `docker run` e la configurazione dell'host siano corrette. Inoltre, controlla che il tuo firewall non stia bloccando il traffico.