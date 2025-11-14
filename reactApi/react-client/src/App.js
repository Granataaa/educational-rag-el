import React, { useState, useEffect } from 'react';

function getVideoSrc(fileTxtName) {
  if (!fileTxtName.includes("clean")) return "";

  const [part1, part2WithExt] = fileTxtName.split("clean");

  const part1Cleaned = part1.slice(0, -1); // rimuove ultimo carattere
  const part2Raw = part2WithExt.replace(".txt", ""); // rimuove .txt
  const part2Cleaned = part2Raw.slice(1); // rimuove primo carattere

  const finalPath = `${part1Cleaned}/${part2Cleaned}.mp4`;
  console.log("finalPath", finalPath);
  return `/video/${finalPath}`;
}

function App() {
  const [query, setQuery] = useState('');  // Stato per il campo di input
  const [risposta, setRisposta] = useState(null);  // Stato per la risposta
  const [loading, setLoading] = useState(false);  // Stato per il caricamento
  const [k_ric, setk_ric] = useState(5) // Stato sul numero di risposte date da Faiss
  const [LLMHelp, setLLMHelp] = useState(false)
  const [config, setConfig] = useState(null);

  useEffect(() => {
    fetch('/config.json')
      .then((res) => res.json())
      .then((data) => setConfig(data))
      .catch((err) => console.error('Errore nel caricamento config:', err));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();  // Preveniamo il comportamento di default del form
    setLoading(true);  // Attiva il caricamento

    try {
      const response = await fetch(`http://${config.server.host}:${config.server.port}/ask?query=${encodeURIComponent(query)}&k_ric=${encodeURIComponent(k_ric)}&LLMHelp=${encodeURIComponent(LLMHelp)}`);
      if (!response.ok) {
        throw new Error(`Errore: ${response.status}`);
      }
      const data = await response.json();
      setRisposta(data);  // Salva la risposta nel state
      console.log(risposta)
    } catch (error) {
      console.error(error);
      alert('Errore durante la richiesta');
    } finally {
      setLoading(false);  // Disattiva il caricamento
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>API RAG UniNettuno</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Inserisci la tua query:
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}  // Aggiorna lo stato della query
            style={{ marginLeft: '10px', padding: '5px', width: '300px' }}
          />
        </label>
        <br/>
        <br/>
        <label>
            Numero massimo di risposte (1-10):
            <input
            type="number"
            value={k_ric}
            onChange={(e) => {
              const value = Math.max(1, Math.min(10, Number(e.target.value))); // Limita il valore tra 1 e 10
              setk_ric(value);
            }}
            placeholder="5"
            style={{ marginLeft: '10px', padding: '5px', width: '50px' }}
            />
        </label>
        <br/>
        <br/>
        <label>
          Sistema testo con LLM:
          <input
            type="checkbox"
            checked={LLMHelp}
            onChange={(e) => setLLMHelp(e.target.checked)} // Aggiorna lo stato LLMHelp
            style={{ marginLeft: '10px' }}
          />
        </label>
        <br/>
        <br/>
        <button type="submit" style={{ marginLeft: '10px', padding: '5px 10px' }}>
          Invia
        </button>
      </form>

      {loading && <p>Caricamento...</p>}

      {risposta && (
        <div style={{ marginTop: '20px' }}>
          <h2>Risultati:</h2>
          <div dangerouslySetInnerHTML={{__html: risposta.testoRisp }} />
          {risposta.chunks.map((item, index) => (
            <div key={`${item.chunk_id}-${index}`} style={{ marginBottom: '20px' }} id={`ris-${index+1}`}>
              <p style={{ marginLeft: '20px' }} ><strong>Risposta n.{index + 1}:</strong></p>
              <p style={{ marginLeft: '20px' }} ><strong>Text:</strong> {item.text}</p>
              <p style={{ marginLeft: '20px' }} ><strong>Source:</strong> {item.source}-{item.chunk_id}</p>
              <video
                width="640"
                controls
                onLoadedMetadata={(e) => {
                  e.currentTarget.currentTime = item.start_time;
                }}
              >
                <source src={getVideoSrc(item.source)} type="video/mp4" />
              </video>
              <br/>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
