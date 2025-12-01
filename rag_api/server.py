import connexion
from flask_cors import CORS
from rag_service import loading
from rag_ner_spacy import loading_spacy
from rag_el import loading_entity_linking
import json

with open('../config.json', 'r') as f:
    config = json.load(f)

host = config['server']['host']
port = config['server']['port']

# Crea l'app Connexion
app = connexion.App(__name__, specification_dir='./swagger')
app.add_api('openapi.yaml', swagger_ui=True)

CORS(app.app)

# Usa Connexion per avviare il server senza uvicorn
if __name__ == "__main__":
    loading_entity_linking()
    app.run(host=host, port=port)