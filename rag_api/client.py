import requests
import json

with open('../config.json', 'r') as f:
    config = json.load(f)

BASE_URL = f"http://{config['server']['host']}:{config['server']['port']}"

def ask_query():
    while True:
        query = input("Operazioni possibili: \n1. Inserisci una query\n2. Esci scrivendo exit\n")
        
        if query.lower() == 'exit':
            print("Uscita in corso...")
            break

        params = {'query': query}
        response = requests.get(f'{BASE_URL}/ask', params=params)

        if response.status_code == 200:
            data = response.json()
            for item in data:
                print(f"\nChunk ID: {item['chunk_id']}")
                print(f"Text: {item['text']}")
                print(f"Source: {item['source']}")
                print(f"Start Time: {item['start_time']}")
                print(f"End Time: {item['end_time']}")
                print("Entities:")
                for entity in item['entities']:
                    print(f"  - {entity['text']} ({entity['label']})")
        else:
            print(f"Errore: {response.status_code} - {response.text}")

if __name__ == "__main__":
    ask_query()
