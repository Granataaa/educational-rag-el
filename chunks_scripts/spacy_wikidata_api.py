import spacy
import requests
import json

with open("chunks_metadata/chunks_metadata300_nouns.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

# nlp = spacy.load("it_core_news_lg")

def wikidata_search(entity, sentence_span):
    url = "https://www.wikidata.org/w/api.php"

    keywords = [t.text for t in sentence_span if t.pos_ in ("PROPN","NOUN")]
    search_query = " ".join([entity] + keywords[:1])  # prendi solo le prime 2 keywords
    print("\nQuery di ricerca:", search_query)

    # passo la entità e basta perchè con il contesto non funziona

    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "it",
        "search": entity,
    }
    headers = {
        "User-Agent": "MyEntityLinker/1.0 (francesco@example.com)"  
    }
    
    resp = requests.get(url, params=params, headers=headers)
    
    if resp.status_code != 200:
        print(f"Errore HTTP {resp.status_code} per query '{entity}'")
        return None, None, None
    
    try:
        data = resp.json()
    except ValueError:
        print(f"Risposta non JSON per query '{entity}':", resp.text[:200])
        return None, None, None
    
    if data.get("search"):
        first = data["search"][0]
        return first["id"], first.get("label"), first.get("description")
    
    return None, None, None

for chunk in chunk_data[:5]:  # Limitiamoci ai primi 5 chunk per il test
    #doc = nlp(chunk["text"])

    for noun in chunk["noun_chunk"]:
        qid, label, desc = wikidata_search(noun, "")
        # print(ent.sent.text)
        if qid:
            print(noun, "->", qid, "|", label, "|", desc)
        else:
            print(noun, "-> Nessun match")
