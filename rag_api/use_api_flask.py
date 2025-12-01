import requests

def ask_query(user_query: str, k_ric: int = 5, LLMHelp: bool = True):
    url = "http://10.10.11.141:5005/ask"
    
    payload = {
        'query': user_query,
        'k_ric': k_ric,
        'LLMHelp': 'true' if LLMHelp else 'false'
    }

    headers = {
        "accept": "application/json"
    }

    try:
        response = requests.get(url, params=payload, headers=headers)
        
        print(f"Request URL: {response.url}")
        print("Status code:", response.status_code)

        response.raise_for_status() # Solleva un errore se status != 200

        print("Response JSON:", response.json())

    except requests.exceptions.RequestException as e:
        print("❌ Request failed:", e)
        # Se c'è un body di errore nel server, proviamo a stamparlo
        if 'response' in locals() and response.content:
             print("Server message:", response.text)

if __name__ == "__main__":
    ask_query("cos'è la mano invisibile di smith?")