from flask import request, jsonify
# from rag_service import query_rag
from rag_el import query_rag, query_entity_linking_rerank, query_entity_linking_rerank_RRF, query_rag_with_cross_encoder
from models.models import RagResponse
import json
import os

def ask_get():
    print(request)
    query = request.args.get('query')
    k_ric = request.args.get('k_ric')
    LLMHelp = request.args.get('LLMHelp')

    if not query:
        return {"error": "Missing 'query' parameter"}, 400
    if not k_ric:
        return {"error": "Missing 'k_ric' parameter"}, 400
    if not LLMHelp:
        return {"error": "Missing 'LLMHelp' parameter"}, 400

    # res = query_rag(query=query, k_ric=k_ric, LLMHelp=LLMHelp)
    # res = query_entity_linking_rerank(query=query, k_final=int(k_ric), k_initial_retrieval=30, BETA=0.5, LLMHelp=LLMHelp)
    res = query_entity_linking_rerank_RRF(query=query, k_final=int(k_ric), k_initial_retrieval=30, LLMHelp=LLMHelp)
    # res = query_rag_with_cross_encoder(query=query, k_ric=int(k_ric), LLMHelp=LLMHelp)
    # print(res)

    x = RagResponse(**res).model_dump()
    print(x)

    log_file_path = "request_log.json"

    request_data = {
        "query_params": dict(request.args),
        "method": request.method,
        "path": request.path,
        "request": str(request) # Convert request to string for logging
        # Add other relevant request details if needed, e.g., headers, body
    }
    
    entry_to_log = {
        "request": request_data,
        "response_data": x
    }

    all_logs = []
    # Check if the file exists and is not empty before trying to read
    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                all_logs = json.load(f)
        except json.JSONDecodeError:
            # Handle case where file exists but is empty or corrupted JSON
            print(f"Warning: {log_file_path} is empty or contains invalid JSON. Starting a new log.")
            all_logs = []

    all_logs.append(entry_to_log)

    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)
    
    print(f"Logged request and response to {log_file_path}")
    # --- End of JSON file appending logic ---

    return x