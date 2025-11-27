import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import internal modules
# Ensure we are in the correct directory or path is set up
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rag_el
from models.models import RagResponse

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        HOST = config['server']['host']
        # Use 5006 for FastAPI to allow running side-by-side with Flask (5005)
        PORT = 5006 
except Exception as e:
    print(f"Error loading config.json: {e}. Using defaults.")
    HOST = "0.0.0.0"
    PORT = 5006

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Load RAG models
    print("Starting up... Loading RAG models.")
    rag_el.loading_entity_linking()
    yield
    # Shutdown: Clean up resources if needed
    print("Shutting down...")

app = FastAPI(
    title="UniNettuno RAG API",
    description="API for Retrieval-Augmented Generation system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/ask", response_model=RagResponse, summary="Ask a query to the RAG system")
async def ask(
    query: str = Query(..., description="The query to send to the RAG system"),
    k_ric: int = Query(..., description="Number of results to retrieve"),
    LLMHelp: str = Query(..., description="Whether to use LLM to refine results (true/false)")
):
    """
    Retrieves data from the RAG system for the provided query.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' parameter")
    
    # Call the RAG function
    # Using query_entity_linking_rerank_RRF as it seems to be the main one used in the existing controller
    try:
        # Note: rag_el functions are synchronous, so they might block the event loop.
        # For high load, consider running in a threadpool, but for now direct call is fine.
        res = rag_el.query_entity_linking_rerank_RRF(
            query=query, 
            k_final=k_ric, 
            k_initial_retrieval=30, 
            LLMHelp=LLMHelp
        )
    except Exception as e:
        print(f"Error in RAG execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Validate and serialize response
    try:
        response_model = RagResponse(**res)
        response_data = response_model.model_dump()
    except Exception as e:
        print(f"Validation error: {e}")
        # If validation fails, we might still want to return the raw result or handle it
        # For now, let's try to return what we have, but it might fail schema validation
        response_data = res

    # Logging request and response
    log_request(query, k_ric, LLMHelp, response_data)

    return response_data

def log_request(query, k_ric, llm_help, response_data):
    log_file_path = "request_log.json"
    request_data = {
        "query_params": {"query": query, "k_ric": k_ric, "LLMHelp": llm_help},
        "method": "GET",
        "path": "/ask"
    }
    
    entry_to_log = {
        "request": request_data,
        "response_data": response_data
    }

    try:
        all_logs = []
        if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                all_logs = json.load(f)
        
        all_logs.append(entry_to_log)
        
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_logs, f, ensure_ascii=False, indent=2)
    except Exception as log_error:
        print(f"Logging error: {log_error}")

def get_local_ip():
    """
    Utility to get the machine's local IP address (the one visible on the network/VPN).
    """
    import socket
    try:
        # We don't actually connect, just determine the route
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"\n🚀 Server starting!")
    print(f"📡 Accessible at: http://{local_ip}:{PORT}")
    print(f"📄 Docs available at: http://{local_ip}:{PORT}/docs\n")
    
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
