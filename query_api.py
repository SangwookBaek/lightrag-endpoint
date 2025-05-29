from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from jose import JWTError, jwt
import os
import hashlib
import time
from typing import Literal, List

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.llm.gemini import gemini_2_0_flash_complete, gemini_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from neo4j import GraphDatabase

# App initialization
app = FastAPI(title="LightRAG Query API with Auth")

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET", "secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 60 * 60
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# In-memory user store
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": hashlib.sha256("ossca2727".encode()).hexdigest(),
        "api_key": "admin123"
    }
}

# Global instances
rag: LightRAG = None
driver = None
executed_queries: List[dict] = []

# Application startup: initialize LightRAG and Neo4j driver
@app.on_event("startup")
async def startup_event():
    global rag, driver
    WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myKG")
    os.makedirs(WORKING_DIR, exist_ok=True)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_2_0_flash_complete,
        llm_model_max_token_size=32768,
        embedding_func=gemini_embed,
        chunk_token_size=512,
        chunk_overlap_token_size=128,
        vector_storage="PGVectorStorage",
        kv_storage = "PGKVStorage",
        graph_storage = "Neo4JStorage",
        doc_status_storage = "PGDocStatusStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASS = os.getenv("NEO4J_PASS")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Utility functions
def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload.update({"exp": time.time() + ACCESS_TOKEN_EXPIRE_SECONDS})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def verify_token(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username or username not in users_db:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# Request model
class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Forward query to LightRAG engine")
    mode: Literal["naive", "local", "global", "hybrid"] = Field(
        default="hybrid", description="RAG query mode"
    )

# Endpoints
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or user["hashed_password"] != hashlib.sha256(form_data.password.encode()).hexdigest():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect credentials")
    access_token = create_access_token({"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/run_neo4j_query")
async def run_neo4j_query(query: str, username: str = Depends(verify_token)):
    start = time.time()
    with driver.session() as session:
        records = session.run(query).data()
    elapsed_ms = int((time.time() - start) * 1000)
    executed_queries.append({"query": query, "duration_ms": elapsed_ms})
    return {"elapsed_ms": elapsed_ms, "rows": len(records)}

@app.post("/run_query")
async def run_rag_query(request: RAGQueryRequest, username: str = Depends(verify_token)):
    start = time.time()
    response = await rag.aquery(request.query, param=QueryParam(mode=request.mode))
    elapsed_ms = int((time.time() - start) * 1000)
    executed_queries.append({"query": request.query, "duration_ms": elapsed_ms})
    return {"elapsed_ms": elapsed_ms, "result": response}