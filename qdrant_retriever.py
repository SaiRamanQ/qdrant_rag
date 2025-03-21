from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import time
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Retrieval API",
    description="API for retrieving relevant document chunks from a Qdrant vector database",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Qdrant client and embedding model
qdrant_client = QdrantClient("localhost", port=6333)
collection_name = "documents"
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Request and response models
class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 3
    threshold: Optional[float] = 0.7  # Minimum similarity score

class DocumentChunk(BaseModel):
    text: str
    file_name: str
    score: float
    page_count: Optional[int] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None

class QueryResponse(BaseModel):
    documents: List[DocumentChunk]
    query: str
    took_ms: float

@app.get("/")
def read_root():
    """Root endpoint that provides basic API information."""
    return {
        "status": "online",
        "api": "Document Retrieval API",
        "version": "1.0.0",
        "endpoints": {
            "/retrieve": "POST - Retrieve relevant document chunks",
            "/health": "GET - Check API health status"
        }
    }

@app.get("/health")
def health_check():
    """Endpoint to check the health of the API and its dependencies."""
    try:
        # Check Qdrant connection
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        # Check if our collection exists
        if collection_name not in collection_names:
            return {
                "status": "warning",
                "message": f"Collection '{collection_name}' does not exist in Qdrant",
                "collections": collection_names
            }
            
        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        
        return {
            "status": "healthy",
            "qdrant": {
                "status": "connected",
                "collection": collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count
            },
            "embedding_model": {
                "name": embedding_model.get_sentence_embedding_dimension()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/retrieve", response_model=QueryResponse)
def retrieve_docs(request: QueryRequest):
    """
    Retrieve relevant document chunks from the vector database.
    
    Parameters:
    - query: The search query text
    - limit: Maximum number of results to return (default: 3)
    - threshold: Minimum similarity score (0-1) for results (default: 0.7)
    
    Returns:
    - List of relevant document chunks with metadata
    """
    start_time = time.time()
    
    try:
        # Generate embedding for the query
        logger.info(f"Processing query: {request.query}")
        query_embedding = embedding_model.encode(request.query).tolist()
        
        # Perform vector search in Qdrant
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=request.limit,
            score_threshold=request.threshold
        )
        
        # Process search results
        documents = []
        for hit in search_results:
            try:
                # Extract metadata
                metadata = hit.payload.get("metadata", {})
                
                document = DocumentChunk(
                    text=hit.payload["text"],
                    file_name=metadata.get("file_name", "Unknown"),
                    score=float(hit.score),  # Convert to float to ensure JSON serialization
                    page_count=metadata.get("page_count"),
                    chunk_index=hit.payload.get("chunk_index"),
                    total_chunks=hit.payload.get("total_chunks")
                )
                documents.append(document)
            except KeyError as e:
                logger.warning(f"Missing field in search result payload: {e}")
        
        # Calculate processing time
        took_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Retrieved {len(documents)} documents in {took_ms:.2f}ms")
        
        return QueryResponse(
            documents=documents,
            query=request.query,
            took_ms=took_ms
        )
        
    except UnexpectedResponse as e:
        logger.error(f"Qdrant error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Qdrant search error: {str(e)}")
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Document Retrieval API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)