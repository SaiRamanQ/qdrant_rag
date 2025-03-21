import os
import io
import json
import hashlib
import time
from typing import Dict, Any, List
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Configuration
MINIO_URL = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minio123"
BUCKET_NAME = "index"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500  # Characters per chunk
PROCESSED_FILES_PATH = "processed_files.json"
FORCE_REINDEX = True  # Set to True to force reindexing all documents

# Initialize clients
minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

qdrant_client = QdrantClient("localhost", port=6333)

# Load embedding model
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def get_file_hash(file_stream):
    """Generate hash for file to detect changes."""
    file_stream.seek(0)
    file_hash = hashlib.md5(file_stream.read()).hexdigest()
    file_stream.seek(0)
    return file_hash

def load_processed_files():
    """Load record of processed files."""
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_processed_files(processed_files):
    """Save record of processed files."""
    with open(PROCESSED_FILES_PATH, 'w') as f:
        json.dump(processed_files, f)

def extract_text_from_pdf(file_stream):
    """Extract text from PDF file."""
    reader = PdfReader(file_stream)
    text = ""
    
    print(f"PDF has {len(reader.pages)} pages")
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
        else:
            print(f"Warning: Page {i+1} returned no text")
    
    print(f"Extracted {len(text)} characters from PDF")
    return text.strip(), len(reader.pages)

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks of approximately equal size."""
    if not text:
        print("Warning: No text to chunk")
        return []
        
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    
    print(f"Split text into {len(chunks)} chunks")
    return chunks

def reset_collection():
    """Delete and recreate the collection."""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME in collection_names:
            print(f"Deleting existing collection '{COLLECTION_NAME}'")
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        
        vector_size = embedding_model.get_sentence_embedding_dimension()
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created fresh collection '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"Error resetting collection: {str(e)}")

def ensure_collection_exists():
    """Create Qdrant collection if it doesn't exist."""
    collections = qdrant_client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        vector_size = embedding_model.get_sentence_embedding_dimension()
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created collection '{COLLECTION_NAME}'")

def check_collection_status():
    """Check and print the status of the collection."""
    try:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        print("\nCollection Status:")
        print(f"Points count: {collection_info.points_count}")
        print(f"Vectors count: {collection_info.vectors_count}")
        print(f"Segments count: {collection_info.segments_count}")
        print(f"Status: {collection_info.status}")
    except Exception as e:
        print(f"Error checking collection status: {str(e)}")

def index_document(file_name):
    """Process a single document from MinIO."""
    try:
        print(f"\nProcessing file: {file_name}")
        
        # Get file from MinIO
        print("Fetching file from MinIO...")
        file_data = minio_client.get_object(BUCKET_NAME, file_name)
        file_stream = io.BytesIO(file_data.read())
        print(f"File size: {file_stream.getbuffer().nbytes} bytes")
        
        # Check if file has changed since last indexing
        file_hash = get_file_hash(file_stream)
        processed_files = load_processed_files()
        
        if not FORCE_REINDEX and file_name in processed_files and processed_files[file_name]["hash"] == file_hash:
            print(f"File {file_name} already indexed and unchanged. Skipping.")
            return
        
        # Extract text based on file type
        print("Extracting text...")
        if file_name.lower().endswith('.pdf'):
            text, page_count = extract_text_from_pdf(file_stream)
            metadata = {
                "file_name": file_name,
                "mime_type": "application/pdf",
                "page_count": page_count,
                "source": f"minio://{BUCKET_NAME}/{file_name}"
            }
        else:
            print(f"Unsupported file type: {file_name}")
            return
        
        # Split text into chunks
        print("Chunking text...")
        text_chunks = split_text_into_chunks(text)
        
        if not text_chunks:
            print(f"No text chunks generated from {file_name}")
            return
        
        # Create vector embeddings and store in Qdrant
        print("Generating embeddings and storing in Qdrant...")
        points = []
        for i, chunk in enumerate(text_chunks):
            if i % 50 == 0:
                print(f"Processing chunk {i}/{len(text_chunks)}")
                
            # Generate embedding
            embedding = embedding_model.encode(chunk).tolist()
            
            # Create unique ID for this chunk
            chunk_id = int(hashlib.md5(f"{file_name}-{i}".encode()).hexdigest()[:16], 16)
            
            # Prepare payload
            payload = {
                "text": chunk,
                "metadata": metadata,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "indexed_at": time.time()
            }
            
            points.append(PointStruct(id=chunk_id, vector=embedding, payload=payload))
            
            # Upload in batches to avoid memory issues
            if len(points) >= 100:
                print(f"Uploading batch of {len(points)} points to Qdrant...")
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []
        
        # Upload any remaining points
        if points:
            print(f"Uploading final batch of {len(points)} points to Qdrant...")
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        # Update processed files record
        processed_files[file_name] = {"hash": file_hash, "last_indexed": time.time()}
        save_processed_files(processed_files)
        
        print(f"Successfully indexed {file_name}: {len(text_chunks)} chunks")
        
    except Exception as e:
        print(f"Error indexing {file_name}: {str(e)}")
        import traceback
        traceback.print_exc()

def index_all_documents():
    """Index all documents in the MinIO bucket."""
    # Ensure bucket exists
    if not minio_client.bucket_exists(BUCKET_NAME):
        print(f"Bucket '{BUCKET_NAME}' doesn't exist")
        return
    
    # Reset collection if force reindex is enabled
    if FORCE_REINDEX:
        reset_collection()
    else:
        ensure_collection_exists()
    
    # List and process all objects
    print("Listing objects in bucket...")
    objects = list(minio_client.list_objects(BUCKET_NAME, recursive=True))
    print(f"Found {len(objects)} objects in bucket")
    
    for obj in objects:
        index_document(obj.object_name)
    
    # Check final status
    check_collection_status()
    
    print("\nIndexing complete")

if __name__ == "__main__":
    index_all_documents()