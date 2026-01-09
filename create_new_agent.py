import uuid
from vector_store import vector_store

async def create_agent(knowledge: str, collection_name: str):
    """Create an agent by storing text embeddings in Qdrant"""
    agent_id = str(uuid.uuid4())
    try:
        # Create collection if it doesn't exist
        if not vector_store.collection_exists(collection_name):
            vector_store.create_collection(collection_name)
        
        # Chunk the text into smaller pieces
        chunks = vector_store.chunk_text(knowledge)
        
        # Create metadata and IDs for each chunk
        base_metadata = { "agent_id": agent_id }
        
        metadatas = []
        ids = []
        for i, chunk in enumerate(chunks):
            # Create unique metadata for each chunk
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            metadatas.append(chunk_metadata)
            
            uid = uuid.uuid4()
            # Create unique ID for each chunk
            ids.append(f"{uid}")
        
        # Store all chunks in vector database
        success = vector_store.add_documents(
            collection_name=collection_name,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        if success:
            return {
                "success": True,
                "status_code": 201,
                "agent_id": agent_id,
                "collection_name": collection_name,
                "stored_count": len(chunks),
                "message": f"Agent created successfully with {len(chunks)} chunks"
                }
        else:
            return { "success": False, "status_code": 500, "error": "Failed to store agent embeddings" }
    except Exception as e:
        print(f"Error creating agent : {e}")
        return {"success": False, "status_code": 500, "error": str(e)}