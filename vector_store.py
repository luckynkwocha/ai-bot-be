from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config as configs    


class VectorStore:
    def __init__(self):
        self.client = None
        self.embedding_model = None
        self._initialize_client()
        self._initialize_embedding_model()

    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            self.client = QdrantClient(
                url=configs['qdrant_url'],
                api_key=configs['qdrant_api_key'],
                timeout=600
            )

            print(f"Connected to Qdrant at {configs['qdrant_url']}")
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer model"""
        self.embedding_model = OpenAIEmbeddings(
            model=configs["embedding_model"],
            api_key=configs["openai_api_key"]
        )
        print(f"Initialized OpenAI Embedding model: {configs['embedding_model']}")

    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1200, 
        chunk_overlap: int = 200
    ) -> List[str]:
        """Chunk text into smaller pieces"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            print(f"Chunked text into {len(chunks)} pieces")
            return chunks
        except Exception as e:
            print(f"Failed to chunk text: {e}")
            return [text]  # Return original text if chunking fails
    
    def create_collection(self, collection_name: str, vector_size: int = 1536) -> bool:
        """Create a new collection for an agent"""
        try:
            # return collection if already exists
            if self.collection_exists(collection_name):
                return True
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {collection_name}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Collection {collection_name} already exists")
                return True
            print(f"Failed to create collection {collection_name}: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Failed to delete collection {collection_name}: {e}")
            return False
  
    def add_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str]
    ) -> bool:
        """Add documents to a collection"""
        try:
            # Generate embeddings using OpenAI
            embeddings = self.embedding_model.embed_documents(documents)
            
            # Prepare points for Qdrant
            points = []
            for i, (doc, metadata, doc_id) in enumerate(zip(documents, metadatas, ids)):
                point = models.PointStruct(
                    id=doc_id,
                    vector=embeddings[i],
                    payload={
                        "content": doc,
                        **metadata
                    }
                )
                points.append(point)
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            print(f"Added {len(documents)} documents to collection {collection_name}")
            return True
        except Exception as e:
            print(f"Failed to add documents to collection {collection_name}: {e}")
            return False
    
    def query_documents(
        self, 
        collection_name: str, 
        query_text: str, 
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Query documents from a collection using similarity search"""
        try:
            # Generate query embedding using OpenAI
            query_embedding = self.embedding_model.embed_query(query_text)
            
            # Build filter
            query_filter = None
            if filter_dict:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_dict.items()
                    ]
                )
            
            # Search in Qdrant - returns most similar chunks
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=n_results,
                with_payload=True,
                score_threshold=0.3  # Only return results with similarity > 0.3
            )
            # Format results - returns only the relevant chunks
            documents = []
            metadatas = []
            distances = []
            
            for hit in search_result.points:
                # Return only the chunk content (not the full document)
                documents.append(hit.payload.get("content", ""))
                # Remove content from metadata to avoid duplication
                metadata = {k: v for k, v in hit.payload.items() if k != "content"}
                metadatas.append(metadata)
                distances.append(1 - hit.score)  # Convert similarity to distance
            
            print(f"Queried collection {collection_name} with {len(documents)} relevant chunks")
            return {
                "documents": [documents],
                "metadatas": [metadatas], 
                "distances": [distances]
            }
        except Exception as e:
            print(f"Failed to query collection {collection_name}: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

# Global vector store instance
vector_store = VectorStore()