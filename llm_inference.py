import time

from vector_store import vector_store
from llm_service import llm_service

async def process_inference_request(
    request_data: dict,
    stream: bool = False,
    ):
    """Process inference request (non-streaming)"""
    start_time = time.time()
    
    try:
        # Retrieve relevant documents using RAG
        collection_name = request_data["collection_name"]
        conversation_id = request_data["conversation"]

        model = request_data.get("model", "gpt-4o")
        retrieved_docs = []
        message = request_data["message"]
        
        if vector_store.collection_exists(collection_name):
            query_result = vector_store.query_documents(
                collection_name=collection_name,
                query_text=message,
                n_results=3
            )
            
            if query_result.get("documents") and query_result["documents"][0]:
                for i, doc in enumerate(query_result["documents"][0]):
                    metadata = query_result["metadatas"][0][i] if query_result.get("metadatas") else {}
                    distance = query_result["distances"][0][i] if query_result.get("distances") else 0
                    
                    retrieved_docs.append({
                        "content": doc,
                        "metadata": metadata,
                        "score": 1 - distance
                    })

        print(f"Retrieved {len(retrieved_docs)} documents for RAG")
        print(f"Retrieved documents: {retrieved_docs}")

        # Add current user message with context
        context_text = ""
        if retrieved_docs:
            context_text = "\n\n".join([
                f"Context {i+1}: {doc['content']}"
                for i, doc in enumerate(retrieved_docs)
            ])
            
        if context_text:
            user_content = f"Context:\n{context_text}\n\nUser Question: {message}"
        else:
            user_content = message
        
        # Generate response
        llm_response = await llm_service.generate_response(
            message=user_content,
            conversation_id=conversation_id,
            model=model,
            temperature=0.5,
            stream=False
        )
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "message": llm_response["content"],
            "response_time_in_milliseconds": response_time_ms,
            "conversation_id": conversation_id,
            }
    except Exception as e:
        print(f"Error processing inference request : {e}")
        return {
            "success": False,
            "status_code": 500,
            "error": str(e)
            }