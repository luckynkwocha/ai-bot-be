from dotenv import load_dotenv
load_dotenv()  

from flask import Flask, request
from web_scraper import scrape_text
from flask import jsonify
from pypdf import PdfReader
from llm_inference import process_inference_request
from create_new_agent import create_agent
from llm_service import llm_service


app = Flask(__name__)

@app.post('/create_agent')
async def create_new_agent():
    file = request.files["file"]

# Option A: extract directly from stream (no need to save)
    try:
        reader = PdfReader(file.stream)

        pages_text = ""
        for page in reader.pages:
            pages_text += "\n" + page.extract_text() or ""

        all_text = "\n\n".join(pages_text).strip()

    except Exception as e:
        return jsonify({"error": "Failed to read PDF", "details": str(e)}), 500


    knowledge = scrape_text("https://hdfund.org/") 
    knowledge_base = knowledge + "\n\n" + all_text  
    response = await create_agent(knowledge_base, "hdf_collection")
    print(f"Create Agent Response: {response}")
    return response
    
@app.get('/health')
def health():
    return {
        "status": "ok",
        "message": "HDF AI Assistant is running"
    }


@app.get('/')
def home():
    return {
        "status": "ok",
        "message": "HDF AI Assistant is running"
    }

@app.post("/inference")
async def inference():
    request_data = request.get_json()
    """Generate response using agent embeddings and conversation history"""
    if request_data.get("conversation") is None:
        prompt = request_data.get("prompt", "Hello!")
        convo = await llm_service.create_conversation(prompt)
        request_data["conversation"] = convo["conversation_id"]
    if request_data.get("stream"):
        return {
        "status_code": "400",
        "message": "For streaming responses, use the /inference/stream endpoint"
        }  
    return await process_inference_request(request_data, stream=False)

# if __name__ == "__main__":
#     app.run(debug=True)