from openai import OpenAI
from config import config as configs

client = OpenAI(api_key=configs["openai_api_key"])

def create_conversation(prompt="Hello!"):
    """
    Create a new conversation stored on OpenAI, optionally with:
      - metadata: dict
      - items: list (initial items, up to 20)
    Returns:
      - conversation_id
      - full conversation object (optional)
    """
    items = [{"type": "message", "role": "user", "content": prompt}]

    try:
        conversation = client.conversations.create(
            items=items
        )

        print(f"Created conversation with ID: {conversation}" and conversation.object)

        return {
            "conversation_id": conversation.id,
        }

    except Exception as e:
        # In production, log the exception + return a safer error message
        print(f"Error creating conversation: {e}")
        return {
            "error": "Failed to create conversation",
            "details": str(e)
        }
