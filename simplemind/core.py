from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

app = FastAPI(title="SimpleMind AI API", description="AI for humans, replacing LangGraph and LangChain for Python users.")



@app.post("/generate", response_model=AIResponse)
def generate_response(request: AIRequest):
    try:
        # Placeholder for AI generation logic
        response = {"message": "This would be the AI response."}
        metadata = {"tokens_used": 50}
        return AIResponse(response=response, metadata=metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
