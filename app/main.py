from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.service import GuardrailService

app = FastAPI(title="RAG Guardrail & Evaluation Engine")
guardrail_service = GuardrailService()

# Define the incoming JSON request contract
class GuardrailRequest(BaseModel):
   llm_response_vector: List[float]       # Vector of the generated answer
   retrieved_chunk_vectors: List[List[float]] # List of vectors from the Vector DB
   threshold: Optional[float] = 0.25

@app.post("/api/v1/validate")
async def validate_rag_output(payload: GuardrailRequest):
    try:
        # Run our mathematical grounding evaluation
        evaluation = guardrail_service.evaluate_groundedness(
            llm_response_vector=payload.llm_response_vector,
            retrieved_chunk_vectors=payload.retrieved_chunk_vectors,
            threshold=payload.threshold
        )
        
        # Step 5: The Deterministic Fallback Router
        # If the guardrail flags a hallucination, we intercept the flow completely
        if evaluation["is_hallucinated"]:
            return {
                "evaluation": evaluation,
                "action_taken": "FALLBACK_TRIGGERED",
                "output": "System Note: Secure fallback generated due to factual divergence. "
                          f"Direct source context verified: {payload.retrieved_chunk_vectors[0]}"
            }
            
        return {
            "evaluation": evaluation,
            "action_taken": "ALLOW_ORIGINAL",
            "output": payload.llm_response_vector
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))