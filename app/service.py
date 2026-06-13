import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class GuardrailService:
    def __init__(self):
       #self.model = SentenceTransformer('all-MiniLM-L6-v2')
       pass

    def _calculate_cosine_distance(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        Calculates 1 - Cosine Similarity using pure linear algebra.
        """
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            return 1.0

        similarity = dot_product / (norm_a * norm_b)
        return float(1.0 - similarity)
    
    def evaluate_groundedness(self, llm_response_vector: List[float], retrieved_chunk_vectors: List[List[float]], threshold: float = 0.25) -> Dict[str, Any]:
        """
        Validates whether the LLM response vector matches any retrieved vector.
        """
        # 1. Convert incoming list of floats straight to a NumPy array
        response_np = np.array(llm_response_vector)
        
        distances = []
        for chunk in retrieved_chunk_vectors:
            # 2. Convert each chunk list to a NumPy array
            chunk_np = np.array(chunk)
            
            # 3. Calculate distance between the pre-existing vectors
            distance = self._calculate_cosine_distance(response_np, chunk_np)
            distances.append(distance)
    
        min_distance = min(distances) if distances else 1.0
        is_hallucinated = min_distance > threshold
    
        return {
            "cosine_distance": round(min_distance, 4),
            "is_hallucinated": is_hallucinated,
            "status": "BLOCKED" if is_hallucinated else "PASSED"
        }