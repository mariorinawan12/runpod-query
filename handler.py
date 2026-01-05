"""
RunPod Serverless Worker - QUERY Endpoint (ONNX CPU)
=====================================================

Lightweight, fast worker for query embedding only.
Uses ONNX Runtime on CPU for:
- Fast cold start (no CUDA initialization)
- Low latency inference
- Cost effective (CPU-only pods)

Model: BAAI/bge-m3 (1024 dimensions) - ONNX optimized
"""
import runpod
import numpy as np
from typing import List, Dict, Any
import os

# =============================================================================
# GLOBAL MODEL (loaded once on cold start)
# =============================================================================
ONNX_SESSION = None
TOKENIZER = None

print("🚀 Starting QUERY worker (ONNX CPU)...")


def load_model():
    """Load ONNX model and tokenizer (called once on cold start)."""
    global ONNX_SESSION, TOKENIZER
    
    if ONNX_SESSION is None:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        print("🔄 Loading BAAI/bge-m3 (ONNX)...")
        
        # Set ONNX to use CPU only with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()  # Use all CPU cores
        sess_options.inter_op_num_threads = os.cpu_count()
        
        # Load ONNX model (optimum exports to model.onnx in the directory)
        model_dir = "/app/model"
        model_path = os.path.join(model_dir, "model.onnx")
        
        # Fallback: check for onnx subdirectory (some optimum versions)
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "onnx", "model.onnx")
        
        if not os.path.exists(model_path):
            # List available files for debugging
            import glob
            files = glob.glob(f"{model_dir}/**/*.onnx", recursive=True)
            if files:
                model_path = files[0]
                print(f"📁 Found ONNX model at: {model_path}")
            else:
                raise FileNotFoundError(f"No ONNX model found in {model_dir}")
        
        ONNX_SESSION = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']  # CPU only!
        )
        
        # Load tokenizer from same directory
        TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
        
        print(f"✅ ONNX model loaded on CPU ({os.cpu_count()} cores)")
    
    return ONNX_SESSION, TOKENIZER


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Mean pooling - take mean of token embeddings weighted by attention mask.
    Same as SentenceTransformer's default pooling.
    """
    # Expand attention mask to match embedding dimensions
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape).astype(float)
    
    # Sum embeddings weighted by mask
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    
    return sum_embeddings / sum_mask


def normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    return embeddings / norms


def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encode texts to embeddings using ONNX model.
    
    Returns:
        np.ndarray of shape (len(texts), 1024)
    """
    session, tokenizer = load_model()
    
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="np"
    )
    
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    
    # Run ONNX inference
    # BGE-M3 ONNX model expects: input_ids, attention_mask
    # Returns: token embeddings (last_hidden_state)
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    # Get output names from model
    output_names = [o.name for o in session.get_outputs()]
    
    # Run inference
    outputs = session.run(output_names, ort_inputs)
    
    # First output is usually last_hidden_state (token embeddings)
    token_embeddings = outputs[0]
    
    # Mean pooling + normalize (same as SentenceTransformer)
    embeddings = mean_pooling(token_embeddings, attention_mask)
    embeddings = normalize(embeddings)
    
    return embeddings


# =============================================================================
# HANDLER FUNCTIONS
# =============================================================================

def embed_query(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Embed user query/queries.
    
    NOTE: BGE-M3 does NOT need query prefix (unlike E5 models).
    Same behavior as LOCAL/INGEST to ensure consistency.
    
    Input:
        query: str (single query)
        OR
        queries: List[str] (batch queries)
    
    Output:
        embedding: List[float] (single)
        OR
        embeddings: List[List[float]] (batch)
    """
    query = input_data.get("query", "")
    queries = input_data.get("queries", [])
    
    # Handle single or batch
    if queries:
        input_texts = queries
        is_batch = True
    elif query:
        input_texts = [query]
        is_batch = False
    else:
        return {"error": "No query provided"}
    
    # BGE-M3 does NOT need prefix (same as LOCAL/INGEST behavior)
    embeddings = encode_texts(input_texts)
    
    if is_batch:
        return {"embeddings": embeddings.tolist()}
    else:
        return {"embedding": embeddings[0].tolist()}


def embed_batch(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch embedding (alias for compatibility).
    
    NOTE: BGE-M3 does NOT need query prefix (unlike E5 models).
    
    Input:
        texts: List[str]
        is_query: bool (default True for this endpoint)
    
    Output:
        embeddings: List[List[float]]
    """
    texts = input_data.get("texts", [])
    # is_query param ignored for BGE-M3 (no prefix needed)
    
    if not texts:
        return {"error": "No texts provided", "embeddings": []}
    
    # BGE-M3 does NOT need prefix (same as LOCAL/INGEST behavior)
    embeddings = encode_texts(texts)
    
    return {"embeddings": embeddings.tolist()}


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler for QUERY endpoint.
    
    Actions:
    - embed_query: Single/batch query embedding (default)
    - embed_batch: Batch embedding (compatibility)
    - health_check: Health check
    """
    try:
        input_data = event.get("input", {})
        action = input_data.get("action", "embed_query")  # Default to embed_query
        
        if action == "embed_query":
            return embed_query(input_data)
        elif action == "embed_batch":
            return embed_batch(input_data)
        elif action == "health_check":
            return {"status": "ok", "endpoint": "query", "runtime": "onnx-cpu"}
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start RunPod serverless
if __name__ == "__main__":
    print("🎉 QUERY worker ready (ONNX CPU)!")
    runpod.serverless.start({"handler": handler})
