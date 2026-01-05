# RUNPOD-QUERY: ONNX CPU-based Query Embedding
# =============================================
# Fast cold start, low latency, cost effective

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and convert model to ONNX at build time
# This makes cold start MUCH faster
RUN mkdir -p /app/model

# Download BGE-M3 and export to ONNX
RUN python -c "
from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

print('📥 Downloading BGE-M3...')
model_name = 'BAAI/bge-m3'

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('/app/model')

# Export to ONNX using optimum
print('🔄 Converting to ONNX...')
ort_model = ORTModelForFeatureExtraction.from_pretrained(
    model_name,
    export=True,
    provider='CPUExecutionProvider'
)
ort_model.save_pretrained('/app/model')

print('✅ Model exported to ONNX!')
"

# Copy handler
COPY handler.py .

# RunPod serverless entry point
CMD ["python", "-u", "handler.py"]
