# RUNPOD-QUERY: ONNX CPU Query Embedding

Fast, lightweight query embedding endpoint using ONNX Runtime on CPU.

## Why ONNX CPU?

| Aspect | PyTorch CUDA | ONNX CPU |
|--------|--------------|----------|
| Cold Start | ~15-30s (CUDA init) | ~3-5s |
| Inference | ~10ms | ~20-50ms |
| Cost | GPU required | CPU-only pod |
| Memory | ~4GB VRAM | ~2GB RAM |

For query embedding (single short text), the cold start savings outweigh the slightly slower inference.

## Model

- **Model**: BAAI/bge-m3
- **Dimensions**: 1024
- **Runtime**: ONNX Runtime (CPU)
- **Optimization**: Graph optimization enabled, multi-threaded

## Setup GitHub Actions

### 1. Create Docker Hub Access Token

1. Go to [Docker Hub](https://hub.docker.com/) → Account Settings → Security
2. Create new Access Token with Read/Write permissions
3. Copy the token

### 2. Add GitHub Secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret Name | Value |
|-------------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token |

### 3. Push to Main

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

GitHub Actions will automatically build and push the Docker image.

## Create RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `query-embedding`
   - **Docker Image**: `yourusername/runpod-query:latest`
   - **Pod Type**: CPU (no GPU needed!)
   - **Min Workers**: 1 (always warm for low latency)
   - **Max Workers**: 3-5 (based on traffic)
   - **Idle Timeout**: 60 seconds (keep warm longer)
   - **Execution Timeout**: 30 seconds

4. Copy the Endpoint ID for your app config

## API Actions

### `embed_query` (default)
```json
// Single query
{
  "input": {
    "action": "embed_query",
    "query": "What is machine learning?"
  }
}
// Response: {"embedding": [0.1, 0.2, ...]}

// Batch queries
{
  "input": {
    "action": "embed_query",
    "queries": ["Query 1", "Query 2"]
  }
}
// Response: {"embeddings": [[0.1, ...], [0.2, ...]]}
```

### `embed_batch`
```json
{
  "input": {
    "action": "embed_batch",
    "texts": ["Text 1", "Text 2"]
  }
}
// Response: {"embeddings": [[0.1, ...], [0.2, ...]]}
```

### `health_check`
```json
{
  "input": {
    "action": "health_check"
  }
}
// Response: {"status": "ok", "endpoint": "query", "runtime": "onnx-cpu"}
```

## Local Testing

```bash
# Build (takes ~10-15 min for ONNX export)
docker build -t runpod-query .

# Run
docker run -p 8000:8000 runpod-query
```

## Environment Variables

Set these in your main app:

```bash
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_QUERY_ENDPOINT_ID=your_endpoint_id
BACKEND_EMBED_QUERY=runpod
```

## Notes

- BGE-M3 does NOT need query prefix (unlike E5 models)
- Output is L2 normalized (same as SentenceTransformer)
- Model is pre-converted to ONNX at Docker build time
- Uses all available CPU cores for inference
- Keep min workers = 1 for always-warm low latency
