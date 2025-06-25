# Auro-PAI Platform Backend

A comprehensive FastAPI-based backend for the Auro-PAI platform, providing AI assistance with context-aware capabilities through local LLM integration, RAG (Retrieval-Augmented Generation), and external tool access.

## Features

- **Local LLM Integration**: Seamless integration with llama.cpp servers (LLaVA + Mixtral)
- **Multi-Provider LLM Support**: Support for local, OpenAI, and Gemini models
- **RAG System**: ChromaDB-based vector storage for codebase and document retrieval
- **External Tools**: Web search and URL fetching capabilities
- **Context Management**: Session-based conversation context with memory management
- **Transparent AI Assistance**: Structured, reviewable AI suggestions
- **RESTful API**: Comprehensive FastAPI-based REST API
- **Health Monitoring**: Built-in health checks and monitoring endpoints

## Architecture

The backend follows a modular, layered architecture:

```
┌─────────────────────────┐
│ Client Applications     │
│ (VS Code Extension,     │
│  Web UI, CLI tools)     │
└──────────┬──────────────┘
           │ HTTP/JSON API
           ▼
┌─────────────────────────┐
│ FastAPI Backend         │
│ - API Endpoints         │
│ - Request/Response      │
│ - Agentic Orchestration│
│ - Context Management    │
└─────┬──────┬──────┬─────┘
      │      │      │
      ▼      ▼      ▼
┌─────────┐ ┌─────┐ ┌─────────┐
│ LLM     │ │ RAG │ │ Tools   │
│ Manager │ │ Svc │ │ Service │
└─────────┘ └─────┘ └─────────┘
      │      │      │
      ▼      ▼      ▼
┌─────────┐ ┌─────┐ ┌─────────┐
│llama.cpp│ │Chroma│ │Web APIs │
│ Server  │ │ DB  │ │ Search  │
└─────────┘ └─────┘ └─────────┘
```

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **llama.cpp server** running with LLaVA + Mixtral model
3. **ChromaDB** (optional, falls back to in-memory)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd auro-pai-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

### Docker Setup (Alternative)

```bash
# Build the image
docker build -t auro-pai-backend .

# Run the container
docker run -p 8000:8000 --env-file .env auro-pai-backend
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# LLM Configuration
LLAMACPP_SERVER_URL=http://localhost:8080
LLAMACPP_MODEL_NAME=mixtral-8x7b-instruct

# RAG Configuration
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Tool Configuration
WEB_SEARCH_ENABLED=true
WEB_SEARCH_ENGINE=duckduckgo

# Optional: Alternative LLM providers
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

### llama.cpp Server Setup

1. Install and build llama.cpp
2. Download LLaVA + Mixtral 8x7B GGUF model
3. Start server:
```bash
./server -m /path/to/mixtral-8x7b-instruct.gguf --host 0.0.0.0 --port 8080
```

### ChromaDB Setup

```bash
# Install ChromaDB
pip install chromadb

# Start ChromaDB server (optional)
chroma run --host localhost --port 8000
```

## API Documentation

### Interactive API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### Chat API
- `POST /api/v1/chat` - Main chat interaction
- `GET /api/v1/chat/sessions` - List active sessions
- `GET /api/v1/chat/sessions/{id}/history` - Get conversation history
- `DELETE /api/v1/chat/sessions/{id}` - Delete session

#### RAG API
- `POST /api/v1/rag/index` - Index directory for RAG
- `POST /api/v1/rag/search` - Search indexed documents
- `GET /api/v1/rag/stats` - Get collection statistics

#### Tools API
- `POST /api/v1/tools/search` - Web search
- `POST /api/v1/tools/fetch` - Fetch URL content
- `POST /api/v1/tools/search-and-fetch` - Combined search and fetch

#### Health API
- `GET /api/v1/health` - Comprehensive health check
- `GET /api/v1/health/readiness` - Readiness probe
- `GET /api/v1/health/liveness` - Liveness probe

### Example Usage

#### Chat Interaction
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Help me refactor this Python function",
    "include_rag": true,
    "include_tools": true
  }'
```

#### Index Codebase for RAG
```bash
curl -X POST "http://localhost:8000/api/v1/rag/index" \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/path/to/your/codebase"
  }'
```

#### Web Search
```bash
curl -X POST "http://localhost:8000/api/v1/tools/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "FastAPI async best practices",
    "max_results": 5
  }'
```

## Services Overview

### LLM Manager
- Manages multiple LLM providers (local llama.cpp, OpenAI, Gemini)
- Handles streaming and non-streaming responses
- Provider failover and load balancing
- Token usage tracking

### RAG Service
- ChromaDB integration for vector storage
- Automatic document chunking and indexing
- Semantic search across codebase and documents
- Support for multiple file types
- Context retrieval for AI responses

### Tool Service
- Web search using DuckDuckGo (privacy-focused)
- URL content fetching and text extraction
- Combined search-and-fetch operations
- Rate limiting and safety controls

### Context Manager
- Session-based conversation management
- Context length management
- Memory optimization
- Session cleanup and expiration

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
isort .
flake8 .
```

### Adding New Features

1. **New LLM Provider**: Extend `LLMManager` in `services/llm_manager.py`
2. **New Tool**: Add to `ToolService` in `services/tool_service.py`
3. **New API Endpoint**: Create route in `api/routes/`

## Deployment

### Production Considerations

1. **Security**: Enable API key authentication
2. **Rate Limiting**: Configure appropriate limits
3. **Monitoring**: Set up logging and metrics
4. **Scaling**: Use multiple workers with Gunicorn
5. **SSL**: Enable HTTPS in production

### Example Production Deployment
```bash
# Using Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With SSL
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:443 \
  --keyfile /path/to/key.pem \
  --certfile /path/to/cert.pem
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auro-pai-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: auro-pai-backend
  template:
    metadata:
      labels:
        app: auro-pai-backend
    spec:
      containers:
      - name: backend
        image: auro-pai-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLAMACPP_SERVER_URL
          value: "http://llama-service:8080"
        readinessProbe:
          httpGet:
            path: /api/v1/health/readiness
            port: 8000
        livenessProbe:
          httpGet:
            path: /api/v1/health/liveness
            port: 8000
```

## Troubleshooting

### Common Issues

1. **llama.cpp Connection Failed**
   - Verify server is running on correct host/port
   - Check firewall settings
   - Ensure model is properly loaded

2. **ChromaDB Connection Issues**
   - Check ChromaDB server status
   - Verify host/port configuration
   - Falls back to in-memory mode if unavailable

3. **Web Search Not Working**
   - Ensure `duckduckgo-search` package is installed
   - Check internet connectivity
   - Verify `WEB_SEARCH_ENABLED=true`

4. **High Memory Usage**
   - Reduce `MAX_SESSIONS` and `SESSION_TIMEOUT`
   - Decrease `CHUNK_SIZE` for RAG
   - Monitor context length limits

### Logs and Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

Check service health:
```bash
curl http://localhost:8000/api/v1/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run code quality checks
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the health endpoints for service status
