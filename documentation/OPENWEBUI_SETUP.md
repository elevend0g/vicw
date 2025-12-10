# Connecting VICW API to OpenWebUI

The VICW API now includes OpenAI-compatible endpoints, allowing it to work with any OpenAI-compatible client, including OpenWebUI.

## Available Endpoints

Your VICW API now supports the following OpenAI-compatible endpoints:

- **GET /v1/models** - List available models
- **POST /v1/chat/completions** - Chat completions (streaming and non-streaming)

## Connection Details

### For Local Development (Docker)

If OpenWebUI is running on the same machine:

- **Base URL**: `http://localhost:8000`
- **API Key**: Not required (can leave empty or use any dummy value)
- **Model**: `openai/gpt-oss-120b`

### For OpenWebUI Running in Docker

If OpenWebUI is running in a Docker container on the same machine as VICW:

- **Base URL**: `http://host.docker.internal:8000`
  - On Linux: Use `http://172.17.0.1:8000` (Docker bridge network IP)
  - On Mac/Windows: Use `http://host.docker.internal:8000`

### For Remote OpenWebUI

If OpenWebUI is on a different machine:

- **Base URL**: `http://YOUR_SERVER_IP:8000`
- Make sure port 8000 is accessible from the OpenWebUI machine

## OpenWebUI Setup Instructions

### Method 1: Add as OpenAI-Compatible Endpoint

1. Open OpenWebUI settings
2. Go to **Admin Panel** → **Settings** → **Connections**
3. Add a new connection:
   - **API Type**: OpenAI API
   - **Base URL**: `http://localhost:8000/v1` (or appropriate URL from above)
   - **API Key**: Leave empty or enter any value (not validated)
4. Save the settings
5. The model `vicw-openai/gpt-oss-120b` should now appear in your model list

### Method 2: Use Environment Variables (Docker Compose)

If running OpenWebUI with Docker Compose, add these environment variables:

```yaml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    environment:
      - OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
      - OPENAI_API_KEY=dummy  # Not validated, but required by OpenWebUI
    ports:
      - "3000:8080"
```

### Method 3: Use as Additional OpenAI Endpoint

In OpenWebUI's admin settings:

1. Navigate to **Settings** → **Connections** → **OpenAI API**
2. Click **Add OpenAI API**
3. Fill in:
   - **Name**: VICW
   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: `sk-dummy` (any value, not validated)
4. Save and refresh

## Testing the Connection

### Test with curl

```bash
# Test models endpoint
curl http://localhost:8000/v1/models

# Test chat completion (non-streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'

# Test chat completion (streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

### Test with Python (OpenAI SDK)

```python
from openai import OpenAI

# Initialize client pointing to VICW API
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not validated
)

# Test non-streaming
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print(response.choices[0].message.content)

# Test streaming
stream = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Features Supported

✅ **Supported:**
- Chat completions (non-streaming)
- Chat completions (streaming via Server-Sent Events)
- Model listing
- RAG (Retrieval-Augmented Generation) - always enabled
- Echo Guard (duplicate response detection)
- State tracking (loop prevention)

⚠️ **Limitations:**
- API key validation is not implemented (any key is accepted)
- Only one model is exposed: `openai/gpt-oss-120b`
- Temperature, top_p, and other parameters are accepted but not used (VICW uses backend LLM defaults)
- Token counts are approximate estimates

## Troubleshooting

### "Connection refused" error

Make sure the VICW API is running:
```bash
docker-compose ps
curl http://localhost:8000/health
```

### "404 Not Found" on /v1/models

The server needs to be rebuilt with the new code:
```bash
docker-compose up -d --build vicw_api
```

### OpenWebUI can't reach localhost:8000

If OpenWebUI is in Docker, it can't use `localhost` to reach the host machine. Use:
- Linux: `http://172.17.0.1:8000`
- Mac/Windows: `http://host.docker.internal:8000`

### Models not showing up in OpenWebUI

1. Check that the connection is saved correctly
2. Try refreshing the page
3. Check OpenWebUI logs for connection errors
4. Verify the endpoint works with curl first

## Architecture Notes

The OpenAI-compatible endpoints are a thin wrapper around VICW's internal `/chat` endpoint:

1. Request comes in OpenAI format
2. Converted to VICW internal format
3. Processed through VICW pipeline (RAG, echo guard, state tracking)
4. Response converted back to OpenAI format
5. Returned as JSON (non-streaming) or SSE stream (streaming)

The VICW context manager maintains state across requests, providing infinite context capabilities beyond standard OpenAI limits.
