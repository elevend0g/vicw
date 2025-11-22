# Quick Start Guide - VICW Phase 2

Get up and running with VICW in under 5 minutes!

## Prerequisites

1. Docker and Docker Compose installed
2. An API key for an OpenAI-compatible LLM service (OpenRouter, OpenAI, etc.)

## Step 1: Get Your API Key

### Option A: OpenRouter (Recommended)
1. Go to https://openrouter.ai/
2. Sign up for an account
3. Navigate to "Keys" and create a new API key
4. Copy your API key (starts with `sk-or-v1-...`)

### Option B: OpenAI
1. Go to https://platform.openai.com/
2. Sign up or log in
3. Navigate to API keys and create a new key
4. Copy your API key (starts with `sk-...`)

## Step 2: Configure VICW

```bash
# Clone or navigate to the project directory
cd vicw_phase2

# Copy the example environment file
cp .env.example .env

# Edit the .env file
nano .env  # or use your preferred editor
```

**Minimum required configuration:**
```bash
# Add your API key
VICW_LLM_API_KEY=your_api_key_here

# For OpenRouter (recommended):
VICW_LLM_API_URL=https://api.openrouter.ai/api/v1/chat/completions
VICW_LLM_MODEL_NAME=mistralai/mistral-7b-instruct

# For OpenAI:
# VICW_LLM_API_URL=https://api.openai.com/v1/chat/completions
# VICW_LLM_MODEL_NAME=gpt-3.5-turbo

# Set Neo4j password
NEO4J_PASSWORD=your_secure_password_here
```

Save and close the file.

## Step 3: Start the Stack

```bash
# Start all services
docker-compose up -d

# Wait for services to initialize (about 30-60 seconds)
# You can watch the logs:
docker-compose logs -f vicw_api
```

Wait until you see:
```
INFO:     VICW Phase 2 API Server ready!
```

## Step 4: Test the System

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "system": "VICW Phase 2",
  "model": "mistralai/mistral-7b-instruct",
  "context_initialized": true,
  "llm_initialized": true
}
```

### Test 2: Send a Chat Message
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! Can you explain what the VICW system is?",
    "use_rag": true
  }'
```

You should get a response from the LLM!

### Test 3: Check Statistics
```bash
curl http://localhost:8000/stats
```

This shows context usage, queue status, and more.

## Step 5: Start Chatting!

You can now:

1. **Use the API**: Send POST requests to `http://localhost:8000/chat`
2. **Use the web UI** (if you've set one up): Navigate to `http://localhost:8000/docs` for the auto-generated API docs
3. **Use the CLI**: Run `docker-compose exec vicw_api python main.py`

## Common Commands

```bash
# View logs
docker-compose logs -f vicw_api

# Stop the system
docker-compose down

# Stop and remove all data
docker-compose down -v

# Restart a service
docker-compose restart vicw_api

# Check service status
docker-compose ps

# View resource usage
docker stats
```

## Troubleshooting

### "Connection refused" errors
Services may still be starting up. Wait 30-60 seconds and try again.

### "Authentication failed" for Neo4j
Check that `NEO4J_PASSWORD` in your `.env` file matches what you set.

### LLM API errors
- Verify your API key is correct
- Check you have credits/access to the model
- Try a different model

### High memory usage
Default configuration uses about 2-4GB RAM. To reduce:
```bash
# In .env:
MAX_CONTEXT_TOKENS=2048  # Instead of 4096
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Customize your system prompt in `system_prompt.txt`
- Adjust configuration in `.env` for your use case

## Getting Help

If you encounter issues:
1. Check the logs: `docker-compose logs -f`
2. Verify all services are healthy: `docker-compose ps`
3. Review the troubleshooting section in README.md

Enjoy your Virtual Infinite Context Window! 🚀
