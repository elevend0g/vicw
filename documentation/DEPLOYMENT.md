# Production Deployment Guide - VICW Phase 2

This guide covers deploying VICW Phase 2 in a production environment with proper security, monitoring, and scalability.

## Table of Contents
1. [Infrastructure Requirements](#infrastructure-requirements)
2. [Security Best Practices](#security-best-practices)
3. [Scaling Configuration](#scaling-configuration)
4. [Monitoring and Logging](#monitoring-and-logging)
5. [Backup and Recovery](#backup-and-recovery)
6. [Performance Tuning](#performance-tuning)

## Infrastructure Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Disk**: 50 GB SSD
- **Network**: 100 Mbps

### Recommended for Production
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Disk**: 100+ GB NVMe SSD
- **Network**: 1 Gbps

### Service Resource Allocation

```yaml
# Recommended docker-compose resource limits
services:
  vicw_api:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          cpus: '2'
          memory: 2G
  
  redis:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
  
  qdrant:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
  
  neo4j:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

## Security Best Practices

### 1. API Key Management

**Never commit API keys to version control!**

```bash
# Use a secrets manager in production
# Example with Docker Swarm secrets:
echo "your_api_key_here" | docker secret create llm_api_key -

# Reference in docker-compose.yml:
secrets:
  - llm_api_key
```

### 2. Network Security

```yaml
# docker-compose.yml - Production networking
services:
  vicw_api:
    ports:
      - "127.0.0.1:8000:8000"  # Only bind to localhost
    networks:
      - vicw_internal
      - vicw_external
  
  redis:
    ports: []  # Don't expose externally
    networks:
      - vicw_internal
  
  qdrant:
    ports: []
    networks:
      - vicw_internal
  
  neo4j:
    ports: []
    networks:
      - vicw_internal

networks:
  vicw_internal:
    internal: true
  vicw_external:
    driver: bridge
```

### 3. Reverse Proxy with TLS

Use Nginx or Traefik with SSL/TLS:

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long LLM generation
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }
}
```

### 4. Rate Limiting

Add rate limiting to prevent abuse:

```python
# In api_server.py, add:
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("10/minute")  # 10 requests per minute
async def chat(request: Request, chat_request: ChatRequest):
    # ... existing code
```

## Scaling Configuration

### Horizontal Scaling with Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose-prod.yml vicw

# Scale API service
docker service scale vicw_vicw_api=3
```

### Load Balancing

```yaml
# docker-compose-prod.yml
version: '3.8'
services:
  vicw_api:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        max_attempts: 3
  
  nginx_lb:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - vicw_api
```

### Database Scaling

**Redis Cluster**:
```bash
# For high availability, use Redis Sentinel or Cluster
# Example with Redis Sentinel:
docker-compose -f docker-compose.redis-sentinel.yml up -d
```

**Qdrant Scaling**:
```yaml
# For large datasets, use Qdrant with multiple nodes
qdrant:
  environment:
    - QDRANT__CLUSTER__ENABLED=true
    - QDRANT__CLUSTER__P2P__PORT=6335
```

**Neo4j Clustering**:
```yaml
# For high availability, use Neo4j Causal Cluster
# Requires Neo4j Enterprise Edition
neo4j:
  image: neo4j:5-enterprise
  environment:
    - NEO4J_EDITION=ENTERPRISE
    - NEO4J_dbms_mode=CORE
```

## Monitoring and Logging

### 1. Application Metrics

Add Prometheus metrics:

```python
# Add to requirements.txt:
# prometheus-client==0.19.0

# In api_server.py:
from prometheus_client import Counter, Histogram, generate_latest

chat_requests = Counter('vicw_chat_requests_total', 'Total chat requests')
chat_duration = Histogram('vicw_chat_duration_seconds', 'Chat request duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 2. Structured Logging

```python
# Use structlog for better log parsing
import structlog

logger = structlog.get_logger()
logger.info("chat_request", user_id="123", tokens=2048, rag_enabled=True)
```

### 3. Monitoring Stack

```yaml
# monitoring/docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  prometheus_data:
  grafana_data:
```

### 4. Log Aggregation

Use ELK stack or Loki for centralized logging:

```yaml
loki:
  image: grafana/loki
  ports:
    - "3100:3100"

promtail:
  image: grafana/promtail
  volumes:
    - /var/log:/var/log
    - ./promtail-config.yml:/etc/promtail/config.yml
```

## Backup and Recovery

### 1. Database Backups

**Redis**:
```bash
# Automated backup script
#!/bin/bash
docker exec vicw_redis redis-cli BGSAVE
docker cp vicw_redis:/data/dump.rdb ./backups/redis-$(date +%Y%m%d).rdb
```

**Qdrant**:
```bash
# Backup Qdrant snapshots
curl -X POST "http://localhost:6333/collections/vicw_memory/snapshots"
```

**Neo4j**:
```bash
# Backup Neo4j database
docker exec vicw_neo4j neo4j-admin database backup neo4j \
  --to-path=/backups/neo4j-$(date +%Y%m%d)
```

### 2. Automated Backup Schedule

```bash
# crontab -e
0 2 * * * /opt/vicw/backup.sh  # Daily at 2 AM
```

### 3. Disaster Recovery

```bash
# Restore from backup
docker-compose down
docker volume create vicw_redis_data
docker run --rm -v vicw_redis_data:/data \
  -v ./backups:/backups alpine \
  cp /backups/redis-20250120.rdb /data/dump.rdb
docker-compose up -d
```

## Performance Tuning

### 1. Context Window Sizing

```bash
# Adjust based on your use case and LLM pricing
MAX_CONTEXT_TOKENS=8192  # For longer conversations
# or
MAX_CONTEXT_TOKENS=2048  # For cost optimization
```

### 2. Offload Tuning

```bash
# Aggressive offloading (more cost-effective)
OFFLOAD_THRESHOLD=0.70
TARGET_AFTER_RELIEF=0.40

# Conservative offloading (more context retained)
OFFLOAD_THRESHOLD=0.90
TARGET_AFTER_RELIEF=0.70
```

### 3. Cold Path Optimization

```bash
# More workers for faster processing
COLD_PATH_WORKERS=8
COLD_PATH_BATCH_SIZE=5

# Fewer workers to reduce CPU usage
COLD_PATH_WORKERS=2
COLD_PATH_BATCH_SIZE=2
```

### 4. Database Tuning

**Redis**:
```bash
# In redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
```

**Qdrant**:
```yaml
environment:
  - QDRANT__SERVICE__MAX_CONCURRENT_REQUESTS=10
  - QDRANT__STORAGE__PERFORMANCE__OPTIMIZERS__DELETED_THRESHOLD=0.2
```

**Neo4j**:
```yaml
environment:
  - NEO4J_dbms_memory_pagecache_size=2G
  - NEO4J_dbms_memory_heap_max__size=4G
  - NEO4J_dbms_query__cache__size=1000
```

## Health Checks and Alerts

### 1. Automated Health Checks

```bash
# healthcheck.sh
#!/bin/bash
response=$(curl -s http://localhost:8000/health)
if ! echo "$response" | grep -q '"status":"healthy"'; then
    echo "VICW API is unhealthy!"
    # Send alert (email, Slack, PagerDuty, etc.)
    curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
      -d '{"text":"VICW API health check failed"}'
fi
```

### 2. Resource Monitoring

```bash
# Monitor resource usage
docker stats --no-stream --format \
  "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
```

## Maintenance

### 1. Regular Updates

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Update Docker images
docker-compose pull
docker-compose up -d
```

### 2. Log Rotation

```yaml
# docker-compose.yml
services:
  vicw_api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 3. Database Maintenance

```bash
# Redis: Compact and optimize
docker exec vicw_redis redis-cli BGREWRITEAOF

# Neo4j: Consistency check
docker exec vicw_neo4j neo4j-admin check-consistency \
  --database=neo4j --verbose
```

## Checklist for Production

- [ ] Secure API keys in secrets manager
- [ ] Enable TLS/SSL with valid certificates
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting
- [ ] Implement automated backups
- [ ] Configure log rotation
- [ ] Set resource limits
- [ ] Use a reverse proxy
- [ ] Test disaster recovery procedures
- [ ] Document runbooks for common issues
- [ ] Set up health checks
- [ ] Enable firewall rules
- [ ] Review and harden database security
- [ ] Implement authentication/authorization
- [ ] Set up CI/CD for deployments

## Support

For production support:
- Review logs: `docker-compose logs -f`
- Check metrics: `curl http://localhost:8000/stats`
- Monitor resources: `docker stats`
- Review backup status: Check backup directory

---

Remember: Production deployments require ongoing maintenance, monitoring, and security updates. Always test changes in a staging environment first!
