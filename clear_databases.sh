#!/bin/bash
# Clear all VICW databases and restart services

echo "🗑️  Clearing VICW databases..."

# Clear Neo4j
echo "Clearing Neo4j..."
docker exec vicw_neo4j cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n" 2>&1 | grep -v "^$"

# Clear Redis
echo "Clearing Redis..."
docker exec vicw_redis redis-cli FLUSHALL

# Delete and recreate Qdrant collection
echo "Clearing Qdrant..."
curl -s -X DELETE "http://localhost:6333/collections/vicw_memory" > /dev/null 2>&1

# Restart API to recreate Qdrant collection
echo "Restarting API to recreate Qdrant collection..."
docker-compose restart vicw_api > /dev/null 2>&1

# Wait for API to be ready
echo "Waiting for API to be ready..."
sleep 10

# Verify everything is cleared
echo ""
echo "✅ Verification:"
echo "---------------"

# Check Neo4j
NEO4J_COUNT=$(docker exec vicw_neo4j cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n) AS count" 2>/dev/null | grep -E "^[0-9]+$" | head -1)
echo "Neo4j nodes: $NEO4J_COUNT"

# Check Redis
REDIS_COUNT=$(docker exec vicw_redis redis-cli DBSIZE 2>/dev/null | grep -o "[0-9]*")
echo "Redis keys: $REDIS_COUNT"

# Check Qdrant
QDRANT_COUNT=$(curl -s "http://localhost:6333/collections/vicw_memory" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['result']['points_count'])" 2>/dev/null)
echo "Qdrant points: $QDRANT_COUNT"

# Check API health
API_STATUS=$(curl -s http://localhost:8000/health 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
echo "API status: $API_STATUS"

echo ""
echo "✅ All databases cleared and services ready!"
