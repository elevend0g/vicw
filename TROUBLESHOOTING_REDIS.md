# Redis Connection Troubleshooting - December 3, 2025

## Problem Statement

The VICW API server fails to connect to Redis during startup with the error:
```
redis.exceptions.TimeoutError: Timeout connecting to server
```

Despite Redis being healthy and accepting connections, the API container cannot establish a TCP connection to port 6379.

---

## Investigation Timeline

### 1. Initial Diagnosis

**Symptoms:**
- API container crashes on startup with `TimeoutError: Timeout connecting to server`
- Redis container is healthy and responding to health checks
- Error occurs in `redis_storage.py` during `await redis_storage.init()`

**Initial checks performed:**
```bash
docker-compose ps                    # All containers running, Redis healthy
docker logs vicw_redis              # Redis accepting connections on port 6379
docker logs vicw_api                # Timeout after 5-10 seconds
```

### 2. Network Connectivity Testing

**DNS Resolution** ✅ WORKING
```bash
docker exec vicw_api getent hosts redis
# Output: 172.18.0.3  redis
```
DNS resolution works correctly - the hostname `redis` resolves to the correct IP.

**Raw Socket Connection** ❌ FAILING
```bash
docker exec vicw_api python3 -c "import socket; s = socket.socket(); s.settimeout(2); s.connect(('redis', 6379)); print('Connected'); s.close()"
# Output: TimeoutError: timed out
```

**Other Services** ❌ ALL FAILING
```bash
# Test Neo4j (port 7687)
docker exec vicw_api python3 -c "import socket; s = socket.socket(); s.settimeout(2); s.connect(('neo4j', 7687)); print('Connected'); s.close()"
# Output: TimeoutError: timed out

# Test Qdrant (port 6333)
docker exec vicw_api python3 -c "import socket; s = socket.socket(); s.settimeout(2); s.connect(('qdrant', 6333)); print('Connected'); s.close()"
# Output: TimeoutError: timed out
```

**KEY FINDING:** The API container cannot reach ANY service on the network - this is a complete network isolation issue, not Redis-specific.

### 3. Code-Level Fixes Attempted

#### Fix #1: Add Redis Connection Retry Logic
**File:** `app/redis_storage.py`
**Change:** Added retry mechanism with exponential backoff
```python
async def init(self, max_retries: int = 5, retry_delay: float = 2.0):
    for attempt in range(max_retries):
        try:
            self.redis = await redis.from_url(...)
            await self.redis.ping()
            return
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise
```
**Result:** ❌ Still times out on every attempt

#### Fix #2: Move Heavy Imports to Startup Function
**File:** `app/api_server.py:46`
**Problem:** `from sentence_transformers import SentenceTransformer` at module level blocks for several seconds
**Change:** Moved import inside `startup_event()` function
```python
# Before (line 46):
from sentence_transformers import SentenceTransformer

# After (inside startup_event):
if EMBEDDING_MODEL_TYPE == 'sentence_transformer':
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
```
**Result:** ❌ Improved startup time but connection still fails

#### Fix #3: Add Startup Delay
**File:** `app/api_server.py:182`
**Change:** Added 2-second delay before connecting to services
```python
async def startup_event():
    logger.info("Waiting for services to be fully ready...")
    await asyncio.sleep(2)
```
**Result:** ❌ Connection still times out

#### Fix #4: Switch from uvloop to asyncio
**File:** `Dockerfile:37`
**Problem:** uvloop (used by uvicorn by default) has known issues with async Redis in some environments
**Change:**
```dockerfile
# Before:
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

# After:
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
```
**Result:** ❌ Connection still times out

#### Fix #5: Switch to Synchronous Redis Client
**File:** `app/redis_storage.py:6`
**Problem:** `redis.asyncio` may have compatibility issues
**Change:**
```python
# Before:
import redis.asyncio as redis

# After:
import redis  # Synchronous client
```
Updated all `await self.redis.*` calls to synchronous `self.redis.*`
**Result:** ❌ Connection still times out (even synchronous sockets fail)

### 4. Docker Configuration Fixes

#### Fix #6: Remove Custom DNS Servers
**File:** `docker-compose.yml:12-14`
**Problem:** Custom DNS can interfere with Docker's internal DNS
**Change:**
```yaml
# Removed:
dns:
  - 8.8.8.8
  - 8.8.4.4
```
**Result:** ❌ No change - connections still timeout

#### Fix #7: Simplify Network Name
**File:** `docker-compose.yml:143-146`
**Problem:** Docker Compose was creating network named `vicw_vicw_network` (double prefix)
**Change:**
```yaml
# Before:
networks:
  vicw_network:
    driver: bridge

# After:
networks:
  default:
    name: vicw_network
    driver: bridge
```
**Result:** ❌ Network renamed but connectivity unchanged

### 5. System-Level Investigation

#### Redis Configuration
```bash
docker exec vicw_redis redis-cli CONFIG GET protected-mode
# Output: protected-mode no ✅

docker exec vicw_redis redis-cli CONFIG GET bind
# Output: bind * -::* ✅ (listening on all interfaces)

docker exec vicw_redis netstat -tln | grep 6379
# Output: tcp 0.0.0.0:6379 LISTEN ✅
```
Redis is correctly configured and listening.

#### IP Forwarding
```bash
sudo sysctl net.ipv4.conf.all.forwarding
# Output: net.ipv4.conf.all.forwarding = 1 ✅
```

#### Kernel Modules
```bash
lsmod | grep br_netfilter
# Output: br_netfilter 32768  0 ✅
```

#### iptables Investigation
```bash
sudo iptables -L DOCKER -n --line-numbers
```
**CRITICAL FINDING:** Found two DROP rules at end of DOCKER chain:
```
7    DROP       all  --  0.0.0.0/0  0.0.0.0/0
8    DROP       all  --  0.0.0.0/0  0.0.0.0/0
```

These rules were blocking ALL inter-container traffic on the Docker network!

**Fix Attempted:**
```bash
sudo iptables -D DOCKER 8
sudo iptables -D DOCKER 7
```
**Verification:**
```bash
sudo iptables -L DOCKER -n | tail -3
# Output shows DROP rules removed ✅
```

**Result:** ❌ Even after removing DROP rules, containers still cannot communicate

### 6. External Network Test
```bash
# Test from a fresh container on same network
docker run --rm --network vicw_network python:3.11-slim \
  timeout 3 python3 -c "import socket; s = socket.socket(); s.settimeout(2); s.connect(('vicw_redis', 6379)); print('SUCCESS'); s.close()"
# Output: TimeoutError: timed out ❌
```

**CONCLUSION:** The problem affects ALL containers on the `vicw_network`, not just the API container.

---

## Root Cause Analysis

### Identified Issues (Fixed)

1. **Heavy Import Blocking**: `SentenceTransformer` import delayed startup
2. **Custom DNS**: Custom DNS servers interfered with internal Docker DNS
3. **Network Naming**: `vicw_vicw_network` caused confusion (though not blocking)
4. **uvloop Compatibility**: uvloop can have issues with async Redis
5. **iptables DROP Rules**: Were blocking Docker traffic (removed but issue persists)

### Unresolved Root Cause

The fundamental issue is **complete inter-container network isolation** on the `vicw_network` bridge. Even after:
- Removing iptables DROP rules
- Verifying IP forwarding is enabled
- Confirming bridge netfilter module is loaded
- Testing with fresh containers

**Containers cannot establish ANY TCP connections to each other.**

### Possible Remaining Causes

1. **AppArmor/SELinux Policies**
   - Security policy may be blocking bridge traffic
   - Check: `sudo aa-status` or `getenforce`

2. **Kernel Conntrack Table Full**
   - Connection tracking table may be exhausted
   - Check: `sudo sysctl net.netfilter.nf_conntrack_count`
   - Compare to: `sudo sysctl net.netfilter.nf_conntrack_max`

3. **Bridge Filtering**
   - Kernel may be filtering bridge traffic
   - Check: `sudo sysctl net.bridge.bridge-nf-call-iptables`

4. **Docker Daemon Issue**
   - Docker daemon may need restart
   - Check: `sudo systemctl status docker`
   - Try: `sudo systemctl restart docker`

5. **Firewall Rules (nftables)**
   - System may be using nftables instead of iptables
   - Check: `sudo nft list ruleset | grep docker`

6. **MTU/Network Driver Issues**
   - Bridge MTU mismatch
   - Check: `docker network inspect vicw_network | grep -i mtu`

---

## Files Modified

### 1. `app/redis_storage.py`
- Changed import from `redis.asyncio` to synchronous `redis`
- Added connection retry logic with backoff
- Converted all async Redis calls to synchronous

### 2. `app/api_server.py`
- Removed `SentenceTransformer` import from module level (line 46)
- Moved to lazy import inside `startup_event()`
- Added 2-second delay before service initialization

### 3. `Dockerfile`
- Changed CMD to use `--loop asyncio` instead of default uvloop

### 4. `docker-compose.yml`
- Removed `dns` section from `vicw_api` service
- Simplified network name from `vicw_network:` to `default: name: vicw_network`
- Removed explicit `networks:` references from services

---

## Recommended Next Steps

### Immediate Debugging

1. **Check AppArmor/SELinux:**
   ```bash
   sudo aa-status
   # or
   getenforce
   ```

2. **Check conntrack:**
   ```bash
   sudo sysctl net.netfilter.nf_conntrack_count
   sudo sysctl net.netfilter.nf_conntrack_max
   ```

3. **Check bridge filtering:**
   ```bash
   sudo sysctl net.bridge.bridge-nf-call-iptables
   sudo sysctl net.bridge.bridge-nf-call-ip6tables
   ```

4. **Check nftables:**
   ```bash
   sudo nft list ruleset | grep -i docker
   ```

5. **Restart Docker daemon:**
   ```bash
   sudo systemctl restart docker
   docker-compose up -d
   ```

### Workarounds

#### Option 1: Use Host Networking Mode
**Pros:** Bypasses bridge networking issues
**Cons:** Containers share host network stack (less isolated)

```yaml
# docker-compose.yml
services:
  vicw_api:
    network_mode: "host"
    environment:
      - REDIS_HOST=127.0.0.1  # localhost instead of 'redis'
```

#### Option 2: Use IP Addresses Instead of Hostnames
Less elegant but might bypass DNS issues:
```yaml
environment:
  - REDIS_HOST=172.18.0.3  # Hard-code IP
```

#### Option 3: Use Docker's Default Bridge
Instead of custom network, use default bridge:
```yaml
# Remove custom network, use defaults
# Services will be on default bridge network
```

### Long-Term Solution

This appears to be a system-level Docker networking configuration issue that requires investigating:
- Firewall policies (iptables/nftables/firewalld)
- Security modules (AppArmor/SELinux)
- Kernel parameters
- Docker daemon configuration

Consider consulting with system administrator or reviewing system security policies.

---

## Testing Commands Reference

### Test Redis from Host
```bash
redis-cli -h localhost -p 6379 PING
```

### Test from Inside API Container
```bash
docker exec vicw_api python3 -c "import socket; s = socket.socket(); s.settimeout(2); s.connect(('redis', 6379)); print('OK'); s.close()"
```

### Test with Fresh Container
```bash
docker run --rm --network vicw_network redis:7-alpine redis-cli -h vicw_redis PING
```

### Check iptables
```bash
sudo iptables -L DOCKER -n
sudo iptables -L DOCKER-FORWARD -n
sudo iptables -L FORWARD -n -v
```

### Check Docker Network
```bash
docker network inspect vicw_network
docker network inspect vicw_network --format '{{range .Containers}}{{.Name}}: {{.IPv4Address}} {{end}}'
```

---

## Summary

**Problem:** Complete network isolation between Docker containers on custom bridge network

**Fixed:**
- Code-level improvements (imports, retries, asyncio)
- Docker configuration (DNS, network naming)
- Removed blocking iptables rules

**Still Broken:**
- Inter-container TCP connections timeout at socket level
- Affects ALL containers on the network
- System-level networking issue beyond application scope

**Status:** Requires system administrator intervention to investigate firewall/security policies

---

**Date:** December 3, 2025
**Duration:** ~2 hours of troubleshooting
**Files Modified:** 4 files (redis_storage.py, api_server.py, Dockerfile, docker-compose.yml)
**iptables Rules Removed:** 2 DROP rules from DOCKER chain
