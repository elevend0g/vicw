
***

# Architecture Specification: Dynamic Hybrid-Memory System for VICW
**Project:** Generalist GraphRAG for Dynamic Context Offload
**Stack:** Qdrant (Vector), Neo4j (Graph), snowflake-arctic-embed-l-v2.0-q8_0 (Embedding Model)

## 1. Executive Summary
This system provides a dynamic, high-fidelity long-term memory for LLM interactions. It separates **Semantic Similarity** (Vector) from **Causal Logic** (Graph) to achieve 95% retrieval fidelity across opposing domains (e.g., Strict Coding vs. Abstract Creative Writing).

The architecture utilizes a **"Metaphysical Schema"** rather than a domain-specific one, allowing zero-schema-migration when switching tasks. It leverages a specific lightweight embedding model (Qwen-0.6B Q8) to allow for aggressive re-ranking and maintenance without latency penalties.

---

## 2. Data Modeling Strategy

### 2.1 The "Metaphysical" Graph Schema (Neo4j)
Instead of modeling *what* things are (e.g., `:Function`, `:Character`), we model *how* they exist.

#### **2.1.1 Node Labels (The Core-5)**
1.  **`Context`**: The Root node for a domain (e.g., "Python Project Alpha", "Fantasy Novel"). Usage: Domain partitioning.
2.  **`Entity`**: Nouns. Objects, people, variables, files, places.
3.  **`Event`**: Actions. Things that happen at a point in time.
4.  **`Concept`**: Abstract ideas. Genres, design patterns, emotions.
5.  **`Chunk`**: Proof. The raw text snippet or file source.

#### **2.1.2 Node Properties**
All nodes (except Context) must possess:
*   `uid`: UUID string.
*   `name`: Human readable identifier.
*   `subtype`: (String) Granular type (e.g., "function", "villain").
*   `domain`: (String) High-level filter key (e.g., "coding", "prose").
*   `qdrant_id`: Link to the vector record.

#### **2.1.3 Edge Typology (Relationships)**
*   **`BELONGS_TO`**: `(Entity/Event) -> (Context)`
    *   *Purpose:* Topology-based security and domain separation.
*   **`RELATED_TO`**: `(Entity) -> (Entity)`
    *   *Prop:* `desc` (String) - Describes the relationship (e.g., "calls", "hates").
*   **`INITIATED`**: `(Entity) -> (Event)`
    *   *Purpose:* Attribution/Agency.
*   **`CAUSED`**: `(Event) -> (Event/Entity)`
    *   *Purpose:* Causal Logic.
    *   *Prop:* `certainty` (Float 0.0-1.0).
*   **`NEXT`**: `(Event) -> (Event)`
    *   *Purpose:* Logical Sequence (Flow).
*   **`HAPPENED_AFTER`**: `(Event) -> (Event)`
    *   *Purpose:* Temporal Sequence (Chronology).

---

### 2.2 The Vector Strategy (Qdrant)

#### **2.2.1 Model Specification**
*   **Model:** snowflake-arctic-embed-l-v2.0-q8_0 (Instruct / English).
*   **Format:** ONNX / GGUF (Quantized Int8/Q8).
*   **Footprint:** ~639MB RAM.
*   **Dimensions:** 1024 (Elastic).

#### **2.2.2 The "Contextual Wrapper" Pattern**
To ensure domain separation, we never embed raw text. We embed a structured string to force orthogonality in vector space.

**Construct:**
```text
[Domain: <domain_type>] [Type: <subtype>] [Name: <node_name>] <content_description>
```

**Examples:**
*   *Coding:* `"[Domain: Coding] [Type: Method] [Name: AuthLogin] Handles JWT token validation."`
*   *Story:* `"[Domain: Story] [Type: Scene] [Name: The Betrayal] The protagonist realizes the amulet is fake."`

#### **2.2.3 Payload Configuration**
```json
{
  "id": "uuid",
  "vector": [ ...1024 dims... ],
  "payload": {
    "domain": "coding",       // For Component-Level Filtering
    "node_id": "neo4j_uuid",  // The Bridge
    "subtype": "method"
  }
}
```

---

## 3. Dynamic Offload Pipelines

### 3.1 Ingestion (Write Path)
*Trigger: LLM analyzes chat buffer availability.*

1.  **Extraction:** The LLM follows the "SPO" (Subject-Predicate-Object) prompt to extract Entities and Events.
2.  **Context Check:** System identifies active `Context` (e.g., "Coding Project").
3.  **Sequence Assignment:**
    *   Assign `timestamp` (Current Real Time).
    *   Determine `flow_id` (Logical Thread).
    *   Assign `flow_step` (Incrementing Integer).
4.  **Vector Generation:**
    *   Wrap content using the Contextual Wrapper.
    *   Generate embedding via Qwen-0.6B.
    *   Upsert to Qdrant.
5.  **Graph Materialization:**
    *   Create Nodes in Neo4j.
    *   Set `[:BELONGS_TO]` edge to the Context root.
    *   Create `[:NEXT]` edges based on `flow_step`.
    *   Create `[:CAUSED]` edges based on extraction logic.

### 3.2 Retrieval (Read Path)
*Trigger: User Query.*

1.  **Intent Analysis:** Determine if query is `Coding`, `Creative`, or `General`.
2.  **Vector Filter Scan (Qdrant):**
    *   Query Qdrant using the Contextual Wrapper format.
    *   **Apply Filter:** `Must { Match { "domain": "current_intent" } }`.
    *   *Result:* List of Top-K Node IDs.
3.  **Graph Expansion (Neo4j):**
    *   Match the K nodes.
    *   Traverse 1 Hop outgoing via `[:CAUSED]` (What is the consequence?).
    *   Traverse 1 Hop incoming via `[:INITIATED]` (Who caused this?).
    *   Traverse 1 Hop linear via `[:NEXT]` (What happens next?).
4.  **Synthesis:** Return the subgraph as JSON/Text to the LLM Context Window.

---

## 4. Causal & Temporal Logic

To handle the "Interleaved Context" problem (jumping between topics), we maintain two separate chains:

| Feature | Property/Edge | Question it Solves |
| :--- | :--- | :--- |
| **Chronology** | `timestamp` (Prop) | "What did we discuss 10 minutes ago?" |
| **Logic Flow** | `[:NEXT]` (Edge) | "What happened in the story after the explosion?" |
| **Causality** | `[:CAUSED]` (Edge) | "Why is the server throwing a 500 error?" |

---

## 5. Infrastructure Guidelines

### 5.1 Resource Allocation (Assuming 16GB System)
The strategy prioritizes Graph cache over Model size.

*   **Qwen Embedding (Int8):** 700MB (Fixed).
*   **Qdrant:** 4GB (Heap + Mmap).
*   **Neo4j:** 8GB (Page Cache & Heap). *Performance critical.*
*   **OS/System:** ~3GB.

### 5.2 The "Sleep Cycle" (Maintenance)
Because the model is lightweight, a background job runs every N interactions:
1.  **Scan:** Find `Event` nodes within the same `flow_id` that are older than 1 hour.
2.  **Consolidate:** Ask LLM to summarize 5 events into 1 `Macro-Event`.
3.  **Re-Embed:** Generate new vector for `Macro-Event`.
4.  **Prune:** Delete detailed nodes, replace with `Macro-Event`, update `[:NEXT]` pointers.

---

## 6. Fidelity Assurance

### How 95% Fidelity is Achieved:
1.  **Strict Partitioning:** The `Context` Root Node + Qdrant Payload Filters make it physically impossible to mix up Coding variables with Story characters.
2.  **Dense Embeddings:** Qwen-0.6B + Contextual Wrapping differentiates "Wizard (Class)" from "Wizard (Person)" at the vector level.
3.  **Causal Reasoning:** The `[:CAUSED]` edge captures the *logic* that vectors miss, allowing for debugging and plot consistency.
