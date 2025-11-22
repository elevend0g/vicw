"""Data models for VICW Phase 2"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import time
import uuid


@dataclass
class OffloadJob:
    """Represents a job to be offloaded to cold path"""
    job_id: str
    chunk_text: str
    metadata: Dict[str, Any]
    timestamp: float
    token_count: int
    message_count: int
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def create(cls, chunk_text: str, token_count: int, message_count: int, metadata: Optional[Dict] = None):
        """Factory method to create a new offload job"""
        return cls(
            job_id=f"job_{uuid.uuid4().hex}",
            chunk_text=chunk_text,
            metadata=metadata or {},
            timestamp=time.time(),
            token_count=token_count,
            message_count=message_count
        )


@dataclass
class Message:
    """Chat message with metadata"""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0

    def to_dict(self) -> Dict[str, str]:
        """Convert to LLM API format"""
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass
class PinnedHeader:
    """Persistent context header that never gets offloaded"""
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    definitions: Dict[str, str] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=lambda: {
        "step_id": "init",
        "next": None,
        "blockers": []
    })
    active_entities: List[str] = field(default_factory=list)
    active_artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_context_message(self) -> Dict[str, str]:
        """Convert to context message format"""
        content_parts = []
        
        if self.goals:
            content_parts.append(f"GOALS: {', '.join(self.goals)}")
        
        if self.constraints:
            content_parts.append(f"CONSTRAINTS: {', '.join(self.constraints)}")
        
        if self.definitions:
            defs = [f"{k}: {v}" for k, v in self.definitions.items()]
            content_parts.append(f"DEFINITIONS: {', '.join(defs)}")
        
        if self.plan and self.plan.get('step_id') != 'init':
            content_parts.append(f"CURRENT PLAN: {self.plan}")
        
        if self.active_entities:
            content_parts.append(f"ACTIVE ENTITIES: {', '.join(self.active_entities)}")
        
        if self.active_artifacts:
            content_parts.append(f"ACTIVE ARTIFACTS: {', '.join(self.active_artifacts)}")
        
        if not content_parts:
            return None
        
        return {
            "role": "system",
            "content": "[PINNED STATE]\n" + "\n".join(content_parts) + "\n[END PINNED STATE]"
        }


@dataclass
class OffloadResult:
    """Result of successful offload processing"""
    job_id: str
    summary: str
    embedding: List[float]
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class RAGResult:
    """Result from RAG retrieval"""
    semantic_chunks: List[str] = field(default_factory=list)
    relational_facts: List[str] = field(default_factory=list)
    retrieval_time_ms: float = 0.0

    @property
    def total_items(self) -> int:
        """Total number of retrieved items"""
        return len(self.semantic_chunks) + len(self.relational_facts)

    def is_empty(self) -> bool:
        """Check if no results were retrieved"""
        return self.total_items == 0

    def to_context_message(self) -> Optional[Dict[str, str]]:
        """Convert to context message format"""
        if self.is_empty():
            return None
        
        content_parts = ["[RETRIEVED LONG-TERM KNOWLEDGE]"]
        
        if self.semantic_chunks:
            content_parts.append("\n[SEMANTIC MEMORY]")
            for i, chunk in enumerate(self.semantic_chunks, 1):
                content_parts.append(f"{i}. {chunk}")
        
        if self.relational_facts:
            content_parts.append("\n[RELATIONAL STATE]")
            content_parts.extend(self.relational_facts)
        
        content_parts.append("\n[END RETRIEVED KNOWLEDGE]")
        
        return {
            "role": "system",
            "content": "\n".join(content_parts)
        }
