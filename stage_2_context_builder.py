# -*- coding: utf-8 -*-
"""
Stage 2: Enhanced Context Builder + Prompt Versioning for AI Research Agent
Advanced context management with memory tiers and adaptive packing
"""

import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import re
from jinja2 import Template, Environment, FileSystemLoader

class MemoryTier(Enum):
    SHORT_TERM = "short_term"      # Current conversation
    EPISODIC = "episodic"          # Current session
    LONG_TERM = "long_term"        # Persistent across sessions
    GRAPH_MEMORY = "graph_memory"  # Semantic graph storage

class TaskType(Enum):
    QA = "question_answering"
    CODE_REPAIR = "code_repair"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"

class ContextPackingStrategy(Enum):
    RECENCY_FIRST = "recency_first"
    DIVERSITY_FIRST = "diversity_first"
    RELEVANCE_FIRST = "relevance_first"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"

@dataclass
class MemoryItem:
    """Individual memory item with tier classification"""
    id: str
    content: str
    memory_tier: MemoryTier
    timestamp: datetime
    relevance_score: float
    access_count: int
    last_accessed: datetime
    metadata: Dict[str, Any]
    token_count: int
    importance_score: float

@dataclass
class PromptTemplate:
    """Versioned prompt template"""
    template_id: str
    name: str
    version: str
    template_content: str
    variables: List[str]
    task_types: List[TaskType]
    created_at: datetime
    performance_metrics: Dict[str, float]
    active: bool

@dataclass
class ContextPackingResult:
    """Result of context packing operation"""
    packed_items: List[MemoryItem]
    total_tokens: int
    packing_strategy: ContextPackingStrategy
    diversity_score: float
    recency_score: float
    relevance_score: float
    compression_ratio: float
    metadata: Dict[str, Any]

class MemoryTierManager:
    """Manages different tiers of memory with intelligent retrieval"""
    
    def __init__(self, max_tokens_per_tier: Dict[MemoryTier, int] = None):
        self.max_tokens_per_tier = max_tokens_per_tier or {
            MemoryTier.SHORT_TERM: 2000,
            MemoryTier.EPISODIC: 4000,
            MemoryTier.LONG_TERM: 8000,
            MemoryTier.GRAPH_MEMORY: 6000
        }
        
        # Memory storage by tier
        self.memory_tiers: Dict[MemoryTier, List[MemoryItem]] = {
            tier: [] for tier in MemoryTier
        }
        
        # Access patterns for optimization
        self.access_patterns: Dict[str, List[datetime]] = {}
        
        print("🧠 Memory Tier Manager initialized")
        print(f"   Token limits: {dict(self.max_tokens_per_tier)}")
    
    def store_memory(self, content: str, memory_tier: MemoryTier,
                    relevance_score: float = 0.5, metadata: Dict[str, Any] = None) -> str:
        """Store a memory item in the appropriate tier"""
        
        import uuid
        memory_id = str(uuid.uuid4())
        
        # Estimate token count (rough approximation)
        token_count = len(content.split()) * 1.3  # Account for tokenization
        
        # Calculate importance score
        importance_score = self._calculate_importance(content, relevance_score, metadata or {})
        
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            memory_tier=memory_tier,
            timestamp=datetime.now(),
            relevance_score=relevance_score,
            access_count=0,
            last_accessed=datetime.now(),
            metadata=metadata or {},
            token_count=int(token_count),
            importance_score=importance_score
        )
        
        # Add to appropriate tier
        self.memory_tiers[memory_tier].append(memory_item)
        
        # Manage tier capacity
        self._manage_tier_capacity(memory_tier)
        
        return memory_id
    
    def retrieve_memories(self, query: str, memory_tiers: List[MemoryTier] = None,
                         max_items: int = 10, relevance_threshold: float = 0.3) -> List[MemoryItem]:
        """Retrieve relevant memories from specified tiers"""
        
        if memory_tiers is None:
            memory_tiers = list(MemoryTier)
        
        all_candidates = []
        
        # Collect candidates from all specified tiers
        for tier in memory_tiers:
            for memory_item in self.memory_tiers[tier]:
                # Simple relevance scoring (in practice, use embeddings)
                relevance = self._calculate_query_relevance(query, memory_item.content)
                
                if relevance >= relevance_threshold:
                    # Update access patterns
                    memory_item.access_count += 1
                    memory_item.last_accessed = datetime.now()
                    
                    # Track access pattern
                    if memory_item.id not in self.access_patterns:
                        self.access_patterns[memory_item.id] = []
                    self.access_patterns[memory_item.id].append(datetime.now())
                    
                    all_candidates.append((memory_item, relevance))
        
        # Sort by relevance and recency
        all_candidates.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
        
        # Return top candidates
        return [item for item, _ in all_candidates[:max_items]]
    
    def promote_memory(self, memory_id: str, target_tier: MemoryTier) -> bool:
        """Promote a memory item to a higher tier"""
        
        # Find the memory item
        source_tier = None
        memory_item = None
        
        for tier, items in self.memory_tiers.items():
            for item in items:
                if item.id == memory_id:
                    source_tier = tier
                    memory_item = item
                    break
            if memory_item:
                break
        
        if not memory_item or source_tier == target_tier:
            return False
        
        # Remove from source tier
        self.memory_tiers[source_tier].remove(memory_item)
        
        # Update tier and add to target
        memory_item.memory_tier = target_tier
        memory_item.importance_score += 0.1  # Boost importance
        self.memory_tiers[target_tier].append(memory_item)
        
        # Manage target tier capacity
        self._manage_tier_capacity(target_tier)
        
        return True
    
    def get_tier_statistics(self) -> Dict[str, Any]:
        """Get statistics for all memory tiers"""
        
        stats = {}
        
        for tier, items in self.memory_tiers.items():
            total_tokens = sum(item.token_count for item in items)
            avg_relevance = sum(item.relevance_score for item in items) / max(len(items), 1)
            avg_importance = sum(item.importance_score for item in items) / max(len(items), 1)
            
            stats[tier.value] = {
                "item_count": len(items),
                "total_tokens": total_tokens,
                "token_utilization": total_tokens / self.max_tokens_per_tier[tier],
                "avg_relevance": avg_relevance,
                "avg_importance": avg_importance,
                "oldest_item": min(items, key=lambda x: x.timestamp).timestamp.isoformat() if items else None,
                "most_accessed": max(items, key=lambda x: x.access_count).access_count if items else 0
            }
        
        return stats
    
    def _calculate_importance(self, content: str, relevance_score: float, metadata: Dict[str, Any]) -> float:
        """Calculate importance score for a memory item"""
        
        # Base importance from relevance
        importance = relevance_score
        
        # Boost for certain content types
        if "research_finding" in metadata:
            importance += 0.2
        if "hypothesis" in metadata:
            importance += 0.15
        if "tool_result" in metadata:
            importance += 0.1
        
        # Boost for longer, more detailed content
        content_length_factor = min(len(content) / 1000, 0.2)
        importance += content_length_factor
        
        # Boost for structured content
        if any(marker in content for marker in ["1.", "2.", "•", "-", "**"]):
            importance += 0.1
        
        return min(importance, 1.0)
    
    def _calculate_query_relevance(self, query: str, content: str) -> float:
        """Calculate relevance between query and content (simplified)"""
        
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _manage_tier_capacity(self, tier: MemoryTier):
        """Manage capacity of a memory tier"""
        
        items = self.memory_tiers[tier]
        max_tokens = self.max_tokens_per_tier[tier]
        
        # Calculate current token usage
        current_tokens = sum(item.token_count for item in items)
        
        # If over capacity, remove least important items
        while current_tokens > max_tokens and items:
            # Sort by importance (considering recency and access patterns)
            items.sort(key=lambda x: (
                x.importance_score,
                x.access_count,
                x.timestamp
            ))
            
            # Remove least important item
            removed_item = items.pop(0)
            current_tokens -= removed_item.token_count
            
            # If removing from short-term, consider promoting to episodic
            if tier == MemoryTier.SHORT_TERM and removed_item.access_count > 2:
                self.memory_tiers[MemoryTier.EPISODIC].append(removed_item)
                removed_item.memory_tier = MemoryTier.EPISODIC

class AdaptiveContextPacker:
    """Intelligent context packing with token awareness and trade-offs"""
    
    def __init__(self, max_context_tokens: int = 8000):
        self.max_context_tokens = max_context_tokens
        self.packing_history: List[ContextPackingResult] = []
        
        print(f"📦 Adaptive Context Packer initialized (max tokens: {max_context_tokens})")
    
    def pack_context(self, memory_items: List[MemoryItem], task_type: TaskType,
                    strategy: ContextPackingStrategy = ContextPackingStrategy.ADAPTIVE,
                    diversity_weight: float = 0.3, recency_weight: float = 0.3,
                    relevance_weight: float = 0.4) -> ContextPackingResult:
        """Pack context items optimally within token limits"""
        
        if strategy == ContextPackingStrategy.ADAPTIVE:
            strategy = self._select_adaptive_strategy(memory_items, task_type)
        
        # Apply packing strategy
        if strategy == ContextPackingStrategy.RECENCY_FIRST:
            packed_items = self._pack_recency_first(memory_items)
        elif strategy == ContextPackingStrategy.DIVERSITY_FIRST:
            packed_items = self._pack_diversity_first(memory_items)
        elif strategy == ContextPackingStrategy.RELEVANCE_FIRST:
            packed_items = self._pack_relevance_first(memory_items)
        else:  # BALANCED
            packed_items = self._pack_balanced(memory_items, diversity_weight, recency_weight, relevance_weight)
        
        # Calculate metrics
        total_tokens = sum(item.token_count for item in packed_items)
        diversity_score = self._calculate_diversity_score(packed_items)
        recency_score = self._calculate_recency_score(packed_items)
        relevance_score = sum(item.relevance_score for item in packed_items) / max(len(packed_items), 1)
        compression_ratio = len(packed_items) / max(len(memory_items), 1)
        
        result = ContextPackingResult(
            packed_items=packed_items,
            total_tokens=total_tokens,
            packing_strategy=strategy,
            diversity_score=diversity_score,
            recency_score=recency_score,
            relevance_score=relevance_score,
            compression_ratio=compression_ratio,
            metadata={
                "original_item_count": len(memory_items),
                "task_type": task_type.value,
                "token_utilization": total_tokens / self.max_context_tokens
            }
        )
        
        # Store packing history
        self.packing_history.append(result)
        
        return result
    
    def _select_adaptive_strategy(self, memory_items: List[MemoryItem], task_type: TaskType) -> ContextPackingStrategy:
        """Adaptively select the best packing strategy"""
        
        # Analyze memory items
        avg_age = sum((datetime.now() - item.timestamp).total_seconds() for item in memory_items) / max(len(memory_items), 1)
        avg_relevance = sum(item.relevance_score for item in memory_items) / max(len(memory_items), 1)
        
        # Task-specific strategy selection
        if task_type in [TaskType.QA, TaskType.RESEARCH]:
            if avg_relevance > 0.8:
                return ContextPackingStrategy.RELEVANCE_FIRST
            else:
                return ContextPackingStrategy.BALANCED
        
        elif task_type == TaskType.CODE_REPAIR:
            return ContextPackingStrategy.RECENCY_FIRST
        
        elif task_type in [TaskType.ANALYSIS, TaskType.SYNTHESIS]:
            return ContextPackingStrategy.DIVERSITY_FIRST
        
        else:
            return ContextPackingStrategy.BALANCED
    
    def _pack_recency_first(self, memory_items: List[MemoryItem]) -> List[MemoryItem]:
        """Pack items prioritizing recency"""
        
        # Sort by timestamp (most recent first)
        sorted_items = sorted(memory_items, key=lambda x: x.timestamp, reverse=True)
        
        packed_items = []
        current_tokens = 0
        
        for item in sorted_items:
            if current_tokens + item.token_count <= self.max_context_tokens:
                packed_items.append(item)
                current_tokens += item.token_count
            else:
                break
        
        return packed_items
    
    def _pack_diversity_first(self, memory_items: List[MemoryItem]) -> List[MemoryItem]:
        """Pack items prioritizing diversity"""
        
        packed_items = []
        current_tokens = 0
        used_content_hashes = set()
        
        # Sort by importance to start with high-quality items
        sorted_items = sorted(memory_items, key=lambda x: x.importance_score, reverse=True)
        
        for item in sorted_items:
            # Calculate content hash for diversity check
            content_hash = hashlib.md5(item.content[:100].encode()).hexdigest()[:8]
            
            # Check if we have similar content
            if content_hash not in used_content_hashes:
                if current_tokens + item.token_count <= self.max_context_tokens:
                    packed_items.append(item)
                    current_tokens += item.token_count
                    used_content_hashes.add(content_hash)
                else:
                    break
        
        return packed_items
    
    def _pack_relevance_first(self, memory_items: List[MemoryItem]) -> List[MemoryItem]:
        """Pack items prioritizing relevance"""
        
        # Sort by relevance score
        sorted_items = sorted(memory_items, key=lambda x: x.relevance_score, reverse=True)
        
        packed_items = []
        current_tokens = 0
        
        for item in sorted_items:
            if current_tokens + item.token_count <= self.max_context_tokens:
                packed_items.append(item)
                current_tokens += item.token_count
            else:
                break
        
        return packed_items
    
    def _pack_balanced(self, memory_items: List[MemoryItem], diversity_weight: float,
                      recency_weight: float, relevance_weight: float) -> List[MemoryItem]:
        """Pack items using balanced scoring"""
        
        # Calculate composite scores
        now = datetime.now()
        max_age = max((now - item.timestamp).total_seconds() for item in memory_items) if memory_items else 1
        
        scored_items = []
        for item in memory_items:
            age_seconds = (now - item.timestamp).total_seconds()
            recency_score = 1.0 - (age_seconds / max_age)
            
            # Simple diversity score (could be improved with embeddings)
            diversity_score = 1.0 / (1.0 + item.access_count * 0.1)
            
            composite_score = (
                relevance_weight * item.relevance_score +
                recency_weight * recency_score +
                diversity_weight * diversity_score
            )
            
            scored_items.append((item, composite_score))
        
        # Sort by composite score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        packed_items = []
        current_tokens = 0
        
        for item, score in scored_items:
            if current_tokens + item.token_count <= self.max_context_tokens:
                packed_items.append(item)
                current_tokens += item.token_count
            else:
                break
        
        return packed_items
    
    def _calculate_diversity_score(self, items: List[MemoryItem]) -> float:
        """Calculate diversity score for packed items"""
        
        if len(items) <= 1:
            return 1.0
        
        # Simple diversity based on content similarity
        content_hashes = set()
        for item in items:
            content_hash = hashlib.md5(item.content[:100].encode()).hexdigest()[:8]
            content_hashes.add(content_hash)
        
        return len(content_hashes) / len(items)
    
    def _calculate_recency_score(self, items: List[MemoryItem]) -> float:
        """Calculate recency score for packed items"""
        
        if not items:
            return 0.0
        
        now = datetime.now()
        ages = [(now - item.timestamp).total_seconds() for item in items]
        max_age = max(ages) if ages else 1
        
        # Average recency (normalized)
        avg_recency = sum(1.0 - (age / max_age) for age in ages) / len(ages)
        return avg_recency

class PromptTemplateManager:
    """Manages versioned prompt templates with A/B testing"""
    
    def __init__(self, templates_dir: str = "extensions/prompt_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        self.templates: Dict[str, PromptTemplate] = {}
        self.template_performance: Dict[str, List[Dict[str, Any]]] = {}
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Load existing templates
        self._load_templates()
        
        print(f"📝 Prompt Template Manager initialized")
        print(f"   Templates loaded: {len(self.templates)}")
        print(f"   Templates directory: {self.templates_dir}")
    
    def create_template(self, name: str, template_content: str, task_types: List[TaskType],
                       variables: List[str] = None, version: str = "1.0.0") -> str:
        """Create a new prompt template"""
        
        import uuid
        template_id = str(uuid.uuid4())
        
        # Extract variables from template if not provided
        if variables is None:
            variables = self._extract_variables(template_content)
        
        template = PromptTemplate(
            template_id=template_id,
            name=name,
            version=version,
            template_content=template_content,
            variables=variables,
            task_types=task_types,
            created_at=datetime.now(),
            performance_metrics={},
            active=True
        )
        
        self.templates[template_id] = template
        
        # Save to file
        self._save_template(template)
        
        print(f"📝 Created template: {name} v{version}")
        print(f"   Variables: {variables}")
        print(f"   Task types: {[t.value for t in task_types]}")
        
        return template_id
    
    def get_template(self, task_type: TaskType, version: str = None,
                    performance_threshold: float = 0.0) -> Optional[PromptTemplate]:
        """Get the best template for a task type"""
        
        # Filter templates by task type
        candidates = [
            template for template in self.templates.values()
            if task_type in template.task_types and template.active
        ]
        
        if not candidates:
            return None
        
        # Filter by version if specified
        if version:
            candidates = [t for t in candidates if t.version == version]
        
        # Filter by performance threshold
        if performance_threshold > 0:
            candidates = [
                t for t in candidates
                if t.performance_metrics.get("avg_score", 0) >= performance_threshold
            ]
        
        if not candidates:
            return candidates[0] if candidates else None
        
        # Return best performing template
        return max(candidates, key=lambda t: t.performance_metrics.get("avg_score", 0))
    
    def render_template(self, template_id: str, variables: Dict[str, Any]) -> str:
        """Render a template with provided variables"""
        
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        
        # Create Jinja2 template
        jinja_template = Template(template.template_content)
        
        # Render with variables
        rendered = jinja_template.render(**variables)
        
        return rendered
    
    def record_template_performance(self, template_id: str, performance_metrics: Dict[str, float]):
        """Record performance metrics for a template"""
        
        if template_id not in self.templates:
            return
        
        # Add timestamp to metrics
        metrics_with_timestamp = {
            **performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in performance history
        if template_id not in self.template_performance:
            self.template_performance[template_id] = []
        
        self.template_performance[template_id].append(metrics_with_timestamp)
        
        # Update template's average performance
        template = self.templates[template_id]
        all_scores = [m.get("score", 0) for m in self.template_performance[template_id]]
        template.performance_metrics["avg_score"] = sum(all_scores) / len(all_scores)
        template.performance_metrics["total_uses"] = len(all_scores)
        template.performance_metrics["last_updated"] = datetime.now().isoformat()
    
    def create_template_variant(self, base_template_id: str, modifications: Dict[str, Any],
                               version: str = None) -> str:
        """Create a variant of an existing template for A/B testing"""
        
        if base_template_id not in self.templates:
            raise ValueError(f"Base template {base_template_id} not found")
        
        base_template = self.templates[base_template_id]
        
        # Generate new version if not provided
        if version is None:
            base_version = base_template.version.split(".")
            base_version[-1] = str(int(base_version[-1]) + 1)
            version = ".".join(base_version)
        
        # Apply modifications to template content
        modified_content = base_template.template_content
        for key, value in modifications.items():
            if key == "content_replacement":
                for old, new in value.items():
                    modified_content = modified_content.replace(old, new)
            elif key == "variable_additions":
                # Add new variables to template
                for var_name, var_placeholder in value.items():
                    modified_content += f"\n{var_placeholder}"
        
        # Create variant template
        variant_id = self.create_template(
            name=f"{base_template.name}_variant",
            template_content=modified_content,
            task_types=base_template.task_types,
            version=version
        )
        
        return variant_id
    
    def get_template_analytics(self) -> Dict[str, Any]:
        """Get analytics for all templates"""
        
        analytics = {
            "total_templates": len(self.templates),
            "active_templates": len([t for t in self.templates.values() if t.active]),
            "templates_by_task": {},
            "performance_summary": {},
            "version_distribution": {}
        }
        
        # Templates by task type
        for task_type in TaskType:
            count = len([t for t in self.templates.values() if task_type in t.task_types])
            analytics["templates_by_task"][task_type.value] = count
        
        # Performance summary
        for template_id, template in self.templates.items():
            if template.performance_metrics:
                analytics["performance_summary"][template.name] = {
                    "avg_score": template.performance_metrics.get("avg_score", 0),
                    "total_uses": template.performance_metrics.get("total_uses", 0),
                    "version": template.version
                }
        
        # Version distribution
        versions = [t.version for t in self.templates.values()]
        for version in set(versions):
            analytics["version_distribution"][version] = versions.count(version)
        
        return analytics
    
    def _load_templates(self):
        """Load templates from files"""
        
        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                template = PromptTemplate(
                    template_id=template_data["template_id"],
                    name=template_data["name"],
                    version=template_data["version"],
                    template_content=template_data["template_content"],
                    variables=template_data["variables"],
                    task_types=[TaskType(t) for t in template_data["task_types"]],
                    created_at=datetime.fromisoformat(template_data["created_at"]),
                    performance_metrics=template_data.get("performance_metrics", {}),
                    active=template_data.get("active", True)
                )
                
                self.templates[template.template_id] = template
                
            except Exception as e:
                print(f"⚠️ Failed to load template {template_file}: {e}")
    
    def _save_template(self, template: PromptTemplate):
        """Save template to file"""
        
        template_data = {
            "template_id": template.template_id,
            "name": template.name,
            "version": template.version,
            "template_content": template.template_content,
            "variables": template.variables,
            "task_types": [t.value for t in template.task_types],
            "created_at": template.created_at.isoformat(),
            "performance_metrics": template.performance_metrics,
            "active": template.active
        }
        
        filename = f"{template.name}_{template.version}.yaml"
        filepath = self.templates_dir / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(template_data, f, default_flow_style=False)
    
    def _extract_variables(self, template_content: str) -> List[str]:
        """Extract Jinja2 variables from template content"""
        
        # Find all {{ variable }} patterns
        pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
        variables = re.findall(pattern, template_content)
        
        return list(set(variables))

class ContextCompressionEngine:
    """Sentence-level extractive summarization for context compression"""
    
    def __init__(self):
        self.compression_history: List[Dict[str, Any]] = []
        
        print("🗜️ Context Compression Engine initialized")
    
    def compress_context(self, content: str, target_ratio: float = 0.5,
                        preserve_structure: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Compress content while preserving key information"""
        
        # Split into sentences
        sentences = self._split_sentences(content)
        
        if len(sentences) <= 3:  # Too short to compress meaningfully
            return content, {"compression_ratio": 1.0, "sentences_kept": len(sentences)}
        
        # Score sentences by importance
        sentence_scores = self._score_sentences(sentences, content)
        
        # Select top sentences based on target ratio
        target_count = max(1, int(len(sentences) * target_ratio))
        
        # Sort by score and select top sentences
        scored_sentences = list(zip(sentences, sentence_scores, range(len(sentences))))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        selected_sentences = scored_sentences[:target_count]
        
        # Reorder by original position if preserving structure
        if preserve_structure:
            selected_sentences.sort(key=lambda x: x[2])
        
        # Reconstruct compressed content
        compressed_content = " ".join([sent[0] for sent in selected_sentences])
        
        # Calculate metrics
        compression_metrics = {
            "original_length": len(content),
            "compressed_length": len(compressed_content),
            "compression_ratio": len(compressed_content) / len(content),
            "sentences_original": len(sentences),
            "sentences_kept": len(selected_sentences),
            "sentence_retention_ratio": len(selected_sentences) / len(sentences)
        }
        
        # Store compression history
        self.compression_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": compression_metrics,
            "target_ratio": target_ratio
        })
        
        return compressed_content, compression_metrics
    
    def _split_sentences(self, content: str) -> List[str]:
        """Split content into sentences"""
        
        # Simple sentence splitting (could be improved with NLTK)
        import re
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', content)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _score_sentences(self, sentences: List[str], full_content: str) -> List[float]:
        """Score sentences by importance"""
        
        scores = []
        
        # Calculate word frequencies in full content
        words = full_content.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        for sentence in sentences:
            sentence_words = sentence.lower().split()
            
            if not sentence_words:
                scores.append(0.0)
                continue
            
            # Base score from word frequencies
            freq_score = sum(word_freq.get(word, 0) for word in sentence_words) / len(sentence_words)
            
            # Position bonus (first and last sentences often important)
            position_bonus = 0.0
            sentence_idx = sentences.index(sentence)
            if sentence_idx == 0 or sentence_idx == len(sentences) - 1:
                position_bonus = 0.2
            
            # Length bonus (moderate length sentences often more informative)
            length_score = min(len(sentence_words) / 20, 0.3)
            
            # Keyword bonus
            keyword_bonus = 0.0
            keywords = ["important", "key", "significant", "conclusion", "result", "finding"]
            if any(keyword in sentence.lower() for keyword in keywords):
                keyword_bonus = 0.3
            
            total_score = freq_score + position_bonus + length_score + keyword_bonus
            scores.append(total_score)
        
        return scores

# Integration class for Stage 2
class EnhancedContextBuilder:
    """Main integration class for Stage 2 enhancements"""
    
    def __init__(self, max_context_tokens: int = 8000):
        self.memory_manager = MemoryTierManager()
        self.context_packer = AdaptiveContextPacker(max_context_tokens)
        self.template_manager = PromptTemplateManager()
        self.compression_engine = ContextCompressionEngine()
        
        # Create default templates
        self._create_default_templates()
        
        print("🏗️ Enhanced Context Builder initialized")
    
    def build_context(self, query: str, task_type: TaskType,
                     max_tokens: int = None, compression_ratio: float = 0.8) -> Dict[str, Any]:
        """Build optimized context for a query"""
        
        # Retrieve relevant memories
        memories = self.memory_manager.retrieve_memories(
            query=query,
            memory_tiers=[MemoryTier.SHORT_TERM, MemoryTier.EPISODIC, MemoryTier.LONG_TERM],
            max_items=20
        )
        
        # Pack context optimally
        packing_result = self.context_packer.pack_context(
            memory_items=memories,
            task_type=task_type,
            strategy=ContextPackingStrategy.ADAPTIVE
        )
        
        # Get appropriate template
        template = self.template_manager.get_template(task_type)
        
        # Build context content
        context_content = ""
        for item in packing_result.packed_items:
            context_content += f"{item.content}\n\n"
        
        # Compress if needed
        if max_tokens and len(context_content.split()) * 1.3 > max_tokens:
            context_content, compression_metrics = self.compression_engine.compress_context(
                context_content, target_ratio=compression_ratio
            )
        else:
            compression_metrics = {"compression_ratio": 1.0}
        
        # Render template if available
        rendered_prompt = None
        if template:
            try:
                rendered_prompt = self.template_manager.render_template(
                    template.template_id,
                    {
                        "query": query,
                        "context": context_content,
                        "task_type": task_type.value
                    }
                )
            except Exception as e:
                print(f"⚠️ Template rendering failed: {e}")
        
        return {
            "context_content": context_content,
            "rendered_prompt": rendered_prompt,
            "packing_result": packing_result,
            "compression_metrics": compression_metrics,
            "template_used": template.name if template else None,
            "memory_items_used": len(packing_result.packed_items),
            "total_tokens": packing_result.total_tokens
        }
    
    def store_interaction(self, query: str, response: str, task_type: TaskType,
                         relevance_score: float = 0.7, metadata: Dict[str, Any] = None):
        """Store an interaction in memory"""
        
        # Store query in short-term memory
        self.memory_manager.store_memory(
            content=f"Query: {query}",
            memory_tier=MemoryTier.SHORT_TERM,
            relevance_score=relevance_score,
            metadata={**(metadata or {}), "type": "query", "task_type": task_type.value}
        )
        
        # Store response in episodic memory
        self.memory_manager.store_memory(
            content=f"Response: {response}",
            memory_tier=MemoryTier.EPISODIC,
            relevance_score=relevance_score,
            metadata={**(metadata or {}), "type": "response", "task_type": task_type.value}
        )
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        
        return {
            "memory_tiers": self.memory_manager.get_tier_statistics(),
            "context_packing": {
                "total_packings": len(self.context_packer.packing_history),
                "avg_compression_ratio": sum(p.compression_ratio for p in self.context_packer.packing_history) / max(len(self.context_packer.packing_history), 1)
            },
            "templates": self.template_manager.get_template_analytics(),
            "compression": {
                "total_compressions": len(self.compression_engine.compression_history),
                "avg_compression_ratio": sum(c["metrics"]["compression_ratio"] for c in self.compression_engine.compression_history) / max(len(self.compression_engine.compression_history), 1)
            }
        }
    
    def _create_default_templates(self):
        """Create default prompt templates"""
        
        # Research template
        research_template = """
# Research Query Analysis

**Query**: {{ query }}
**Task Type**: {{ task_type }}

## Relevant Context
{{ context }}

## Instructions
Based on the provided context, conduct a thorough analysis of the research query. Consider multiple perspectives and provide evidence-based insights.

## Expected Output Format
1. Key findings
2. Analysis and interpretation
3. Implications and recommendations
4. Areas for further investigation
"""
        
        self.template_manager.create_template(
            name="research_analysis",
            template_content=research_template,
            task_types=[TaskType.RESEARCH, TaskType.ANALYSIS],
            version="1.0.0"
        )
        
        # QA template
        qa_template = """
# Question Answering

**Question**: {{ query }}

## Context Information
{{ context }}

## Instructions
Provide a clear, accurate, and comprehensive answer to the question based on the available context. If the context is insufficient, indicate what additional information would be helpful.

## Answer Format
- Direct answer
- Supporting evidence from context
- Confidence level
- Additional considerations
"""
        
        self.template_manager.create_template(
            name="question_answering",
            template_content=qa_template,
            task_types=[TaskType.QA],
            version="1.0.0"
        )

# Global instance
_enhanced_context_builder = None

def get_enhanced_context_builder() -> EnhancedContextBuilder:
    """Get the global enhanced context builder instance"""
    global _enhanced_context_builder
    if _enhanced_context_builder is None:
        _enhanced_context_builder = EnhancedContextBuilder()
    return _enhanced_context_builder