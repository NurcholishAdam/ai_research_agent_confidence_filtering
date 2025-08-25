# -*- coding: utf-8 -*-
"""
Stage 3: Enhanced Semantic Graph with Multi-Source Fusion
Advanced graph operations with hybrid retrieval and reasoning write-back
"""

import json
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
from pathlib import Path
import numpy as np
from collections import defaultdict, deque

class NodeType(Enum):
    CONCEPT = "concept"
    PAPER = "paper"
    AUTHOR = "author"
    INSTITUTION = "institution"
    OBSERVATION = "observation"
    CLAIM = "claim"
    EXPERIMENT = "experiment"
    REPAIR_ACTION = "repair_action"
    HYPOTHESIS = "hypothesis"
    TOOL_RESULT = "tool_result"
    RESEARCH_FINDING = "research_finding"
    CODE_SNIPPET = "code_snippet"
    DATASET = "dataset"

class EdgeType(Enum):
    CITES = "cites"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    MENTIONS = "mentions"
    AUTHORED_BY = "authored_by"
    AFFILIATED_WITH = "affiliated_with"
    IMPLEMENTS = "implements"
    USES = "uses"
    VALIDATES = "validates"
    REFUTES = "refutes"
    EXTENDS = "extends"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"

class SourceType(Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    GITHUB = "github"
    WIKIPEDIA = "wikipedia"
    PUBMED = "pubmed"
    INTERNAL = "internal"
    USER_INPUT = "user_input"
    TOOL_OUTPUT = "tool_output"

@dataclass
class GraphNode:
    """Enhanced graph node with rich metadata"""
    node_id: str
    node_type: NodeType
    content: str
    title: Optional[str]
    source_type: SourceType
    source_id: Optional[str]
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    importance_score: float
    confidence_score: float
    created_at: datetime
    last_updated: datetime
    access_count: int
    tags: List[str]

@dataclass
class GraphEdge:
    """Enhanced graph edge with relationship metadata"""
    edge_id: str
    source_node: str
    target_node: str
    edge_type: EdgeType
    weight: float
    confidence: float
    evidence: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    last_updated: datetime

@dataclass
class RetrievalResult:
    """Result from graph-based retrieval"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paths: List[List[str]]
    relevance_scores: Dict[str, float]
    retrieval_metadata: Dict[str, Any]

class SemanticGraphManager:
    """Advanced semantic graph with multi-source fusion and hybrid retrieval"""
    
    def __init__(self, graph_storage_path: str = "extensions/semantic_graph"):
        self.storage_path = Path(graph_storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # NetworkX graph for structure
        self.graph = nx.MultiDiGraph()
        
        # Enhanced storage
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        
        # Source tracking
        self.source_mappings: Dict[SourceType, Dict[str, str]] = defaultdict(dict)
        
        # Indexing for fast retrieval
        self.content_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.type_index: Dict[NodeType, Set[str]] = defaultdict(set)
        
        # Reasoning history
        self.reasoning_steps: List[Dict[str, Any]] = []
        
        # Load existing graph
        self._load_graph()
        
        print("🕸️ Semantic Graph Manager initialized")
        print(f"   Nodes: {len(self.nodes)}")
        print(f"   Edges: {len(self.edges)}")
        print(f"   Sources: {len(self.source_mappings)}")
    
    def add_node(self, content: str, node_type: NodeType, source_type: SourceType,
                title: str = None, source_id: str = None, embedding: List[float] = None,
                metadata: Dict[str, Any] = None, tags: List[str] = None,
                importance_score: float = 0.5, confidence_score: float = 0.8) -> str:
        """Add a new node to the semantic graph"""
        
        node_id = str(uuid.uuid4())
        
        # Check for duplicates
        existing_node = self._find_duplicate_node(content, node_type, source_type, source_id)
        if existing_node:
            return existing_node
        
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            title=title,
            source_type=source_type,
            source_id=source_id,
            embedding=embedding,
            metadata=metadata or {},
            importance_score=importance_score,
            confidence_score=confidence_score,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            access_count=0,
            tags=tags or []
        )
        
        # Store node
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
        
        # Update indices
        self._update_indices_for_node(node)
        
        # Update source mapping
        if source_id:
            self.source_mappings[source_type][source_id] = node_id
        
        return node_id
    
    def add_edge(self, source_node: str, target_node: str, edge_type: EdgeType,
                weight: float = 1.0, confidence: float = 0.8, evidence: List[str] = None,
                metadata: Dict[str, Any] = None) -> str:
        """Add an edge between two nodes"""
        
        if source_node not in self.nodes or target_node not in self.nodes:
            raise ValueError("Both nodes must exist in the graph")
        
        edge_id = str(uuid.uuid4())
        
        edge = GraphEdge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            edge_type=edge_type,
            weight=weight,
            confidence=confidence,
            evidence=evidence or [],
            metadata=metadata or {},
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Store edge
        self.edges[edge_id] = edge
        self.graph.add_edge(source_node, target_node, key=edge_id, **asdict(edge))
        
        return edge_id
    
    def multi_source_fusion(self, sources_data: Dict[SourceType, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Fuse data from multiple sources with deduplication"""
        
        fusion_stats = {
            "nodes_added": 0,
            "edges_added": 0,
            "duplicates_merged": 0,
            "conflicts_resolved": 0
        }
        
        # Process each source
        for source_type, items in sources_data.items():
            print(f"🔄 Processing {len(items)} items from {source_type.value}")
            
            for item in items:
                try:
                    # Extract node information
                    node_data = self._extract_node_from_source(item, source_type)
                    
                    # Add or merge node
                    node_id = self._add_or_merge_node(node_data, fusion_stats)
                    
                    # Extract and add relationships
                    relationships = self._extract_relationships_from_source(item, source_type, node_id)
                    
                    for rel in relationships:
                        self._add_or_update_edge(rel, fusion_stats)
                
                except Exception as e:
                    print(f"⚠️ Failed to process item from {source_type.value}: {e}")
        
        # Post-fusion optimization
        self._optimize_graph_structure()
        
        print(f"✅ Multi-source fusion complete:")
        print(f"   Nodes added: {fusion_stats['nodes_added']}")
        print(f"   Edges added: {fusion_stats['edges_added']}")
        print(f"   Duplicates merged: {fusion_stats['duplicates_merged']}")
        
        return fusion_stats
    
    def hybrid_retrieval(self, query: str, retrieval_types: List[str] = None,
                        max_nodes: int = 20, similarity_threshold: float = 0.7) -> RetrievalResult:
        """Perform hybrid retrieval combining multiple strategies"""
        
        if retrieval_types is None:
            retrieval_types = ["semantic", "structural", "path_constrained"]
        
        all_results = {}
        
        # Semantic similarity retrieval
        if "semantic" in retrieval_types:
            semantic_results = self._semantic_retrieval(query, max_nodes, similarity_threshold)
            all_results["semantic"] = semantic_results
        
        # Structural retrieval (based on graph structure)
        if "structural" in retrieval_types:
            structural_results = self._structural_retrieval(query, max_nodes)
            all_results["structural"] = structural_results
        
        # Path-constrained retrieval
        if "path_constrained" in retrieval_types:
            path_results = self._path_constrained_retrieval(query, max_nodes)
            all_results["path_constrained"] = path_results
        
        # Combine and rank results
        combined_result = self._combine_retrieval_results(all_results, query)
        
        return combined_result
    
    def reasoning_writeback(self, reasoning_step: Dict[str, Any], 
                           create_nodes: bool = True, create_edges: bool = True) -> Dict[str, Any]:
        """Write back reasoning steps as graph nodes and edges"""
        
        writeback_result = {
            "nodes_created": [],
            "edges_created": [],
            "reasoning_id": str(uuid.uuid4())
        }
        
        # Extract reasoning components
        reasoning_type = reasoning_step.get("type", "inference")
        premises = reasoning_step.get("premises", [])
        conclusion = reasoning_step.get("conclusion", "")
        confidence = reasoning_step.get("confidence", 0.7)
        evidence = reasoning_step.get("evidence", [])
        
        if create_nodes and conclusion:
            # Create node for conclusion
            conclusion_node_id = self.add_node(
                content=conclusion,
                node_type=NodeType.CLAIM,
                source_type=SourceType.INTERNAL,
                metadata={
                    "reasoning_type": reasoning_type,
                    "reasoning_id": writeback_result["reasoning_id"],
                    "derived": True
                },
                confidence_score=confidence,
                tags=["reasoning", reasoning_type]
            )
            writeback_result["nodes_created"].append(conclusion_node_id)
        
        if create_edges and premises:
            # Create edges from premises to conclusion
            for premise in premises:
                premise_node_id = self._find_or_create_premise_node(premise)
                
                if premise_node_id and conclusion_node_id:
                    edge_id = self.add_edge(
                        source_node=premise_node_id,
                        target_node=conclusion_node_id,
                        edge_type=EdgeType.SUPPORTS,
                        confidence=confidence,
                        evidence=evidence,
                        metadata={
                            "reasoning_type": reasoning_type,
                            "reasoning_id": writeback_result["reasoning_id"]
                        }
                    )
                    writeback_result["edges_created"].append(edge_id)
        
        # Store reasoning step
        reasoning_step_record = {
            **reasoning_step,
            "reasoning_id": writeback_result["reasoning_id"],
            "timestamp": datetime.now().isoformat(),
            "writeback_result": writeback_result
        }
        self.reasoning_steps.append(reasoning_step_record)
        
        return writeback_result
    
    def get_node_neighbors(self, node_id: str, max_depth: int = 2,
                          edge_types: List[EdgeType] = None) -> Dict[str, Dict[str, Any]]:
        """Get neighbors of a node within specified depth"""
        
        if node_id not in self.nodes:
            return {}
        
        neighbors = {}
        visited = set()
        queue = deque([(node_id, 0, [])])
        
        while queue:
            current_node, depth, path = queue.popleft()
            
            if depth > max_depth or current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Get all edges from current node
            for edge_id, edge in self.edges.items():
                if edge.source_node == current_node:
                    target = edge.target_node
                    
                    # Filter by edge type if specified
                    if edge_types and edge.edge_type not in edge_types:
                        continue
                    
                    if target not in visited and target != node_id:
                        neighbors[target] = {
                            "distance": depth + 1,
                            "path": path + [edge_id],
                            "relationship": edge.edge_type.value,
                            "confidence": edge.confidence
                        }
                        
                        if depth + 1 < max_depth:
                            queue.append((target, depth + 1, path + [edge_id]))
        
        return neighbors
    
    def find_paths(self, source_node: str, target_node: str, 
                  max_length: int = 4, path_types: List[EdgeType] = None) -> List[List[str]]:
        """Find paths between two nodes with constraints"""
        
        if source_node not in self.nodes or target_node not in self.nodes:
            return []
        
        paths = []
        
        try:
            # Use NetworkX for path finding
            if path_types:
                # Create filtered graph
                filtered_edges = []
                for edge_id, edge in self.edges.items():
                    if edge.edge_type in path_types:
                        filtered_edges.append((edge.source_node, edge.target_node, {"edge_id": edge_id}))
                
                temp_graph = nx.MultiDiGraph()
                temp_graph.add_edges_from(filtered_edges)
                
                if temp_graph.has_node(source_node) and temp_graph.has_node(target_node):
                    nx_paths = nx.all_simple_paths(temp_graph, source_node, target_node, cutoff=max_length)
                    paths = list(nx_paths)
            else:
                nx_paths = nx.all_simple_paths(self.graph, source_node, target_node, cutoff=max_length)
                paths = list(nx_paths)
        
        except nx.NetworkXNoPath:
            pass
        
        return paths[:10]  # Limit to top 10 paths
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        
        stats = {
            "nodes": {
                "total": len(self.nodes),
                "by_type": {},
                "by_source": {}
            },
            "edges": {
                "total": len(self.edges),
                "by_type": {}
            },
            "connectivity": {
                "density": nx.density(self.graph),
                "connected_components": nx.number_weakly_connected_components(self.graph),
                "avg_clustering": nx.average_clustering(self.graph.to_undirected())
            },
            "reasoning": {
                "total_steps": len(self.reasoning_steps),
                "nodes_from_reasoning": len([n for n in self.nodes.values() if n.metadata.get("derived", False)])
            }
        }
        
        # Count by node type
        for node in self.nodes.values():
            node_type = node.node_type.value
            stats["nodes"]["by_type"][node_type] = stats["nodes"]["by_type"].get(node_type, 0) + 1
            
            source_type = node.source_type.value
            stats["nodes"]["by_source"][source_type] = stats["nodes"]["by_source"].get(source_type, 0) + 1
        
        # Count by edge type
        for edge in self.edges.values():
            edge_type = edge.edge_type.value
            stats["edges"]["by_type"][edge_type] = stats["edges"]["by_type"].get(edge_type, 0) + 1
        
        return stats
    
    def _find_duplicate_node(self, content: str, node_type: NodeType, 
                           source_type: SourceType, source_id: str = None) -> Optional[str]:
        """Find duplicate nodes based on content and source"""
        
        # Check source mapping first
        if source_id and source_type in self.source_mappings:
            if source_id in self.source_mappings[source_type]:
                return self.source_mappings[source_type][source_id]
        
        # Check content similarity
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        for node_id, node in self.nodes.items():
            if (node.node_type == node_type and 
                node.source_type == source_type and
                hashlib.md5(node.content.encode()).hexdigest() == content_hash):
                return node_id
        
        return None
    
    def _update_indices_for_node(self, node: GraphNode):
        """Update search indices for a node"""
        
        # Content index (word-based)
        words = node.content.lower().split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                self.content_index[word].add(node.node_id)
        
        # Tag index
        for tag in node.tags:
            self.tag_index[tag].add(node.node_id)
        
        # Type index
        self.type_index[node.node_type].add(node.node_id)
    
    def _extract_node_from_source(self, item: Dict[str, Any], source_type: SourceType) -> Dict[str, Any]:
        """Extract node data from source-specific item format"""
        
        if source_type == SourceType.ARXIV:
            return {
                "content": item.get("abstract", ""),
                "title": item.get("title", ""),
                "node_type": NodeType.PAPER,
                "source_id": item.get("id", ""),
                "metadata": {
                    "authors": item.get("authors", []),
                    "categories": item.get("categories", []),
                    "published": item.get("published", "")
                },
                "tags": item.get("categories", [])
            }
        
        elif source_type == SourceType.GITHUB:
            return {
                "content": item.get("description", "") + "\n" + item.get("readme", ""),
                "title": item.get("name", ""),
                "node_type": NodeType.CODE_SNIPPET,
                "source_id": item.get("full_name", ""),
                "metadata": {
                    "language": item.get("language", ""),
                    "stars": item.get("stargazers_count", 0),
                    "forks": item.get("forks_count", 0)
                },
                "tags": item.get("topics", [])
            }
        
        else:
            # Generic extraction
            return {
                "content": item.get("content", str(item)),
                "title": item.get("title", ""),
                "node_type": NodeType.CONCEPT,
                "source_id": item.get("id", ""),
                "metadata": item,
                "tags": item.get("tags", [])
            }
    
    def _extract_relationships_from_source(self, item: Dict[str, Any], 
                                         source_type: SourceType, node_id: str) -> List[Dict[str, Any]]:
        """Extract relationships from source item"""
        
        relationships = []
        
        if source_type == SourceType.ARXIV:
            # Create author relationships
            authors = item.get("authors", [])
            for author in authors:
                author_node_id = self._find_or_create_author_node(author)
                if author_node_id:
                    relationships.append({
                        "source": author_node_id,
                        "target": node_id,
                        "type": EdgeType.AUTHORED_BY,
                        "confidence": 0.9
                    })
            
            # Create citation relationships (if available)
            references = item.get("references", [])
            for ref in references:
                ref_node_id = self._find_or_create_reference_node(ref)
                if ref_node_id:
                    relationships.append({
                        "source": node_id,
                        "target": ref_node_id,
                        "type": EdgeType.CITES,
                        "confidence": 0.8
                    })
        
        return relationships
    
    def _add_or_merge_node(self, node_data: Dict[str, Any], stats: Dict[str, Any]) -> str:
        """Add new node or merge with existing"""
        
        existing_node_id = self._find_duplicate_node(
            node_data["content"], 
            node_data["node_type"], 
            node_data.get("source_type", SourceType.INTERNAL),
            node_data.get("source_id")
        )
        
        if existing_node_id:
            # Merge with existing node
            self._merge_node_data(existing_node_id, node_data)
            stats["duplicates_merged"] += 1
            return existing_node_id
        else:
            # Create new node
            node_id = self.add_node(**node_data)
            stats["nodes_added"] += 1
            return node_id
    
    def _merge_node_data(self, existing_node_id: str, new_data: Dict[str, Any]):
        """Merge new data into existing node"""
        
        existing_node = self.nodes[existing_node_id]
        
        # Update metadata
        existing_node.metadata.update(new_data.get("metadata", {}))
        
        # Merge tags
        new_tags = new_data.get("tags", [])
        existing_node.tags = list(set(existing_node.tags + new_tags))
        
        # Update confidence if higher
        new_confidence = new_data.get("confidence_score", 0)
        if new_confidence > existing_node.confidence_score:
            existing_node.confidence_score = new_confidence
        
        # Update timestamp
        existing_node.last_updated = datetime.now()
    
    def _add_or_update_edge(self, relationship: Dict[str, Any], stats: Dict[str, Any]):
        """Add new edge or update existing"""
        
        source = relationship["source"]
        target = relationship["target"]
        edge_type = relationship["type"]
        
        # Check for existing edge
        existing_edge = None
        for edge in self.edges.values():
            if (edge.source_node == source and 
                edge.target_node == target and 
                edge.edge_type == edge_type):
                existing_edge = edge
                break
        
        if existing_edge:
            # Update confidence if higher
            new_confidence = relationship.get("confidence", 0.8)
            if new_confidence > existing_edge.confidence:
                existing_edge.confidence = new_confidence
                existing_edge.last_updated = datetime.now()
        else:
            # Create new edge
            self.add_edge(
                source_node=source,
                target_node=target,
                edge_type=edge_type,
                confidence=relationship.get("confidence", 0.8),
                evidence=relationship.get("evidence", [])
            )
            stats["edges_added"] += 1
    
    def _semantic_retrieval(self, query: str, max_nodes: int, threshold: float) -> List[str]:
        """Retrieve nodes based on semantic similarity"""
        
        query_words = set(query.lower().split())
        scored_nodes = []
        
        for node_id, node in self.nodes.items():
            # Simple word overlap similarity (in practice, use embeddings)
            node_words = set(node.content.lower().split())
            
            if query_words and node_words:
                intersection = query_words.intersection(node_words)
                union = query_words.union(node_words)
                similarity = len(intersection) / len(union)
                
                if similarity >= threshold:
                    scored_nodes.append((node_id, similarity))
        
        # Sort by similarity and return top nodes
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in scored_nodes[:max_nodes]]
    
    def _structural_retrieval(self, query: str, max_nodes: int) -> List[str]:
        """Retrieve nodes based on graph structure"""
        
        # Find nodes with high centrality measures
        centrality_scores = {}
        
        try:
            # PageRank centrality
            pagerank = nx.pagerank(self.graph)
            centrality_scores.update(pagerank)
        except:
            pass
        
        # Sort by centrality and return top nodes
        sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in sorted_nodes[:max_nodes]]
    
    def _path_constrained_retrieval(self, query: str, max_nodes: int) -> List[str]:
        """Retrieve nodes using path constraints"""
        
        # Find query-relevant nodes first
        query_nodes = self._semantic_retrieval(query, 5, 0.3)
        
        # Find nodes connected to query nodes
        connected_nodes = set()
        
        for query_node in query_nodes:
            neighbors = self.get_node_neighbors(query_node, max_depth=2)
            connected_nodes.update(neighbors.keys())
        
        return list(connected_nodes)[:max_nodes]
    
    def _combine_retrieval_results(self, all_results: Dict[str, List[str]], 
                                 query: str) -> RetrievalResult:
        """Combine results from different retrieval methods"""
        
        # Collect all unique nodes
        all_node_ids = set()
        for results in all_results.values():
            all_node_ids.update(results)
        
        # Calculate combined relevance scores
        relevance_scores = {}
        for node_id in all_node_ids:
            score = 0.0
            
            # Weight different retrieval methods
            if node_id in all_results.get("semantic", []):
                score += 0.5
            if node_id in all_results.get("structural", []):
                score += 0.3
            if node_id in all_results.get("path_constrained", []):
                score += 0.2
            
            relevance_scores[node_id] = score
        
        # Sort by relevance
        sorted_nodes = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top nodes and their data
        top_node_ids = [node_id for node_id, _ in sorted_nodes[:20]]
        nodes = [self.nodes[node_id] for node_id in top_node_ids if node_id in self.nodes]
        
        # Get relevant edges
        relevant_edges = []
        for edge in self.edges.values():
            if edge.source_node in top_node_ids and edge.target_node in top_node_ids:
                relevant_edges.append(edge)
        
        # Find paths between top nodes
        paths = []
        for i, node1 in enumerate(top_node_ids[:5]):
            for node2 in top_node_ids[i+1:6]:
                node_paths = self.find_paths(node1, node2, max_length=3)
                paths.extend(node_paths[:2])  # Limit paths per pair
        
        return RetrievalResult(
            nodes=nodes,
            edges=relevant_edges,
            paths=paths,
            relevance_scores=relevance_scores,
            retrieval_metadata={
                "query": query,
                "methods_used": list(all_results.keys()),
                "total_candidates": len(all_node_ids)
            }
        )
    
    def _find_or_create_premise_node(self, premise: str) -> Optional[str]:
        """Find existing premise node or create new one"""
        
        # Simple search for existing premise
        for node_id, node in self.nodes.items():
            if premise.lower() in node.content.lower():
                return node_id
        
        # Create new premise node
        return self.add_node(
            content=premise,
            node_type=NodeType.CLAIM,
            source_type=SourceType.INTERNAL,
            metadata={"premise": True}
        )
    
    def _find_or_create_author_node(self, author: str) -> Optional[str]:
        """Find or create author node"""
        
        # Check if author already exists
        for node_id, node in self.nodes.items():
            if (node.node_type == NodeType.AUTHOR and 
                author.lower() in node.content.lower()):
                return node_id
        
        # Create new author node
        return self.add_node(
            content=author,
            node_type=NodeType.AUTHOR,
            source_type=SourceType.INTERNAL,
            title=author
        )
    
    def _find_or_create_reference_node(self, reference: str) -> Optional[str]:
        """Find or create reference node"""
        
        # Simplified reference handling
        return self.add_node(
            content=reference,
            node_type=NodeType.PAPER,
            source_type=SourceType.INTERNAL,
            metadata={"reference": True}
        )
    
    def _optimize_graph_structure(self):
        """Optimize graph structure after fusion"""
        
        # Remove low-confidence edges
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.confidence < 0.3:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
            # Also remove from NetworkX graph
            edge = self.edges.get(edge_id)
            if edge:
                self.graph.remove_edge(edge.source_node, edge.target_node, key=edge_id)
        
        # Update importance scores based on connectivity
        self._update_importance_scores()
    
    def _update_importance_scores(self):
        """Update node importance scores based on graph structure"""
        
        try:
            # Calculate centrality measures
            pagerank_scores = nx.pagerank(self.graph)
            
            for node_id, node in self.nodes.items():
                if node_id in pagerank_scores:
                    # Combine original importance with centrality
                    centrality_score = pagerank_scores[node_id]
                    node.importance_score = 0.7 * node.importance_score + 0.3 * centrality_score
        except:
            pass  # Skip if graph is empty or has issues
    
    def _load_graph(self):
        """Load graph from storage"""
        
        nodes_file = self.storage_path / "nodes.json"
        edges_file = self.storage_path / "edges.json"
        
        # Load nodes
        if nodes_file.exists():
            try:
                with open(nodes_file, 'r') as f:
                    nodes_data = json.load(f)
                
                for node_data in nodes_data:
                    node = GraphNode(
                        node_id=node_data["node_id"],
                        node_type=NodeType(node_data["node_type"]),
                        content=node_data["content"],
                        title=node_data.get("title"),
                        source_type=SourceType(node_data["source_type"]),
                        source_id=node_data.get("source_id"),
                        embedding=node_data.get("embedding"),
                        metadata=node_data.get("metadata", {}),
                        importance_score=node_data.get("importance_score", 0.5),
                        confidence_score=node_data.get("confidence_score", 0.8),
                        created_at=datetime.fromisoformat(node_data["created_at"]),
                        last_updated=datetime.fromisoformat(node_data["last_updated"]),
                        access_count=node_data.get("access_count", 0),
                        tags=node_data.get("tags", [])
                    )
                    
                    self.nodes[node.node_id] = node
                    self.graph.add_node(node.node_id, **asdict(node))
                    self._update_indices_for_node(node)
            
            except Exception as e:
                print(f"⚠️ Failed to load nodes: {e}")
        
        # Load edges
        if edges_file.exists():
            try:
                with open(edges_file, 'r') as f:
                    edges_data = json.load(f)
                
                for edge_data in edges_data:
                    edge = GraphEdge(
                        edge_id=edge_data["edge_id"],
                        source_node=edge_data["source_node"],
                        target_node=edge_data["target_node"],
                        edge_type=EdgeType(edge_data["edge_type"]),
                        weight=edge_data.get("weight", 1.0),
                        confidence=edge_data.get("confidence", 0.8),
                        evidence=edge_data.get("evidence", []),
                        metadata=edge_data.get("metadata", {}),
                        created_at=datetime.fromisoformat(edge_data["created_at"]),
                        last_updated=datetime.fromisoformat(edge_data["last_updated"])
                    )
                    
                    self.edges[edge.edge_id] = edge
                    if edge.source_node in self.nodes and edge.target_node in self.nodes:
                        self.graph.add_edge(edge.source_node, edge.target_node, 
                                          key=edge.edge_id, **asdict(edge))
            
            except Exception as e:
                print(f"⚠️ Failed to load edges: {e}")
    
    def save_graph(self):
        """Save graph to storage"""
        
        # Save nodes
        nodes_data = []
        for node in self.nodes.values():
            node_dict = asdict(node)
            node_dict["node_type"] = node.node_type.value
            node_dict["source_type"] = node.source_type.value
            node_dict["created_at"] = node.created_at.isoformat()
            node_dict["last_updated"] = node.last_updated.isoformat()
            nodes_data.append(node_dict)
        
        with open(self.storage_path / "nodes.json", 'w') as f:
            json.dump(nodes_data, f, indent=2)
        
        # Save edges
        edges_data = []
        for edge in self.edges.values():
            edge_dict = asdict(edge)
            edge_dict["edge_type"] = edge.edge_type.value
            edge_dict["created_at"] = edge.created_at.isoformat()
            edge_dict["last_updated"] = edge.last_updated.isoformat()
            edges_data.append(edge_dict)
        
        with open(self.storage_path / "edges.json", 'w') as f:
            json.dump(edges_data, f, indent=2)

# Integration function
def integrate_semantic_graph():
    """Integration point for semantic graph system"""
    
    graph_manager = SemanticGraphManager()
    
    print("🕸️ Stage 3: Enhanced Semantic Graph integrated")
    print("   Features:")
    print("   - Multi-source fusion with deduplication")
    print("   - Hybrid retrieval (semantic + structural + path-constrained)")
    print("   - Reasoning write-back capabilities")
    print("   - Advanced graph analytics")
    
    return graph_manager

if __name__ == "__main__":
    # Demo usage
    graph_manager = integrate_semantic_graph()
    
    # Add some sample nodes
    concept_id = graph_manager.add_node(
        content="Machine learning is a subset of artificial intelligence",
        node_type=NodeType.CONCEPT,
        source_type=SourceType.INTERNAL,
        title="Machine Learning Definition"
    )
    
    paper_id = graph_manager.add_node(
        content="Deep learning has revolutionized computer vision tasks",
        node_type=NodeType.PAPER,
        source_type=SourceType.ARXIV,
        title="Deep Learning in Computer Vision"
    )
    
    # Add relationship
    graph_manager.add_edge(
        source_node=paper_id,
        target_node=concept_id,
        edge_type=EdgeType.MENTIONS
    )
    
    # Test retrieval
    results = graph_manager.hybrid_retrieval("machine learning")
    print(f"\nRetrieval Results: {len(results.nodes)} nodes found")
    
    # Test reasoning writeback
    reasoning_step = {
        "type": "deduction",
        "premises": ["Machine learning is a subset of AI", "Deep learning is a subset of ML"],
        "conclusion": "Deep learning is a subset of AI",
        "confidence": 0.9
    }
    
    writeback_result = graph_manager.reasoning_writeback(reasoning_step)
    print(f"Reasoning writeback: {len(writeback_result['nodes_created'])} nodes, {len(writeback_result['edges_created'])} edges")
    
    # Show statistics
    stats = graph_manager.get_graph_statistics()
    print(f"\nGraph Statistics: {stats}")