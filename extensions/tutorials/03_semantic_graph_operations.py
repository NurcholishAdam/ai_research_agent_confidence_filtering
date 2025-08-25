#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 3: Semantic Graph Operations
====================================

This tutorial covers advanced semantic graph operations including multi-source
data fusion, hybrid retrieval strategies, and reasoning write-back capabilities.

Prerequisites:
- Completed Tutorials 1 and 2
- Understanding of graph concepts

Run this tutorial:
    python extensions/tutorials/03_semantic_graph_operations.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add extensions to path
sys.path.append(str(Path(__file__).parent.parent))

from stage_3_semantic_graph import (
    SemanticGraphManager, NodeType, EdgeType, SourceType
)

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

async def tutorial_semantic_graph_operations():
    print_section("Semantic Graph Operations Tutorial")
    
    # Initialize graph manager
    graph_manager = SemanticGraphManager()
    
    # Multi-source data fusion
    print("\n🔄 Multi-Source Data Fusion")
    
    # Simulate arXiv data
    arxiv_data = [
        {
            "id": "2301.12345",
            "title": "Attention Is All You Need",
            "abstract": "We propose the Transformer architecture...",
            "authors": ["Vaswani", "Shazeer"],
            "categories": ["cs.CL", "cs.LG"]
        }
    ]
    
    # Simulate GitHub data  
    github_data = [
        {
            "full_name": "huggingface/transformers",
            "description": "Transformers library for NLP",
            "language": "Python",
            "topics": ["nlp", "pytorch", "tensorflow"]
        }
    ]
    
    sources_data = {
        SourceType.ARXIV: arxiv_data,
        SourceType.GITHUB: github_data
    }
    
    fusion_stats = graph_manager.multi_source_fusion(sources_data)
    print(f"✅ Fusion complete: {fusion_stats}")
    
    # Hybrid retrieval
    print("\n🔍 Hybrid Retrieval")
    
    results = graph_manager.hybrid_retrieval(
        query="transformer attention mechanisms",
        retrieval_types=["semantic", "structural"],
        max_nodes=10
    )
    
    print(f"✅ Retrieved {len(results.nodes)} nodes")
    
    # Reasoning write-back
    print("\n🧠 Reasoning Write-Back")
    
    reasoning_step = {
        "type": "deduction",
        "premises": ["Transformers use attention", "Attention enables parallelization"],
        "conclusion": "Transformers enable parallel processing",
        "confidence": 0.9
    }
    
    writeback_result = graph_manager.reasoning_writeback(reasoning_step)
    print(f"✅ Reasoning stored: {writeback_result}")
    
    print("\n🎉 Tutorial 3 Complete!")

if __name__ == "__main__":
    asyncio.run(tutorial_semantic_graph_operations())