#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 7: Confidence Filtering with DeepConf Integration
=========================================================

This tutorial demonstrates how to use the confidence filtering system
with DeepConf methodology for enhanced response quality and reliability.

Topics covered:
1. Basic confidence filtering setup
2. Token-level confidence scoring
3. Early termination logic
4. Confidence-aware voting
5. Semantic graph integration
6. RLHF integration
7. Complete workflow examples
8. Performance monitoring and optimization

Prerequisites:
- Basic understanding of AI Research Agent
- Familiarity with confidence concepts
- Python 3.8+ with required dependencies
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Import confidence filtering components
from extensions.stage_7_confidence_filtering import (
    ConfidenceScoringHook,
    EarlyTerminationLogic,
    ConfidenceAwareVoting,
    SemanticGraphAlignment,
    DeepConfIntegration,
    ConfidenceFilterManager,
    ConfidenceStrategy,
    TokenConfidence,
    TraceConfidence,
    GroupConfidence,
    ReasoningPhase,
    integrate_confidence_filtering
)

# Import related components for integration examples
try:
    from extensions.stage_3_semantic_graph import SemanticGraphManager, NodeType, EdgeType
    from extensions.stage_5_rlhf_agentic_rl import ConfidenceRLHFIntegration
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    print("⚠️ Some integration components not available. Basic examples will still work.")
    INTEGRATIONS_AVAILABLE = False

class ConfidenceFilteringTutorial:
    """Interactive tutorial for confidence filtering"""
    
    def __init__(self):
        self.tutorial_data = {}
        self.results = {}
        
    def print_section(self, title: str, description: str = ""):
        """Print formatted section header"""
        print(f"\n{'='*60}")
        print(f"📚 {title}")
        print(f"{'='*60}")
        if description:
            print(f"{description}\n")
    
    def print_step(self, step: str, description: str = ""):
        """Print formatted step"""
        print(f"\n🔹 {step}")
        if description:
            print(f"   {description}")
    
    def print_result(self, result: str):
        """Print formatted result"""
        print(f"   ✅ {result}")
    
    def print_code_example(self, code: str, description: str = ""):
        """Print formatted code example"""
        if description:
            print(f"\n💡 {description}")
        print(f"```python\n{code}\n```")

# =============================================================================
# Section 1: Basic Confidence Filtering Setup
# =============================================================================

def section_1_basic_setup():
    """Section 1: Basic confidence filtering setup"""
    tutorial = ConfidenceFilteringTutorial()
    
    tutorial.print_section(
        "Section 1: Basic Confidence Filtering Setup",
        "Learn how to initialize and configure the confidence filtering system"
    )
    
    # Step 1.1: Initialize basic confidence filter
    tutorial.print_step("Step 1.1: Initialize Basic Confidence Filter")
    
    # Basic configuration
    config = {
        "strategy": "adaptive_threshold",
        "threshold": 15.0,
        "adaptation_rate": 0.1
    }
    
    # Initialize confidence filter manager
    manager = ConfidenceFilterManager(config)
    
    tutorial.print_result(f"Initialized confidence filter with strategy: {manager.strategy.value}")
    tutorial.print_result(f"Initial threshold: {config['threshold']}")
    
    # Step 1.2: Test basic filtering
    tutorial.print_step("Step 1.2: Test Basic Response Filtering")
    
    # Sample response data with logprobs
    test_responses = [
        {
            "name": "High Confidence Response",
            "logprobs": [-0.1, -0.2, -0.15, -0.3, -0.25],  # High confidence
            "expected": "PASS"
        },
        {
            "name": "Medium Confidence Response", 
            "logprobs": [-0.8, -1.0, -0.9, -1.2, -0.7],   # Medium confidence
            "expected": "PASS"
        },
        {
            "name": "Low Confidence Response",
            "logprobs": [-2.0, -2.5, -3.0, -2.8, -2.2],   # Low confidence
            "expected": "FAIL"
        }
    ]
    
    print("\n   Testing different confidence levels:")
    for response in test_responses:
        result = manager.filter_response({"logprobs": response["logprobs"]})
        
        status = "✅ PASSED" if result.passed else "❌ FILTERED"
        print(f"   {response['name']}: {status}")
        print(f"      Confidence Score: {result.confidence_score:.3f}")
        print(f"      Reason: {result.reason}")
    
    # Step 1.3: View statistics
    tutorial.print_step("Step 1.3: View Filtering Statistics")
    
    stats = manager.get_statistics()
    print(f"   Total Filtered: {stats['total_filtered']}")
    print(f"   Pass Rate: {stats['pass_rate']:.1%}")
    print(f"   Average Confidence: {stats['average_confidence']:.3f}")
    
    return manager

# =============================================================================
# Section 2: Token-Level Confidence Scoring
# =============================================================================

def section_2_token_confidence():
    """Section 2: Token-level confidence scoring"""
    tutorial = ConfidenceFilteringTutorial()
    
    tutorial.print_section(
        "Section 2: Token-Level Confidence Scoring",
        "Learn how to compute and track confidence at the token level"
    )
    
    # Step 2.1: Initialize confidence scoring hook
    tutorial.print_step("Step 2.1: Initialize Confidence Scoring Hook")
    
    scoring_hook = ConfidenceScoringHook(enable_real_time=True)
    
    # Register callback for demonstration
    confidence_history = []
    
    def confidence_callback(token_confidence: TokenConfidence):
        confidence_history.append(token_confidence)
        print(f"      Token: '{token_confidence.token}' | "
              f"Confidence: {token_confidence.confidence:.3f} | "
              f"Reliable: {token_confidence.is_reliable}")
    
    scoring_hook.register_generation_callback(confidence_callback)
    tutorial.print_result("Confidence scoring hook initialized with real-time callback")
    
    # Step 2.2: Simulate token generation with confidence scoring
    tutorial.print_step("Step 2.2: Simulate Token Generation with Confidence Scoring")
    
    # Simulate a response about machine learning
    simulated_tokens = [
        ("Machine", -0.2),    # High confidence
        ("learning", -0.1),   # Very high confidence
        ("is", -0.4),         # Good confidence
        ("a", -0.3),          # Good confidence
        ("subset", -0.8),     # Medium confidence
        ("of", -0.5),         # Medium confidence
        ("artificial", -0.6), # Medium confidence
        ("intelligence", -0.3), # Good confidence
        ("that", -1.2),       # Lower confidence
        ("enables", -1.5),    # Low confidence
        ("computers", -0.7),  # Medium confidence
        ("to", -0.4),         # Good confidence
        ("learn", -0.2),      # High confidence
        ("automatically", -1.8), # Very low confidence
        (".", -0.1)           # Very high confidence
    ]
    
    print("\n   Generating tokens with confidence scoring:")
    token_confidences = []
    
    for i, (token, logprob) in enumerate(simulated_tokens):
        token_conf = scoring_hook.compute_token_confidence(token, logprob, i)
        token_confidences.append(token_conf)
    
    # Step 2.3: Analyze token confidence patterns
    tutorial.print_step("Step 2.3: Analyze Token Confidence Patterns")
    
    # Calculate statistics
    confidences = [tc.confidence for tc in token_confidences]
    reliable_count = sum(1 for tc in token_confidences if tc.is_reliable)
    
    print(f"\n   Token Confidence Analysis:")
    print(f"   Total Tokens: {len(token_confidences)}")
    print(f"   Reliable Tokens: {reliable_count}/{len(token_confidences)} ({reliable_count/len(token_confidences):.1%})")
    print(f"   Average Confidence: {np.mean(confidences):.3f}")
    print(f"   Confidence Range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
    
    # Step 2.4: Compute trace-level confidence
    tutorial.print_step("Step 2.4: Compute Trace-Level Confidence")
    
    trace_confidence = scoring_hook.get_trace_confidence(token_confidences)
    print(f"   Overall Trace Confidence: {trace_confidence:.3f}")
    
    # Identify low-confidence regions
    low_confidence_tokens = [tc for tc in token_confidences if tc.confidence < 0.3]
    if low_confidence_tokens:
        print(f"   Low Confidence Tokens:")
        for tc in low_confidence_tokens:
            print(f"      Position {tc.position}: '{tc.token}' (confidence: {tc.confidence:.3f})")
    
    return token_confidences, trace_confidence

# =============================================================================
# Section 3: Early Termination Logic
# =============================================================================

def section_3_early_termination():
    """Section 3: Early termination logic"""
    tutorial = ConfidenceFilteringTutorial()
    
    tutorial.print_section(
        "Section 3: Early Termination Logic",
        "Learn how to implement compute-efficient inference with confidence gates"
    )
    
    # Step 3.1: Initialize early termination logic
    tutorial.print_step("Step 3.1: Initialize Early Termination Logic")
    
    early_termination = EarlyTerminationLogic(
        threshold_percentile=90,
        warmup_traces=16,
        window_size=10
    )
    
    tutorial.print_result("Early termination logic initialized")
    print(f"   Threshold Percentile: 90th")
    print(f"   Warmup Traces Required: 16")
    print(f"   Sliding Window Size: 10")
    
    # Step 3.2: Create mock traces for warmup
    tutorial.print_step("Step 3.2: Create Mock Traces for Offline Warmup")
    
    # Generate mock trace confidences for warmup
    np.random.seed(42)  # For reproducible results
    
    mock_traces = []
    for i in range(20):
        # Simulate different quality traces
        if i < 5:  # High quality traces
            confidence = np.random.normal(0.8, 0.1)
        elif i < 15:  # Medium quality traces
            confidence = np.random.normal(0.6, 0.15)
        else:  # Low quality traces
            confidence = np.random.normal(0.3, 0.1)
        
        confidence = max(0.1, min(0.95, confidence))  # Clamp to reasonable range
        
        # Create mock trace confidence object
        trace_conf = TraceConfidence(
            trace_id=f"warmup_trace_{i}",
            tokens=[],  # Simplified for demo
            group_confidence=GroupConfidence(window_size=5),
            overall_confidence=confidence,
            reasoning_phase=ReasoningPhase.GENERATION
        )
        mock_traces.append(trace_conf)
    
    print(f"   Generated {len(mock_traces)} mock traces for warmup")
    print(f"   Confidence range: {min(t.overall_confidence for t in mock_traces):.3f} - {max(t.overall_confidence for t in mock_traces):.3f}")
    
    # Step 3.3: Perform offline warmup
    tutorial.print_step("Step 3.3: Perform Offline Warmup")
    
    early_termination.offline_warmup(mock_traces)
    
    tutorial.print_result(f"Warmup completed! Threshold set to: {early_termination.threshold:.3f}")
    print(f"   Warmed up: {early_termination.is_warmed_up}")
    
    # Step 3.4: Test early termination decisions
    tutorial.print_step("Step 3.4: Test Early Termination Decisions")
    
    test_scenarios = [
        {"confidence": 0.8, "tokens": 50, "expected": "Continue"},
        {"confidence": 0.6, "tokens": 30, "expected": "Continue"},
        {"confidence": 0.4, "tokens": 25, "expected": "Terminate"},
        {"confidence": 0.2, "tokens": 40, "expected": "Terminate"},
        {"confidence": 0.7, "tokens": 3, "expected": "Continue (too few tokens)"}
    ]
    
    print("\n   Testing early termination decisions:")
    for scenario in test_scenarios:
        should_terminate, reason = early_termination.should_terminate_early(
            scenario["confidence"], 
            scenario["tokens"]
        )
        
        decision = "🛑 TERMINATE" if should_terminate else "✅ CONTINUE"
        print(f"   Confidence: {scenario['confidence']:.1f}, Tokens: {scenario['tokens']} → {decision}")
        print(f"      Reason: {reason}")
    
    # Step 3.5: Demonstrate adaptive threshold adjustment
    tutorial.print_step("Step 3.5: Demonstrate Adaptive Threshold Adjustment")
    
    print(f"   Initial threshold: {early_termination.threshold:.3f}")
    
    # Simulate confidence updates that would trigger adaptation
    for i in range(150):  # Need 100+ samples for adaptation
        # Simulate varying confidence levels
        confidence = np.random.beta(2, 2)  # Beta distribution for realistic confidence
        early_termination.update_confidence_history(confidence)
    
    print(f"   Threshold after adaptation: {early_termination.threshold:.3f}")
    print(f"   Confidence history size: {len(early_termination.confidence_history)}")
    
    return early_termination

# =============================================================================
# Section 4: Confidence-Aware Voting
# =============================================================================

def section_4_confidence_voting():
    """Section 4: Confidence-aware voting"""
    tutorial = ConfidenceFilteringTutorial()
    
    tutorial.print_section(
        "Section 4: Confidence-Aware Voting",
        "Learn how to aggregate multiple reasoning paths with confidence weighting"
    )
    
    # Step 4.1: Initialize confidence-aware voting
    tutorial.print_step("Step 4.1: Initialize Confidence-Aware Voting System")
    
    voting_system = ConfidenceAwareVoting(
        top_n_percent=50,      # Keep top 50% confident traces
        min_confidence=0.1     # Minimum confidence threshold
    )
    
    tutorial.print_result("Confidence-aware voting system initialized")
    print(f"   Top N Percent: {voting_system.top_n_percent}%")
    print(f"   Minimum Confidence: {voting_system.min_confidence}")
    
    # Step 4.2: Create mock reasoning traces
    tutorial.print_step("Step 4.2: Create Mock Reasoning Traces")
    
    # Create diverse reasoning traces with different confidence levels
    reasoning_traces = []
    
    trace_scenarios = [
        {
            "id": "high_conf_correct",
            "confidence": 0.9,
            "semantic_reliability": 0.85,
            "description": "High confidence, semantically reliable"
        },
        {
            "id": "high_conf_uncertain",
            "confidence": 0.85,
            "semantic_reliability": 0.6,
            "description": "High confidence, but semantically uncertain"
        },
        {
            "id": "medium_conf_reliable",
            "confidence": 0.6,
            "semantic_reliability": 0.8,
            "description": "Medium confidence, semantically reliable"
        },
        {
            "id": "medium_conf_average",
            "confidence": 0.55,
            "semantic_reliability": 0.5,
            "description": "Medium confidence, average reliability"
        },
        {
            "id": "low_conf_unreliable",
            "confidence": 0.3,
            "semantic_reliability": 0.2,
            "description": "Low confidence, unreliable"
        },
        {
            "id": "very_low_conf",
            "confidence": 0.15,
            "semantic_reliability": 0.1,
            "description": "Very low confidence"
        }
    ]
    
    for scenario in trace_scenarios:
        trace = TraceConfidence(
            trace_id=scenario["id"],
            tokens=[],  # Simplified for demo
            group_confidence=GroupConfidence(window_size=5),
            overall_confidence=scenario["confidence"],
            reasoning_phase=ReasoningPhase.EVALUATION,
            semantic_reliability=scenario["semantic_reliability"]
        )
        reasoning_traces.append(trace)
        print(f"   Created trace: {scenario['description']}")
    
    # Step 4.3: Filter top confident traces
    tutorial.print_step("Step 4.3: Filter Top Confident Traces")
    
    top_traces = voting_system.filter_top_confident_traces(reasoning_traces)
    
    print(f"\n   Original traces: {len(reasoning_traces)}")
    print(f"   Filtered to top {voting_system.top_n_percent}%: {len(top_traces)}")
    print(f"   Selected traces:")
    
    for trace in top_traces:
        print(f"      {trace.trace_id}: confidence={trace.overall_confidence:.3f}")
    
    # Show filtered traces
    filtered_traces = [t for t in reasoning_traces if t not in top_traces]
    if filtered_traces:
        print(f"   Filtered out:")
        for trace in filtered_traces:
            print(f"      {trace.trace_id}: confidence={trace.overall_confidence:.3f} (reason: {trace.filter_reason})")
    
    # Step 4.4: Compute vote weights
    tutorial.print_step("Step 4.4: Compute Vote Weights")
    
    vote_weights = voting_system.compute_vote_weights(top_traces)
    
    print(f"\n   Vote weights computed:")
    for trace_id, weight in vote_weights.items():
        trace = next(t for t in top_traces if t.trace_id == trace_id)
        print(f"      {trace_id}: weight={weight:.3f} (confidence={trace.overall_confidence:.3f}, semantic={trace.semantic_reliability:.3f})")
    
    # Step 4.5: Aggregate votes for different answers
    tutorial.print_step("Step 4.5: Aggregate Votes for Different Answers")
    
    # Create mock answers corresponding to traces
    candidate_answers = [
        "Neural networks learn through backpropagation and gradient descent",  # high_conf_correct
        "Neural networks use complex mathematical optimization techniques",     # high_conf_uncertain  
        "Neural networks adjust weights based on error feedback",              # medium_conf_reliable
        "Neural networks learn by processing data iteratively",                # medium_conf_average
        "Neural networks use some kind of learning algorithm",                 # low_conf_unreliable
        "Neural networks learn somehow through training"                       # very_low_conf
    ]
    
    # Only use answers for top traces
    top_answers = [candidate_answers[i] for i, trace in enumerate(reasoning_traces) if trace in top_traces]
    
    best_answer, best_confidence, voting_metadata = voting_system.aggregate_votes(
        top_traces, top_answers
    )
    
    print(f"\n   Voting Results:")
    print(f"   Best Answer: '{best_answer[:60]}...'")
    print(f"   Winning Confidence: {best_confidence:.3f}")
    print(f"   Total Traces Considered: {voting_metadata['total_traces']}")
    print(f"   Traces After Filtering: {voting_metadata['filtered_traces']}")
    print(f"   Filter Rate: {voting_metadata['filter_rate']:.1%}")
    
    return voting_system, top_traces, best_answer

# =============================================================================
# Section 5: Semantic Graph Integration
# =============================================================================

def section_5_semantic_integration():
    """Section 5: Semantic graph integration"""
    tutorial = ConfidenceFilteringTutorial()
    
    tutorial.print_section(
        "Section 5: Semantic Graph Integration",
        "Learn how to integrate confidence with semantic graph for reasoning reliability"
    )
    
    if not INTEGRATIONS_AVAILABLE:
        print("⚠️ Semantic graph integration not available. Skipping this section.")
        return None
    
    # Step 5.1: Initialize semantic graph alignment
    tutorial.print_step("Step 5.1: Initialize Semantic Graph Alignment")
    
    semantic_alignment = SemanticGraphAlignment(confidence_threshold=0.7)
    
    tutorial.print_result("Semantic graph alignment initialized")
    print(f"   Confidence threshold: {semantic_alignment.confidence_threshold}")
    
    # Step 5.2: Create mock semantic graph
    tutorial.print_step("Step 5.2: Create Mock Semantic Graph with Nodes")
    
    # Create a simple semantic graph for demonstration
    graph_manager = SemanticGraphManager()
    
    # Add some concept nodes
    concept_nodes = [
        {
            "content": "Machine learning is a subset of artificial intelligence",
            "node_type": NodeType.CONCEPT,
            "title": "ML Definition",
            "confidence": 0.9
        },
        {
            "content": "Neural networks are inspired by biological neurons",
            "node_type": NodeType.CONCEPT, 
            "title": "Neural Network Concept",
            "confidence": 0.8
        },
        {
            "content": "Backpropagation is used to train neural networks",
            "node_type": NodeType.CLAIM,
            "title": "Backpropagation Claim",
            "confidence": 0.85
        },
        {
            "content": "Deep learning uses multiple hidden layers",
            "node_type": NodeType.CONCEPT,
            "title": "Deep Learning Definition", 
            "confidence": 0.75
        }
    ]
    
    node_ids = []
    for node_info in concept_nodes:
        node_id = graph_manager.add_node(
            content=node_info["content"],
            node_type=node_info["node_type"],
            source_type=SourceType.INTERNAL,
            title=node_info["title"]
        )
        node_ids.append(node_id)
        print(f"   Added node: {node_info['title']}")
    
    # Step 5.3: Annotate nodes with confidence-based reliability
    tutorial.print_step("Step 5.3: Annotate Nodes with Confidence-Based Reliability")
    
    for i, (node_id, node_info) in enumerate(zip(node_ids, concept_nodes)):
        # Create mock trace confidence for this node
        trace_confidence = TraceConfidence(
            trace_id=f"trace_for_node_{i}",
            tokens=[],
            group_confidence=GroupConfidence(window_size=5),
            overall_confidence=node_info["confidence"],
            reasoning_phase=ReasoningPhase.EVALUATION,
            semantic_reliability=node_info["confidence"] * 0.9  # Slightly lower than confidence
        )
        
        # Annotate node with reliability
        reliability_score = semantic_alignment.annotate_node_reliability(
            node_id, trace_confidence
        )
        
        print(f"   Node '{node_info['title']}': reliability={reliability_score:.3f}")
    
    # Step 5.4: Create reasoning paths and evaluate confidence
    tutorial.print_step("Step 5.4: Create Reasoning Paths and Evaluate Confidence")
    
    # Define some reasoning paths through the graph
    candidate_paths = [
        [node_ids[0], node_ids[1], node_ids[2]],  # ML -> Neural Networks -> Backprop
        [node_ids[0], node_ids[3], node_ids[2]],  # ML -> Deep Learning -> Backprop
        [node_ids[1], node_ids[3]],               # Neural Networks -> Deep Learning
        [node_ids[0], node_ids[1]]                # ML -> Neural Networks
    ]
    
    path_descriptions = [
        "ML → Neural Networks → Backpropagation",
        "ML → Deep Learning → Backpropagation", 
        "Neural Networks → Deep Learning",
        "ML → Neural Networks"
    ]
    
    # Evaluate path confidence
    scored_paths = semantic_alignment.guide_path_selection(candidate_paths)
    
    print(f"\n   Reasoning Path Confidence Scores:")
    for i, (path, confidence) in enumerate(scored_paths):
        print(f"   {path_descriptions[i]}: confidence={confidence:.3f}")
    
    # Step 5.5: Demonstrate source prioritization
    tutorial.print_step("Step 5.5: Demonstrate Source Prioritization")
    
    # Mock external sources with different reliability characteristics
    mock_sources = [
        {
            "id": "arxiv_paper_1",
            "source_type": "academic",
            "citation_count": 250,
            "recency_score": 0.8,
            "title": "Deep Learning Fundamentals"
        },
        {
            "id": "blog_post_1", 
            "source_type": "blog",
            "citation_count": 5,
            "recency_score": 0.95,
            "title": "ML Explained Simply"
        },
        {
            "id": "official_docs_1",
            "source_type": "official_docs",
            "citation_count": 100,
            "recency_score": 0.7,
            "title": "TensorFlow Documentation"
        },
        {
            "id": "forum_post_1",
            "source_type": "forum",
            "citation_count": 2,
            "recency_score": 0.9,
            "title": "Stack Overflow Answer"
        }
    ]
    
    prioritized_sources = semantic_alignment.prioritize_retrieval_sources(mock_sources)
    
    print(f"\n   Source Prioritization (by reliability):")
    for i, source in enumerate(prioritized_sources):
        print(f"   {i+1}. {source['title']} ({source['source_type']})")
    
    return semantic_alignment, graph_manager

# =============================================================================
# Section 6: RLHF Integration
# =============================================================================

def section_6_rlhf_integration():
    """Section 6: RLHF integration"""
    tutorial = ConfidenceFilteringTutorial()
    
    tutorial.print_section(
        "Section 6: RLHF Integration",
        "Learn how to integrate confidence filtering with RLHF for reward shaping"
    )
    
    if not INTEGRATIONS_AVAILABLE:
        print("⚠️ RLHF integration not available. Skipping this section.")
        return None
    
    # Step 6.1: Initialize RLHF integration
    tutorial.print_step("Step 6.1: Initialize Confidence-RLHF Integration")
    
    rlhf_integration = ConfidenceRLHFIntegration()
    
    tutorial.print_result("Confidence-RLHF integration initialized")
    print("   Features enabled:")
    print("   - Confidence-aware reward shaping")
    print("   - Dynamic exploration based on uncertainty")
    print("   - Calibration-aware alignment evaluation")
    
    # Step 6.2: Demonstrate confidence-aware action selection
    tutorial.print_step("Step 6.2: Demonstrate Confidence-Aware Action Selection")
    
    # Define a research scenario
    research_state = {
        "query": "How do transformer attention mechanisms work?",
        "context_size": 1500,
        "user_expertise": "intermediate",
        "domain": "machine_learning"
    }
    
    available_actions = [
        "provide_detailed_technical_explanation",
        "give_intuitive_overview", 
        "show_step_by_step_example",
        "compare_with_other_mechanisms"
    ]
    
    # Test with different confidence levels
    confidence_scenarios = [
        {
            "name": "High Confidence Scenario",
            "confidence_metrics": {
                "confidence_score": 0.9,
                "uncertainty_score": 0.1,
                "mean_logprob": -0.3,
                "variance": 0.05
            }
        },
        {
            "name": "Medium Confidence Scenario",
            "confidence_metrics": {
                "confidence_score": 0.6,
                "uncertainty_score": 0.4,
                "mean_logprob": -1.0,
                "variance": 0.2
            }
        },
        {
            "name": "Low Confidence Scenario",
            "confidence_metrics": {
                "confidence_score": 0.3,
                "uncertainty_score": 0.7,
                "mean_logprob": -2.0,
                "variance": 0.5
            }
        }
    ]
    
    print(f"\n   Testing action selection with different confidence levels:")
    
    for scenario in confidence_scenarios:
        print(f"\n   {scenario['name']}:")
        
        result = rlhf_integration.process_research_action(
            state=research_state,
            available_actions=available_actions,
            confidence_metrics=scenario["confidence_metrics"]
        )
        
        print(f"      Selected Action: {result['selected_action']}")
        print(f"      Confidence Score: {result['action_metadata']['confidence_score']:.3f}")
        print(f"      Exploration Rate: {result['action_metadata']['exploration_rate']:.3f}")
        print(f"      Predicted Reward: {result['action_metadata']['predicted_reward']:.3f}")
    
    # Step 6.3: Demonstrate confidence-aware reward shaping
    tutorial.print_step("Step 6.3: Demonstrate Confidence-Aware Reward Shaping")
    
    # Simulate action outcomes with different quality levels
    action_outcomes = [
        {
            "name": "High Quality Outcome",
            "outcome": {
                "quality_score": 0.9,
                "response_time": 1.5,
                "factuality_score": 0.95,
                "user_satisfaction": 0.85,
                "session_id": "demo_session"
            },
            "confidence_metrics": {
                "confidence_score": 0.9,
                "uncertainty_score": 0.1,
                "mean_logprob": -0.3,
                "variance": 0.05
            }
        },
        {
            "name": "Overconfident Poor Outcome",
            "outcome": {
                "quality_score": 0.3,  # Poor quality
                "response_time": 2.0,
                "factuality_score": 0.4,
                "user_satisfaction": 0.2,
                "session_id": "demo_session"
            },
            "confidence_metrics": {
                "confidence_score": 0.9,  # But high confidence (overconfident)
                "uncertainty_score": 0.1,
                "mean_logprob": -0.4,
                "variance": 0.08
            }
        },
        {
            "name": "Appropriately Uncertain Outcome",
            "outcome": {
                "quality_score": 0.6,
                "response_time": 1.8,
                "factuality_score": 0.7,
                "user_satisfaction": 0.6,
                "session_id": "demo_session"
            },
            "confidence_metrics": {
                "confidence_score": 0.5,  # Appropriately uncertain
                "uncertainty_score": 0.5,
                "mean_logprob": -1.2,
                "variance": 0.3
            }
        }
    ]
    
    print(f"\n   Testing reward shaping with different outcome scenarios:")
    
    for i, scenario in enumerate(action_outcomes):
        action_id = f"demo_action_{i}"
        
        print(f"\n   {scenario['name']}:")
        print(f"      Base Quality Score: {scenario['outcome']['quality_score']:.3f}")
        print(f"      Confidence Score: {scenario['confidence_metrics']['confidence_score']:.3f}")
        
        # Record outcome with confidence-aware reward shaping
        reward_result = rlhf_integration.record_action_outcome(
            action_id=action_id,
            outcome=scenario["outcome"],
            confidence_metrics=scenario["confidence_metrics"]
        )
        
        print(f"      Reward Signals Created: {reward_result['reward_signals_count']}")
        print(f"      Confidence Shaped: {reward_result['confidence_shaped']}")
    
    # Step 6.4: Demonstrate confidence-weighted preference collection
    tutorial.print_step("Step 6.4: Demonstrate Confidence-Weighted Preference Collection")
    
    # Example preference scenario
    query = "Explain the attention mechanism in transformers"
    
    response_a = """
    The attention mechanism allows the model to focus on different parts of the input sequence 
    when processing each token. It computes attention weights using query, key, and value matrices.
    """
    
    response_b = """
    Attention in transformers works by calculating similarity scores between tokens, then using 
    these scores to create weighted combinations of token representations for better context understanding.
    """
    
    # Different confidence levels for each response
    confidence_a = {
        "confidence_score": 0.7,
        "uncertainty_score": 0.3,
        "mean_logprob": -0.8,
        "variance": 0.15
    }
    
    confidence_b = {
        "confidence_score": 0.9,
        "uncertainty_score": 0.1,
        "mean_logprob": -0.4,
        "variance": 0.08
    }
    
    # Collect preference with confidence weighting
    preference_id = rlhf_integration.collect_preference_with_confidence(
        query=query,
        response_a=response_a,
        response_b=response_b,
        preference=1,  # Prefer response B
        confidence_a=confidence_a,
        confidence_b=confidence_b
    )
    
    print(f"\n   Preference Collection Results:")
    print(f"   Preference ID: {preference_id}")
    print(f"   Preferred Response: B (higher confidence)")
    print(f"   Response A Confidence: {confidence_a['confidence_score']:.3f}")
    print(f"   Response B Confidence: {confidence_b['confidence_score']:.3f}")
    
    # Step 6.5: View integration statistics
    tutorial.print_step("Step 6.5: View Integration Statistics")
    
    stats = rlhf_integration.get_integration_statistics()
    
    print(f"\n   Integration Statistics:")
    print(f"   High Confidence Actions: {stats['integration_stats']['high_confidence_actions']}")
    print(f"   Low Confidence Explorations: {stats['integration_stats']['low_confidence_explorations']}")
    print(f"   Confidence Shaped Rewards: {stats['integration_stats']['confidence_shaped_rewards']}")
    print(f"   Calibration Adjustments: {stats['integration_stats']['calibration_adjustments']}")
    
    return rlhf_integration

# =============================================================================
# Section 7: Complete Workflow Examples
# =============================================================================

async def section_7_complete_workflow():
    """Section 7: Complete workflow examples"""
    tutorial = ConfidenceFilteringTutorial()
    
    tutorial.print_section(
        "Section 7: Complete Workflow Examples",
        "Learn how to use confidence filtering in end-to-end research workflows"
    )
    
    # Step 7.1: Initialize complete DeepConf integration
    tutorial.print_step("Step 7.1: Initialize Complete DeepConf Integration")
    
    config = {
        "strategy": "adaptive_threshold",
        "threshold": 15.0,
        "adaptation_rate": 0.1,
        "enable_real_time": True,
        "top_n_percent": 30,
        "semantic_threshold": 0.7,
        "confidence_weight": 0.3,
        "uncertainty_penalty": 0.2
    }
    
    integration = DeepConfIntegration(config)
    
    tutorial.print_result("Complete DeepConf integration initialized")
    print("   Features enabled:")
    print("   - Real-time confidence scoring")
    print("   - Early termination logic")
    print("   - Confidence-aware voting")
    print("   - Semantic graph alignment")
    print("   - RLHF reward shaping")
    
    # Step 7.2: Research workflow example
    tutorial.print_step("Step 7.2: Research Workflow Example")
    
    research_requests = [
        {
            "session_id": "research_session_1",
            "query": "What are the latest developments in transformer architectures?",
            "expected_quality": "high"
        },
        {
            "session_id": "research_session_2",
            "query": "How does quantum computing work?",
            "expected_quality": "medium"
        },
        {
            "session_id": "research_session_3",
            "query": "Explain the concept of machine learning",
            "expected_quality": "high"
        }
    ]
    
    print(f"\n   Processing research requests with confidence filtering:")
    
    workflow_results = []
    for i, request in enumerate(research_requests):
        print(f"\n   Request {i+1}: {request['query'][:50]}...")
        
        # Process request with full confidence integration
        result = await integration.process_research_request(request)
        workflow_results.append(result)
        
        # Display results
        status = "✅ SUCCESS" if result['success'] else "❌ FILTERED"
        print(f"      Status: {status}")
        print(f"      Confidence Score: {result['confidence_score']:.3f}")
        print(f"      Filter Passed: {result['filter_result']['passed']}")
        
        if result['success']:
            print(f"      Answer Preview: {result['answer'][:80]}...")
        else:
            print(f"      Filter Reason: {result['filter_result']['reason']}")
        
        # Show trace metadata
        trace_meta = result['trace_metadata']
        print(f"      Semantic Reliability: {trace_meta['semantic_reliability']:.3f}")
        print(f"      Early Termination: {trace_meta['early_termination']}")
    
    # Step 7.3: Analyze workflow performance
    tutorial.print_step("Step 7.3: Analyze Workflow Performance")
    
    # Calculate workflow statistics
    successful_requests = [r for r in workflow_results if r['success']]
    filtered_requests = [r for r in workflow_results if not r['success']]
    
    avg_confidence = np.mean([r['confidence_score'] for r in workflow_results])
    avg_semantic_reliability = np.mean([r['trace_metadata']['semantic_reliability'] for r in workflow_results])
    
    print(f"\n   Workflow Performance Analysis:")
    print(f"   Total Requests: {len(workflow_results)}")
    print(f"   Successful: {len(successful_requests)} ({len(successful_requests)/len(workflow_results):.1%})")
    print(f"   Filtered: {len(filtered_requests)} ({len(filtered_requests)/len(workflow_results):.1%})")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   Average Semantic Reliability: {avg_semantic_reliability:.3f}")
    
    # Step 7.4: Integration status and metrics
    tutorial.print_step("Step 7.4: Integration Status and Metrics")
    
    status = integration.get_integration_status()
    
    print(f"\n   Integration Status:")
    print(f"   Active Traces: {status['active_traces']}")
    print(f"   Total Requests Processed: {status['integration_metrics']['total_requests']}")
    print(f"   Early Terminations: {status['integration_metrics']['early_terminations']}")
    print(f"   Semantic Alignments: {status['integration_metrics']['semantic_alignments']}")
    print(f"   Voting Decisions: {status['integration_metrics']['voting_decisions']}")
    
    # Early termination statistics
    et_stats = status['early_termination_stats']
    print(f"\n   Early Termination Status:")
    print(f"   Threshold: {et_stats['threshold']:.3f}")
    print(f"   Warmed Up: {et_stats['is_warmed_up']}")
    print(f"   Confidence History Size: {et_stats['confidence_history_size']}")
    
    return integration, workflow_results

# =============================================================================
# Section 8: Performance Monitoring and Optimization
# =============================================================================

def section_8_performance_monitoring():
    """Section 8: Performance monitoring and optimization"""
    tutorial = ConfidenceFilteringTutorial()
    
    tutorial.print_section(
        "Section 8: Performance Monitoring and Optimization",
        "Learn how to monitor and optimize confidence filtering performance"
    )
    
    # Step 8.1: Initialize monitoring system
    tutorial.print_step("Step 8.1: Initialize Performance Monitoring")
    
    # Create a confidence filter manager for monitoring
    config = {
        "strategy": "adaptive_threshold",
        "threshold": 15.0,
        "enable_statistics": True
    }
    
    manager = ConfidenceFilterManager(config)
    
    # Generate some test data for monitoring
    test_data = [
        {"logprobs": [-0.1, -0.2, -0.15], "expected": True},   # High confidence
        {"logprobs": [-0.8, -1.0, -0.9], "expected": True},   # Medium confidence
        {"logprobs": [-2.0, -2.5, -3.0], "expected": False},  # Low confidence
        {"logprobs": [-0.3, -0.4, -0.2], "expected": True},   # High confidence
        {"logprobs": [-1.5, -1.8, -2.0], "expected": False},  # Low confidence
    ]
    
    print(f"   Processing {len(test_data)} test samples for monitoring...")
    
    results = []
    for i, sample in enumerate(test_data):
        result = manager.filter_response({"logprobs": sample["logprobs"]})
        results.append({
            "sample_id": i,
            "passed": result.passed,
            "confidence": result.confidence_score,
            "expected": sample["expected"],
            "correct": result.passed == sample["expected"]
        })
    
    tutorial.print_result("Test data processed for monitoring")
    
    # Step 8.2: Analyze performance metrics
    tutorial.print_step("Step 8.2: Analyze Performance Metrics")
    
    # Calculate key metrics
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r["correct"])
    accuracy = correct_predictions / total_samples
    
    passed_samples = [r for r in results if r["passed"]]
    failed_samples = [r for r in results if not r["passed"]]
    
    avg_confidence_passed = np.mean([r["confidence"] for r in passed_samples]) if passed_samples else 0
    avg_confidence_failed = np.mean([r["confidence"] for r in failed_samples]) if failed_samples else 0
    
    print(f"\n   Performance Metrics:")
    print(f"   Total Samples: {total_samples}")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Samples Passed: {len(passed_samples)} ({len(passed_samples)/total_samples:.1%})")
    print(f"   Samples Failed: {len(failed_samples)} ({len(failed_samples)/total_samples:.1%})")
    print(f"   Avg Confidence (Passed): {avg_confidence_passed:.3f}")
    print(f"   Avg Confidence (Failed): {avg_confidence_failed:.3f}")
    
    # Step 8.3: Calibration analysis
    tutorial.print_step("Step 8.3: Calibration Analysis")
    
    # Simple calibration analysis
    confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print(f"\n   Confidence Calibration Analysis:")
    print(f"   Bin Range    | Count | Accuracy | Avg Confidence")
    print(f"   -------------|-------|----------|---------------")
    
    for i in range(len(confidence_bins) - 1):
        bin_min, bin_max = confidence_bins[i], confidence_bins[i + 1]
        
        bin_results = [r for r in results if bin_min <= r["confidence"] < bin_max]
        
        if bin_results:
            bin_accuracy = sum(1 for r in bin_results if r["correct"]) / len(bin_results)
            bin_avg_conf = np.mean([r["confidence"] for r in bin_results])
            
            print(f"   {bin_min:.1f} - {bin_max:.1f}    | {len(bin_results):5d} | {bin_accuracy:8.1%} | {bin_avg_conf:13.3f}")
    
    # Step 8.4: Optimization recommendations
    tutorial.print_step("Step 8.4: Optimization Recommendations")
    
    # Generate optimization recommendations based on performance
    recommendations = []
    
    if accuracy < 0.8:
        recommendations.append("Consider adjusting confidence threshold - current accuracy is low")
    
    if len(passed_samples) / total_samples < 0.5:
        recommendations.append("Filter may be too conservative - consider lowering threshold")
    elif len(passed_samples) / total_samples > 0.9:
        recommendations.append("Filter may be too permissive - consider raising threshold")
    
    if abs(avg_confidence_passed - avg_confidence_failed) < 0.2:
        recommendations.append("Confidence scores may not be well-calibrated - review scoring method")
    
    print(f"\n   Optimization Recommendations:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("   ✅ Performance looks good! No immediate optimizations needed.")
    
    # Step 8.5: Adaptive threshold demonstration
    tutorial.print_step("Step 8.5: Adaptive Threshold Demonstration")
    
    # Simulate adaptive threshold adjustment
    adaptive_manager = ConfidenceFilterManager({
        "strategy": "adaptive_threshold",
        "threshold": 15.0,
        "adaptation_rate": 0.1
    })
    
    print(f"   Initial threshold: {adaptive_manager.filters['adaptive'].threshold:.3f}")
    
    # Simulate performance feedback
    performance_scenarios = [
        {"performance": 0.6, "description": "Poor performance - should lower threshold"},
        {"performance": 0.95, "description": "Excellent performance - should raise threshold"},
        {"performance": 0.8, "description": "Good performance - minor adjustment"}
    ]
    
    for scenario in performance_scenarios:
        print(f"\n   Scenario: {scenario['description']}")
        print(f"   Performance Score: {scenario['performance']:.2f}")
        
        # Update threshold based on performance
        for _ in range(10):  # Multiple updates to see effect
            adaptive_manager.update_performance_feedback(scenario["performance"])
        
        new_threshold = adaptive_manager.filters['adaptive'].threshold
        print(f"   New Threshold: {new_threshold:.3f}")
    
    # Step 8.6: Monitoring dashboard simulation
    tutorial.print_step("Step 8.6: Monitoring Dashboard Simulation")
    
    # Get comprehensive statistics
    stats = manager.get_statistics()
    
    print(f"\n   📊 Confidence Filtering Dashboard")
    print(f"   {'='*50}")
    print(f"   Strategy: {stats['strategy']}")
    print(f"   Total Filtered: {stats['total_filtered']}")
    print(f"   Pass Rate: {stats['pass_rate']:.1%}")
    print(f"   Fail Rate: {stats['fail_rate']:.1%}")
    print(f"   Average Confidence: {stats['average_confidence']:.3f}")
    
    # Recent activity
    if stats['filter_history']:
        recent_activity = stats['filter_history'][-3:]  # Last 3 results
        print(f"\n   Recent Activity:")
        for i, activity in enumerate(recent_activity, 1):
            status = "✅ PASS" if activity['passed'] else "❌ FAIL"
            print(f"   {i}. {status} | Confidence: {activity['confidence']:.3f}")
    
    return manager, results

# =============================================================================
# Main Tutorial Runner
# =============================================================================

async def run_confidence_filtering_tutorial():
    """Run the complete confidence filtering tutorial"""
    
    print("🎯 Welcome to the Confidence Filtering Tutorial!")
    print("=" * 60)
    print("This tutorial will guide you through all aspects of confidence filtering")
    print("with the DeepConf methodology for enhanced AI response quality.\n")
    
    # Run all sections
    try:
        print("Starting tutorial sections...")
        
        # Section 1: Basic Setup
        manager = section_1_basic_setup()
        
        # Section 2: Token-Level Confidence
        token_confidences, trace_confidence = section_2_token_confidence()
        
        # Section 3: Early Termination
        early_termination = section_3_early_termination()
        
        # Section 4: Confidence Voting
        voting_system, top_traces, best_answer = section_4_confidence_voting()
        
        # Section 5: Semantic Integration (if available)
        semantic_result = section_5_semantic_integration()
        
        # Section 6: RLHF Integration (if available)
        rlhf_result = section_6_rlhf_integration()
        
        # Section 7: Complete Workflow
        integration, workflow_results = await section_7_complete_workflow()
        
        # Section 8: Performance Monitoring
        monitor_manager, monitor_results = section_8_performance_monitoring()
        
        # Tutorial completion summary
        print("\n" + "=" * 60)
        print("🎉 Tutorial Completed Successfully!")
        print("=" * 60)
        print("\nWhat you've learned:")
        print("✅ Basic confidence filtering setup and configuration")
        print("✅ Token-level confidence scoring and analysis")
        print("✅ Early termination logic for compute efficiency")
        print("✅ Confidence-aware voting for multi-trace aggregation")
        if semantic_result:
            print("✅ Semantic graph integration for reasoning reliability")
        if rlhf_result:
            print("✅ RLHF integration for reward shaping and calibration")
        print("✅ Complete workflow examples and best practices")
        print("✅ Performance monitoring and optimization techniques")
        
        print("\nNext steps:")
        print("1. Experiment with different confidence thresholds")
        print("2. Integrate with your existing AI research workflows")
        print("3. Monitor performance metrics and adjust as needed")
        print("4. Explore advanced configuration options")
        
        print(f"\n📚 For more information, see:")
        print(f"   - CONFIDENCE_FILTERING_INTEGRATION.md")
        print(f"   - CONFIDENCE_INTEGRATION_SUMMARY.md")
        print(f"   - API_REFERENCE.md")
        
    except Exception as e:
        print(f"\n❌ Tutorial error: {e}")
        print("Please check your setup and try again.")
        raise

# =============================================================================
# Interactive Demo Functions
# =============================================================================

def interactive_confidence_demo():
    """Interactive demo for confidence filtering"""
    
    print("\n🎮 Interactive Confidence Filtering Demo")
    print("=" * 50)
    
    # Initialize basic filter
    manager = integrate_confidence_filtering({
        "strategy": "adaptive_threshold",
        "threshold": 15.0
    })
    
    while True:
        print("\nOptions:")
        print("1. Test response filtering")
        print("2. View statistics")
        print("3. Adjust threshold")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                # Test response filtering
                print("\nEnter logprobs (comma-separated, e.g., -0.5,-0.3,-0.8):")
                logprobs_input = input("Logprobs: ").strip()
                
                try:
                    logprobs = [float(x.strip()) for x in logprobs_input.split(",")]
                    result = manager.filter_response({"logprobs": logprobs})
                    
                    status = "✅ PASSED" if result.passed else "❌ FILTERED"
                    print(f"\nResult: {status}")
                    print(f"Confidence Score: {result.confidence_score:.3f}")
                    print(f"Reason: {result.reason}")
                    
                except ValueError:
                    print("❌ Invalid input. Please enter comma-separated numbers.")
            
            elif choice == "2":
                # View statistics
                stats = manager.get_statistics()
                print(f"\n📊 Statistics:")
                print(f"Total Filtered: {stats['total_filtered']}")
                print(f"Pass Rate: {stats['pass_rate']:.1%}")
                print(f"Average Confidence: {stats['average_confidence']:.3f}")
            
            elif choice == "3":
                # Adjust threshold
                current_threshold = manager.config.get("threshold", 15.0)
                print(f"\nCurrent threshold: {current_threshold}")
                
                try:
                    new_threshold = float(input("Enter new threshold: ").strip())
                    manager.config["threshold"] = new_threshold
                    print(f"✅ Threshold updated to: {new_threshold}")
                except ValueError:
                    print("❌ Invalid threshold. Please enter a number.")
            
            elif choice == "4":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Confidence Filtering Tutorial")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    parser.add_argument("--section", type=int, help="Run specific section (1-8)")
    parser.add_argument("--quick", action="store_true", help="Run quick demo")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_confidence_demo()
    elif args.section:
        if args.section == 1:
            section_1_basic_setup()
        elif args.section == 2:
            section_2_token_confidence()
        elif args.section == 3:
            section_3_early_termination()
        elif args.section == 4:
            section_4_confidence_voting()
        elif args.section == 5:
            section_5_semantic_integration()
        elif args.section == 6:
            section_6_rlhf_integration()
        elif args.section == 7:
            asyncio.run(section_7_complete_workflow())
        elif args.section == 8:
            section_8_performance_monitoring()
        else:
            print("❌ Invalid section. Please choose 1-8.")
    elif args.quick:
        # Quick demo
        print("🚀 Quick Confidence Filtering Demo")
        manager = section_1_basic_setup()
        section_2_token_confidence()
        print("\n✅ Quick demo completed!")
    else:
        # Run full tutorial
        asyncio.run(run_confidence_filtering_tutorial())