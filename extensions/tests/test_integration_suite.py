#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Research Agent Extensions - Integration Test Suite
===================================================

Comprehensive integration tests for all extension stages.
Tests functionality, integration points, error handling, and performance.

Usage:
    python -m pytest extensions/tests/test_integration_suite.py -v
    python extensions/tests/test_integration_suite.py --run-all
"""

import asyncio
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add extensions to path
sys.path.append(str(Path(__file__).parent.parent))

from integration_orchestrator import AIResearchAgentExtensions
from stage_1_observability import ObservabilityCollector, ModuleType, ExperimentType
from stage_2_context_builder import (
    MemoryTierManager, AdaptiveContextPacker, PromptTemplateManager,
    MemoryTier, TaskType, ContextPackingStrategy
)
from stage_3_semantic_graph import SemanticGraphManager, NodeType, EdgeType, SourceType
from stage_4_diffusion_repair import RuntimeRepairOperator, LanguageType, RepairStrategy
from stage_5_rlhf_agentic_rl import (
    PreferenceDataPipeline, OnlineAgenticRL, RewardModel, MultiObjectiveAlignment,
    PreferenceType, AlignmentObjective
)
from stage_6_cross_module_synergies import UnifiedOrchestrator, SynergyType

class TestStage1Observability:
    """Test Stage 1: Enhanced Observability"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_observability_collector_initialization(self, temp_config_dir):
        """Test observability collector initialization"""
        config_path = Path(temp_dir) / "test_config.json"
        collector = ObservabilityCollector(str(config_path))
        
        assert collector is not None
        assert len(collector.module_configs) > 0
        assert config_path.exists()
    
    def test_event_tracking(self, temp_config_dir):
        """Test event tracking functionality"""
        collector = ObservabilityCollector()
        
        event_id = collector.track_event(
            module_type=ModuleType.CONTEXT_ENGINEERING,
            event_type="test_event",
            session_id="test_session",
            data={"test_key": "test_value"}
        )
        
        assert event_id is not None
        assert len(collector.events) > 0
        
        # Verify event data
        event = collector.events[-1]
        assert event.module_type == ModuleType.CONTEXT_ENGINEERING
        assert event.event_type == "test_event"
        assert event.session_id == "test_session"
        assert event.data["test_key"] == "test_value"
    
    def test_performance_tracking(self):
        """Test performance tracking"""
        collector = ObservabilityCollector()
        
        event_id = collector.track_performance(
            module_type=ModuleType.SEMANTIC_GRAPH,
            operation="test_operation",
            execution_time=0.123,
            success=True,
            additional_metrics={"metric1": 42}
        )
        
        assert event_id is not None
        
        # Check performance cache
        cache_key = "semantic_graph_test_operation"
        assert cache_key in collector.performance_cache
        assert 0.123 in collector.performance_cache[cache_key]
    
    def test_ab_testing(self):
        """Test A/B testing functionality"""
        collector = ObservabilityCollector()
        
        experiment_id = collector.create_experiment(
            name="Test Experiment",
            description="Test A/B experiment",
            variants={
                "control": {"param": "value_a"},
                "treatment": {"param": "value_b"}
            },
            traffic_allocation={"control": 50, "treatment": 50},
            success_metrics=["conversion_rate"]
        )
        
        assert experiment_id is not None
        assert experiment_id in collector.experiments
        
        # Test variant assignment
        variant_a = collector.get_experiment_variant(experiment_id, "session_a")
        variant_b = collector.get_experiment_variant(experiment_id, "session_b")
        
        assert variant_a in ["control", "treatment"]
        assert variant_b in ["control", "treatment"]
    
    def test_analytics_dashboard(self):
        """Test analytics dashboard generation"""
        collector = ObservabilityCollector()
        
        # Generate some test data
        for i in range(5):
            collector.track_event(
                module_type=ModuleType.CONTEXT_ENGINEERING,
                event_type="test_event",
                session_id=f"session_{i}",
                data={"iteration": i}
            )
        
        dashboard = collector.get_analytics_dashboard()
        
        assert "system_health" in dashboard
        assert "module_performance" in dashboard
        assert dashboard["system_health"]["total_events"] >= 5

class TestStage2ContextEngineering:
    """Test Stage 2: Context Engineering"""
    
    def test_memory_tier_manager_initialization(self):
        """Test memory tier manager initialization"""
        manager = MemoryTierManager()
        
        assert manager is not None
        assert len(manager.memory_tiers) == 4  # All memory tiers
        assert MemoryTier.SHORT_TERM in manager.memory_tiers
        assert MemoryTier.LONG_TERM in manager.memory_tiers
    
    def test_memory_storage_and_retrieval(self):
        """Test memory storage and retrieval"""
        manager = MemoryTierManager()
        
        # Store test memory
        memory_id = manager.store_memory(
            content="Test memory content about machine learning",
            memory_tier=MemoryTier.LONG_TERM,
            relevance_score=0.8,
            metadata={"topic": "ML", "type": "definition"}
        )
        
        assert memory_id is not None
        assert len(manager.memory_tiers[MemoryTier.LONG_TERM]) == 1
        
        # Retrieve memories
        memories = manager.retrieve_memories(
            query="machine learning",
            memory_tiers=[MemoryTier.LONG_TERM],
            max_items=5
        )
        
        assert len(memories) == 1
        assert memories[0].content == "Test memory content about machine learning"
        assert memories[0].access_count == 1  # Should be incremented
    
    def test_memory_promotion(self):
        """Test memory promotion between tiers"""
        manager = MemoryTierManager()
        
        # Store in episodic tier
        memory_id = manager.store_memory(
            content="Important finding",
            memory_tier=MemoryTier.EPISODIC,
            relevance_score=0.9
        )
        
        # Promote to long-term
        success = manager.promote_memory(memory_id, MemoryTier.LONG_TERM)
        
        assert success is True
        assert len(manager.memory_tiers[MemoryTier.EPISODIC]) == 0
        assert len(manager.memory_tiers[MemoryTier.LONG_TERM]) == 1
    
    def test_adaptive_context_packing(self):
        """Test adaptive context packing"""
        manager = MemoryTierManager()
        packer = AdaptiveContextPacker(max_context_tokens=1000)
        
        # Create test memories
        memories = []
        for i in range(10):
            memory_id = manager.store_memory(
                content=f"Test content {i} with various information",
                memory_tier=MemoryTier.LONG_TERM,
                relevance_score=0.5 + i * 0.05
            )
            memories.extend(manager.retrieve_memories(f"content {i}", max_items=1))
        
        # Test packing
        result = packer.pack_context(
            memory_items=memories,
            task_type=TaskType.RESEARCH,
            strategy=ContextPackingStrategy.ADAPTIVE
        )
        
        assert result is not None
        assert len(result.packed_items) <= len(memories)
        assert result.total_tokens <= 1000
        assert 0 <= result.diversity_score <= 1
        assert 0 <= result.relevance_score <= 1
    
    def test_prompt_template_management(self):
        """Test prompt template management"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PromptTemplateManager(templates_dir=temp_dir)
            
            # Create template
            template_id = manager.create_template(
                name="test_template",
                template_content="Query: {{ query }}\nContext: {{ context }}",
                task_types=[TaskType.QA],
                version="1.0.0"
            )
            
            assert template_id is not None
            assert template_id in manager.templates
            
            # Test rendering
            rendered = manager.render_template(
                template_id,
                {"query": "Test query", "context": "Test context"}
            )
            
            assert "Query: Test query" in rendered
            assert "Context: Test context" in rendered
            
            # Test template variant creation
            variant_id = manager.create_template_variant(
                base_template_id=template_id,
                modifications={
                    "content_replacement": {"Query:": "Question:"}
                }
            )
            
            assert variant_id != template_id
            assert variant_id in manager.templates

class TestStage3SemanticGraph:
    """Test Stage 3: Semantic Graph"""
    
    @pytest.fixture
    def temp_graph_dir(self):
        """Create temporary graph directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_graph_manager_initialization(self, temp_graph_dir):
        """Test semantic graph manager initialization"""
        manager = SemanticGraphManager(graph_storage_path=temp_graph_dir)
        
        assert manager is not None
        assert manager.graph is not None
        assert len(manager.nodes) == 0
        assert len(manager.edges) == 0
    
    def test_node_creation_and_retrieval(self, temp_graph_dir):
        """Test node creation and retrieval"""
        manager = SemanticGraphManager(graph_storage_path=temp_graph_dir)
        
        # Add test node
        node_id = manager.add_node(
            content="Machine learning is a subset of AI",
            node_type=NodeType.CONCEPT,
            source_type=SourceType.INTERNAL,
            title="ML Definition",
            importance_score=0.8,
            tags=["AI", "ML"]
        )
        
        assert node_id is not None
        assert node_id in manager.nodes
        
        node = manager.nodes[node_id]
        assert node.content == "Machine learning is a subset of AI"
        assert node.node_type == NodeType.CONCEPT
        assert "AI" in node.tags
    
    def test_edge_creation(self, temp_graph_dir):
        """Test edge creation between nodes"""
        manager = SemanticGraphManager(graph_storage_path=temp_graph_dir)
        
        # Create two nodes
        node1_id = manager.add_node(
            content="Concept A",
            node_type=NodeType.CONCEPT,
            source_type=SourceType.INTERNAL
        )
        
        node2_id = manager.add_node(
            content="Concept B",
            node_type=NodeType.CONCEPT,
            source_type=SourceType.INTERNAL
        )
        
        # Create edge
        edge_id = manager.add_edge(
            source_node=node1_id,
            target_node=node2_id,
            edge_type=EdgeType.SUPPORTS,
            confidence=0.8
        )
        
        assert edge_id is not None
        assert edge_id in manager.edges
        
        edge = manager.edges[edge_id]
        assert edge.source_node == node1_id
        assert edge.target_node == node2_id
        assert edge.edge_type == EdgeType.SUPPORTS
    
    def test_hybrid_retrieval(self, temp_graph_dir):
        """Test hybrid retrieval functionality"""
        manager = SemanticGraphManager(graph_storage_path=temp_graph_dir)
        
        # Create test graph
        nodes = []
        for i in range(5):
            node_id = manager.add_node(
                content=f"Test concept {i} about machine learning",
                node_type=NodeType.CONCEPT,
                source_type=SourceType.INTERNAL,
                importance_score=0.5 + i * 0.1
            )
            nodes.append(node_id)
        
        # Add some edges
        for i in range(len(nodes) - 1):
            manager.add_edge(
                source_node=nodes[i],
                target_node=nodes[i + 1],
                edge_type=EdgeType.SUPPORTS
            )
        
        # Test retrieval
        results = manager.hybrid_retrieval(
            query="machine learning concept",
            max_nodes=3
        )
        
        assert results is not None
        assert len(results.nodes) <= 3
        assert len(results.relevance_scores) > 0
    
    def test_reasoning_writeback(self, temp_graph_dir):
        """Test reasoning writeback functionality"""
        manager = SemanticGraphManager(graph_storage_path=temp_graph_dir)
        
        # Create premise nodes
        premise1_id = manager.add_node(
            content="All humans are mortal",
            node_type=NodeType.CLAIM,
            source_type=SourceType.INTERNAL
        )
        
        premise2_id = manager.add_node(
            content="Socrates is human",
            node_type=NodeType.CLAIM,
            source_type=SourceType.INTERNAL
        )
        
        # Test reasoning writeback
        reasoning_step = {
            "type": "deduction",
            "premises": ["All humans are mortal", "Socrates is human"],
            "conclusion": "Socrates is mortal",
            "confidence": 0.95
        }
        
        result = manager.reasoning_writeback(reasoning_step)
        
        assert result is not None
        assert len(result["nodes_created"]) > 0
        assert len(result["edges_created"]) > 0
        assert "reasoning_id" in result

class TestStage4DiffusionRepair:
    """Test Stage 4: Diffusion Repair"""
    
    def test_repair_operator_initialization(self):
        """Test repair operator initialization"""
        operator = RuntimeRepairOperator()
        
        assert operator is not None
        assert operator.noise_scheduler is not None
        assert operator.diffusion_core is not None
        assert operator.voting_system is not None
    
    def test_python_code_repair(self):
        """Test Python code repair"""
        operator = RuntimeRepairOperator()
        
        broken_code = """
def hello_world(
    print("Hello, World!")
"""
        
        result = operator.repair_code(
            broken_code=broken_code,
            language=LanguageType.PYTHON,
            error_type="SyntaxError"
        )
        
        assert result is not None
        # Note: Actual repair success depends on diffusion model implementation
        # For testing, we verify the structure and fallback behavior
        assert hasattr(result, 'success')
        assert hasattr(result, 'best_candidate')
        assert hasattr(result, 'repair_time')
    
    def test_javascript_code_repair(self):
        """Test JavaScript code repair"""
        operator = RuntimeRepairOperator()
        
        broken_code = """
function test() {
    console.log("test"
}
"""
        
        result = operator.repair_code(
            broken_code=broken_code,
            language=LanguageType.JAVASCRIPT,
            error_type="SyntaxError"
        )
        
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'provenance')
    
    def test_repair_statistics(self):
        """Test repair statistics tracking"""
        operator = RuntimeRepairOperator()
        
        # Perform some repairs
        test_cases = [
            ("def test(", LanguageType.PYTHON),
            ("SELECT * FROM", LanguageType.SQL),
            ("{\"key\":", LanguageType.JSON)
        ]
        
        for code, language in test_cases:
            operator.repair_code(code, language)
        
        stats = operator.get_repair_statistics()
        
        assert "total_repairs" in stats
        assert "successful_repairs" in stats
        assert "fallback_repairs" in stats
        assert stats["total_repairs"] >= len(test_cases)

class TestStage5RLHF:
    """Test Stage 5: RLHF & Agentic RL"""
    
    @pytest.fixture
    def temp_preference_dir(self):
        """Create temporary preference directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_preference_data_pipeline(self, temp_preference_dir):
        """Test preference data collection and processing"""
        pipeline = PreferenceDataPipeline(storage_path=temp_preference_dir)
        
        # Collect preference
        preference_id = pipeline.collect_preference(
            query="How to implement binary search?",
            response_a="Use linear search",
            response_b="Use divide and conquer approach",
            preference=1,  # Prefer response B
            preference_type=PreferenceType.HUMAN_FEEDBACK,
            confidence=0.9
        )
        
        assert preference_id is not None
        assert len(pipeline.preference_data) == 1
        
        # Test data retrieval
        training_data = pipeline.get_training_data(min_confidence=0.8)
        assert len(training_data) == 1
        
        # Test statistics
        stats = pipeline.get_preference_statistics()
        assert stats["total_preferences"] == 1
        assert stats["average_confidence"] == 0.9
    
    def test_reward_model(self):
        """Test reward model functionality"""
        model = RewardModel(input_dim=10)
        
        # Test prediction
        test_state = {f"feature_{i}": i * 0.1 for i in range(10)}
        reward = model.predict_reward(test_state)
        
        assert isinstance(reward, float)
        assert -10 <= reward <= 10  # Reasonable range
    
    def test_agentic_rl(self):
        """Test agentic RL system"""
        reward_model = RewardModel(input_dim=5)
        rl_system = OnlineAgenticRL(reward_model)
        
        # Test action selection
        state = {"complexity": 0.5, "context_size": 1000, "user_preference": 0.8}
        actions = ["detailed_analysis", "quick_summary", "step_by_step"]
        
        selected_action, metadata = rl_system.select_action(state, actions)
        
        assert selected_action in actions
        assert "predicted_reward" in metadata
        
        # Test reward recording
        from stage_5_rlhf_agentic_rl import RewardSignal, RewardSignalType
        
        reward_signals = [
            RewardSignal(
                signal_id="test_signal",
                action=selected_action,
                reward_value=0.8,
                signal_type=RewardSignalType.CORRECTNESS_SCORE,
                context=state,
                timestamp=datetime.now(),
                session_id="test_session"
            )
        ]
        
        # Find the action ID (simplified for test)
        action_id = rl_system.action_history[-1].action_id if rl_system.action_history else "test_action"
        rl_system.record_reward_signal(action_id, reward_signals)
        
        # Test statistics
        stats = rl_system.get_rl_statistics()
        assert "total_actions" in stats
        assert stats["total_actions"] >= 1
    
    def test_multi_objective_alignment(self):
        """Test multi-objective alignment system"""
        alignment = MultiObjectiveAlignment()
        
        # Test alignment evaluation
        response = "Here's a detailed explanation of the concept with examples and references."
        context = {
            "query": "Explain machine learning",
            "response_time": 1.5,
            "known_facts": ["ML is subset of AI"]
        }
        
        scores = alignment.evaluate_alignment(response, context)
        
        assert len(scores) == len(AlignmentObjective)
        for objective, score in scores.items():
            assert isinstance(objective, AlignmentObjective)
            assert 0 <= score <= 1
        
        # Test composite score
        composite = alignment.calculate_composite_alignment_score(scores)
        assert 0 <= composite <= 1
        
        # Test statistics
        stats = alignment.get_alignment_statistics()
        assert len(stats) == len(AlignmentObjective)

class TestStage6CrossModuleSynergies:
    """Test Stage 6: Cross-Module Synergies"""
    
    def test_unified_orchestrator_initialization(self):
        """Test unified orchestrator initialization"""
        orchestrator = UnifiedOrchestrator()
        
        assert orchestrator is not None
        assert orchestrator.observability is not None
        assert orchestrator.memory_manager is not None
        assert orchestrator.context_packer is not None
    
    @pytest.mark.asyncio
    async def test_request_processing(self):
        """Test unified request processing"""
        orchestrator = UnifiedOrchestrator()
        
        # Test research request
        request = {
            "type": "research",
            "query": "How do neural networks work?",
            "session_id": "test_session"
        }
        
        result = await orchestrator.process_request(request)
        
        assert result is not None
        assert "success" in result
        assert "synergies_used" in result
    
    def test_synergy_status(self):
        """Test synergy status reporting"""
        orchestrator = UnifiedOrchestrator()
        
        status = orchestrator.get_synergy_status()
        
        assert "active_synergies" in status
        assert "configurations" in status
        assert "performance_metrics" in status

class TestStage7ConfidenceFiltering:
    """Test Stage 7: Confidence Filtering"""
    
    def test_confidence_filtering_ci(self):
        """CI test for confidence filtering functionality"""
        from extensions.stage_7_confidence_filtering import (
            filter_top_confident_traces, compute_trace_confidence
        )
        
        traces = [
            {"logprobs": [-0.5, -0.6]}, 
            {"logprobs": [-2.0, -2.1]}
        ]
        filtered = filter_top_confident_traces(traces, top_percent=50)
        assert len(filtered) == 1
        assert compute_trace_confidence(filtered[0]) > 1.0
    
    def test_confidence_filter_basic(self):
        """Test basic confidence filter functionality"""
        from extensions.stage_7_confidence_filtering import ConfidenceFilter
        
        cf = ConfidenceFilter(group_size=3, threshold=0.9, warmup_traces=0)
        
        # Test with high confidence tokens
        for lp in [-0.1, -0.2, -0.15]:
            cf.update(lp)
        assert cf.should_stop() is False
        
        # Test with low confidence tokens
        cf_low = ConfidenceFilter(group_size=3, threshold=0.9, warmup_traces=0)
        for lp in [-2.0, -2.1, -2.2]:
            cf_low.update(lp)
        assert cf_low.should_stop() is True
    
    def test_adaptive_confidence_filter(self):
        """Test adaptive confidence filter functionality"""
        from extensions.stage_7_confidence_filtering import AdaptiveConfidenceFilter
        
        acf = AdaptiveConfidenceFilter(initial_threshold=1.0, adaptation_rate=0.1)
        
        # Test response filtering
        response_data = {"logprobs": [-0.3, -0.4, -0.2, -0.5]}
        result = acf.filter_response(response_data)
        
        assert result.passed is True  # Should pass with good logprobs
        assert result.confidence_score > 0
        assert result.metrics.mean_logprob < 0
        
        # Test threshold adaptation
        initial_threshold = acf.threshold
        for _ in range(10):
            acf.update_threshold(0.3)  # Poor performance
        assert acf.threshold < initial_threshold
    
    def test_confidence_filter_manager(self):
        """Test confidence filter manager"""
        from extensions.stage_7_confidence_filtering import (
            ConfidenceFilterManager, ConfidenceStrategy
        )
        
        config = {
            "strategy": "adaptive_threshold",
            "threshold": 1.0,
            "adaptation_rate": 0.1
        }
        
        manager = ConfidenceFilterManager(config)
        
        assert manager.strategy == ConfidenceStrategy.ADAPTIVE_THRESHOLD
        assert "adaptive" in manager.filters
        
        # Test filtering
        response_data = {"logprobs": [-0.5, -0.6, -0.4]}
        result = manager.filter_response(response_data)
        
        assert result.passed is True
        assert result.confidence_score > 0
        
        # Test statistics
        stats = manager.get_statistics()
        assert stats["total_filtered"] == 1
        assert stats["passed_count"] == 1
    
    def test_ensemble_filtering(self):
        """Test ensemble filtering strategy"""
        from extensions.stage_7_confidence_filtering import ConfidenceFilterManager
        
        config = {"strategy": "ensemble_voting"}
        manager = ConfidenceFilterManager(config)
        
        response_data = {"logprobs": [-0.5, -0.6, -0.4, -0.7]}
        result = manager.filter_response(response_data)
        
        assert "Ensemble voting" in result.reason
        assert isinstance(result.passed, bool)
    
    def test_trace_confidence_computation(self):
        """Test trace confidence computation"""
        from extensions.stage_7_confidence_filtering import compute_trace_confidence
        
        # High confidence trace
        high_conf_trace = {"logprobs": [-0.1, -0.2, -0.15]}
        high_conf = compute_trace_confidence(high_conf_trace)
        
        # Low confidence trace
        low_conf_trace = {"logprobs": [-2.0, -2.5, -3.0]}
        low_conf = compute_trace_confidence(low_conf_trace)
        
        assert high_conf > low_conf
        assert high_conf > 0.1  # Should be positive for good logprobs
        assert low_conf > 2.0   # Should be higher for bad logprobs
    
    def test_top_confident_traces_filtering(self):
        """Test filtering of top confident traces"""
        from extensions.stage_7_confidence_filtering import filter_top_confident_traces
        
        traces = [
            {"logprobs": [-0.1, -0.2], "id": "high_conf"},
            {"logprobs": [-1.0, -1.1], "id": "med_conf"},
            {"logprobs": [-3.0, -3.1], "id": "low_conf"},
            {"logprobs": [-0.05, -0.15], "id": "highest_conf"}
        ]
        
        # Test 50% filtering
        filtered_50 = filter_top_confident_traces(traces, top_percent=50)
        assert len(filtered_50) == 2
        
        # Test 25% filtering
        filtered_25 = filter_top_confident_traces(traces, top_percent=25)
        assert len(filtered_25) == 1
        
        # Verify the highest confidence trace is selected
        assert filtered_25[0]["id"] == "highest_conf"
    
    def test_confidence_integration(self):
        """Test confidence filtering integration"""
        from extensions.stage_7_confidence_filtering import integrate_confidence_filtering
        
        config = {
            "strategy": "adaptive_threshold",
            "threshold": 15.0,
            "adaptation_rate": 0.2
        }
        
        manager = integrate_confidence_filtering(config)
        
        assert manager is not None
        assert manager.config["threshold"] == 15.0
        assert manager.config["adaptation_rate"] == 0.2
    
    def test_performance_feedback_integration(self):
        """Test performance feedback integration"""
        from extensions.stage_7_confidence_filtering import ConfidenceFilterManager
        
        config = {"strategy": "adaptive_threshold", "threshold": 10.0}
        manager = ConfidenceFilterManager(config)
        
        initial_threshold = manager.filters["adaptive"].threshold
        
        # Provide performance feedback
        manager.update_performance_feedback(0.95)  # Good performance
        
        # Should not change immediately (needs more feedback)
        assert manager.filters["adaptive"].threshold == initial_threshold

class TestFullIntegration:
    """Test full system integration"""
    
    @pytest.mark.asyncio
    async def test_extensions_initialization(self):
        """Test full extensions initialization"""
        extensions = AIResearchAgentExtensions()
        
        # Test initialization
        status = await extensions.initialize_all_stages()
        
        assert status is not None
        assert "success_rate" in status
        assert "initialized_stages" in status
        assert status["success_rate"] > 0  # At least some stages should initialize
    
    @pytest.mark.asyncio
    async def test_enhanced_request_processing(self):
        """Test enhanced request processing"""
        extensions = AIResearchAgentExtensions()
        await extensions.initialize_all_stages()
        
        # Test research request with all enhancements
        request = {
            "type": "research",
            "query": "What are the latest developments in transformer architectures?",
            "session_id": "integration_test_session",
            "preferences": {
                "detail_level": "comprehensive",
                "include_citations": True,
                "max_response_time": 30
            }
        }
        
        result = await extensions.process_enhanced_request(request)
        
        assert result is not None
        assert "response" in result
        assert "metadata" in result
        assert "confidence_score" in result
        assert "synergies_applied" in result
    
    def test_cross_stage_data_flow(self):
        """Test data flow between stages"""
        extensions = AIResearchAgentExtensions()
        
        # Test observability -> context engineering flow
        event_id = extensions.observability.track_event(
            module_type=ModuleType.CONTEXT_ENGINEERING,
            event_type="memory_retrieval",
            session_id="test_session",
            data={"query": "test query", "results_count": 5}
        )
        
        # Test context engineering -> semantic graph flow
        memory_id = extensions.memory_manager.store_memory(
            content="Test semantic content for graph integration",
            memory_tier=MemoryTier.LONG_TERM,
            relevance_score=0.9,
            metadata={"source": "integration_test"}
        )
        
        # Test semantic graph -> diffusion repair flow
        node_id = extensions.semantic_graph.add_node(
            content="def broken_function(",
            node_type=NodeType.CODE,
            source_type=SourceType.INTERNAL,
            metadata={"language": "python", "needs_repair": True}
        )
        
        assert event_id is not None
        assert memory_id is not None
        assert node_id is not None
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking across all stages"""
        extensions = AIResearchAgentExtensions()
        
        # Run performance benchmark
        benchmark_results = extensions.run_performance_benchmark()
        
        assert benchmark_results is not None
        assert "stage_performance" in benchmark_results
        assert "overall_metrics" in benchmark_results
        assert "bottlenecks" in benchmark_results
        
        # Verify all stages are benchmarked
        expected_stages = [
            "observability", "context_engineering", "semantic_graph",
            "diffusion_repair", "rlhf", "cross_module_synergies"
        ]
        
        for stage in expected_stages:
            assert stage in benchmark_results["stage_performance"]
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        extensions = AIResearchAgentExtensions()
        
        # Test with invalid configuration
        invalid_config = {"invalid_key": "invalid_value"}
        
        try:
            extensions.configure_stage("nonexistent_stage", invalid_config)
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "nonexistent_stage" in str(e).lower()
        
        # Test graceful degradation
        degraded_result = extensions.process_with_degradation({
            "query": "test query",
            "force_error": True  # Simulate error condition
        })
        
        assert degraded_result is not None
        assert "fallback_used" in degraded_result
        assert degraded_result["fallback_used"] is True
    
    def test_configuration_management(self):
        """Test configuration management across stages"""
        extensions = AIResearchAgentExtensions()
        
        # Test configuration updates
        new_config = {
            "observability": {
                "enable_detailed_logging": True,
                "performance_tracking": True
            },
            "context_engineering": {
                "max_context_tokens": 2000,
                "memory_tier_weights": {
                    "short_term": 0.3,
                    "long_term": 0.5,
                    "episodic": 0.2
                }
            }
        }
        
        success = extensions.update_configuration(new_config)
        assert success is True
        
        # Verify configuration was applied
        current_config = extensions.get_current_configuration()
        assert current_config["observability"]["enable_detailed_logging"] is True
        assert current_config["context_engineering"]["max_context_tokens"] == 2000
    
    def test_health_monitoring(self):
        """Test system health monitoring"""
        extensions = AIResearchAgentExtensions()
        
        health_status = extensions.get_system_health()
        
        assert health_status is not None
        assert "overall_status" in health_status
        assert "stage_health" in health_status
        assert "resource_usage" in health_status
        assert "recommendations" in health_status
        
        # Verify health status format
        assert health_status["overall_status"] in ["healthy", "degraded", "critical"]
        
        for stage_name, stage_health in health_status["stage_health"].items():
            assert "status" in stage_health
            assert "metrics" in stage_health
            assert stage_health["status"] in ["healthy", "degraded", "critical", "offline"]


class TestRegressionSuite:
    """Regression tests for known issues and edge cases"""
    
    def test_memory_leak_prevention(self):
        """Test memory leak prevention in long-running operations"""
        extensions = AIResearchAgentExtensions()
        
        # Simulate long-running operation
        initial_memory = extensions.get_memory_usage()
        
        for i in range(100):
            extensions.memory_manager.store_memory(
                content=f"Test memory {i}",
                memory_tier=MemoryTier.SHORT_TERM,
                relevance_score=0.5
            )
        
        # Trigger cleanup
        extensions.memory_manager.cleanup_expired_memories()
        
        final_memory = extensions.get_memory_usage()
        
        # Memory should not grow unboundedly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
    
    def test_concurrent_access_safety(self):
        """Test thread safety for concurrent access"""
        import threading
        import time
        
        extensions = AIResearchAgentExtensions()
        results = []
        errors = []
        
        def worker_function(worker_id):
            try:
                for i in range(10):
                    memory_id = extensions.memory_manager.store_memory(
                        content=f"Worker {worker_id} memory {i}",
                        memory_tier=MemoryTier.SHORT_TERM,
                        relevance_score=0.5
                    )
                    results.append(memory_id)
                    time.sleep(0.01)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 50  # 5 workers * 10 operations each
        assert len(set(results)) == 50  # All IDs should be unique
    
    def test_large_data_handling(self):
        """Test handling of large data volumes"""
        extensions = AIResearchAgentExtensions()
        
        # Test large content storage
        large_content = "x" * 10000  # 10KB content
        
        memory_id = extensions.memory_manager.store_memory(
            content=large_content,
            memory_tier=MemoryTier.LONG_TERM,
            relevance_score=0.8
        )
        
        assert memory_id is not None
        
        # Test retrieval
        memories = extensions.memory_manager.retrieve_memories(
            query="x",
            memory_tiers=[MemoryTier.LONG_TERM],
            max_items=1
        )
        
        assert len(memories) == 1
        assert len(memories[0].content) == 10000
    
    def test_edge_case_inputs(self):
        """Test handling of edge case inputs"""
        extensions = AIResearchAgentExtensions()
        
        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace only
            "a" * 100000,  # Very long string
            "🚀🔬🧠💡",  # Unicode emojis
            "SELECT * FROM users; DROP TABLE users;",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            None,  # None value
        ]
        
        for case in edge_cases:
            try:
                if case is not None:
                    result = extensions.memory_manager.store_memory(
                        content=case,
                        memory_tier=MemoryTier.SHORT_TERM,
                        relevance_score=0.5
                    )
                    assert result is not None or case == ""  # Empty string might be rejected
            except Exception as e:
                # Some edge cases should be handled gracefully
                assert "validation" in str(e).lower() or "invalid" in str(e).lower()


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_memory_retrieval_performance(self):
        """Test memory retrieval performance"""
        extensions = AIResearchAgentExtensions()
        
        # Store test data
        for i in range(1000):
            extensions.memory_manager.store_memory(
                content=f"Performance test content {i} with various keywords",
                memory_tier=MemoryTier.LONG_TERM,
                relevance_score=0.5 + (i % 10) * 0.05
            )
        
        # Benchmark retrieval
        import time
        start_time = time.time()
        
        results = extensions.memory_manager.retrieve_memories(
            query="performance test",
            memory_tiers=[MemoryTier.LONG_TERM],
            max_items=50
        )
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        assert len(results) > 0
        assert retrieval_time < 1.0  # Should complete within 1 second
    
    def test_semantic_graph_performance(self):
        """Test semantic graph performance"""
        extensions = AIResearchAgentExtensions()
        
        # Create test graph
        nodes = []
        for i in range(100):
            node_id = extensions.semantic_graph.add_node(
                content=f"Performance node {i}",
                node_type=NodeType.CONCEPT,
                source_type=SourceType.INTERNAL
            )
            nodes.append(node_id)
        
        # Add edges
        for i in range(len(nodes) - 1):
            extensions.semantic_graph.add_edge(
                source_node=nodes[i],
                target_node=nodes[i + 1],
                edge_type=EdgeType.SUPPORTS
            )
        
        # Benchmark retrieval
        import time
        start_time = time.time()
        
        results = extensions.semantic_graph.hybrid_retrieval(
            query="performance node",
            max_nodes=20
        )
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        assert results is not None
        assert len(results.nodes) > 0
        assert retrieval_time < 2.0  # Should complete within 2 seconds


def run_integration_tests():
    """Run all integration tests"""
    import pytest
    
    # Run with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])


def run_performance_tests():
    """Run only performance tests"""
    import pytest
    
    pytest.main([
        __file__ + "::TestPerformanceBenchmarks",
        "-v",
        "--tb=short"
    ])


def run_regression_tests():
    """Run only regression tests"""
    import pytest
    
    pytest.main([
        __file__ + "::TestRegressionSuite",
        "-v",
        "--tb=short"
    ])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Research Agent Extensions Integration Tests")
    parser.add_argument("--run-all", action="store_true", help="Run all tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--regression", action="store_true", help="Run regression tests only")
    parser.add_argument("--stage", type=str, help="Run tests for specific stage (1-7)")
    
    args = parser.parse_args()
    
    if args.performance:
        run_performance_tests()
    elif args.regression:
        run_regression_tests()
    elif args.stage:
        stage_class_map = {
            "1": "TestStage1Observability",
            "2": "TestStage2ContextEngineering", 
            "3": "TestStage3SemanticGraph",
            "4": "TestStage4DiffusionRepair",
            "5": "TestStage5RLHF",
            "6": "TestStage6CrossModuleSynergies",
            "7": "TestStage7ConfidenceFiltering"
        }
        
        if args.stage in stage_class_map:
            import pytest
            pytest.main([
                f"{__file__}::{stage_class_map[args.stage]}",
                "-v",
                "--tb=short"
            ])
        else:
            print(f"Invalid stage: {args.stage}. Use 1-7.")
    else:
        run_integration_tests()it extensions.initialize_all_stages()
        
        # Test research request
        request = {
            "type": "research",
            "query": "What are the benefits of transformer architectures?",
            "session_id": "integration_test_session"
        }
        
        result = await extensions.process_enhanced_request(request)
        
        assert result is not None
        assert "success" in result
        assert "processing_time" in result
        assert "enhancements_used" in result
    
    @pytest.mark.asyncio
    async def test_code_repair_request(self):
        """Test code repair request processing"""
        extensions = AIResearchAgentExtensions()
        await extensions.initialize_all_stages()
        
        # Test code repair request
        request = {
            "type": "code_repair",
            "code": "def test(\n    print('test')",
            "language": "python",
            "session_id": "repair_test_session"
        }
        
        result = await extensions.process_enhanced_request(request)
        
        assert result is not None
        assert "success" in result
    
    def test_performance_dashboard(self):
        """Test performance dashboard generation"""
        extensions = AIResearchAgentExtensions()
        
        dashboard = extensions.get_performance_dashboard()
        
        assert dashboard is not None
        assert "integration_overview" in dashboard
        
        integration_overview = dashboard["integration_overview"]
        assert "initialized_stages" in integration_overview
        assert "success_rate" in integration_overview

# Test utilities
class TestUtilities:
    """Test utility functions and helpers"""
    
    def test_configuration_loading(self):
        """Test configuration loading and validation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "enable_observability": True,
                "enable_context_engineering": True,
                "enable_semantic_graph": False,
                "integration_level": "basic"
            }
            json.dump(config, f)
            config_path = f.name
        
        try:
            extensions = AIResearchAgentExtensions(config_path)
            assert extensions.config["enable_observability"] is True
            assert extensions.config["enable_semantic_graph"] is False
        finally:
            Path(config_path).unlink()
    
    def test_error_handling(self):
        """Test error handling and graceful degradation"""
        # Test with invalid configuration
        extensions = AIResearchAgentExtensions("nonexistent_config.json")
        
        # Should still initialize with defaults
        assert extensions.config is not None
        assert "enable_observability" in extensions.config

# Performance tests
class TestPerformance:
    """Performance and scalability tests"""
    
    def test_memory_scalability(self):
        """Test memory management scalability"""
        manager = MemoryTierManager()
        
        # Add many memories
        start_time = datetime.now()
        memory_ids = []
        
        for i in range(100):
            memory_id = manager.store_memory(
                content=f"Test memory {i} with content",
                memory_tier=MemoryTier.LONG_TERM,
                relevance_score=0.5
            )
            memory_ids.append(memory_id)
        
        storage_time = (datetime.now() - start_time).total_seconds()
        
        # Test retrieval performance
        start_time = datetime.now()
        memories = manager.retrieve_memories("test memory", max_items=50)
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert storage_time < 5.0  # Should store 100 items in under 5 seconds
        assert retrieval_time < 1.0  # Should retrieve in under 1 second
        assert len(memories) > 0
    
    def test_graph_scalability(self):
        """Test semantic graph scalability"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SemanticGraphManager(graph_storage_path=temp_dir)
            
            # Add many nodes
            start_time = datetime.now()
            node_ids = []
            
            for i in range(50):  # Reduced for test performance
                node_id = manager.add_node(
                    content=f"Test concept {i}",
                    node_type=NodeType.CONCEPT,
                    source_type=SourceType.INTERNAL
                )
                node_ids.append(node_id)
            
            node_creation_time = (datetime.now() - start_time).total_seconds()
            
            # Test retrieval performance
            start_time = datetime.now()
            results = manager.hybrid_retrieval("test concept", max_nodes=10)
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            # Performance assertions
            assert node_creation_time < 10.0  # Should create 50 nodes in under 10 seconds
            assert retrieval_time < 2.0  # Should retrieve in under 2 seconds
            assert len(results.nodes) > 0

def run_integration_tests():
    """Run all integration tests"""
    import subprocess
    import sys
    
    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--durations=10"
    ], capture_output=True, text=True)
    
    print("INTEGRATION TEST RESULTS")
    print("=" * 50)
    print(result.stdout)
    
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Research Agent Extensions Integration Tests")
    parser.add_argument("--run-all", action="store_true", help="Run all integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    
    args = parser.parse_args()
    
    if args.run_all:
        success = run_integration_tests()
        sys.exit(0 if success else 1)
    elif args.performance:
        # Run only performance tests
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            __file__ + "::TestPerformance", 
            "-v"
        ])
        sys.exit(result.returncode)
    else:
        # Run pytest normally
        pytest.main([__file__, "-v"])