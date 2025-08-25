# -*- coding: utf-8 -*-
"""
Tests for Stage 7: Confidence Filtering
"""

import pytest
import numpy as np
from datetime import datetime
from extensions.stage_7_confidence_filtering import (
    ConfidenceFilter, AdaptiveConfidenceFilter, ConfidenceFilterManager,
    ConfidenceStrategy, compute_trace_confidence, filter_top_confident_traces,
    integrate_confidence_filtering
)

class TestConfidenceFilter:
    """Test basic confidence filter functionality"""
    
    def test_group_confidence_no_stop(self):
        """Test that high confidence doesn't trigger stop"""
        cf = ConfidenceFilter(group_size=5, threshold=0.5)
        tokens = [-1.2, -0.8, -1.0, -0.9, -1.1]
        for lp in tokens:
            cf.update(lp)
        assert cf.should_stop() is False
        assert cf.get_current_confidence() > 0.5

    def test_low_confidence_trigger(self):
        """Test that low confidence triggers stop"""
        cf = ConfidenceFilter(group_size=3, threshold=0.9, warmup_traces=0)
        for lp in [-2.0, -2.1, -2.2]:
            cf.update(lp)
        assert cf.should_stop() is True

    def test_warmup_period(self):
        """Test that warmup period prevents early stopping"""
        cf = ConfidenceFilter(group_size=3, threshold=0.9, warmup_traces=10)
        for lp in [-2.0, -2.1, -2.2]:
            cf.update(lp)
        assert cf.should_stop() is False  # Should not stop during warmup

    def test_reset_functionality(self):
        """Test filter reset"""
        cf = ConfidenceFilter(group_size=3, threshold=0.9, warmup_traces=0)
        for lp in [-2.0, -2.1, -2.2]:
            cf.update(lp)
        assert cf.should_stop() is True
        
        cf.reset()
        assert cf.should_stop() is False
        assert len(cf.buffer) == 0

class TestAdaptiveConfidenceFilter:
    """Test adaptive confidence filter"""
    
    def test_threshold_adaptation_poor_performance(self):
        """Test threshold lowering with poor performance"""
        acf = AdaptiveConfidenceFilter(initial_threshold=17.0, adaptation_rate=0.1)
        initial_threshold = acf.threshold
        
        # Simulate poor performance
        for _ in range(10):
            acf.update_threshold(0.5)  # Poor performance score
        
        assert acf.threshold < initial_threshold

    def test_threshold_adaptation_good_performance(self):
        """Test threshold raising with good performance"""
        acf = AdaptiveConfidenceFilter(initial_threshold=17.0, adaptation_rate=0.1)
        initial_threshold = acf.threshold
        
        # Simulate good performance
        for _ in range(10):
            acf.update_threshold(0.95)  # Good performance score
        
        assert acf.threshold > initial_threshold

    def test_filter_response_with_logprobs(self):
        """Test response filtering with logprobs"""
        acf = AdaptiveConfidenceFilter(initial_threshold=1.0)
        
        response_data = {
            "logprobs": [-0.5, -0.6, -0.4, -0.7, -0.3]
        }
        
        result = acf.filter_response(response_data)
        
        assert result.passed is True  # Should pass with good logprobs
        assert result.confidence_score > 0
        assert result.metrics.mean_logprob < 0
        assert isinstance(result.timestamp, datetime)

    def test_filter_response_no_logprobs(self):
        """Test response filtering without logprobs"""
        acf = AdaptiveConfidenceFilter()
        
        response_data = {}
        
        result = acf.filter_response(response_data)
        
        assert result.passed is False
        assert result.confidence_score == 0.0
        assert "No logprobs" in result.reason

class TestConfidenceFilterManager:
    """Test confidence filter manager"""
    
    def test_adaptive_strategy_initialization(self):
        """Test adaptive strategy initialization"""
        config = {
            "strategy": "adaptive_threshold",
            "threshold": 15.0,
            "adaptation_rate": 0.2
        }
        
        manager = ConfidenceFilterManager(config)
        
        assert manager.strategy == ConfidenceStrategy.ADAPTIVE_THRESHOLD
        assert "adaptive" in manager.filters
        assert manager.filters["adaptive"].threshold == 15.0

    def test_ensemble_strategy_initialization(self):
        """Test ensemble strategy initialization"""
        config = {
            "strategy": "ensemble_voting"
        }
        
        manager = ConfidenceFilterManager(config)
        
        assert manager.strategy == ConfidenceStrategy.ENSEMBLE_VOTING
        assert len(manager.filters) == 3  # conservative, moderate, liberal

    def test_basic_filtering(self):
        """Test basic filtering functionality"""
        config = {
            "strategy": "logprob_threshold",
            "threshold": 1.0
        }
        
        manager = ConfidenceFilterManager(config)
        
        response_data = {
            "logprobs": [-0.5, -0.6, -0.4]
        }
        
        result = manager.filter_response(response_data)
        
        assert isinstance(result.passed, bool)
        assert result.confidence_score > 0
        assert "Basic threshold" in result.reason

    def test_ensemble_filtering(self):
        """Test ensemble filtering"""
        config = {
            "strategy": "ensemble_voting"
        }
        
        manager = ConfidenceFilterManager(config)
        
        response_data = {
            "logprobs": [-0.5, -0.6, -0.4, -0.7, -0.3]
        }
        
        result = manager.filter_response(response_data)
        
        assert isinstance(result.passed, bool)
        assert "Ensemble voting" in result.reason

    def test_statistics_tracking(self):
        """Test statistics tracking"""
        config = {"strategy": "adaptive_threshold"}
        manager = ConfidenceFilterManager(config)
        
        # Process some responses
        for i in range(5):
            response_data = {
                "logprobs": [-0.5 - i * 0.1, -0.6 - i * 0.1, -0.4 - i * 0.1]
            }
            manager.filter_response(response_data)
        
        stats = manager.get_statistics()
        
        assert stats["total_filtered"] == 5
        assert stats["passed_count"] + stats["failed_count"] == 5
        assert 0 <= stats["pass_rate"] <= 1
        assert 0 <= stats["fail_rate"] <= 1
        assert stats["strategy"] == "adaptive_threshold"

    def test_performance_feedback(self):
        """Test performance feedback integration"""
        config = {"strategy": "adaptive_threshold"}
        manager = ConfidenceFilterManager(config)
        
        initial_threshold = manager.filters["adaptive"].threshold
        
        # Provide poor performance feedback
        manager.update_performance_feedback(0.3)
        
        # Threshold should remain the same until enough feedback is collected
        assert manager.filters["adaptive"].threshold == initial_threshold

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_compute_trace_confidence(self):
        """Test trace confidence computation"""
        trace_good = {"logprobs": [-0.5, -0.6, -0.4]}
        trace_bad = {"logprobs": [-2.0, -2.1, -2.2]}
        trace_empty = {"logprobs": []}
        trace_no_logprobs = {}
        
        conf_good = compute_trace_confidence(trace_good)
        conf_bad = compute_trace_confidence(trace_bad)
        conf_empty = compute_trace_confidence(trace_empty)
        conf_none = compute_trace_confidence(trace_no_logprobs)
        
        assert conf_good > conf_bad
        assert conf_empty == float("-inf")
        assert conf_none == float("-inf")

    def test_filter_top_confident_traces(self):
        """Test filtering top confident traces"""
        traces = [
            {"logprobs": [-0.5, -0.6, -0.4]},  # Good confidence
            {"logprobs": [-2.0, -2.1, -2.2]},  # Poor confidence
            {"logprobs": [-0.3, -0.4, -0.2]},  # Best confidence
            {"logprobs": [-1.5, -1.6, -1.4]}   # Medium confidence
        ]
        
        # Filter top 50%
        filtered = filter_top_confident_traces(traces, top_percent=50)
        
        assert len(filtered) == 2
        
        # Check that the best traces are selected
        confidences = [compute_trace_confidence(t) for t in filtered]
        assert all(c > 1.0 for c in confidences)  # Should be high confidence

    def test_filter_top_confident_traces_edge_cases(self):
        """Test edge cases for trace filtering"""
        # Empty traces
        assert filter_top_confident_traces([], top_percent=50) == []
        
        # Single trace
        single_trace = [{"logprobs": [-0.5, -0.6]}]
        filtered = filter_top_confident_traces(single_trace, top_percent=10)
        assert len(filtered) == 1
        
        # Top 0% should still return at least 1
        traces = [{"logprobs": [-0.5, -0.6]}, {"logprobs": [-1.0, -1.1]}]
        filtered = filter_top_confident_traces(traces, top_percent=0)
        assert len(filtered) == 1

class TestIntegration:
    """Test integration function"""
    
    def test_integrate_confidence_filtering_default(self):
        """Test integration with default config"""
        manager = integrate_confidence_filtering()
        
        assert isinstance(manager, ConfidenceFilterManager)
        assert manager.strategy == ConfidenceStrategy.ADAPTIVE_THRESHOLD

    def test_integrate_confidence_filtering_custom_config(self):
        """Test integration with custom config"""
        config = {
            "strategy": "ensemble_voting",
            "threshold": 20.0
        }
        
        manager = integrate_confidence_filtering(config)
        
        assert manager.strategy == ConfidenceStrategy.ENSEMBLE_VOTING
        assert manager.config["threshold"] == 20.0

    def test_end_to_end_filtering_workflow(self):
        """Test complete filtering workflow"""
        # Initialize manager
        manager = integrate_confidence_filtering({
            "strategy": "adaptive_threshold",
            "threshold": 1.0
        })
        
        # Process multiple responses
        responses = [
            {"logprobs": [-0.3, -0.4, -0.2]},  # High confidence
            {"logprobs": [-1.5, -1.6, -1.4]},  # Medium confidence
            {"logprobs": [-3.0, -3.1, -2.9]},  # Low confidence
        ]
        
        results = []
        for response in responses:
            result = manager.filter_response(response)
            results.append(result)
        
        # Check results
        assert len(results) == 3
        assert results[0].passed is True   # High confidence should pass
        assert results[2].passed is False  # Low confidence should fail
        
        # Check statistics
        stats = manager.get_statistics()
        assert stats["total_filtered"] == 3
        assert stats["passed_count"] >= 1
        assert stats["failed_count"] >= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
