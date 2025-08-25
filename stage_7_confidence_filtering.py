# -*- coding: utf-8 -*-
"""
Stage 7: DeepConf-Enhanced Confidence Filtering
Advanced confidence-based filtering for AI Research Agent responses
Based on DeepConf methodology with token-level confidence, group confidence,
early termination, confidence-aware voting, and semantic graph alignment.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import asyncio
from collections import deque, defaultdict
import statistics

class ConfidenceStrategy(Enum):
    """Different confidence filtering strategies based on DeepConf methodology"""
    TOKEN_LEVEL = "token_level"  # Fine-grained trace pruning
    GROUP_CONFIDENCE = "group_confidence"  # Sliding window confidence
    EARLY_TERMINATION = "early_termination"  # Compute-efficient inference
    CONFIDENCE_VOTING = "confidence_voting"  # Accuracy boost via trace prioritization
    TAIL_CONFIDENCE = "tail_confidence"  # Robust filtering of weak traces
    SEMANTIC_ALIGNMENT = "semantic_alignment"  # Graph-based reasoning reliability
    ADAPTIVE_THRESHOLD = "adaptive_threshold"  # Adaptive threshold adjustment
    LOGPROB_THRESHOLD = "logprob_threshold"  # Basic logprob threshold
    ENSEMBLE_VOTING = "ensemble_voting"  # Ensemble of multiple filters

class ConfidenceLevel(Enum):
    """Confidence levels for different filtering strategies"""
    HIGH = "high"  # Top 10% confident traces
    MEDIUM = "medium"  # Middle 80% traces
    LOW = "low"  # Bottom 10% traces (hallucination-prone)

class ReasoningPhase(Enum):
    """Different phases of agent reasoning for confidence integration"""
    GENERATION = "generation"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    EVALUATION = "evaluation"
    FINAL_ANSWER = "final_answer"

@dataclass
class TokenConfidence:
    """Token-level confidence information"""
    token: str
    logprob: float
    confidence: float
    position: int
    is_reliable: bool = True

@dataclass
class GroupConfidence:
    """Group confidence using sliding window"""
    window_size: int
    confidence_scores: List[float] = field(default_factory=list)
    mean_confidence: float = 0.0
    variance: float = 0.0
    percentile_90: float = 0.0
    is_above_threshold: bool = False

@dataclass
class TraceConfidence:
    """Complete trace confidence assessment"""
    trace_id: str
    tokens: List[TokenConfidence]
    group_confidence: GroupConfidence
    overall_confidence: float
    reasoning_phase: ReasoningPhase
    semantic_reliability: float = 0.0
    vote_weight: float = 1.0
    is_filtered: bool = False
    filter_reason: str = ""

@dataclass
class ConfidenceMetrics:
    """Enhanced metrics for confidence assessment"""
    mean_logprob: float
    variance: float
    entropy: float
    perplexity: float
    confidence_score: float
    uncertainty_score: float
    # DeepConf specific metrics
    token_level_confidence: float = 0.0
    group_confidence_score: float = 0.0
    tail_confidence: float = 0.0
    semantic_alignment_score: float = 0.0
    early_termination_triggered: bool = False
    tokens_saved: int = 0

@dataclass
class FilterResult:
    """Result of confidence filtering"""
    passed: bool
    confidence_score: float
    reason: str
    metrics: ConfidenceMetrics
    timestamp: datetime

class ConfidenceScoringHook:
    """Token-level logprob-based confidence scoring hook for generation module"""
    
    def __init__(self, enable_real_time=True):
        self.enable_real_time = enable_real_time
        self.token_confidences = []
        self.generation_callbacks = []
    
    def register_generation_callback(self, callback: Callable):
        """Register callback for generation events"""
        self.generation_callbacks.append(callback)
    
    def compute_token_confidence(self, token: str, logprob: float, position: int) -> TokenConfidence:
        """Compute token-level confidence from logprob"""
        # Convert logprob to confidence (higher logprob = higher confidence)
        confidence = np.exp(logprob)  # Convert to probability
        is_reliable = logprob > -2.0  # Threshold for reliability
        
        token_conf = TokenConfidence(
            token=token,
            logprob=logprob,
            confidence=confidence,
            position=position,
            is_reliable=is_reliable
        )
        
        if self.enable_real_time:
            self.token_confidences.append(token_conf)
            # Trigger callbacks for real-time processing
            for callback in self.generation_callbacks:
                callback(token_conf)
        
        return token_conf
    
    def get_trace_confidence(self, tokens: List[TokenConfidence]) -> float:
        """Compute overall trace confidence from token confidences"""
        if not tokens:
            return 0.0
        
        # Weighted average with position decay
        weights = [1.0 / (1.0 + 0.1 * i) for i in range(len(tokens))]
        weighted_conf = sum(t.confidence * w for t, w in zip(tokens, weights))
        total_weight = sum(weights)
        
        return weighted_conf / total_weight if total_weight > 0 else 0.0

class EarlyTerminationLogic:
    """Early termination logic with confidence gates in agent's reasoning loop"""
    
    def __init__(self, threshold_percentile=90, warmup_traces=16, window_size=10):
        self.threshold_percentile = threshold_percentile
        self.warmup_traces = warmup_traces
        self.window_size = window_size
        self.confidence_history = deque(maxlen=1000)  # Store for offline warmup
        self.threshold = None
        self.is_warmed_up = False
        self.termination_triggered = False
        self.tokens_saved = 0
    
    def offline_warmup(self, initial_traces: List[TraceConfidence]):
        """Set threshold via offline warmup using 90th percentile of initial traces"""
        if not initial_traces:
            self.threshold = 0.5  # Default threshold
            return
        
        confidences = [trace.overall_confidence for trace in initial_traces]
        self.threshold = np.percentile(confidences, self.threshold_percentile)
        self.is_warmed_up = True
        
        print(f"🎯 Early termination threshold set to {self.threshold:.3f} "
              f"({self.threshold_percentile}th percentile of {len(initial_traces)} traces)")
    
    def should_terminate_early(self, current_confidence: float, tokens_generated: int) -> Tuple[bool, str]:
        """Check if generation should terminate early based on confidence gate"""
        if not self.is_warmed_up or tokens_generated < 5:  # Minimum tokens before termination
            return False, "Not warmed up or insufficient tokens"
        
        if current_confidence < self.threshold:
            self.termination_triggered = True
            reason = f"Confidence {current_confidence:.3f} below threshold {self.threshold:.3f}"
            return True, reason
        
        return False, "Confidence above threshold"
    
    def update_confidence_history(self, confidence: float):
        """Update confidence history for adaptive threshold adjustment"""
        self.confidence_history.append(confidence)
        
        # Adaptive threshold adjustment every 100 samples
        if len(self.confidence_history) % 100 == 0 and len(self.confidence_history) >= 100:
            recent_confidences = list(self.confidence_history)[-100:]
            new_threshold = np.percentile(recent_confidences, self.threshold_percentile)
            
            # Smooth threshold updates
            if self.threshold is not None:
                self.threshold = 0.9 * self.threshold + 0.1 * new_threshold
            else:
                self.threshold = new_threshold

class ConfidenceAwareVoting:
    """Confidence-aware voting system for aggregating multiple reasoning paths"""
    
    def __init__(self, top_n_percent=50, min_confidence=0.1):
        self.top_n_percent = top_n_percent
        self.min_confidence = min_confidence
        self.voting_history = []
    
    def filter_top_confident_traces(self, traces: List[TraceConfidence]) -> List[TraceConfidence]:
        """Filter top-n% confident traces before voting"""
        if not traces:
            return []
        
        # Sort by confidence
        sorted_traces = sorted(traces, key=lambda t: t.overall_confidence, reverse=True)
        
        # Take top n%
        cutoff = max(1, int(len(sorted_traces) * self.top_n_percent / 100))
        top_traces = sorted_traces[:cutoff]
        
        # Mark filtered traces
        for trace in traces:
            if trace not in top_traces:
                trace.is_filtered = True
                trace.filter_reason = f"Below top {self.top_n_percent}% confidence threshold"
        
        return top_traces
    
    def compute_vote_weights(self, traces: List[TraceConfidence]) -> Dict[str, float]:
        """Compute vote weights based on confidence scores"""
        weights = {}
        
        for trace in traces:
            # Weight by confidence with minimum threshold
            weight = max(trace.overall_confidence, self.min_confidence)
            
            # Boost weight for high semantic reliability
            if trace.semantic_reliability > 0.8:
                weight *= 1.2
            
            # Penalize low tail confidence (potential hallucination)
            if hasattr(trace, 'tail_confidence') and trace.tail_confidence < 0.3:
                weight *= 0.8
            
            weights[trace.trace_id] = weight
            trace.vote_weight = weight
        
        return weights
    
    def aggregate_votes(self, traces: List[TraceConfidence], answers: List[str]) -> Tuple[str, float, Dict]:
        """Aggregate votes with confidence weighting"""
        if not traces or not answers or len(traces) != len(answers):
            return "", 0.0, {}
        
        # Filter top confident traces
        filtered_traces = self.filter_top_confident_traces(traces)
        filtered_answers = [answers[i] for i, trace in enumerate(traces) if trace in filtered_traces]
        
        # Compute vote weights
        weights = self.compute_vote_weights(filtered_traces)
        
        # Aggregate by answer similarity (simplified - could use semantic similarity)
        answer_weights = defaultdict(float)
        answer_traces = defaultdict(list)
        
        for trace, answer in zip(filtered_traces, filtered_answers):
            answer_weights[answer] += weights[trace.trace_id]
            answer_traces[answer].append(trace)
        
        # Select best answer
        if not answer_weights:
            return "", 0.0, {}
        
        best_answer = max(answer_weights.keys(), key=lambda a: answer_weights[a])
        best_confidence = answer_weights[best_answer] / sum(answer_weights.values())
        
        voting_metadata = {
            "total_traces": len(traces),
            "filtered_traces": len(filtered_traces),
            "answer_weights": dict(answer_weights),
            "winning_confidence": best_confidence,
            "filter_rate": (len(traces) - len(filtered_traces)) / len(traces)
        }
        
        self.voting_history.append(voting_metadata)
        
        return best_answer, best_confidence, voting_metadata

class SemanticGraphAlignment:
    """Use group confidence to annotate semantic graph nodes with reasoning reliability"""
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.node_reliability_cache = {}
        self.path_confidence_cache = {}
    
    def annotate_node_reliability(self, node_id: str, trace_confidence: TraceConfidence) -> float:
        """Annotate semantic graph node with reasoning reliability"""
        reliability_score = self._compute_reliability_score(trace_confidence)
        
        # Cache for future path selection
        self.node_reliability_cache[node_id] = {
            "reliability": reliability_score,
            "confidence": trace_confidence.overall_confidence,
            "timestamp": datetime.now(),
            "reasoning_phase": trace_confidence.reasoning_phase
        }
        
        return reliability_score
    
    def _compute_reliability_score(self, trace_confidence: TraceConfidence) -> float:
        """Compute reliability score for semantic graph annotation"""
        base_score = trace_confidence.overall_confidence
        
        # Boost for consistent group confidence
        if trace_confidence.group_confidence.variance < 0.1:
            base_score *= 1.1
        
        # Boost for high semantic alignment
        if trace_confidence.semantic_reliability > 0.8:
            base_score *= 1.2
        
        # Penalize for low tail confidence
        if hasattr(trace_confidence, 'tail_confidence'):
            if trace_confidence.tail_confidence < 0.3:
                base_score *= 0.8
        
        return min(base_score, 1.0)  # Cap at 1.0
    
    def guide_path_selection(self, candidate_paths: List[List[str]]) -> List[Tuple[List[str], float]]:
        """Guide path selection during planning based on node reliability"""
        scored_paths = []
        
        for path in candidate_paths:
            path_confidence = self._compute_path_confidence(path)
            scored_paths.append((path, path_confidence))
        
        # Sort by confidence (highest first)
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        return scored_paths
    
    def _compute_path_confidence(self, path: List[str]) -> float:
        """Compute confidence score for a reasoning path"""
        if not path:
            return 0.0
        
        # Get cached reliabilities
        reliabilities = []
        for node_id in path:
            if node_id in self.node_reliability_cache:
                reliabilities.append(self.node_reliability_cache[node_id]["reliability"])
            else:
                reliabilities.append(0.5)  # Default reliability
        
        # Path confidence is geometric mean (all nodes must be reliable)
        if reliabilities:
            path_confidence = np.prod(reliabilities) ** (1.0 / len(reliabilities))
        else:
            path_confidence = 0.0
        
        # Cache result
        path_key = "->".join(path)
        self.path_confidence_cache[path_key] = path_confidence
        
        return path_confidence
    
    def prioritize_retrieval_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize retrieval from external sources based on reliability"""
        scored_sources = []
        
        for source in sources:
            source_id = source.get("id", "unknown")
            
            # Get reliability from cache or compute default
            if source_id in self.node_reliability_cache:
                reliability = self.node_reliability_cache[source_id]["reliability"]
            else:
                # Compute based on source metadata
                reliability = self._compute_source_reliability(source)
            
            scored_sources.append((source, reliability))
        
        # Sort by reliability (highest first)
        scored_sources.sort(key=lambda x: x[1], reverse=True)
        
        return [source for source, _ in scored_sources]
    
    def _compute_source_reliability(self, source: Dict[str, Any]) -> float:
        """Compute reliability score for external source"""
        base_reliability = 0.5
        
        # Boost for high-quality sources
        if source.get("source_type") == "academic":
            base_reliability += 0.2
        elif source.get("source_type") == "official_docs":
            base_reliability += 0.15
        
        # Boost for recent sources
        if source.get("recency_score", 0) > 0.8:
            base_reliability += 0.1
        
        # Boost for high citation count
        if source.get("citation_count", 0) > 100:
            base_reliability += 0.1
        
        return min(base_reliability, 1.0)

class ValidationStrategy:
    """Validation strategy to ensure robustness with offline evaluations"""
    
    def __init__(self, validation_threshold=0.8, min_samples=100):
        self.validation_threshold = validation_threshold
        self.min_samples = min_samples
        self.validation_results = []
        self.offline_metrics = {}
    
    def validate_filtering_performance(self, ground_truth: List[bool], predictions: List[bool]) -> Dict[str, float]:
        """Validate filtering performance against ground truth"""
        if len(ground_truth) != len(predictions):
            raise ValueError("Ground truth and predictions must have same length")
        
        # Compute validation metrics
        tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt and pred)
        tn = sum(1 for gt, pred in zip(ground_truth, predictions) if not gt and not pred)
        fp = sum(1 for gt, pred in zip(ground_truth, predictions) if not gt and pred)
        fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt and not pred)
        
        accuracy = (tp + tn) / len(ground_truth) if ground_truth else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        }
        
        self.validation_results.append(metrics)
        return metrics
    
    def run_offline_evaluation(self, test_traces: List[TraceConfidence], expected_outcomes: List[bool]) -> Dict[str, Any]:
        """Run comprehensive offline evaluation"""
        if len(test_traces) < self.min_samples:
            return {"error": f"Insufficient samples: {len(test_traces)} < {self.min_samples}"}
        
        # Test different confidence strategies
        strategies = [
            ConfidenceStrategy.TOKEN_LEVEL,
            ConfidenceStrategy.GROUP_CONFIDENCE,
            ConfidenceStrategy.EARLY_TERMINATION,
            ConfidenceStrategy.CONFIDENCE_VOTING
        ]
        
        strategy_results = {}
        
        for strategy in strategies:
            # Simulate filtering with this strategy
            predictions = self._simulate_strategy_filtering(test_traces, strategy)
            metrics = self.validate_filtering_performance(expected_outcomes, predictions)
            strategy_results[strategy.value] = metrics
        
        # Find best performing strategy
        best_strategy = max(strategy_results.keys(), 
                          key=lambda s: strategy_results[s]["f1_score"])
        
        self.offline_metrics = {
            "best_strategy": best_strategy,
            "strategy_results": strategy_results,
            "total_samples": len(test_traces),
            "validation_threshold": self.validation_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.offline_metrics
    
    def _simulate_strategy_filtering(self, traces: List[TraceConfidence], strategy: ConfidenceStrategy) -> List[bool]:
        """Simulate filtering with given strategy"""
        predictions = []
        
        for trace in traces:
            if strategy == ConfidenceStrategy.TOKEN_LEVEL:
                # Token-level filtering
                reliable_tokens = sum(1 for token in trace.tokens if token.is_reliable)
                prediction = reliable_tokens / len(trace.tokens) > 0.7 if trace.tokens else False
            
            elif strategy == ConfidenceStrategy.GROUP_CONFIDENCE:
                # Group confidence filtering
                prediction = trace.group_confidence.is_above_threshold
            
            elif strategy == ConfidenceStrategy.EARLY_TERMINATION:
                # Early termination would have kept this trace
                prediction = trace.overall_confidence > 0.6
            
            elif strategy == ConfidenceStrategy.CONFIDENCE_VOTING:
                # Voting-based filtering
                prediction = trace.vote_weight > 0.5
            
            else:
                # Default to overall confidence
                prediction = trace.overall_confidence > 0.5
            
            predictions.append(prediction)
        
        return predictions
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not self.validation_results:
            return {"error": "No validation results available"}
        
        # Aggregate metrics across all validation runs
        avg_metrics = {}
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            values = [result[metric] for result in self.validation_results]
            avg_metrics[f"avg_{metric}"] = np.mean(values)
            avg_metrics[f"std_{metric}"] = np.std(values)
        
        return {
            "validation_runs": len(self.validation_results),
            "average_metrics": avg_metrics,
            "latest_metrics": self.validation_results[-1] if self.validation_results else {},
            "offline_evaluation": self.offline_metrics
        }

class ConfidenceFilter:
    """Basic confidence filter using logprob thresholds"""
    
    def __init__(self, threshold=17.0, group_size=2048, warmup_traces=16):
        self.threshold = threshold
        self.group_size = group_size
        self.warmup_traces = warmup_traces
        self.current_logprobs = []
        self.current_confidence = 0.0
        self.processed_tokens = 0
    
    def update(self, logprob: float):
        """Update filter with new logprob"""
        self.current_logprobs.append(logprob)
        self.processed_tokens += 1
        
        # Update running confidence
        if self.current_logprobs:
            self.current_confidence = -np.mean(self.current_logprobs)
    
    def should_stop(self) -> bool:
        """Check if generation should stop based on confidence"""
        if self.processed_tokens < self.warmup_traces:
            return False
        
        return self.current_confidence < self.threshold
    
    def get_current_confidence(self) -> float:
        """Get current confidence score"""
        return self.current_confidence
    
    def reset(self):
        """Reset filter state"""
        self.current_logprobs = []
        self.current_confidence = 0.0
        self.processed_tokens = 0

class AdaptiveConfidenceFilter:
    """Adaptive confidence filter that adjusts thresholds based on performance"""
    
    def __init__(self, initial_threshold=17.0, adaptation_rate=0.1):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.base_filter = ConfidenceFilter(threshold=initial_threshold)
    
    def update_threshold(self, performance_score: float):
        """Update threshold based on performance feedback"""
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-10:])
            
            # Adjust threshold based on performance
            if recent_performance < 0.7:  # Poor performance, lower threshold
                self.threshold *= (1 - self.adaptation_rate)
            elif recent_performance > 0.9:  # Good performance, raise threshold
                self.threshold *= (1 + self.adaptation_rate)
            
            self.base_filter.threshold = self.threshold

    def filter_response(self, response_data: Dict[str, Any]) -> FilterResult:
        """Filter response based on adaptive confidence"""
        logprobs = response_data.get("logprobs", [])
        
        if not logprobs:
            return FilterResult(
                passed=False,
                confidence_score=0.0,
                reason="No logprobs available",
                metrics=ConfidenceMetrics(0, 0, 0, 0, 0, 1.0),
                timestamp=datetime.now()
            )
        
        metrics = self._compute_metrics(logprobs)
        passed = metrics.confidence_score >= self.threshold
        
        return FilterResult(
            passed=passed,
            confidence_score=metrics.confidence_score,
            reason=f"Confidence {'above' if passed else 'below'} threshold {self.threshold:.2f}",
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def _compute_metrics(self, logprobs: List[float]) -> ConfidenceMetrics:
        """Compute comprehensive confidence metrics"""
        logprobs_array = np.array(logprobs)
        
        mean_logprob = np.mean(logprobs_array)
        variance = np.var(logprobs_array)
        
        # Convert to probabilities for entropy calculation
        probs = np.exp(logprobs_array)
        probs = probs / np.sum(probs)  # Normalize
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        perplexity = np.exp(-mean_logprob)
        confidence_score = -mean_logprob  # Higher is better
        uncertainty_score = entropy / np.log(len(logprobs))  # Normalized entropy
        
        return ConfidenceMetrics(
            mean_logprob=mean_logprob,
            variance=variance,
            entropy=entropy,
            perplexity=perplexity,
            confidence_score=confidence_score,
            uncertainty_score=uncertainty_score
        )
    
    def __init__(self, initial_threshold=17.0, adaptation_rate=0.1):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.base_filter = ConfidenceFilter(threshold=initial_threshold)
    
    def update_threshold(self, performance_score: float):
        """Update threshold based on performance feedback"""
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-10:])
            
            # Adjust threshold based on performance
            if recent_performance < 0.7:  # Poor performance, lower threshold
                self.threshold *= (1 - self.adaptation_rate)
            elif recent_performance > 0.9:  # Good performance, raise threshold
                self.threshold *= (1 + self.adaptation_rate)
            
            self.base_filter.threshold = self.threshold

    def filter_response(self, response_data: Dict[str, Any]) -> FilterResult:
        """Filter response based on adaptive confidence"""
        logprobs = response_data.get("logprobs", [])
        
        if not logprobs:
            return FilterResult(
                passed=False,
                confidence_score=0.0,
                reason="No logprobs available",
                metrics=ConfidenceMetrics(0, 0, 0, 0, 0, 1.0),
                timestamp=datetime.now()
            )
        
        metrics = self._compute_metrics(logprobs)
        passed = metrics.confidence_score >= self.threshold
        
        return FilterResult(
            passed=passed,
            confidence_score=metrics.confidence_score,
            reason=f"Confidence {'above' if passed else 'below'} threshold {self.threshold:.2f}",
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def _compute_metrics(self, logprobs: List[float]) -> ConfidenceMetrics:
        """Compute comprehensive confidence metrics"""
        logprobs_array = np.array(logprobs)
        
        mean_logprob = np.mean(logprobs_array)
        variance = np.var(logprobs_array)
        
        # Convert to probabilities for entropy calculation
        probs = np.exp(logprobs_array)
        probs = probs / np.sum(probs)  # Normalize
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        perplexity = np.exp(-mean_logprob)
        confidence_score = -mean_logprob  # Higher is better
        uncertainty_score = entropy / np.log(len(logprobs))  # Normalized entropy
        
        return ConfidenceMetrics(
            mean_logprob=mean_logprob,
            variance=variance,
            entropy=entropy,
            perplexity=perplexity,
            confidence_score=confidence_score,
            uncertainty_score=uncertainty_score
        )

class ConfidenceFilterManager:
    """Manager for different confidence filtering strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = ConfidenceStrategy(config.get("strategy", "adaptive_threshold"))
        self.filters = {}
        self.statistics = {
            "total_filtered": 0,
            "passed_count": 0,
            "failed_count": 0,
            "average_confidence": 0.0,
            "filter_history": []
        }
        
        self._initialize_filters()
    
    def _initialize_filters(self):
        """Initialize filters based on strategy"""
        if self.strategy == ConfidenceStrategy.ADAPTIVE_THRESHOLD:
            self.filters["adaptive"] = AdaptiveConfidenceFilter(
                initial_threshold=self.config.get("threshold", 17.0),
                adaptation_rate=self.config.get("adaptation_rate", 0.1)
            )
        elif self.strategy == ConfidenceStrategy.LOGPROB_THRESHOLD:
            self.filters["basic"] = ConfidenceFilter(
                threshold=self.config.get("threshold", 17.0),
                group_size=self.config.get("group_size", 2048)
            )
        elif self.strategy == ConfidenceStrategy.ENSEMBLE_VOTING:
            # Initialize multiple filters for ensemble
            self.filters["conservative"] = ConfidenceFilter(threshold=20.0)
            self.filters["moderate"] = ConfidenceFilter(threshold=17.0)
            self.filters["liberal"] = ConfidenceFilter(threshold=14.0)
    
    def filter_response(self, response_data: Dict[str, Any]) -> FilterResult:
        """Apply confidence filtering to response"""
        if self.strategy == ConfidenceStrategy.ADAPTIVE_THRESHOLD:
            result = self.filters["adaptive"].filter_response(response_data)
        elif self.strategy == ConfidenceStrategy.ENSEMBLE_VOTING:
            result = self._ensemble_filter(response_data)
        else:
            result = self._basic_filter(response_data)
        
        # Update statistics
        self._update_statistics(result)
        
        return result
    
    def _ensemble_filter(self, response_data: Dict[str, Any]) -> FilterResult:
        """Ensemble voting across multiple filters"""
        logprobs = response_data.get("logprobs", [])
        
        if not logprobs:
            return FilterResult(
                passed=False,
                confidence_score=0.0,
                reason="No logprobs for ensemble filtering",
                metrics=ConfidenceMetrics(0, 0, 0, 0, 0, 1.0),
                timestamp=datetime.now()
            )
        
        votes = []
        confidences = []
        
        for name, filter_obj in self.filters.items():
            for logprob in logprobs:
                filter_obj.update(logprob)
            
            confidence = filter_obj.get_current_confidence()
            passed = not filter_obj.should_stop()
            
            votes.append(passed)
            confidences.append(confidence)
            filter_obj.reset()
        
        # Majority voting
        passed_votes = sum(votes)
        ensemble_passed = passed_votes >= len(votes) / 2
        ensemble_confidence = np.mean(confidences)
        
        metrics = AdaptiveConfidenceFilter()._compute_metrics(logprobs)
        
        return FilterResult(
            passed=ensemble_passed,
            confidence_score=ensemble_confidence,
            reason=f"Ensemble voting: {passed_votes}/{len(votes)} filters passed",
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def _basic_filter(self, response_data: Dict[str, Any]) -> FilterResult:
        """Basic logprob threshold filtering"""
        logprobs = response_data.get("logprobs", [])
        
        if not logprobs:
            return FilterResult(
                passed=False,
                confidence_score=0.0,
                reason="No logprobs for basic filtering",
                metrics=ConfidenceMetrics(0, 0, 0, 0, 0, 1.0),
                timestamp=datetime.now()
            )
        
        filter_obj = self.filters["basic"]
        for logprob in logprobs:
            filter_obj.update(logprob)
        
        confidence = filter_obj.get_current_confidence()
        passed = not filter_obj.should_stop()
        
        metrics = AdaptiveConfidenceFilter()._compute_metrics(logprobs)
        
        filter_obj.reset()
        
        return FilterResult(
            passed=passed,
            confidence_score=confidence,
            reason=f"Basic threshold filter: {confidence:.2f} vs {filter_obj.threshold}",
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def _update_statistics(self, result: FilterResult):
        """Update filtering statistics"""
        self.statistics["total_filtered"] += 1
        
        if result.passed:
            self.statistics["passed_count"] += 1
        else:
            self.statistics["failed_count"] += 1
        
        # Update running average
        total = self.statistics["total_filtered"]
        current_avg = self.statistics["average_confidence"]
        self.statistics["average_confidence"] = (
            (current_avg * (total - 1) + result.confidence_score) / total
        )
        
        # Keep history (last 100 results)
        self.statistics["filter_history"].append({
            "timestamp": result.timestamp.isoformat(),
            "passed": result.passed,
            "confidence": result.confidence_score,
            "reason": result.reason
        })
        
        if len(self.statistics["filter_history"]) > 100:
            self.statistics["filter_history"].pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        total = self.statistics["total_filtered"]
        
        return {
            **self.statistics,
            "pass_rate": self.statistics["passed_count"] / max(total, 1),
            "fail_rate": self.statistics["failed_count"] / max(total, 1),
            "strategy": self.strategy.value,
            "config": self.config
        }
    
    def update_performance_feedback(self, performance_score: float):
        """Update filters with performance feedback"""
        if self.strategy == ConfidenceStrategy.ADAPTIVE_THRESHOLD:
            self.filters["adaptive"].update_threshold(performance_score)

# Utility functions
def compute_trace_confidence(trace: Dict) -> float:
    """Compute confidence score for a trace"""
    logprobs = trace.get("logprobs", [])
    return -np.mean(logprobs) if logprobs else float("-inf")

def filter_top_confident_traces(traces: List[Dict], top_percent: int = 10) -> List[Dict]:
    """Filter traces by confidence, keeping top percentage"""
    scored = [(compute_trace_confidence(t), t) for t in traces]
    scored.sort(reverse=True, key=lambda x: x[0])
    cutoff = max(1, int(len(scored) * top_percent / 100))
    return [t for _, t in scored[:cutoff]]

def integrate_confidence_filtering(config: Dict[str, Any] = None) -> ConfidenceFilterManager:
    """Main integration function for confidence filtering"""
    
    default_config = {
        "strategy": "adaptive_threshold",
        "threshold": 17.0,
        "adaptation_rate": 0.1,
        "group_size": 2048,
        "warmup_traces": 16,
        "enable_statistics": True
    }
    
    if config:
        default_config.update(config)
    
    print("🎯 Initializing Confidence Filtering System")
    print(f"   Strategy: {default_config['strategy']}")
    print(f"   Threshold: {default_config['threshold']}")
    
    manager = ConfidenceFilterManager(default_config)
    
    print("✅ Confidence Filtering System initialized")
    
    return manager

class DeepConfIntegration:
    """Complete DeepConf integration for AI Research Agent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize all components
        self.scoring_hook = ConfidenceScoringHook(
            enable_real_time=self.config.get("enable_real_time", True)
        )
        
        self.early_termination = EarlyTerminationLogic(
            threshold_percentile=self.config.get("threshold_percentile", 90),
            warmup_traces=self.config.get("warmup_traces", 16)
        )
        
        self.confidence_voting = ConfidenceAwareVoting(
            top_n_percent=self.config.get("top_n_percent", 50),
            min_confidence=self.config.get("min_confidence", 0.1)
        )
        
        self.semantic_alignment = SemanticGraphAlignment(
            confidence_threshold=self.config.get("semantic_threshold", 0.7)
        )
        
        self.validation_strategy = ValidationStrategy(
            validation_threshold=self.config.get("validation_threshold", 0.8),
            min_samples=self.config.get("min_samples", 100)
        )
        
        self.filter_manager = ConfidenceFilterManager(self.config)
        
        # Integration state
        self.active_traces = {}
        self.session_statistics = defaultdict(dict)
        self.integration_metrics = {
            "total_requests": 0,
            "filtered_requests": 0,
            "early_terminations": 0,
            "semantic_alignments": 0,
            "voting_decisions": 0
        }
    
    async def process_research_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process research request with full DeepConf integration"""
        session_id = request.get("session_id", "default")
        query = request.get("query", "")
        
        self.integration_metrics["total_requests"] += 1
        
        # Phase 1: Initialize trace confidence tracking
        trace_id = f"trace_{session_id}_{datetime.now().timestamp()}"
        trace_confidence = TraceConfidence(
            trace_id=trace_id,
            tokens=[],
            group_confidence=GroupConfidence(window_size=10),
            overall_confidence=0.0,
            reasoning_phase=ReasoningPhase.GENERATION
        )
        
        self.active_traces[trace_id] = trace_confidence
        
        # Phase 2: Token-level confidence scoring during generation
        generated_tokens = await self._simulate_generation(query, trace_confidence)
        
        # Phase 3: Early termination check
        should_terminate, termination_reason = self.early_termination.should_terminate_early(
            trace_confidence.overall_confidence, len(generated_tokens)
        )
        
        if should_terminate:
            self.integration_metrics["early_terminations"] += 1
            return {
                "success": False,
                "reason": f"Early termination: {termination_reason}",
                "trace_id": trace_id,
                "tokens_saved": len(generated_tokens)
            }
        
        # Phase 4: Semantic graph alignment
        semantic_score = self.semantic_alignment.annotate_node_reliability(
            f"node_{trace_id}", trace_confidence
        )
        trace_confidence.semantic_reliability = semantic_score
        
        if semantic_score > self.config.get("semantic_threshold", 0.7):
            self.integration_metrics["semantic_alignments"] += 1
        
        # Phase 5: Confidence-aware voting (if multiple traces)
        if len(self.active_traces) > 1:
            traces = list(self.active_traces.values())
            answers = [f"Answer for {t.trace_id}" for t in traces]  # Simplified
            
            best_answer, voting_confidence, voting_metadata = self.confidence_voting.aggregate_votes(
                traces, answers
            )
            
            self.integration_metrics["voting_decisions"] += 1
            
            # Use voting result
            final_answer = best_answer
            final_confidence = voting_confidence
        else:
            final_answer = f"Generated answer for {query}"
            final_confidence = trace_confidence.overall_confidence
        
        # Phase 6: Final confidence filtering
        response_data = {
            "logprobs": [token.logprob for token in trace_confidence.tokens],
            "answer": final_answer,
            "trace_id": trace_id
        }
        
        filter_result = self.filter_manager.filter_response(response_data)
        
        if not filter_result.passed:
            self.integration_metrics["filtered_requests"] += 1
        
        # Update session statistics
        self.session_statistics[session_id].update({
            "last_confidence": final_confidence,
            "last_filter_result": filter_result.passed,
            "total_requests": self.session_statistics[session_id].get("total_requests", 0) + 1
        })
        
        # Cleanup
        if trace_id in self.active_traces:
            del self.active_traces[trace_id]
        
        return {
            "success": filter_result.passed,
            "answer": final_answer if filter_result.passed else "Response filtered due to low confidence",
            "confidence_score": final_confidence,
            "filter_result": {
                "passed": filter_result.passed,
                "reason": filter_result.reason,
                "metrics": {
                    "mean_logprob": filter_result.metrics.mean_logprob,
                    "confidence_score": filter_result.metrics.confidence_score,
                    "uncertainty_score": filter_result.metrics.uncertainty_score
                }
            },
            "trace_metadata": {
                "trace_id": trace_id,
                "semantic_reliability": semantic_score,
                "early_termination": should_terminate,
                "reasoning_phase": trace_confidence.reasoning_phase.value
            },
            "integration_metrics": self.integration_metrics.copy()
        }
    
    async def _simulate_generation(self, query: str, trace_confidence: TraceConfidence) -> List[TokenConfidence]:
        """Simulate token generation with confidence tracking"""
        # Simulate token generation (in real implementation, this would hook into LLM)
        simulated_tokens = [
            ("The", -0.5), ("answer", -0.3), ("to", -0.8), ("your", -0.4),
            ("question", -0.6), ("about", -0.7), (query.split()[0] if query else "topic", -0.9),
            ("is", -0.5), ("complex", -1.2), ("and", -0.4), ("requires", -1.0),
            ("careful", -1.1), ("analysis", -0.8), (".", -0.3)
        ]
        
        tokens = []
        for i, (token, logprob) in enumerate(simulated_tokens):
            token_conf = self.scoring_hook.compute_token_confidence(token, logprob, i)
            tokens.append(token_conf)
            
            # Update trace confidence
            trace_confidence.tokens.append(token_conf)
            
            # Update group confidence (sliding window)
            if len(trace_confidence.tokens) >= trace_confidence.group_confidence.window_size:
                recent_confidences = [
                    t.confidence for t in trace_confidence.tokens[-trace_confidence.group_confidence.window_size:]
                ]
                trace_confidence.group_confidence.confidence_scores = recent_confidences
                trace_confidence.group_confidence.mean_confidence = np.mean(recent_confidences)
                trace_confidence.group_confidence.variance = np.var(recent_confidences)
                trace_confidence.group_confidence.percentile_90 = np.percentile(recent_confidences, 90)
                trace_confidence.group_confidence.is_above_threshold = (
                    trace_confidence.group_confidence.mean_confidence > 0.5
                )
            
            # Update overall trace confidence
            trace_confidence.overall_confidence = self.scoring_hook.get_trace_confidence(tokens)
            
            # Update early termination logic
            self.early_termination.update_confidence_history(trace_confidence.overall_confidence)
        
        return tokens
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        return {
            "active_traces": len(self.active_traces),
            "integration_metrics": self.integration_metrics,
            "session_count": len(self.session_statistics),
            "early_termination_stats": {
                "threshold": self.early_termination.threshold,
                "is_warmed_up": self.early_termination.is_warmed_up,
                "confidence_history_size": len(self.early_termination.confidence_history)
            },
            "filter_manager_stats": self.filter_manager.get_statistics(),
            "validation_summary": self.validation_strategy.get_validation_summary(),
            "config": self.config
        }
    
    def run_offline_validation(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run offline validation with test data"""
        # Convert test data to trace confidences
        test_traces = []
        expected_outcomes = []
        
        for item in test_data:
            # Create mock trace confidence
            tokens = [
                TokenConfidence(
                    token=token,
                    logprob=logprob,
                    confidence=np.exp(logprob),
                    position=i,
                    is_reliable=logprob > -2.0
                )
                for i, (token, logprob) in enumerate(zip(item.get("tokens", []), item.get("logprobs", [])))
            ]
            
            trace = TraceConfidence(
                trace_id=f"test_{len(test_traces)}",
                tokens=tokens,
                group_confidence=GroupConfidence(window_size=5),
                overall_confidence=np.mean([t.confidence for t in tokens]) if tokens else 0.0,
                reasoning_phase=ReasoningPhase.EVALUATION
            )
            
            test_traces.append(trace)
            expected_outcomes.append(item.get("expected_pass", True))
        
        return self.validation_strategy.run_offline_evaluation(test_traces, expected_outcomes)

def demo_confidence_filtering():
    """Demonstration of confidence filtering capabilities"""
    print("🎯 DeepConf-Enhanced Confidence Filtering Demo")
    print("=" * 50)
    
    # Initialize integration
    config = {
        "strategy": "adaptive_threshold",
        "threshold": 15.0,
        "adaptation_rate": 0.1,
        "enable_real_time": True,
        "top_n_percent": 30,
        "semantic_threshold": 0.7
    }
    
    integration = DeepConfIntegration(config)
    
    # Demo requests
    demo_requests = [
        {
            "session_id": "demo_session_1",
            "query": "How do neural networks learn?",
            "expected_quality": "high"
        },
        {
            "session_id": "demo_session_2", 
            "query": "What is quantum computing?",
            "expected_quality": "medium"
        },
        {
            "session_id": "demo_session_3",
            "query": "Explain machine learning",
            "expected_quality": "high"
        }
    ]
    
    async def run_demo():
        results = []
        
        for i, request in enumerate(demo_requests):
            print(f"\n📝 Processing Request {i+1}: {request['query']}")
            
            result = await integration.process_research_request(request)
            results.append(result)
            
            print(f"   ✅ Success: {result['success']}")
            print(f"   🎯 Confidence: {result['confidence_score']:.3f}")
            print(f"   🔍 Filter Passed: {result['filter_result']['passed']}")
            
            if not result['success']:
                print(f"   ❌ Reason: {result['filter_result']['reason']}")
        
        # Show integration status
        print(f"\n📊 Integration Status:")
        status = integration.get_integration_status()
        
        print(f"   Total Requests: {status['integration_metrics']['total_requests']}")
        print(f"   Filtered Requests: {status['integration_metrics']['filtered_requests']}")
        print(f"   Early Terminations: {status['integration_metrics']['early_terminations']}")
        print(f"   Semantic Alignments: {status['integration_metrics']['semantic_alignments']}")
        
        # Run offline validation
        print(f"\n🧪 Running Offline Validation...")
        test_data = [
            {
                "tokens": ["Good", "answer", "with", "high", "confidence"],
                "logprobs": [-0.1, -0.2, -0.3, -0.2, -0.1],
                "expected_pass": True
            },
            {
                "tokens": ["Poor", "answer", "with", "low", "confidence"],
                "logprobs": [-2.0, -2.5, -3.0, -2.8, -2.2],
                "expected_pass": False
            }
        ]
        
        validation_results = integration.run_offline_validation(test_data)
        
        if "error" not in validation_results:
            print(f"   Best Strategy: {validation_results['best_strategy']}")
            print(f"   Total Samples: {validation_results['total_samples']}")
        else:
            print(f"   Validation Error: {validation_results['error']}")
        
        return results
    
    # Run the demo
    import asyncio
    results = asyncio.run(run_demo())
    
    print(f"\n✅ Demo completed with {len(results)} requests processed")
    return results

if __name__ == "__main__":
    # Run demonstration
    demo_results = demo_confidence_filtering()
    
    # Additional testing
    print(f"\n🧪 Running Additional Tests...")
    
    # Test basic confidence filtering
    cf = ConfidenceFilter(threshold=15.0, group_size=5)
    test_logprobs = [-0.5, -0.3, -0.8, -0.4, -0.6]
    
    for logprob in test_logprobs:
        cf.update(logprob)
    
    print(f"   Basic Filter Confidence: {cf.get_current_confidence():.3f}")
    print(f"   Should Stop: {cf.should_stop()}")
    
    # Test adaptive filtering
    acf = AdaptiveConfidenceFilter(initial_threshold=15.0)
    response_data = {"logprobs": test_logprobs}
    result = acf.filter_response(response_data)
    
    print(f"   Adaptive Filter Passed: {result.passed}")
    print(f"   Adaptive Filter Confidence: {result.confidence_score:.3f}")
    
    # Test trace confidence computation
    test_trace = {"logprobs": [-0.2, -0.3, -0.1, -0.4]}
    trace_conf = compute_trace_confidence(test_trace)
    print(f"   Trace Confidence: {trace_conf:.3f}")
    
    # Test top confident traces filtering
    test_traces = [
        {"logprobs": [-0.1, -0.2], "id": "high"},
        {"logprobs": [-1.0, -1.1], "id": "medium"},
        {"logprobs": [-2.0, -2.1], "id": "low"}
    ]
    
    filtered = filter_top_confident_traces(test_traces, top_percent=50)
    print(f"   Top 50% Traces: {[t['id'] for t in filtered]}")
    
    print(f"\n🎉 All tests completed successfully!")