# -*- coding: utf-8 -*-
"""
Stage 5: RLHF & Agentic Reinforcement Learning
Advanced preference learning with multi-objective alignment
"""

import json
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PreferenceType(Enum):
    HUMAN_FEEDBACK = "human_feedback"
    AUTOMATED_METRIC = "automated_metric"
    TOOL_SUCCESS = "tool_success"
    QUALITY_SCORE = "quality_score"
    EFFICIENCY_SCORE = "efficiency_score"
    SAFETY_SCORE = "safety_score"

class RewardSignalType(Enum):
    COMPILE_SUCCESS = "compile_success"
    RETRIEVAL_QUALITY = "retrieval_quality"
    FACTUALITY_SCORE = "factuality_score"
    LATENCY_PENALTY = "latency_penalty"
    COST_PENALTY = "cost_penalty"
    CORRECTNESS_SCORE = "correctness_score"

class AlignmentObjective(Enum):
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"

@dataclass
class PreferenceData:
    """Individual preference data point"""
    preference_id: str
    query: str
    response_a: str
    response_b: str
    preference: int  # 0 for A, 1 for B, -1 for tie
    preference_type: PreferenceType
    confidence: float
    annotator_id: Optional[str]
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class RewardSignal:
    """Individual reward signal"""
    signal_id: str
    action: str
    reward_value: float
    signal_type: RewardSignalType
    context: Dict[str, Any]
    timestamp: datetime
    session_id: str

@dataclass
class PolicyAction:
    """Action taken by the policy"""
    action_id: str
    state_representation: Dict[str, Any]
    action_type: str
    action_parameters: Dict[str, Any]
    predicted_reward: float
    actual_reward: Optional[float]
    timestamp: datetime

class PreferenceDataPipeline:
    """Pipeline for collecting and processing preference data"""
    
    def __init__(self, storage_path: str = "extensions/preference_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.preference_data: List[PreferenceData] = []
        self.preference_queue = deque(maxlen=1000)  # Recent preferences
        
        # Load existing data
        self._load_preference_data()
        
        print("📊 Preference Data Pipeline initialized")
        print(f"   Loaded preferences: {len(self.preference_data)}")
    
    def collect_preference(self, query: str, response_a: str, response_b: str,
                          preference: int, preference_type: PreferenceType,
                          confidence: float = 1.0, annotator_id: str = None,
                          metadata: Dict[str, Any] = None) -> str:
        """Collect a new preference data point"""
        
        preference_id = str(uuid.uuid4())
        
        preference_data = PreferenceData(
            preference_id=preference_id,
            query=query,
            response_a=response_a,
            response_b=response_b,
            preference=preference,
            preference_type=preference_type,
            confidence=confidence,
            annotator_id=annotator_id,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        self.preference_data.append(preference_data)
        self.preference_queue.append(preference_data)
        
        # Save to storage
        self._save_preference_data(preference_data)
        
        return preference_id
    
    def get_training_data(self, min_confidence: float = 0.5,
                         preference_types: List[PreferenceType] = None,
                         max_age_days: int = 30) -> List[PreferenceData]:
        """Get filtered training data"""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        filtered_data = []
        for pref in self.preference_data:
            # Filter by confidence
            if pref.confidence < min_confidence:
                continue
            
            # Filter by type
            if preference_types and pref.preference_type not in preference_types:
                continue
            
            # Filter by age
            if pref.timestamp < cutoff_date:
                continue
            
            filtered_data.append(pref)
        
        return filtered_data
    
    def create_dpo_dataset(self, training_data: List[PreferenceData]) -> 'DPODataset':
        """Create dataset for Direct Preference Optimization"""
        return DPODataset(training_data)
    
    def get_preference_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected preferences"""
        
        if not self.preference_data:
            return {"total_preferences": 0}
        
        # Count by type
        type_counts = {}
        for pref_type in PreferenceType:
            count = len([p for p in self.preference_data if p.preference_type == pref_type])
            type_counts[pref_type.value] = count
        
        # Count by preference
        preference_distribution = {
            "prefer_a": len([p for p in self.preference_data if p.preference == 0]),
            "prefer_b": len([p for p in self.preference_data if p.preference == 1]),
            "tie": len([p for p in self.preference_data if p.preference == -1])
        }
        
        # Average confidence
        avg_confidence = sum(p.confidence for p in self.preference_data) / len(self.preference_data)
        
        return {
            "total_preferences": len(self.preference_data),
            "type_distribution": type_counts,
            "preference_distribution": preference_distribution,
            "average_confidence": avg_confidence,
            "recent_preferences": len(self.preference_queue)
        }
    
    def _load_preference_data(self):
        """Load existing preference data"""
        
        data_file = self.storage_path / "preferences.jsonl"
        if data_file.exists():
            with open(data_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        preference = PreferenceData(
                            preference_id=data["preference_id"],
                            query=data["query"],
                            response_a=data["response_a"],
                            response_b=data["response_b"],
                            preference=data["preference"],
                            preference_type=PreferenceType(data["preference_type"]),
                            confidence=data["confidence"],
                            annotator_id=data.get("annotator_id"),
                            metadata=data.get("metadata", {}),
                            timestamp=datetime.fromisoformat(data["timestamp"])
                        )
                        self.preference_data.append(preference)
                    except Exception as e:
                        print(f"⚠️ Failed to load preference data: {e}")
    
    def _save_preference_data(self, preference: PreferenceData):
        """Save preference data to storage"""
        
        data_file = self.storage_path / "preferences.jsonl"
        
        data = {
            "preference_id": preference.preference_id,
            "query": preference.query,
            "response_a": preference.response_a,
            "response_b": preference.response_b,
            "preference": preference.preference,
            "preference_type": preference.preference_type.value,
            "confidence": preference.confidence,
            "annotator_id": preference.annotator_id,
            "metadata": preference.metadata,
            "timestamp": preference.timestamp.isoformat()
        }
        
        with open(data_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization"""
    
    def __init__(self, preference_data: List[PreferenceData]):
        self.preference_data = preference_data
    
    def __len__(self):
        return len(self.preference_data)
    
    def __getitem__(self, idx):
        pref = self.preference_data[idx]
        
        return {
            "query": pref.query,
            "response_a": pref.response_a,
            "response_b": pref.response_b,
            "preference": pref.preference,
            "confidence": pref.confidence
        }

class ConfidenceAwareRewardShaping:
    """Integrate confidence metrics into RLHF reward shaping"""
    
    def __init__(self, confidence_weight=0.3, uncertainty_penalty=0.2):
        self.confidence_weight = confidence_weight
        self.uncertainty_penalty = uncertainty_penalty
        self.confidence_history = deque(maxlen=1000)
        self.reward_adjustments = []
        
        print(f"🎯 Confidence-Aware Reward Shaping initialized")
        print(f"   Confidence weight: {confidence_weight}")
        print(f"   Uncertainty penalty: {uncertainty_penalty}")
    
    def shape_reward_with_confidence(self, base_reward: float, confidence_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Shape reward based on confidence metrics"""
        
        # Extract confidence metrics
        confidence_score = confidence_metrics.get("confidence_score", 0.5)
        uncertainty_score = confidence_metrics.get("uncertainty_score", 0.5)
        mean_logprob = confidence_metrics.get("mean_logprob", -1.0)
        variance = confidence_metrics.get("variance", 1.0)
        
        # Confidence bonus: reward high-confidence responses
        confidence_bonus = self.confidence_weight * confidence_score
        
        # Uncertainty penalty: penalize high-uncertainty responses
        uncertainty_penalty = self.uncertainty_penalty * uncertainty_score
        
        # Consistency bonus: reward consistent token probabilities (low variance)
        consistency_bonus = 0.1 * max(0, 1.0 - variance)
        
        # Calibration adjustment: penalize overconfident wrong answers
        calibration_adjustment = self._compute_calibration_adjustment(
            confidence_score, base_reward
        )
        
        # Compute shaped reward
        shaped_reward = (
            base_reward + 
            confidence_bonus - 
            uncertainty_penalty + 
            consistency_bonus + 
            calibration_adjustment
        )
        
        # Store for analysis
        adjustment_info = {
            "base_reward": base_reward,
            "shaped_reward": shaped_reward,
            "confidence_bonus": confidence_bonus,
            "uncertainty_penalty": uncertainty_penalty,
            "consistency_bonus": consistency_bonus,
            "calibration_adjustment": calibration_adjustment,
            "confidence_metrics": confidence_metrics,
            "timestamp": datetime.now()
        }
        
        self.reward_adjustments.append(adjustment_info)
        self.confidence_history.append(confidence_score)
        
        return adjustment_info
    
    def _compute_calibration_adjustment(self, confidence_score: float, base_reward: float) -> float:
        """Compute calibration adjustment to penalize overconfident wrong answers"""
        
        # If base reward is low but confidence is high, apply penalty
        if base_reward < 0.3 and confidence_score > 0.8:
            return -0.2 * (confidence_score - 0.5)  # Overconfidence penalty
        
        # If base reward is high and confidence is appropriately high, give bonus
        elif base_reward > 0.7 and confidence_score > 0.7:
            return 0.1 * confidence_score  # Well-calibrated bonus
        
        return 0.0
    
    def get_confidence_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about confidence-based reward shaping"""
        
        if not self.reward_adjustments:
            return {"total_adjustments": 0}
        
        # Calculate statistics
        base_rewards = [adj["base_reward"] for adj in self.reward_adjustments]
        shaped_rewards = [adj["shaped_reward"] for adj in self.reward_adjustments]
        confidence_bonuses = [adj["confidence_bonus"] for adj in self.reward_adjustments]
        
        return {
            "total_adjustments": len(self.reward_adjustments),
            "average_base_reward": np.mean(base_rewards),
            "average_shaped_reward": np.mean(shaped_rewards),
            "average_confidence_bonus": np.mean(confidence_bonuses),
            "reward_improvement": np.mean(shaped_rewards) - np.mean(base_rewards),
            "confidence_correlation": np.corrcoef(
                [adj["confidence_metrics"]["confidence_score"] for adj in self.reward_adjustments],
                shaped_rewards
            )[0, 1] if len(self.reward_adjustments) > 1 else 0.0
        }

class RewardModel(nn.Module):
    """Neural reward model for quality assessment with confidence integration"""

    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, confidence_integration: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.confidence_integration = confidence_integration
        
        # Main reward prediction network
        self.reward_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Single reward value
        )
        
        # Confidence-aware components
        if confidence_integration:
            # Confidence prediction head
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output confidence between 0 and 1
            )
            
            # Uncertainty estimation head
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output uncertainty between 0 and 1
            )
        
        # Initialize confidence-aware reward shaping
        self.reward_shaper = ConfidenceAwareRewardShaping()
        
        print(f"🏆 Confidence-Aware Reward Model initialized")
        print(f"   Input dim: {input_dim}, Hidden dim: {hidden_dim}")
        print(f"   Confidence integration: {confidence_integration}")
    
    def forward(self, state_representation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass to predict reward and confidence metrics"""
        
        # Get intermediate representation
        x = state_representation
        for i, layer in enumerate(self.reward_layers[:-1]):
            x = layer(x)
        
        # Final reward prediction
        reward = self.reward_layers[-1](x)
        
        outputs = {"reward": reward}
        
        # Add confidence predictions if enabled
        if self.confidence_integration:
            confidence = self.confidence_head(x)
            uncertainty = self.uncertainty_head(x)
            outputs.update({
                "confidence": confidence,
                "uncertainty": uncertainty
            })
        
        return outputs
    
    def predict_reward(self, state_dict: Dict[str, Any]) -> float:
        """Predict reward from state dictionary"""
        
        # Convert state dict to tensor
        state_vector = self._state_dict_to_vector(state_dict)
        
        with torch.no_grad():
            outputs = self.forward(state_vector)
            return outputs["reward"].item()
    
    def predict_reward_with_confidence(self, state_dict: Dict[str, Any], 
                                     confidence_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Predict reward with confidence-aware shaping"""
        
        # Convert state dict to tensor
        state_vector = self._state_dict_to_vector(state_dict)
        
        with torch.no_grad():
            outputs = self.forward(state_vector)
            
            base_reward = outputs["reward"].item()
            
            # Get confidence metrics from model or external source
            if confidence_metrics is None and self.confidence_integration:
                confidence_metrics = {
                    "confidence_score": outputs["confidence"].item(),
                    "uncertainty_score": outputs["uncertainty"].item(),
                    "mean_logprob": state_dict.get("mean_logprob", -1.0),
                    "variance": state_dict.get("logprob_variance", 1.0)
                }
            elif confidence_metrics is None:
                confidence_metrics = {
                    "confidence_score": 0.5,
                    "uncertainty_score": 0.5,
                    "mean_logprob": -1.0,
                    "variance": 1.0
                }
            
            # Apply confidence-aware reward shaping
            shaping_result = self.reward_shaper.shape_reward_with_confidence(
                base_reward, confidence_metrics
            )
            
            return {
                "base_reward": base_reward,
                "shaped_reward": shaping_result["shaped_reward"],
                "confidence_metrics": confidence_metrics,
                "shaping_details": shaping_result
            }
    
    def select_traces_for_rlhf(self, traces: List[Dict[str, Any]], 
                              top_percent: int = 10) -> List[Dict[str, Any]]:
        """Select top confident traces for RLHF training"""
        
        # Import confidence filtering function
        try:
            from extensions.stage_7_confidence_filtering import filter_top_confident_traces
            return filter_top_confident_traces(traces, top_percent=top_percent)
        except ImportError:
            # Fallback implementation
            return self._fallback_trace_selection(traces, top_percent)
    
    def _fallback_trace_selection(self, traces: List[Dict[str, Any]], 
                                 top_percent: int) -> List[Dict[str, Any]]:
        """Fallback trace selection based on reward model confidence"""
        
        scored_traces = []
        
        for trace in traces:
            # Extract state information from trace
            state_dict = {
                "logprobs": trace.get("logprobs", []),
                "response_length": len(trace.get("response", "")),
                "context_size": trace.get("context_size", 0)
            }
            
            # Get confidence-aware reward
            result = self.predict_reward_with_confidence(state_dict)
            confidence_score = result["confidence_metrics"]["confidence_score"]
            
            scored_traces.append((confidence_score, trace))
        
        # Sort by confidence and take top percentage
        scored_traces.sort(key=lambda x: x[0], reverse=True)
        cutoff = max(1, int(len(scored_traces) * top_percent / 100))
        
        return [trace for _, trace in scored_traces[:cutoff]]
    
    def _state_dict_to_vector(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        """Convert state dictionary to vector representation"""
        
        # Simplified state encoding
        features = []
        
        # Add numeric features
        for key, value in state_dict.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple string hash feature
                features.append(float(hash(value) % 1000) / 1000.0)
        
        # Pad or truncate to input_dim
        while len(features) < self.input_dim:
            features.append(0.0)
        features = features[:self.input_dim]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

class OnlineAgenticRL:
    """Online reinforcement learning for agentic behavior"""
    
    def __init__(self, reward_model: RewardModel):
        self.reward_model = reward_model
        self.action_history: List[PolicyAction] = []
        self.reward_signals: List[RewardSignal] = []
        
        # RL parameters
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        
        # Policy network (simplified)
        self.policy_network = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 possible actions
        )
        
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        print("🤖 Online Agentic RL initialized")
    
    def select_action(self, state: Dict[str, Any], available_actions: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select action using current policy with confidence awareness"""
        
        # Convert state to tensor
        state_tensor = self.reward_model._state_dict_to_vector(state)
        
        # Get action probabilities
        with torch.no_grad():
            action_logits = self.policy_network(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
        
        # Get confidence-aware reward prediction
        reward_prediction = self.reward_model.predict_reward_with_confidence(state)
        predicted_reward = reward_prediction["shaped_reward"]
        confidence_score = reward_prediction["confidence_metrics"]["confidence_score"]
        
        # Confidence-aware exploration: explore more when confidence is low
        dynamic_exploration_rate = self.exploration_rate * (1.0 + (1.0 - confidence_score))
        
        if np.random.random() < dynamic_exploration_rate:
            # Random action (more likely when confidence is low)
            action_idx = np.random.randint(len(available_actions))
        else:
            # Greedy action
            action_idx = torch.argmax(action_probs).item()
            action_idx = min(action_idx, len(available_actions) - 1)
        
        selected_action = available_actions[action_idx]
        
        # Record action with confidence information
        action = PolicyAction(
            action_id=str(uuid.uuid4()),
            state_representation=state,
            action_type=selected_action,
            action_parameters={
                "confidence_score": confidence_score,
                "exploration_rate_used": dynamic_exploration_rate
            },
            predicted_reward=predicted_reward,
            actual_reward=None,
            timestamp=datetime.now()
        )
        
        self.action_history.append(action)
        
        return selected_action, {
            "predicted_reward": predicted_reward,
            "confidence_score": confidence_score,
            "exploration_rate": dynamic_exploration_rate,
            "reward_details": reward_prediction
        }
    
    def record_reward_signal(self, action_id: str, reward_signals: List[RewardSignal], 
                           confidence_metrics: Dict[str, Any] = None):
        """Record reward signals for an action with confidence integration"""
        
        # Find the action
        action = None
        for a in self.action_history:
            if a.action_id == action_id:
                action = a
                break
        
        if not action:
            return
        
        # Calculate base composite reward
        base_reward = self._calculate_composite_reward(reward_signals)
        
        # Apply confidence-aware reward shaping if metrics available
        if confidence_metrics:
            shaping_result = self.reward_model.reward_shaper.shape_reward_with_confidence(
                base_reward, confidence_metrics
            )
            shaped_reward = shaping_result["shaped_reward"]
            
            # Store shaping details in action parameters
            action.action_parameters.update({
                "base_reward": base_reward,
                "shaped_reward": shaped_reward,
                "confidence_shaping": shaping_result
            })
        else:
            shaped_reward = base_reward
        
        action.actual_reward = shaped_reward
        
        # Store reward signals
        self.reward_signals.extend(reward_signals)
        
        # Update policy if we have enough data
        if len(self.action_history) % 10 == 0:  # Update every 10 actions
            self._update_policy()
    
    def _calculate_composite_reward(self, signals: List[RewardSignal]) -> float:
        """Calculate composite reward from multiple signals"""
        
        if not signals:
            return 0.0
        
        # Weighted combination of different reward types
        weights = {
            RewardSignalType.COMPILE_SUCCESS: 0.3,
            RewardSignalType.RETRIEVAL_QUALITY: 0.2,
            RewardSignalType.FACTUALITY_SCORE: 0.25,
            RewardSignalType.CORRECTNESS_SCORE: 0.25,
            RewardSignalType.LATENCY_PENALTY: -0.1,
            RewardSignalType.COST_PENALTY: -0.05
        }
        
        weighted_reward = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = weights.get(signal.signal_type, 0.1)
            weighted_reward += weight * signal.reward_value
            total_weight += abs(weight)
        
        return weighted_reward / max(total_weight, 1.0)
    
    def _update_policy(self):
        """Update policy network using recent experiences"""
        
        # Get recent actions with rewards
        recent_actions = [a for a in self.action_history[-50:] if a.actual_reward is not None]
        
        if len(recent_actions) < 5:
            return
        
        # Prepare training data
        states = []
        actions = []
        rewards = []
        
        for action in recent_actions:
            state_tensor = self.reward_model._state_dict_to_vector(action.state_representation)
            states.append(state_tensor)
            
            # Convert action type to index (simplified)
            action_idx = hash(action.action_type) % 10
            actions.append(action_idx)
            
            rewards.append(action.actual_reward)
        
        # Convert to tensors
        states_tensor = torch.cat(states, dim=0)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Normalize rewards
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Policy gradient update
        self.optimizer.zero_grad()
        
        action_logits = self.policy_network(states_tensor)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Calculate policy loss
        selected_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        policy_loss = -torch.mean(torch.log(selected_probs + 1e-8) * rewards_tensor)
        
        policy_loss.backward()
        self.optimizer.step()
    
    def get_rl_statistics(self) -> Dict[str, Any]:
        """Get RL training statistics"""
        
        actions_with_rewards = [a for a in self.action_history if a.actual_reward is not None]
        
        if not actions_with_rewards:
            return {"total_actions": len(self.action_history), "actions_with_rewards": 0}
        
        avg_reward = sum(a.actual_reward for a in actions_with_rewards) / len(actions_with_rewards)
        
        # Reward by signal type
        signal_stats = {}
        for signal_type in RewardSignalType:
            signals = [s for s in self.reward_signals if s.signal_type == signal_type]
            if signals:
                signal_stats[signal_type.value] = {
                    "count": len(signals),
                    "avg_reward": sum(s.reward_value for s in signals) / len(signals)
                }
        
        return {
            "total_actions": len(self.action_history),
            "actions_with_rewards": len(actions_with_rewards),
            "average_reward": avg_reward,
            "exploration_rate": self.exploration_rate,
            "signal_statistics": signal_stats
        }

class MultiObjectiveAlignment:
    """Multi-objective alignment system"""
    
    def __init__(self):
        self.objectives = {
            AlignmentObjective.HELPFULNESS: 0.3,
            AlignmentObjective.HARMLESSNESS: 0.25,
            AlignmentObjective.HONESTY: 0.2,
            AlignmentObjective.EFFICIENCY: 0.15,
            AlignmentObjective.ACCURACY: 0.1
        }
        
        self.objective_scores: Dict[AlignmentObjective, List[float]] = {
            obj: [] for obj in AlignmentObjective
        }
        
        print("🎯 Multi-Objective Alignment initialized")
        print(f"   Objectives: {list(self.objectives.keys())}")
    
    def evaluate_alignment(self, response: str, context: Dict[str, Any]) -> Dict[AlignmentObjective, float]:
        """Evaluate response against all alignment objectives with confidence awareness"""
        
        scores = {}
        
        # Extract confidence metrics if available
        confidence_metrics = context.get("confidence_metrics", {})
        confidence_score = confidence_metrics.get("confidence_score", 0.5)
        uncertainty_score = confidence_metrics.get("uncertainty_score", 0.5)
        
        # Helpfulness score
        scores[AlignmentObjective.HELPFULNESS] = self._evaluate_helpfulness(response, context)
        
        # Harmlessness score
        scores[AlignmentObjective.HARMLESSNESS] = self._evaluate_harmlessness(response, context)
        
        # Honesty score (confidence-aware)
        scores[AlignmentObjective.HONESTY] = self._evaluate_honesty_with_confidence(
            response, context, confidence_score, uncertainty_score
        )
        
        # Efficiency score
        scores[AlignmentObjective.EFFICIENCY] = self._evaluate_efficiency(response, context)
        
        # Accuracy score (confidence-aware)
        scores[AlignmentObjective.ACCURACY] = self._evaluate_accuracy_with_confidence(
            response, context, confidence_score
        )
        
        # Store scores for tracking
        for objective, score in scores.items():
            self.objective_scores[objective].append(score)
        
        return scores
    
    def _evaluate_honesty_with_confidence(self, response: str, context: Dict[str, Any], 
                                        confidence_score: float, uncertainty_score: float) -> float:
        """Evaluate honesty with confidence calibration"""
        
        base_score = self._evaluate_honesty(response, context)
        
        # Reward well-calibrated confidence
        # High confidence + high quality = bonus
        # High confidence + low quality = penalty
        # Low confidence + uncertain content = bonus for honesty
        
        quality_indicators = context.get("quality_score", 0.5)
        
        if confidence_score > 0.8 and quality_indicators > 0.7:
            # Well-calibrated high confidence
            calibration_bonus = 0.2
        elif confidence_score > 0.8 and quality_indicators < 0.4:
            # Overconfident poor response
            calibration_penalty = -0.3
        elif confidence_score < 0.4 and uncertainty_score > 0.6:
            # Appropriately uncertain
            calibration_bonus = 0.15
        else:
            calibration_bonus = 0.0
            calibration_penalty = 0.0
        
        adjusted_score = base_score + calibration_bonus + calibration_penalty
        return max(0.0, min(1.0, adjusted_score))
    
    def _evaluate_accuracy_with_confidence(self, response: str, context: Dict[str, Any], 
                                         confidence_score: float) -> float:
        """Evaluate accuracy with confidence weighting"""
        
        base_accuracy = self._evaluate_accuracy(response, context)
        
        # Weight accuracy by confidence - high confidence claims should be more accurate
        if confidence_score > 0.7:
            # High confidence responses should be penalized more for inaccuracy
            confidence_weight = 1.2
        elif confidence_score < 0.3:
            # Low confidence responses get some leeway
            confidence_weight = 0.8
        else:
            confidence_weight = 1.0
        
        weighted_accuracy = base_accuracy * confidence_weight
        return max(0.0, min(1.0, weighted_accuracy))
    
    def calculate_composite_alignment_score(self, objective_scores: Dict[AlignmentObjective, float]) -> float:
        """Calculate weighted composite alignment score"""
        
        composite_score = 0.0
        
        for objective, score in objective_scores.items():
            weight = self.objectives.get(objective, 0.0)
            composite_score += weight * score
        
        return composite_score
    
    def _evaluate_helpfulness(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate helpfulness of response"""
        
        score = 0.5  # Base score
        
        # Check if response addresses the query
        query = context.get("query", "")
        if query and any(word in response.lower() for word in query.lower().split()):
            score += 0.2
        
        # Check for actionable information
        if any(indicator in response.lower() for indicator in ["how to", "step", "example", "solution"]):
            score += 0.2
        
        # Check response length (not too short, not too long)
        if 50 <= len(response) <= 1000:
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_harmlessness(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate harmlessness of response"""
        
        score = 1.0  # Start with perfect score
        
        # Check for harmful content indicators
        harmful_indicators = [
            "violence", "illegal", "dangerous", "harmful", "toxic",
            "discriminatory", "offensive", "inappropriate"
        ]
        
        for indicator in harmful_indicators:
            if indicator in response.lower():
                score -= 0.2
        
        return max(score, 0.0)
    
    def _evaluate_honesty(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate honesty of response"""
        
        score = 0.5  # Base score
        
        # Check for uncertainty expressions when appropriate
        uncertainty_indicators = ["might", "could", "possibly", "uncertain", "not sure"]
        if any(indicator in response.lower() for indicator in uncertainty_indicators):
            score += 0.2
        
        # Check for factual claims without sources
        if "according to" in response.lower() or "source:" in response.lower():
            score += 0.2
        
        # Penalize overconfident claims
        overconfident_indicators = ["definitely", "certainly", "absolutely", "guaranteed"]
        if any(indicator in response.lower() for indicator in overconfident_indicators):
            score -= 0.1
        
        return max(min(score, 1.0), 0.0)
    
    def _evaluate_efficiency(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate efficiency of response"""
        
        # Based on response time and resource usage
        response_time = context.get("response_time", 1.0)
        token_count = len(response.split())
        
        # Efficiency score based on time and conciseness
        time_score = max(0, 1.0 - (response_time - 1.0) / 10.0)  # Penalize slow responses
        conciseness_score = max(0, 1.0 - (token_count - 100) / 1000.0)  # Penalize very long responses
        
        return (time_score + conciseness_score) / 2.0
    
    def _evaluate_accuracy(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate accuracy of response"""
        
        # This would typically involve fact-checking against knowledge base
        # For now, use simple heuristics
        
        score = 0.5  # Base score
        
        # Check for specific, detailed information
        if any(char.isdigit() for char in response):  # Contains numbers/data
            score += 0.2
        
        # Check for structured information
        if any(marker in response for marker in ["1.", "2.", "•", "-"]):
            score += 0.1
        
        # Check against known facts (simplified)
        known_facts = context.get("known_facts", [])
        if known_facts:
            fact_matches = sum(1 for fact in known_facts if fact.lower() in response.lower())
            score += min(fact_matches * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def get_alignment_statistics(self) -> Dict[str, Any]:
        """Get alignment statistics"""
        
        stats = {}
        
        for objective, scores in self.objective_scores.items():
            if scores:
                stats[objective.value] = {
                    "count": len(scores),
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "weight": self.objectives[objective]
                }
        
        return stats

class ConfidenceRLHFIntegration:
    """Complete integration of confidence metrics with RLHF system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize core components with confidence integration
        self.preference_pipeline = PreferenceDataPipeline()
        self.reward_model = RewardModel(confidence_integration=True)
        self.agentic_rl = OnlineAgenticRL(self.reward_model)
        self.alignment_system = MultiObjectiveAlignment()
        
        # Integration metrics
        self.integration_stats = {
            "confidence_shaped_rewards": 0,
            "high_confidence_actions": 0,
            "low_confidence_explorations": 0,
            "calibration_adjustments": 0
        }
        
        print("🎯 Confidence-RLHF Integration initialized")
        print("   - Confidence-aware reward shaping")
        print("   - Dynamic exploration based on confidence")
        print("   - Calibration-aware alignment evaluation")
    
    def process_research_action(self, state: Dict[str, Any], available_actions: List[str],
                              confidence_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process research action with confidence-aware RLHF"""
        
        # Add confidence metrics to state if provided
        if confidence_metrics:
            state.update({
                "confidence_score": confidence_metrics.get("confidence_score", 0.5),
                "uncertainty_score": confidence_metrics.get("uncertainty_score", 0.5),
                "mean_logprob": confidence_metrics.get("mean_logprob", -1.0),
                "logprob_variance": confidence_metrics.get("variance", 1.0)
            })
        
        # Select action with confidence awareness
        selected_action, action_metadata = self.agentic_rl.select_action(state, available_actions)
        
        # Update integration statistics
        confidence_score = action_metadata.get("confidence_score", 0.5)
        if confidence_score > 0.7:
            self.integration_stats["high_confidence_actions"] += 1
        elif confidence_score < 0.3:
            self.integration_stats["low_confidence_explorations"] += 1
        
        return {
            "selected_action": selected_action,
            "action_metadata": action_metadata,
            "confidence_integration": True,
            "integration_stats": self.integration_stats.copy()
        }
    
    def record_action_outcome(self, action_id: str, outcome: Dict[str, Any],
                            confidence_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Record action outcome with confidence-aware reward shaping"""
        
        # Create reward signals from outcome
        reward_signals = self._create_reward_signals_from_outcome(action_id, outcome)
        
        # Record with confidence integration
        self.agentic_rl.record_reward_signal(action_id, reward_signals, confidence_metrics)
        
        # Update integration statistics
        if confidence_metrics:
            self.integration_stats["confidence_shaped_rewards"] += 1
            
            # Check for calibration adjustments
            base_reward = outcome.get("quality_score", 0.5)
            confidence_score = confidence_metrics.get("confidence_score", 0.5)
            
            if abs(base_reward - confidence_score) > 0.3:  # Poor calibration
                self.integration_stats["calibration_adjustments"] += 1
        
        return {
            "reward_signals_count": len(reward_signals),
            "confidence_shaped": confidence_metrics is not None,
            "integration_stats": self.integration_stats.copy()
        }
    
    def evaluate_response_alignment(self, response: str, context: Dict[str, Any],
                                  confidence_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate response alignment with confidence awareness"""
        
        # Add confidence metrics to context
        if confidence_metrics:
            context["confidence_metrics"] = confidence_metrics
        
        # Evaluate alignment
        alignment_scores = self.alignment_system.evaluate_alignment(response, context)
        
        # Calculate composite score
        composite_score = self.alignment_system.calculate_composite_alignment_score(alignment_scores)
        
        return {
            "alignment_scores": {obj.value: score for obj, score in alignment_scores.items()},
            "composite_score": composite_score,
            "confidence_aware": confidence_metrics is not None,
            "confidence_metrics": confidence_metrics or {}
        }
    
    def collect_preference_with_confidence(self, query: str, response_a: str, response_b: str,
                                         preference: int, confidence_a: Dict[str, Any] = None,
                                         confidence_b: Dict[str, Any] = None) -> str:
        """Collect preference data with confidence information"""
        
        # Create metadata with confidence information
        metadata = {}
        if confidence_a:
            metadata["response_a_confidence"] = confidence_a
        if confidence_b:
            metadata["response_b_confidence"] = confidence_b
        
        # Calculate preference confidence based on response confidences
        if confidence_a and confidence_b:
            conf_a = confidence_a.get("confidence_score", 0.5)
            conf_b = confidence_b.get("confidence_score", 0.5)
            
            # Higher confidence difference = more reliable preference
            confidence_diff = abs(conf_a - conf_b)
            preference_confidence = 0.5 + (confidence_diff * 0.5)
        else:
            preference_confidence = 0.8  # Default confidence
        
        # Collect preference
        preference_id = self.preference_pipeline.collect_preference(
            query=query,
            response_a=response_a,
            response_b=response_b,
            preference=preference,
            preference_type=PreferenceType.HUMAN_FEEDBACK,
            confidence=preference_confidence,
            metadata=metadata
        )
        
        return preference_id
    
    def _create_reward_signals_from_outcome(self, action_id: str, outcome: Dict[str, Any]) -> List[RewardSignal]:
        """Create reward signals from action outcome"""
        
        signals = []
        
        # Quality-based reward
        if "quality_score" in outcome:
            signals.append(RewardSignal(
                signal_id=f"{action_id}_quality",
                action=action_id,
                reward_value=outcome["quality_score"],
                signal_type=RewardSignalType.CORRECTNESS_SCORE,
                context=outcome,
                timestamp=datetime.now(),
                session_id=outcome.get("session_id", "default")
            ))
        
        # Efficiency-based reward
        if "response_time" in outcome:
            # Reward faster responses (inverse relationship)
            efficiency_reward = max(0, 1.0 - (outcome["response_time"] / 10.0))
            signals.append(RewardSignal(
                signal_id=f"{action_id}_efficiency",
                action=action_id,
                reward_value=efficiency_reward,
                signal_type=RewardSignalType.LATENCY_PENALTY,
                context=outcome,
                timestamp=datetime.now(),
                session_id=outcome.get("session_id", "default")
            ))
        
        # Factuality-based reward
        if "factuality_score" in outcome:
            signals.append(RewardSignal(
                signal_id=f"{action_id}_factuality",
                action=action_id,
                reward_value=outcome["factuality_score"],
                signal_type=RewardSignalType.FACTUALITY_SCORE,
                context=outcome,
                timestamp=datetime.now(),
                session_id=outcome.get("session_id", "default")
            ))
        
        return signals
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        
        return {
            "integration_stats": self.integration_stats,
            "reward_model_stats": self.reward_model.reward_shaper.get_confidence_reward_statistics(),
            "rl_stats": self.agentic_rl.get_rl_statistics(),
            "alignment_stats": self.alignment_system.get_alignment_statistics(),
            "preference_stats": self.preference_pipeline.get_preference_statistics()
        }

# Integration function
def integrate_rlhf_agentic_rl():
    """Integration point for RLHF and Agentic RL system with confidence awareness"""
    
    # Initialize confidence-aware integration
    integration = ConfidenceRLHFIntegration()
    
    print("🎯 Stage 5: Confidence-Aware RLHF & Agentic RL integrated")
    print("   Features:")
    print("   - Confidence-aware reward shaping")
    print("   - Dynamic exploration based on uncertainty")
    print("   - Calibration-aware alignment evaluation")
    print("   - Preference collection with confidence weighting")
    
    return integration

def demo_confidence_rlhf_integration():
    """Demonstrate confidence-RLHF integration"""
    
    print("🎯 Confidence-RLHF Integration Demo")
    print("=" * 50)
    
    # Initialize integration
    integration = integrate_rlhf_agentic_rl()
    
    # Demo 1: Confidence-aware action selection
    print("\n📝 Demo 1: Confidence-Aware Action Selection")
    
    state = {
        "query": "How do neural networks learn?",
        "context_size": 1000,
        "user_expertise": "beginner"
    }
    
    confidence_metrics = {
        "confidence_score": 0.8,
        "uncertainty_score": 0.2,
        "mean_logprob": -0.5,
        "variance": 0.1
    }
    
    available_actions = ["detailed_explanation", "simple_overview", "step_by_step_guide"]
    
    result = integration.process_research_action(state, available_actions, confidence_metrics)
    
    print(f"   Selected Action: {result['selected_action']}")
    print(f"   Confidence Score: {result['action_metadata']['confidence_score']:.3f}")
    print(f"   Exploration Rate: {result['action_metadata']['exploration_rate']:.3f}")
    
    # Demo 2: Confidence-aware reward shaping
    print("\n🏆 Demo 2: Confidence-Aware Reward Shaping")
    
    # Simulate action outcome
    action_id = "demo_action_123"
    outcome = {
        "quality_score": 0.7,
        "response_time": 2.5,
        "factuality_score": 0.8,
        "session_id": "demo_session"
    }
    
    reward_result = integration.record_action_outcome(action_id, outcome, confidence_metrics)
    
    print(f"   Reward Signals: {reward_result['reward_signals_count']}")
    print(f"   Confidence Shaped: {reward_result['confidence_shaped']}")
    
    # Demo 3: Confidence-aware alignment evaluation
    print("\n🎯 Demo 3: Confidence-Aware Alignment Evaluation")
    
    response = "Neural networks learn through backpropagation, adjusting weights based on error gradients."
    context = {
        "query": "How do neural networks learn?",
        "response_time": 1.5,
        "quality_score": 0.8
    }
    
    alignment_result = integration.evaluate_response_alignment(response, context, confidence_metrics)
    
    print(f"   Composite Alignment Score: {alignment_result['composite_score']:.3f}")
    print(f"   Confidence-Aware: {alignment_result['confidence_aware']}")
    
    for objective, score in alignment_result['alignment_scores'].items():
        print(f"   {objective}: {score:.3f}")
    
    # Demo 4: Confidence-weighted preference collection
    print("\n📊 Demo 4: Confidence-Weighted Preference Collection")
    
    query = "Explain machine learning algorithms"
    response_a = "Machine learning uses statistical methods to find patterns in data."
    response_b = "ML algorithms automatically improve through experience without explicit programming."
    
    confidence_a = {"confidence_score": 0.6, "uncertainty_score": 0.4}
    confidence_b = {"confidence_score": 0.9, "uncertainty_score": 0.1}
    
    preference_id = integration.collect_preference_with_confidence(
        query, response_a, response_b, preference=1, 
        confidence_a=confidence_a, confidence_b=confidence_b
    )
    
    print(f"   Preference ID: {preference_id}")
    print(f"   Preferred Response B (higher confidence)")
    
    # Demo 5: Integration statistics
    print("\n📈 Demo 5: Integration Statistics")
    
    stats = integration.get_integration_statistics()
    
    print(f"   High Confidence Actions: {stats['integration_stats']['high_confidence_actions']}")
    print(f"   Low Confidence Explorations: {stats['integration_stats']['low_confidence_explorations']}")
    print(f"   Confidence Shaped Rewards: {stats['integration_stats']['confidence_shaped_rewards']}")
    print(f"   Calibration Adjustments: {stats['integration_stats']['calibration_adjustments']}")
    
    if stats['reward_model_stats']['total_adjustments'] > 0:
        print(f"   Average Reward Improvement: {stats['reward_model_stats']['reward_improvement']:.3f}")
        print(f"   Confidence-Reward Correlation: {stats['reward_model_stats']['confidence_correlation']:.3f}")
    
    print(f"\n✅ Confidence-RLHF Integration Demo completed successfully!")
    
    return integration

if __name__ == "__main__":
    # Run comprehensive demo
    integration = demo_confidence_rlhf_integration()
    
    # Additional testing
    print(f"\n🧪 Additional Testing...")
    
    # Test trace selection for RLHF
    test_traces = [
        {"logprobs": [-0.1, -0.2, -0.3], "response": "High quality response", "context_size": 500},
        {"logprobs": [-1.0, -1.2, -1.5], "response": "Medium quality response", "context_size": 300},
        {"logprobs": [-2.0, -2.5, -3.0], "response": "Low quality response", "context_size": 200}
    ]
    
    selected_traces = integration.reward_model.select_traces_for_rlhf(test_traces, top_percent=50)
    print(f"   Selected {len(selected_traces)} out of {len(test_traces)} traces for RLHF training")
    
    # Test confidence-aware reward prediction
    test_state = {
        "response_quality": 0.8,
        "confidence_score": 0.9,
        "uncertainty_score": 0.1,
        "mean_logprob": -0.3,
        "logprob_variance": 0.05
    }
    
    reward_prediction = integration.reward_model.predict_reward_with_confidence(test_state)
    print(f"   Base Reward: {reward_prediction['base_reward']:.3f}")
    print(f"   Shaped Reward: {reward_prediction['shaped_reward']:.3f}")
    print(f"   Confidence Bonus: {reward_prediction['shaping_details']['confidence_bonus']:.3f}")
    
    print(f"\n🎉 All confidence-RLHF integration tests completed!")