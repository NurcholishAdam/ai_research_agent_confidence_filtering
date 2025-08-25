#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 5: RLHF and Preference Learning
=======================================

This tutorial covers Reinforcement Learning from Human Feedback (RLHF) including
preference data collection, reward model training, online reinforcement learning,
and multi-objective alignment strategies.

Prerequisites:
- Completed Tutorials 1-4
- Basic understanding of machine learning concepts

Run this tutorial:
    python extensions/tutorials/05_rlhf_preference_learning.py
"""

import asyncio
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Add extensions to path
sys.path.append(str(Path(__file__).parent.parent))

from stage_5_rlhf_agentic_rl import (
    PreferenceDataPipeline, RewardModel, OnlineAgenticRL, MultiObjectiveAlignment,
    PreferenceType, RewardSignalType, AlignmentObjective, RewardSignal
)

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_step(step: str, description: str):
    """Print a formatted step"""
    print(f"\n🔹 Step {step}: {description}")
    print("-" * 50)

async def tutorial_rlhf_preference_learning():
    """Main tutorial function for RLHF and preference learning"""
    
    print_section("RLHF and Preference Learning Tutorial")
    
    print("""
Welcome to the RLHF and Preference Learning tutorial!

This tutorial covers:
1. Preference data collection and processing pipelines
2. Neural reward model training and evaluation
3. Online reinforcement learning with policy updates
4. Multi-objective alignment strategies
5. Direct Preference Optimization (DPO) techniques
6. Evaluation and monitoring of RLHF systems
7. Real-world deployment considerations

Let's explore how to build systems that learn from human preferences!
    """)
    
    # Step 1: Preference Data Collection
    print_step("1", "Preference Data Collection and Processing")
    
    preference_pipeline = PreferenceDataPipeline()
    
    print("📊 Setting up preference data collection...")
    
    # Simulate collecting various types of preference data
    preference_scenarios = [
        {
            "query": "How do I implement a binary search algorithm?",
            "response_a": "Use a simple loop to check each element one by one until you find the target.",
            "response_b": "Divide the sorted array in half repeatedly, comparing the target with the middle element to eliminate half the search space each time.",
            "preference": 1,  # Prefer response B
            "type": PreferenceType.HUMAN_FEEDBACK,
            "confidence": 0.9,
            "annotator": "expert_programmer"
        },
        {
            "query": "Explain machine learning in simple terms",
            "response_a": "Machine learning is a complex field involving statistical models, neural networks, gradient descent optimization, and various algorithmic approaches to pattern recognition.",
            "response_b": "Machine learning is like teaching computers to learn patterns from examples, similar to how humans learn from experience.",
            "preference": 1,  # Prefer response B for simplicity
            "type": PreferenceType.HUMAN_FEEDBACK,
            "confidence": 0.8,
            "annotator": "general_user"
        },
        {
            "query": "What's the best way to handle errors in Python?",
            "response_a": "Use try-except blocks to catch and handle specific exceptions gracefully, providing meaningful error messages.",
            "response_b": "Just use print statements to debug and ignore errors when they happen.",
            "preference": 0,  # Prefer response A
            "type": PreferenceType.HUMAN_FEEDBACK,
            "confidence": 0.95,
            "annotator": "senior_developer"
        },
        {
            "query": "How does attention mechanism work in transformers?",
            "response_a": "Attention allows the model to focus on relevant parts of the input sequence when processing each token.",
            "response_b": "Attention computes weighted combinations of input representations using query, key, and value matrices through scaled dot-product operations.",
            "preference": -1,  # Tie - both have merit for different audiences
            "type": PreferenceType.HUMAN_FEEDBACK,
            "confidence": 0.6,
            "annotator": "ml_researcher"
        }
    ]
    
    # Collect preference data
    collected_preferences = []
    
    for scenario in preference_scenarios:
        preference_id = preference_pipeline.collect_preference(
            query=scenario["query"],
            response_a=scenario["response_a"],
            response_b=scenario["response_b"],
            preference=scenario["preference"],
            preference_type=scenario["type"],
            confidence=scenario["confidence"],
            annotator_id=scenario["annotator"],
            metadata={
                "domain": "technical_qa",
                "complexity": "intermediate",
                "timestamp": datetime.now().isoformat()
            }
        )
        collected_preferences.append(preference_id)
        
        print(f"✅ Collected preference: {preference_id[:8]}...")
        print(f"   Query: {scenario['query'][:50]}...")
        print(f"   Preference: {'A' if scenario['preference'] == 0 else 'B' if scenario['preference'] == 1 else 'Tie'}")
        print(f"   Confidence: {scenario['confidence']:.2f}")
        print(f"   Annotator: {scenario['annotator']}")
    
    # Analyze collected preferences
    print(f"\n📈 Preference Collection Analysis:")
    
    stats = preference_pipeline.get_preference_statistics()
    print(f"   Total preferences: {stats['total_preferences']}")
    print(f"   Average confidence: {stats['average_confidence']:.3f}")
    print(f"   Preference distribution: {stats['preference_distribution']}")
    print(f"   Type distribution: {stats['type_distribution']}")
    
    # Get training data with filtering
    training_data = preference_pipeline.get_training_data(
        min_confidence=0.7,
        preference_types=[PreferenceType.HUMAN_FEEDBACK],
        max_age_days=30
    )
    
    print(f"   Training data points: {len(training_data)}")
    print(f"   High-confidence preferences: {len([p for p in training_data if p.confidence > 0.8])}")
    
    # Step 2: Reward Model Training and Evaluation
    print_step("2", "Neural Reward Model Training and Evaluation")
    
    reward_model = RewardModel(input_dim=768, hidden_dim=512)
    
    print("🏆 Testing reward model functionality...")
    
    # Test reward prediction with different state representations
    test_states = [
        {
            "response_length": 150,
            "technical_depth": 0.8,
            "clarity_score": 0.9,
            "completeness": 0.7,
            "user_expertise": 0.6
        },
        {
            "response_length": 50,
            "technical_depth": 0.3,
            "clarity_score": 0.95,
            "completeness": 0.5,
            "user_expertise": 0.2
        },
        {
            "response_length": 300,
            "technical_depth": 0.95,
            "clarity_score": 0.6,
            "completeness": 0.9,
            "user_expertise": 0.9
        }
    ]
    
    print(f"🔍 Testing reward predictions:")
    
    for i, state in enumerate(test_states, 1):
        reward = reward_model.predict_reward(state)
        
        print(f"\n   State {i}:")
        print(f"     Length: {state['response_length']} chars")
        print(f"     Technical depth: {state['technical_depth']:.2f}")
        print(f"     Clarity: {state['clarity_score']:.2f}")
        print(f"     Completeness: {state['completeness']:.2f}")
        print(f"     User expertise: {state['user_expertise']:.2f}")
        print(f"     → Predicted reward: {reward:.3f}")
    
    # Simulate reward model training process
    print(f"\n🎯 Simulating reward model training process:")
    
    # Create synthetic training data from preferences
    training_examples = []
    
    for pref_data in training_data:
        # Create state representations for both responses
        state_a = {
            "response_length": len(pref_data.response_a),
            "technical_depth": 0.5 + np.random.normal(0, 0.1),
            "clarity_score": 0.7 + np.random.normal(0, 0.1),
            "completeness": 0.6 + np.random.normal(0, 0.1),
            "user_expertise": 0.5
        }
        
        state_b = {
            "response_length": len(pref_data.response_b),
            "technical_depth": 0.5 + np.random.normal(0, 0.1),
            "clarity_score": 0.7 + np.random.normal(0, 0.1),
            "completeness": 0.6 + np.random.normal(0, 0.1),
            "user_expertise": 0.5
        }
        
        training_examples.append({
            "state_a": state_a,
            "state_b": state_b,
            "preference": pref_data.preference,
            "confidence": pref_data.confidence
        })
    
    print(f"   Training examples created: {len(training_examples)}")
    print(f"   Average preference confidence: {np.mean([ex['confidence'] for ex in training_examples]):.3f}")
    
    # Simulate training metrics
    training_metrics = {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "final_loss": 0.234,
        "validation_accuracy": 0.847,
        "preference_agreement": 0.823
    }
    
    print(f"   Training completed:")
    for metric, value in training_metrics.items():
        print(f"     {metric}: {value}")
    
    # Step 3: Online Agentic Reinforcement Learning
    print_step("3", "Online Agentic Reinforcement Learning")
    
    agentic_rl = OnlineAgenticRL(reward_model)
    
    print("🤖 Testing online reinforcement learning system...")
    
    # Simulate RL episodes
    rl_episodes = [
        {
            "state": {
                "query_complexity": 0.7,
                "user_expertise": 0.5,
                "context_length": 1500,
                "time_pressure": 0.3,
                "domain": "programming"
            },
            "available_actions": ["detailed_explanation", "quick_summary", "step_by_step", "code_example"]
        },
        {
            "state": {
                "query_complexity": 0.3,
                "user_expertise": 0.2,
                "context_length": 500,
                "time_pressure": 0.8,
                "domain": "general"
            },
            "available_actions": ["simple_explanation", "analogy", "quick_answer"]
        },
        {
            "state": {
                "query_complexity": 0.9,
                "user_expertise": 0.9,
                "context_length": 3000,
                "time_pressure": 0.1,
                "domain": "research"
            },
            "available_actions": ["comprehensive_analysis", "technical_deep_dive", "literature_review", "methodology"]
        }
    ]
    
    episode_results = []
    
    for i, episode in enumerate(rl_episodes, 1):
        print(f"\n🎮 Episode {i}:")
        print(f"   Query complexity: {episode['state']['query_complexity']:.2f}")
        print(f"   User expertise: {episode['state']['user_expertise']:.2f}")
        print(f"   Available actions: {len(episode['available_actions'])}")
        
        # Select action using RL policy
        selected_action, metadata = agentic_rl.select_action(
            state=episode["state"],
            available_actions=episode["available_actions"]
        )
        
        print(f"   → Selected action: {selected_action}")
        print(f"   → Predicted reward: {metadata['predicted_reward']:.3f}")
        
        # Simulate reward signals based on action outcome
        reward_signals = []
        
        # Simulate different types of reward signals
        if selected_action in ["detailed_explanation", "comprehensive_analysis"]:
            # High-quality detailed responses
            reward_signals.extend([
                RewardSignal(
                    signal_id=f"signal_{i}_1",
                    action=selected_action,
                    reward_value=0.8 + np.random.normal(0, 0.1),
                    signal_type=RewardSignalType.CORRECTNESS_SCORE,
                    context=episode["state"],
                    timestamp=datetime.now(),
                    session_id=f"episode_{i}"
                ),
                RewardSignal(
                    signal_id=f"signal_{i}_2",
                    action=selected_action,
                    reward_value=0.7 + np.random.normal(0, 0.1),
                    signal_type=RewardSignalType.FACTUALITY_SCORE,
                    context=episode["state"],
                    timestamp=datetime.now(),
                    session_id=f"episode_{i}"
                )
            ])
        elif selected_action in ["quick_summary", "simple_explanation"]:
            # Quick but potentially less comprehensive responses
            reward_signals.extend([
                RewardSignal(
                    signal_id=f"signal_{i}_3",
                    action=selected_action,
                    reward_value=0.6 + np.random.normal(0, 0.1),
                    signal_type=RewardSignalType.CORRECTNESS_SCORE,
                    context=episode["state"],
                    timestamp=datetime.now(),
                    session_id=f"episode_{i}"
                ),
                RewardSignal(
                    signal_id=f"signal_{i}_4",
                    action=selected_action,
                    reward_value=-0.1,  # Penalty for latency
                    signal_type=RewardSignalType.LATENCY_PENALTY,
                    context=episode["state"],
                    timestamp=datetime.now(),
                    session_id=f"episode_{i}"
                )
            ])
        
        # Record reward signals
        action_id = agentic_rl.action_history[-1].action_id if agentic_rl.action_history else f"action_{i}"
        agentic_rl.record_reward_signal(action_id, reward_signals)
        
        episode_results.append({
            "episode": i,
            "action": selected_action,
            "predicted_reward": metadata["predicted_reward"],
            "actual_rewards": [s.reward_value for s in reward_signals],
            "composite_reward": sum(s.reward_value for s in reward_signals) / len(reward_signals) if reward_signals else 0
        })
        
        print(f"   → Actual rewards: {[f'{r:.3f}' for r in episode_results[-1]['actual_rewards']]}")
        print(f"   → Composite reward: {episode_results[-1]['composite_reward']:.3f}")
    
    # Analyze RL performance
    print(f"\n📊 RL Performance Analysis:")
    
    rl_stats = agentic_rl.get_rl_statistics()
    print(f"   Total actions: {rl_stats['total_actions']}")
    print(f"   Actions with rewards: {rl_stats['actions_with_rewards']}")
    print(f"   Average reward: {rl_stats.get('average_reward', 0):.3f}")
    print(f"   Exploration rate: {rl_stats['exploration_rate']:.3f}")
    
    # Policy improvement analysis
    if len(episode_results) > 1:
        reward_trend = [r["composite_reward"] for r in episode_results]
        print(f"   Reward trend: {' → '.join([f'{r:.3f}' for r in reward_trend])}")
        
        if len(reward_trend) >= 2:
            improvement = reward_trend[-1] - reward_trend[0]
            print(f"   Overall improvement: {improvement:+.3f}")
    
    # Step 4: Multi-Objective Alignment
    print_step("4", "Multi-Objective Alignment Strategies")
    
    alignment_system = MultiObjectiveAlignment()
    
    print("🎯 Testing multi-objective alignment evaluation...")
    
    # Test responses with different characteristics
    test_responses = [
        {
            "name": "Technical Expert Response",
            "response": "The transformer architecture utilizes multi-head self-attention mechanisms with scaled dot-product attention, enabling parallel processing of sequence elements while maintaining positional encoding for sequential information. The model employs residual connections and layer normalization for training stability.",
            "context": {
                "query": "How do transformers work?",
                "user_expertise": 0.9,
                "response_time": 2.5,
                "known_facts": ["Transformers use attention", "Attention enables parallelization"]
            }
        },
        {
            "name": "Beginner-Friendly Response",
            "response": "Transformers are like smart reading systems that can pay attention to different parts of text at the same time. Think of it like having multiple highlighters that can mark important words while reading a book, helping the computer understand what's most relevant.",
            "context": {
                "query": "How do transformers work?",
                "user_expertise": 0.2,
                "response_time": 1.8,
                "known_facts": ["AI helps computers understand text"]
            }
        },
        {
            "name": "Balanced Response",
            "response": "Transformers work by using attention mechanisms to focus on relevant parts of input text. Unlike previous models that process words one by one, transformers can look at all words simultaneously, making them faster and better at understanding context and relationships between words.",
            "context": {
                "query": "How do transformers work?",
                "user_expertise": 0.5,
                "response_time": 2.0,
                "known_facts": ["Transformers are AI models", "Context is important"]
            }
        }
    ]
    
    alignment_results = []
    
    for test_response in test_responses:
        print(f"\n🔍 Evaluating: {test_response['name']}")
        
        # Evaluate alignment across all objectives
        alignment_scores = alignment_system.evaluate_alignment(
            response=test_response["response"],
            context=test_response["context"]
        )
        
        # Calculate composite alignment score
        composite_score = alignment_system.calculate_composite_alignment_score(alignment_scores)
        
        alignment_results.append({
            "name": test_response["name"],
            "scores": alignment_scores,
            "composite": composite_score
        })
        
        print(f"   Alignment Scores:")
        for objective, score in alignment_scores.items():
            print(f"     {objective.value}: {score:.3f}")
        print(f"   → Composite Score: {composite_score:.3f}")
    
    # Compare alignment results
    print(f"\n📈 Alignment Comparison:")
    
    best_overall = max(alignment_results, key=lambda x: x["composite"])
    print(f"   Best overall alignment: {best_overall['name']} ({best_overall['composite']:.3f})")
    
    # Analyze by objective
    for objective in AlignmentObjective:
        scores = [(r["name"], r["scores"][objective]) for r in alignment_results]
        best_for_objective = max(scores, key=lambda x: x[1])
        print(f"   Best {objective.value}: {best_for_objective[0]} ({best_for_objective[1]:.3f})")
    
    # Get alignment statistics
    alignment_stats = alignment_system.get_alignment_statistics()
    
    print(f"\n📊 Alignment System Statistics:")
    for objective, stats in alignment_stats.items():
        print(f"   {objective}:")
        print(f"     Evaluations: {stats['count']}")
        print(f"     Average score: {stats['average']:.3f}")
        print(f"     Score range: {stats['min']:.3f} - {stats['max']:.3f}")
        print(f"     Weight: {stats['weight']:.3f}")
    
    # Step 5: Direct Preference Optimization (DPO)
    print_step("5", "Direct Preference Optimization Techniques")
    
    print("🎯 Implementing Direct Preference Optimization...")
    
    # Create DPO dataset from collected preferences
    dpo_dataset = preference_pipeline.create_dpo_dataset(training_data)
    
    print(f"✅ DPO Dataset created:")
    print(f"   Dataset size: {len(dpo_dataset)}")
    print(f"   Sample data point structure:")
    
    if len(dpo_dataset) > 0:
        sample = dpo_dataset[0]
        print(f"     Query: {sample['query'][:50]}...")
        print(f"     Response A length: {len(sample['response_a'])}")
        print(f"     Response B length: {len(sample['response_b'])}")
        print(f"     Preference: {sample['preference']}")
        print(f"     Confidence: {sample['confidence']:.3f}")
    
    # Simulate DPO training process
    print(f"\n🔄 Simulating DPO training process:")
    
    dpo_training_config = {
        "learning_rate": 0.0001,
        "batch_size": 16,
        "epochs": 20,
        "beta": 0.1,  # DPO temperature parameter
        "reference_model_weight": 0.1
    }
    
    print(f"   Training configuration:")
    for param, value in dpo_training_config.items():
        print(f"     {param}: {value}")
    
    # Simulate training metrics
    dpo_metrics = {
        "initial_loss": 0.693,
        "final_loss": 0.234,
        "preference_accuracy": 0.876,
        "kl_divergence": 0.045,
        "reward_margin": 0.312
    }
    
    print(f"\n   Training results:")
    for metric, value in dpo_metrics.items():
        print(f"     {metric}: {value:.3f}")
    
    # Step 6: Evaluation and Monitoring
    print_step("6", "RLHF System Evaluation and Monitoring")
    
    print("📊 Comprehensive RLHF system evaluation...")
    
    # Evaluation metrics
    evaluation_metrics = {
        "preference_collection": {
            "total_preferences": stats["total_preferences"],
            "collection_rate": f"{stats['total_preferences'] / 7:.1f} per day",  # Assuming 7 days
            "average_confidence": stats["average_confidence"],
            "annotator_agreement": 0.823  # Simulated
        },
        "reward_model": {
            "validation_accuracy": training_metrics["validation_accuracy"],
            "preference_agreement": training_metrics["preference_agreement"],
            "calibration_score": 0.756,  # Simulated
            "robustness_score": 0.689   # Simulated
        },
        "rl_policy": {
            "average_reward": rl_stats.get("average_reward", 0),
            "exploration_rate": rl_stats["exploration_rate"],
            "policy_entropy": 0.234,  # Simulated
            "convergence_rate": 0.045  # Simulated
        },
        "alignment": {
            "composite_alignment": np.mean([r["composite"] for r in alignment_results]),
            "objective_balance": np.std([r["composite"] for r in alignment_results]),
            "consistency_score": 0.812,  # Simulated
            "safety_score": 0.934       # Simulated
        }
    }
    
    print(f"📈 System Evaluation Results:")
    
    for category, metrics in evaluation_metrics.items():
        print(f"\n   {category.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"     {metric.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"     {metric.replace('_', ' ').title()}: {value}")
    
    # Health monitoring
    print(f"\n🏥 System Health Monitoring:")
    
    health_indicators = {
        "preference_quality": "Good" if stats["average_confidence"] > 0.7 else "Needs Attention",
        "reward_model_performance": "Excellent" if training_metrics["validation_accuracy"] > 0.8 else "Good",
        "rl_convergence": "Stable" if rl_stats["exploration_rate"] < 0.2 else "Exploring",
        "alignment_consistency": "High" if evaluation_metrics["alignment"]["consistency_score"] > 0.8 else "Moderate"
    }
    
    for indicator, status in health_indicators.items():
        status_emoji = "✅" if status in ["Good", "Excellent", "Stable", "High"] else "⚠️"
        print(f"   {status_emoji} {indicator.replace('_', ' ').title()}: {status}")
    
    # Step 7: Real-World Deployment Considerations
    print_step("7", "Real-World Deployment and Best Practices")
    
    print("🚀 Real-world deployment considerations for RLHF systems:")
    
    deployment_considerations = {
        "Data Collection": [
            "Implement diverse annotator recruitment strategies",
            "Establish clear annotation guidelines and training",
            "Monitor for annotation bias and quality drift",
            "Implement active learning for efficient data collection",
            "Ensure privacy and ethical data handling"
        ],
        "Model Training": [
            "Use distributed training for large-scale datasets",
            "Implement robust validation and testing procedures",
            "Monitor for reward hacking and specification gaming",
            "Establish model versioning and rollback procedures",
            "Implement continuous learning and adaptation"
        ],
        "Production Deployment": [
            "Implement A/B testing for model updates",
            "Monitor real-time performance and safety metrics",
            "Establish human oversight and intervention protocols",
            "Implement graceful degradation for system failures",
            "Ensure scalability and resource management"
        ],
        "Ethical Considerations": [
            "Implement fairness and bias monitoring",
            "Establish transparency and explainability measures",
            "Ensure user consent and data rights",
            "Monitor for unintended consequences",
            "Implement safety and alignment verification"
        ]
    }
    
    for category, considerations in deployment_considerations.items():
        print(f"\n📋 {category}:")
        for consideration in considerations:
            print(f"   • {consideration}")
    
    # Performance optimization recommendations
    print(f"\n⚡ Performance Optimization Recommendations:")
    
    optimization_tips = [
        "Cache reward model predictions for similar states",
        "Use batch processing for preference data collection",
        "Implement efficient exploration strategies in RL",
        "Optimize alignment evaluation for real-time use",
        "Use model distillation for deployment efficiency",
        "Implement incremental learning for continuous improvement"
    ]
    
    for tip in optimization_tips:
        print(f"   💡 {tip}")
    
    # Monitoring and alerting
    print(f"\n🔔 Monitoring and Alerting Setup:")
    
    monitoring_metrics = [
        "Preference collection rate and quality",
        "Reward model prediction accuracy",
        "RL policy performance and stability",
        "Alignment score distributions",
        "System latency and throughput",
        "Error rates and failure modes"
    ]
    
    for metric in monitoring_metrics:
        print(f"   📊 Monitor: {metric}")
    
    # Final summary
    print_step("8", "Summary and Advanced Topics")
    
    print("🎉 RLHF and Preference Learning Tutorial Complete!")
    
    print(f"\n📚 What you've mastered:")
    print(f"   ✅ Preference data collection and processing pipelines")
    print(f"   ✅ Neural reward model training and evaluation")
    print(f"   ✅ Online reinforcement learning with policy optimization")
    print(f"   ✅ Multi-objective alignment strategies")
    print(f"   ✅ Direct Preference Optimization (DPO) techniques")
    print(f"   ✅ Comprehensive evaluation and monitoring systems")
    print(f"   ✅ Real-world deployment considerations")
    
    print(f"\n🚀 Next Steps:")
    print(f"   • Explore Tutorial 6: Cross-Module Synergies")
    print(f"   • Implement custom alignment objectives")
    print(f"   • Experiment with different RL algorithms")
    print(f"   • Build domain-specific preference models")
    print(f"   • Contribute to RLHF research and development")
    
    print(f"\n🔬 Advanced Research Topics:")
    print(f"   • Constitutional AI and self-supervised alignment")
    print(f"   • Scalable oversight and recursive reward modeling")
    print(f"   • Multi-agent RLHF and cooperative learning")
    print(f"   • Interpretability and explainability in RLHF")
    print(f"   • Robustness and safety in preference learning")
    
    print(f"\n📖 Recommended Reading:")
    print(f"   • 'Training language models to follow instructions with human feedback' (OpenAI)")
    print(f"   • 'Constitutional AI: Harmlessness from AI Feedback' (Anthropic)")
    print(f"   • 'Direct Preference Optimization' (Rafailov et al.)")
    print(f"   • 'Scalable agent alignment via reward modeling' (Leike et al.)")

def main():
    """Run the tutorial"""
    try:
        asyncio.run(tutorial_rlhf_preference_learning())
    except KeyboardInterrupt:
        print("\n\n⚠️ Tutorial interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Tutorial failed: {e}")
        print("💡 Check the troubleshooting guide in README.md")

if __name__ == "__main__":
    main()