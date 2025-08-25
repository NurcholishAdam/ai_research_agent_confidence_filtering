#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 4: Code Repair with Diffusion
======================================

This tutorial covers diffusion-based code repair including language-specific
repair strategies, multi-seed voting, and RLHF enhancement for repair quality.

Prerequisites:
- Completed Tutorials 1-3
- Basic understanding of code repair concepts

Run this tutorial:
    python extensions/tutorials/04_diffusion_code_repair.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add extensions to path
sys.path.append(str(Path(__file__).parent.parent))

from stage_4_diffusion_repair import (
    RuntimeRepairOperator, MultiSeedVotingSystem, LanguageAwareNoiseScheduler,
    DiffusionRepairCore, LanguageType, RepairStrategy
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

async def tutorial_diffusion_code_repair():
    """Main tutorial function for diffusion code repair"""
    
    print_section("Diffusion Code Repair Tutorial")
    
    print("""
Welcome to the Diffusion Code Repair tutorial!

This tutorial covers:
1. Language-aware noise scheduling for different programming languages
2. Multi-seed voting system for repair candidate selection
3. Runtime repair operator with fallback mechanisms
4. Performance analysis and optimization
5. Integration with development workflows
6. RLHF enhancement for repair quality

Let's explore intelligent code repair!
    """)
    
    # Step 1: Language-Aware Noise Scheduling
    print_step("1", "Language-Aware Noise Scheduling")
    
    noise_scheduler = LanguageAwareNoiseScheduler()
    
    print("🔧 Testing noise scheduling for different languages...")
    
    # Test different languages and error types
    test_cases = [
        (LanguageType.PYTHON, "syntax_error"),
        (LanguageType.JAVASCRIPT, "logic_error"),
        (LanguageType.SQL, "syntax_error"),
        (LanguageType.JAVA, "runtime_error")
    ]
    
    for language, error_type in test_cases:
        noise_schedule = noise_scheduler.get_noise_schedule(language, error_type)
        
        print(f"\n📊 {language.value} - {error_type}:")
        print(f"   Noise levels: {len(noise_schedule)} steps")
        print(f"   Start noise: {noise_schedule[0]:.3f}")
        print(f"   End noise: {noise_schedule[-1]:.3f}")
        print(f"   Average noise: {sum(noise_schedule) / len(noise_schedule):.3f}")
    
    # Step 2: Diffusion Repair Core
    print_step("2", "Diffusion Repair Core Engine")
    
    diffusion_core = DiffusionRepairCore()
    
    print("🌊 Testing diffusion repair core...")
    
    # Test code encoding and decoding
    test_code = "def hello_world():\n    print('Hello, World!')"
    
    print(f"Original code: {test_code}")
    
    # Encode code to tensor representation
    code_tensor = diffusion_core.encode_code(test_code)
    print(f"✅ Code encoded to tensor shape: {code_tensor.shape}")
    
    # Add noise to simulate broken code
    noisy_tensor = diffusion_core.add_noise(code_tensor, noise_level=0.3)
    print(f"✅ Noise added (level: 0.3)")
    
    # Perform denoising step
    denoised_tensor = diffusion_core.denoise_step(noisy_tensor, timestep=10)
    print(f"✅ Denoising step completed")
    
    # Generate repair candidates
    broken_code = "def hello_world(\n    print('Hello, World!')"  # Missing closing parenthesis
    
    repair_candidates = diffusion_core.repair_with_diffusion(
        broken_code=broken_code,
        language=LanguageType.PYTHON,
        num_steps=10,
        num_candidates=3
    )
    
    print(f"✅ Generated {len(repair_candidates)} repair candidates")
    for i, candidate in enumerate(repair_candidates):
        print(f"   Candidate {i+1}: {candidate[:50]}...")
    
    # Step 3: Multi-Seed Voting System
    print_step("3", "Multi-Seed Voting System")
    
    voting_system = MultiSeedVotingSystem(num_seeds=5)
    
    print("🗳️ Testing multi-seed voting for repair candidate selection...")
    
    # Generate candidates using multiple seeds
    candidates = voting_system.generate_repair_candidates(
        broken_code=broken_code,
        language=LanguageType.PYTHON,
        diffusion_core=diffusion_core
    )
    
    print(f"✅ Generated {len(candidates)} candidates from {voting_system.num_seeds} seeds")
    
    # Display candidate details
    for i, candidate in enumerate(candidates[:3]):  # Show first 3
        print(f"\n   Candidate {i+1}:")
        print(f"     ID: {candidate.candidate_id}")
        print(f"     Confidence: {candidate.confidence_score:.3f}")
        print(f"     Edit distance: {candidate.edit_distance}")
        print(f"     Strategy: {candidate.repair_strategy.value}")
        print(f"     Code preview: {candidate.repaired_code[:60]}...")
    
    # Vote on candidates
    print(f"\n🗳️ Voting on candidates...")
    
    best_candidate = voting_system.vote_on_candidates(
        candidates=candidates,
        voting_criteria={
            "confidence_weight": 0.4,
            "edit_distance_weight": 0.3,
            "syntax_validity_weight": 0.2,
            "diversity_weight": 0.1
        }
    )
    
    print(f"✅ Best candidate selected:")
    print(f"   ID: {best_candidate.candidate_id}")
    print(f"   Confidence: {best_candidate.confidence_score:.3f}")
    print(f"   Edit distance: {best_candidate.edit_distance}")
    print(f"   Repaired code: {best_candidate.repaired_code}")
    
    # Step 4: Runtime Repair Operator
    print_step("4", "Runtime Repair Operator")
    
    repair_operator = RuntimeRepairOperator()
    
    print("🔧 Testing runtime repair operator with various code examples...")
    
    # Test cases for different languages and error types
    test_repair_cases = [
        {
            "name": "Python Syntax Error",
            "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2",  # Missing closing parenthesis
            "language": LanguageType.PYTHON,
            "error_type": "SyntaxError"
        },
        {
            "name": "JavaScript Missing Brace",
            "code": "function calculateSum(a, b) {\n    return a + b;\n",  # Missing closing brace
            "language": LanguageType.JAVASCRIPT,
            "error_type": "SyntaxError"
        },
        {
            "name": "SQL Incomplete Query",
            "code": "SELECT name, age FROM users WHERE",  # Incomplete WHERE clause
            "language": LanguageType.SQL,
            "error_type": "SyntaxError"
        },
        {
            "name": "JSON Invalid Format",
            "code": '{"name": "John", "age": 30,}',  # Trailing comma
            "language": LanguageType.JSON,
            "error_type": "SyntaxError"
        }
    ]
    
    repair_results = []
    
    for test_case in test_repair_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        print(f"   Original: {test_case['code'][:50]}...")
        
        # Perform repair
        result = repair_operator.repair_code(
            broken_code=test_case["code"],
            language=test_case["language"],
            error_type=test_case["error_type"]
        )
        
        repair_results.append({
            "name": test_case["name"],
            "success": result.success,
            "repair_time": result.repair_time,
            "candidates": len(result.all_candidates),
            "best_candidate": result.best_candidate
        })
        
        if result.success and result.best_candidate:
            print(f"   ✅ Repair successful!")
            print(f"   Repaired: {result.best_candidate.repaired_code[:50]}...")
            print(f"   Confidence: {result.best_candidate.confidence_score:.3f}")
            print(f"   Time: {result.repair_time:.3f}s")
        else:
            print(f"   ❌ Repair failed or used fallback")
            print(f"   Error: {result.error_message}")
    
    # Step 5: Performance Analysis
    print_step("5", "Performance Analysis and Statistics")
    
    # Get repair statistics
    repair_stats = repair_operator.get_repair_statistics()
    
    print("📊 Repair Operation Statistics:")
    print(f"   Total repairs attempted: {repair_stats['total_repairs']}")
    print(f"   Successful repairs: {repair_stats['successful_repairs']}")
    print(f"   Fallback repairs: {repair_stats['fallback_repairs']}")
    print(f"   Failed repairs: {repair_stats['failed_repairs']}")
    print(f"   Success rate: {repair_stats.get('success_rate', 0):.1%}")
    print(f"   Fallback rate: {repair_stats.get('fallback_rate', 0):.1%}")
    
    # Analyze repair results by language
    print(f"\n📈 Repair Results Analysis:")
    
    successful_repairs = [r for r in repair_results if r["success"]]
    failed_repairs = [r for r in repair_results if not r["success"]]
    
    print(f"   Test cases: {len(repair_results)}")
    print(f"   Successful: {len(successful_repairs)}")
    print(f"   Failed: {len(failed_repairs)}")
    
    if successful_repairs:
        avg_time = sum(r["repair_time"] for r in successful_repairs) / len(successful_repairs)
        avg_candidates = sum(r["candidates"] for r in successful_repairs) / len(successful_repairs)
        
        print(f"   Average repair time: {avg_time:.3f}s")
        print(f"   Average candidates generated: {avg_candidates:.1f}")
    
    # Step 6: Advanced Features and Integration
    print_step("6", "Advanced Features and Integration Patterns")
    
    print("🚀 Advanced diffusion repair features:")
    
    # Test batch repair
    print(f"\n🔄 Batch Repair Testing:")
    
    batch_codes = [
        ("def test(", LanguageType.PYTHON),
        ("function test() {", LanguageType.JAVASCRIPT),
        ("SELECT * FROM", LanguageType.SQL)
    ]
    
    batch_results = []
    start_time = datetime.now()
    
    for code, language in batch_codes:
        result = repair_operator.repair_code(code, language)
        batch_results.append(result.success)
    
    batch_time = (datetime.now() - start_time).total_seconds()
    
    print(f"   Batch size: {len(batch_codes)}")
    print(f"   Batch success rate: {sum(batch_results) / len(batch_results):.1%}")
    print(f"   Total batch time: {batch_time:.3f}s")
    print(f"   Average time per repair: {batch_time / len(batch_codes):.3f}s")
    
    # Integration patterns
    print(f"\n🔗 Integration Patterns:")
    
    integration_examples = [
        "IDE Plugin Integration - Real-time repair suggestions",
        "CI/CD Pipeline - Automated code quality improvement",
        "Code Review Tools - Intelligent repair recommendations",
        "Development Workflow - Pre-commit repair hooks",
        "Educational Tools - Learning-oriented repair explanations"
    ]
    
    for example in integration_examples:
        print(f"   • {example}")
    
    # Step 7: Best Practices and Optimization
    print_step("7", "Best Practices and Optimization Tips")
    
    print("💡 Best Practices for Diffusion Code Repair:")
    
    best_practices = [
        {
            "category": "Language Selection",
            "tips": [
                "Use language-specific noise schedules for better results",
                "Adjust confidence thresholds per language complexity",
                "Consider language-specific fallback strategies"
            ]
        },
        {
            "category": "Performance Optimization",
            "tips": [
                "Cache frequently repaired patterns",
                "Use appropriate number of seeds (3-7 optimal)",
                "Implement timeout mechanisms for complex repairs",
                "Monitor memory usage for large codebases"
            ]
        },
        {
            "category": "Quality Assurance",
            "tips": [
                "Validate repairs with syntax checkers",
                "Test repaired code when possible",
                "Collect user feedback for continuous improvement",
                "Monitor repair success rates by error type"
            ]
        },
        {
            "category": "Integration",
            "tips": [
                "Implement graceful fallbacks for critical systems",
                "Provide repair confidence scores to users",
                "Log repair operations for analysis",
                "Support incremental repair for large files"
            ]
        }
    ]
    
    for practice in best_practices:
        print(f"\n📋 {practice['category']}:")
        for tip in practice["tips"]:
            print(f"   • {tip}")
    
    # Performance optimization recommendations
    print(f"\n⚡ Performance Optimization Recommendations:")
    
    current_stats = repair_operator.get_repair_statistics()
    
    if current_stats["total_repairs"] > 0:
        success_rate = current_stats.get("success_rate", 0)
        
        if success_rate < 0.7:
            print(f"   ⚠️ Low success rate ({success_rate:.1%}) - Consider:")
            print(f"     • Increasing number of voting seeds")
            print(f"     • Adjusting confidence thresholds")
            print(f"     • Improving fallback mechanisms")
        elif success_rate > 0.9:
            print(f"   ✅ Excellent success rate ({success_rate:.1%}) - Consider:")
            print(f"     • Reducing computational overhead")
            print(f"     • Optimizing for speed over redundancy")
        else:
            print(f"   👍 Good success rate ({success_rate:.1%}) - System performing well")
    
    # Final summary and next steps
    print_step("8", "Summary and Next Steps")
    
    print("🎉 Diffusion Code Repair Tutorial Complete!")
    
    print(f"\n📚 What you've learned:")
    print(f"   ✅ Language-aware noise scheduling for different programming languages")
    print(f"   ✅ Multi-seed voting system for optimal repair candidate selection")
    print(f"   ✅ Runtime repair operator with comprehensive fallback mechanisms")
    print(f"   ✅ Performance analysis and optimization techniques")
    print(f"   ✅ Integration patterns for development workflows")
    print(f"   ✅ Best practices for production deployment")
    
    print(f"\n🚀 Next Steps:")
    print(f"   • Explore Tutorial 5: RLHF and Preference Learning")
    print(f"   • Experiment with custom repair strategies")
    print(f"   • Integrate with your development environment")
    print(f"   • Contribute repair improvements to the community")
    print(f"   • Build domain-specific repair models")
    
    print(f"\n💡 Advanced Topics to Explore:")
    print(f"   • Custom diffusion model training")
    print(f"   • Language-specific repair pattern learning")
    print(f"   • Integration with static analysis tools")
    print(f"   • Real-time collaborative repair systems")
    print(f"   • Repair quality metrics and evaluation")
    
    print(f"\n🔗 Related Resources:")
    print(f"   • API Reference: extensions/API_REFERENCE.md")
    print(f"   • Integration Examples: extensions/examples/")
    print(f"   • Performance Benchmarks: extensions/benchmarks/")
    print(f"   • Deployment Guide: extensions/DEPLOYMENT_GUIDE.md")

def main():
    """Run the tutorial"""
    try:
        asyncio.run(tutorial_diffusion_code_repair())
    except KeyboardInterrupt:
        print("\n\n⚠️ Tutorial interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Tutorial failed: {e}")
        print("💡 Check the troubleshooting guide in README.md")

if __name__ == "__main__":
    main()