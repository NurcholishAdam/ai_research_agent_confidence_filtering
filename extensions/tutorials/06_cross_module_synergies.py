#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 6: Cross-Module Synergies
=================================

This tutorial covers advanced integration patterns including RLHF-tuned diffusion
repair, graph-aware context packing, unified orchestration, and performance
optimization across all extension modules.

Prerequisites:
- Completed Tutorials 1-5
- Understanding of all extension stages

Run this tutorial:
    python extensions/tutorials/06_cross_module_synergies.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add extensions to path
sys.path.append(str(Path(__file__).parent.parent))

from stage_6_cross_module_synergies import (
    UnifiedOrchestrator, RLHFTunedDiffusionRepair, GraphAwareContextPacking,
    SynergyType, SynergyConfiguration, IntegrationLevel
)
from stage_4_diffusion_repair import RuntimeRepairOperator, LanguageType
from stage_5_rlhf_agentic_rl import PreferenceDataPipeline, OnlineAgenticRL, RewardModel
from integration_orchestrator import AIResearchAgentExtensions

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

async def tutorial_cross_module_synergies():
    print_section("Cross-Module Synergies Tutorial")
    
    print("""
Welcome to the Cross-Module Synergies tutorial!

This tutorial covers:
1. RLHF-tuned diffusion repair for enhanced code quality
2. Graph-aware context packing for intelligent information selection
3. Unified orchestration and coordination across all modules
4. Performance optimization through cross-module collaboration
5. Advanced integration patterns and best practices
6. Real-world deployment of synergistic systems

Let's explore how modules work together for superior performance!
    """)
    
    # Initialize all components
    print("\n🔄 Initializing Cross-Module Synergy System...")
    
    # Initialize base components
    repair_operator = RuntimeRepairOperator()
    preference_pipeline = PreferenceDataPipeline()
    reward_model = RewardModel()
    agentic_rl = OnlineAgenticRL(reward_model)
    
    # Initialize synergy components
    rlhf_repair = RLHFTunedDiffusionRepair(repair_operator, preference_pipeline, agentic_rl)
    
    print("✅ Cross-module synergy system initialized")
    
    # Test RLHF-tuned diffusion repair
    print("\n🔧🎯 Testing RLHF-Tuned Diffusion Repair")
    
    broken_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2
"""
    
    repair_result = rlhf_repair.repair_with_rlhf(
        broken_code=broken_code,
        language=LanguageType.PYTHON,
        context={"user_preference": "minimal_changes", "preserve_logic": True}
    )
    
    print(f"✅ RLHF-Enhanced Repair Result:")
    print(f"   Success: {repair_result['success']}")
    if repair_result['success']:
        print(f"   Repaired code: {repair_result['repaired_code']}")
        print(f"   Confidence: {repair_result['confidence']:.3f}")
        print(f"   RLHF ranking: {repair_result['rlhf_ranking']}")
    
    # Test unified orchestrator
    print("\n🎼 Testing Unified Orchestrator")
    
    orchestrator = UnifiedOrchestrator()
    
    # Configure synergies
    synergy_configs = {
        SynergyType.RLHF_DIFFUSION: SynergyConfiguration(
            synergy_type=SynergyType.RLHF_DIFFUSION,
            integration_level=IntegrationLevel.ADVANCED,
            enabled_modules=["rlhf", "diffusion_repair"],
            synergy_parameters={"rlhf_weight": 0.4},
            performance_threshold=0.8,
            auto_optimization=True
        )
    }
    
    orchestrator.initialize_synergies(synergy_configs)
    
    # Test request processing
    test_request = {
        "type": "code_repair",
        "code": broken_code,
        "language": "python",
        "session_id": "synergy_test"
    }
    
    result = await orchestrator.process_request(test_request)
    
    print(f"✅ Orchestrated Request Result:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Synergies used: {result.get('synergies_used', [])}")
    
    print("\n🎉 Cross-Module Synergies Tutorial Complete!")
    print("All modules working together for enhanced performance!")

if __name__ == "__main__":
    asyncio.run(tutorial_cross_module_synergies())