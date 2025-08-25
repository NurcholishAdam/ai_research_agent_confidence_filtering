#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 1: Getting Started with AI Research Agent Extensions
============================================================

This tutorial covers the basics of setting up and using the AI Research Agent Extensions.
We'll walk through initialization, basic usage, and key concepts.

Prerequisites:
- Python 3.8+
- Required packages: torch, transformers, networkx, jinja2, pyyaml

Run this tutorial:
    python extensions/tutorials/01_getting_started.py
"""

import asyncio
import sys
from pathlib import Path

# Add extensions to path
sys.path.append(str(Path(__file__).parent.parent))

from integration_orchestrator import AIResearchAgentExtensions
from stage_1_observability import ModuleType
from stage_2_context_builder import TaskType, MemoryTier

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_step(step: str, description: str):
    """Print a formatted step"""
    print(f"\n🔹 Step {step}: {description}")
    print("-" * 40)

async def tutorial_getting_started():
    """Main tutorial function"""
    
    print_section("AI Research Agent Extensions - Getting Started Tutorial")
    
    print("""
Welcome to the AI Research Agent Extensions tutorial!

This tutorial will guide you through:
1. Initializing the extension system
2. Understanding the 6-stage architecture
3. Basic usage patterns
4. Monitoring and observability
5. Next steps

Let's begin!
    """)
    
    # Step 1: Initialize Extensions
    print_step("1", "Initialize the Extension System")
    
    try:
        # Create extensions instance
        extensions = AIResearchAgentExtensions()
        print("✅ Extensions orchestrator created")
        
        # Initialize all stages
        print("🔄 Initializing all 6 stages...")
        status = await extensions.initialize_all_stages()
        
        print(f"✅ Initialization complete!")
        print(f"   Success rate: {status['success_rate']:.1%}")
        print(f"   Stages initialized: {len(status['initialized_stages'])}/6")
        
        for stage in status['initialized_stages']:
            print(f"   ✓ {stage}")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install torch transformers networkx jinja2 pyyaml")
        return
    
    # Step 2: Understanding the Architecture
    print_step("2", "Understanding the 6-Stage Architecture")
    
    print("""
The extension system consists of 6 integrated stages:

📊 Stage 1: Enhanced Observability
   - A/B testing framework
   - Performance monitoring
   - Event tracking and analytics

🧠 Stage 2: Context Engineering  
   - Memory tiers (short-term, episodic, long-term)
   - Adaptive context packing
   - Prompt template versioning

🕸️ Stage 3: Semantic Graph
   - Multi-source data fusion
   - Hybrid retrieval (semantic + structural)
   - Reasoning write-back

🔧 Stage 4: Diffusion Repair
   - Language-aware code repair
   - Multi-seed voting system
   - Runtime repair with fallbacks

🎯 Stage 5: RLHF & Agentic RL
   - Preference learning pipeline
   - Online reinforcement learning
   - Multi-objective alignment

🎼 Stage 6: Cross-Module Synergies
   - RLHF-tuned diffusion repair
   - Graph-aware context packing
   - Unified orchestration
    """)
    
    # Step 3: Basic Usage - Memory and Context
    print_step("3", "Basic Usage - Memory and Context Management")
    
    if extensions.memory_manager and extensions.context_packer:
        print("🧠 Testing memory management...")
        
        # Store some memories
        memory_id_1 = extensions.memory_manager.store_memory(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            memory_tier=MemoryTier.LONG_TERM,
            relevance_score=0.9,
            metadata={"topic": "AI", "type": "definition"}
        )
        
        memory_id_2 = extensions.memory_manager.store_memory(
            content="Neural networks are inspired by biological neural networks and consist of interconnected nodes.",
            memory_tier=MemoryTier.EPISODIC,
            relevance_score=0.8,
            metadata={"topic": "neural_networks", "type": "explanation"}
        )
        
        print(f"✅ Stored 2 memories: {memory_id_1[:8]}..., {memory_id_2[:8]}...")
        
        # Retrieve memories
        memories = extensions.memory_manager.retrieve_memories(
            query="machine learning neural networks",
            max_items=5
        )
        
        print(f"✅ Retrieved {len(memories)} relevant memories")
        
        # Test context packing
        if memories:
            packing_result = extensions.context_packer.pack_context(
                memory_items=memories,
                task_type=TaskType.RESEARCH
            )
            
            print(f"✅ Context packing complete:")
            print(f"   Items packed: {len(packing_result.packed_items)}")
            print(f"   Total tokens: {packing_result.total_tokens}")
            print(f"   Strategy used: {packing_result.packing_strategy.value}")
            print(f"   Diversity score: {packing_result.diversity_score:.2f}")
    
    # Step 4: Enhanced Request Processing
    print_step("4", "Enhanced Request Processing")
    
    print("🚀 Testing enhanced request processing...")
    
    # Create a research request
    research_request = {
        "type": "research",
        "query": "How do transformer models work in natural language processing?",
        "session_id": "tutorial_session_001"
    }
    
    # Process with all enhancements
    result = await extensions.process_enhanced_request(research_request)
    
    print(f"✅ Request processed successfully: {result['success']}")
    print(f"   Enhancements used: {result['enhancements_used']}")
    print(f"   Processing time: {result['processing_time']:.3f} seconds")
    
    if 'enhanced_context' in result:
        context_info = result['enhanced_context']
        print(f"   Context items: {context_info.get('memory_items_used', 0)}")
        print(f"   Context tokens: {context_info.get('total_tokens', 0)}")
    
    # Step 5: Observability and Monitoring
    print_step("5", "Observability and Monitoring")
    
    if extensions.observability:
        print("📊 Checking system observability...")
        
        # Get analytics dashboard
        dashboard = extensions.observability.get_analytics_dashboard()
        
        print(f"✅ System health metrics:")
        if 'system_health' in dashboard:
            health = dashboard['system_health']
            print(f"   Total events: {health.get('total_events', 0)}")
            print(f"   Active modules: {health.get('active_modules', 0)}")
            print(f"   Error rate: {health.get('error_rate', 0):.1%}")
        
        # Track a custom event
        event_id = extensions.observability.track_event(
            module_type=ModuleType.MULTI_AGENT,
            event_type="tutorial_completion",
            session_id="tutorial_session_001",
            data={"tutorial": "getting_started", "step": 5}
        )
        
        print(f"✅ Custom event tracked: {event_id[:8]}...")
    
    # Step 6: Performance Dashboard
    print_step("6", "Performance Dashboard")
    
    print("📈 Generating performance dashboard...")
    
    dashboard = extensions.get_performance_dashboard()
    
    print(f"✅ Performance Dashboard Generated:")
    
    integration_overview = dashboard.get('integration_overview', {})
    print(f"   Integration success rate: {integration_overview.get('success_rate', 0):.1%}")
    print(f"   Initialized stages: {len(integration_overview.get('initialized_stages', []))}")
    
    if dashboard.get('memory_statistics'):
        print(f"   Memory tiers active: {len(dashboard['memory_statistics'])}")
    
    if dashboard.get('observability_metrics'):
        obs_metrics = dashboard['observability_metrics']
        system_health = obs_metrics.get('system_health', {})
        print(f"   Events tracked: {system_health.get('total_events', 0)}")
    
    # Step 7: Next Steps
    print_step("7", "Next Steps and Advanced Features")
    
    print("""
🎉 Congratulations! You've completed the getting started tutorial.

Here's what you've learned:
✅ How to initialize the extension system
✅ Understanding the 6-stage architecture
✅ Basic memory and context management
✅ Enhanced request processing
✅ Observability and monitoring
✅ Performance dashboard usage

Next tutorials to explore:

📚 Tutorial 2: Advanced Context Engineering
   - Memory tier optimization
   - Custom prompt templates
   - Context compression techniques

🕸️ Tutorial 3: Semantic Graph Operations
   - Multi-source data fusion
   - Hybrid retrieval strategies
   - Reasoning write-back

🔧 Tutorial 4: Code Repair with Diffusion
   - Language-specific repair
   - Multi-seed voting
   - RLHF enhancement

🎯 Tutorial 5: RLHF and Preference Learning
   - Collecting preference data
   - Training reward models
   - Multi-objective alignment

🎼 Tutorial 6: Cross-Module Synergies
   - Unified orchestration
   - Advanced integration patterns
   - Performance optimization

Run the next tutorial:
    python extensions/tutorials/02_advanced_context_engineering.py

For more information:
- Read the API Reference: extensions/API_REFERENCE.md
- Check the main README: extensions/README.md
- Explore example workflows: extensions/examples/
    """)

def main():
    """Run the tutorial"""
    try:
        asyncio.run(tutorial_getting_started())
    except KeyboardInterrupt:
        print("\n\n⚠️ Tutorial interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Tutorial failed: {e}")
        print("💡 Check the troubleshooting guide in README.md")

if __name__ == "__main__":
    main()
