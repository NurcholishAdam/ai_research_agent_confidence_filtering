#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 2: Advanced Context Engineering
========================================

This tutorial covers advanced features of the context engineering system:
- Memory tier optimization and management
- Custom prompt template creation and versioning
- Context compression and adaptive packing strategies
- Performance monitoring and optimization

Prerequisites:
- Completed Tutorial 1: Getting Started
- Understanding of basic extension concepts

Run this tutorial:
    python extensions/tutorials/02_advanced_context_engineering.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add extensions to path
sys.path.append(str(Path(__file__).parent.parent))

from stage_2_context_builder import (
    MemoryTierManager, AdaptiveContextPacker, PromptTemplateManager,
    ContextCompressionEngine, EnhancedContextBuilder,
    MemoryTier, TaskType, ContextPackingStrategy
)
from stage_1_observability import get_observability_collector, ModuleType

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_step(step: str, description: str):
    """Print a formatted step"""
    print(f"\n🔹 Step {step}: {description}")
    print("-" * 50)

async def tutorial_advanced_context_engineering():
    """Main tutorial function for advanced context engineering"""
    
    print_section("Advanced Context Engineering Tutorial")
    
    print("""
Welcome to the Advanced Context Engineering tutorial!

This tutorial covers:
1. Memory tier optimization and strategies
2. Custom prompt template creation and A/B testing
3. Advanced context packing strategies
4. Context compression techniques
5. Performance monitoring and optimization
6. Real-world usage patterns

Let's dive into advanced context management!
    """)
    
    # Step 1: Advanced Memory Tier Management
    print_step("1", "Advanced Memory Tier Management")
    
    # Create memory manager with custom configuration
    custom_tier_limits = {
        MemoryTier.SHORT_TERM: 3000,    # Increased for demo
        MemoryTier.EPISODIC: 6000,      # Increased for demo
        MemoryTier.LONG_TERM: 12000,    # Increased for demo
        MemoryTier.GRAPH_MEMORY: 8000   # Increased for demo
    }
    
    memory_manager = MemoryTierManager(max_tokens_per_tier=custom_tier_limits)
    
    print("🧠 Created memory manager with custom tier limits:")
    for tier, limit in custom_tier_limits.items():
        print(f"   {tier.value}: {limit} tokens")
    
    # Populate memory tiers with diverse content
    print("\n📝 Populating memory tiers with research content...")
    
    # Long-term memories (foundational knowledge)
    long_term_memories = [
        {
            "content": "Transformers revolutionized NLP by introducing the attention mechanism, allowing models to focus on relevant parts of input sequences without recurrent connections.",
            "metadata": {"type": "foundational", "domain": "NLP", "importance": "high"},
            "relevance": 0.95
        },
        {
            "content": "BERT (Bidirectional Encoder Representations from Transformers) uses masked language modeling to learn bidirectional representations, significantly improving performance on downstream tasks.",
            "metadata": {"type": "model_architecture", "domain": "NLP", "year": 2018},
            "relevance": 0.9
        },
        {
            "content": "GPT models use autoregressive generation, predicting the next token based on previous context, enabling powerful text generation capabilities.",
            "metadata": {"type": "model_architecture", "domain": "generation", "year": 2018},
            "relevance": 0.88
        }
    ]
    
    # Store long-term memories
    long_term_ids = []
    for memory in long_term_memories:
        memory_id = memory_manager.store_memory(
            content=memory["content"],
            memory_tier=MemoryTier.LONG_TERM,
            relevance_score=memory["relevance"],
            metadata=memory["metadata"]
        )
        long_term_ids.append(memory_id)
    
    print(f"✅ Stored {len(long_term_ids)} long-term memories")
    
    # Episodic memories (session-specific knowledge)
    episodic_memories = [
        {
            "content": "Recent research shows that scaling transformer models leads to emergent capabilities, with performance improvements following power laws.",
            "metadata": {"type": "recent_finding", "session": "current", "recency": "high"},
            "relevance": 0.85
        },
        {
            "content": "Chain-of-thought prompting enables large language models to perform complex reasoning by breaking down problems into intermediate steps.",
            "metadata": {"type": "technique", "session": "current", "practical": True},
            "relevance": 0.82
        }
    ]
    
    episodic_ids = []
    for memory in episodic_memories:
        memory_id = memory_manager.store_memory(
            content=memory["content"],
            memory_tier=MemoryTier.EPISODIC,
            relevance_score=memory["relevance"],
            metadata=memory["metadata"]
        )
        episodic_ids.append(memory_id)
    
    print(f"✅ Stored {len(episodic_ids)} episodic memories")
    
    # Short-term memories (immediate context)
    short_term_memories = [
        {
            "content": "User is asking about transformer attention mechanisms and their computational complexity.",
            "metadata": {"type": "user_query", "immediate": True, "context": "current_conversation"},
            "relevance": 0.95
        },
        {
            "content": "Previous response covered basic attention concepts, now need to dive deeper into multi-head attention.",
            "metadata": {"type": "conversation_state", "immediate": True, "follow_up": True},
            "relevance": 0.9
        }
    ]
    
    short_term_ids = []
    for memory in short_term_memories:
        memory_id = memory_manager.store_memory(
            content=memory["content"],
            memory_tier=MemoryTier.SHORT_TERM,
            relevance_score=memory["relevance"],
            metadata=memory["metadata"]
        )
        short_term_ids.append(memory_id)
    
    print(f"✅ Stored {len(short_term_ids)} short-term memories")
    
    # Display tier statistics
    tier_stats = memory_manager.get_tier_statistics()
    print(f"\n📊 Memory tier statistics:")
    for tier, stats in tier_stats.items():
        print(f"   {tier}:")
        print(f"     Items: {stats['item_count']}")
        print(f"     Tokens: {stats['total_tokens']}")
        print(f"     Utilization: {stats['token_utilization']:.1%}")
        print(f"     Avg relevance: {stats['avg_relevance']:.2f}")
    
    # Step 2: Advanced Context Packing Strategies
    print_step("2", "Advanced Context Packing Strategies")
    
    context_packer = AdaptiveContextPacker(max_context_tokens=10000)
    
    # Test different packing strategies
    query = "How do multi-head attention mechanisms work in transformers and what are their computational benefits?"
    
    # Retrieve memories for packing
    all_memories = memory_manager.retrieve_memories(
        query=query,
        memory_tiers=[MemoryTier.SHORT_TERM, MemoryTier.EPISODIC, MemoryTier.LONG_TERM],
        max_items=20,
        relevance_threshold=0.3
    )
    
    print(f"🔍 Retrieved {len(all_memories)} memories for query: '{query[:50]}...'")
    
    # Test different packing strategies
    strategies = [
        ContextPackingStrategy.RELEVANCE_FIRST,
        ContextPackingStrategy.RECENCY_FIRST,
        ContextPackingStrategy.DIVERSITY_FIRST,
        ContextPackingStrategy.BALANCED,
        ContextPackingStrategy.ADAPTIVE
    ]
    
    packing_results = {}
    
    for strategy in strategies:
        result = context_packer.pack_context(
            memory_items=all_memories,
            task_type=TaskType.RESEARCH,
            strategy=strategy
        )
        packing_results[strategy] = result
        
        print(f"\n📦 {strategy.value} strategy:")
        print(f"   Items packed: {len(result.packed_items)}")
        print(f"   Total tokens: {result.total_tokens}")
        print(f"   Diversity score: {result.diversity_score:.3f}")
        print(f"   Recency score: {result.recency_score:.3f}")
        print(f"   Relevance score: {result.relevance_score:.3f}")
        print(f"   Compression ratio: {result.compression_ratio:.3f}")
    
    # Analyze strategy performance
    print(f"\n🎯 Strategy Analysis:")
    best_diversity = max(packing_results.items(), key=lambda x: x[1].diversity_score)
    best_relevance = max(packing_results.items(), key=lambda x: x[1].relevance_score)
    best_balanced = max(packing_results.items(), key=lambda x: (x[1].diversity_score + x[1].relevance_score + x[1].recency_score) / 3)
    
    print(f"   Best diversity: {best_diversity[0].value} ({best_diversity[1].diversity_score:.3f})")
    print(f"   Best relevance: {best_relevance[0].value} ({best_relevance[1].relevance_score:.3f})")
    print(f"   Best balanced: {best_balanced[0].value}")
    
    # Step 3: Custom Prompt Template Creation
    print_step("3", "Custom Prompt Template Creation and Versioning")
    
    template_manager = PromptTemplateManager()
    
    print("📝 Creating custom prompt templates...")
    
    # Create a research analysis template
    research_template = """
# Advanced Research Analysis

**Query**: {{ query }}
**Domain**: {{ domain | default("General") }}
**Analysis Type**: {{ analysis_type | default("Comprehensive") }}

## Context Information
{{ context }}

## Analysis Framework

### 1. Key Concepts Identification
Identify and define the core concepts mentioned in the query.

### 2. Current State Analysis
Analyze the current understanding and state of research in this area.

### 3. Technical Deep Dive
{% if technical_depth == "high" %}
Provide detailed technical explanations including:
- Mathematical formulations where applicable
- Algorithmic descriptions
- Implementation considerations
{% else %}
Provide conceptual explanations focusing on:
- High-level principles
- Practical applications
- Benefits and limitations
{% endif %}

### 4. Recent Developments
Highlight recent advances and breakthrough research.

### 5. Future Directions
Discuss potential future research directions and open questions.

## Expected Output Format
- Clear, structured analysis
- Evidence-based insights
- Actionable recommendations
- References to supporting research

**Confidence Level**: Indicate confidence in the analysis
**Limitations**: Note any limitations or assumptions
"""
    
    research_template_id = template_manager.create_template(
        name="advanced_research_analysis",
        template_content=research_template,
        task_types=[TaskType.RESEARCH, TaskType.ANALYSIS],
        version="1.0.0"
    )
    
    print(f"✅ Created research template: {research_template_id[:8]}...")
    
    # Create a code explanation template
    code_template = """
# Code Analysis and Explanation

**Code Context**: {{ code_context }}
**Language**: {{ language | default("Python") }}
**Complexity Level**: {{ complexity | default("Intermediate") }}

## Code Overview
{{ code_content }}

## Analysis Instructions

### 1. Code Structure Analysis
- Break down the code into logical components
- Identify key functions, classes, and modules
- Analyze the overall architecture

### 2. Functionality Explanation
{% if explain_line_by_line %}
Provide line-by-line explanation of the code functionality.
{% else %}
Provide high-level explanation of what the code does.
{% endif %}

### 3. Best Practices Assessment
- Code quality and readability
- Performance considerations
- Security implications
- Maintainability factors

### 4. Improvement Suggestions
Suggest specific improvements for:
- Code efficiency
- Readability
- Error handling
- Documentation

## Output Requirements
- Clear, technical explanations
- Code examples where helpful
- Performance metrics if applicable
- Testing recommendations
"""
    
    code_template_id = template_manager.create_template(
        name="code_analysis_explanation",
        template_content=code_template,
        task_types=[TaskType.CODE_REPAIR, TaskType.ANALYSIS],
        version="1.0.0"
    )
    
    print(f"✅ Created code template: {code_template_id[:8]}...")
    
    # Test template rendering
    print(f"\n🎨 Testing template rendering...")
    
    # Get the best context packing result for rendering
    best_packing = packing_results[ContextPackingStrategy.ADAPTIVE]
    context_content = "\n\n".join([item.content for item in best_packing.packed_items[:5]])
    
    rendered_research = template_manager.render_template(
        research_template_id,
        {
            "query": query,
            "domain": "Natural Language Processing",
            "analysis_type": "Technical Deep Dive",
            "technical_depth": "high",
            "context": context_content
        }
    )
    
    print(f"✅ Rendered research template ({len(rendered_research)} characters)")
    print(f"   Preview: {rendered_research[:200]}...")
    
    # Create template variant for A/B testing
    print(f"\n🧪 Creating template variant for A/B testing...")
    
    variant_modifications = {
        "content_replacement": {
            "### 1. Key Concepts Identification": "### 1. Fundamental Concepts Analysis",
            "### 2. Current State Analysis": "### 2. State-of-the-Art Review"
        }
    }
    
    variant_id = template_manager.create_template_variant(
        base_template_id=research_template_id,
        modifications=variant_modifications,
        version="1.1.0"
    )
    
    print(f"✅ Created template variant: {variant_id[:8]}...")
    
    # Step 4: Context Compression Techniques
    print_step("4", "Context Compression Techniques")
    
    compression_engine = ContextCompressionEngine()
    
    print("🗜️ Testing context compression...")
    
    # Create a long context for compression testing
    long_context = """
    Transformer models have revolutionized the field of natural language processing through their innovative attention mechanism. The attention mechanism allows the model to focus on different parts of the input sequence when processing each element, eliminating the need for recurrent connections that were prevalent in earlier architectures like RNNs and LSTMs.
    
    The multi-head attention mechanism is a key component of transformers. It works by computing attention weights for different representation subspaces simultaneously. Each attention head learns to focus on different types of relationships in the data. For example, one head might focus on syntactic relationships while another focuses on semantic relationships.
    
    The computational complexity of attention mechanisms is quadratic with respect to sequence length, which can be problematic for very long sequences. However, various optimization techniques have been developed to address this limitation, including sparse attention patterns, linear attention approximations, and hierarchical attention structures.
    
    Recent research has shown that the scaling properties of transformer models follow predictable patterns. As model size increases, performance improvements follow power law relationships. This has led to the development of increasingly large models like GPT-3, GPT-4, and PaLM, which demonstrate emergent capabilities not seen in smaller models.
    
    The training process for large transformer models involves several key techniques. Pre-training on large corpora of text using self-supervised objectives like masked language modeling or autoregressive generation provides the model with broad knowledge. Fine-tuning on specific tasks then adapts this knowledge to particular applications.
    """
    
    # Test different compression ratios
    compression_ratios = [0.3, 0.5, 0.7]
    
    for ratio in compression_ratios:
        compressed_content, metrics = compression_engine.compress_context(
            content=long_context,
            target_ratio=ratio,
            preserve_structure=True
        )
        
        print(f"\n📊 Compression ratio {ratio}:")
        print(f"   Original length: {metrics['original_length']} chars")
        print(f"   Compressed length: {metrics['compressed_length']} chars")
        print(f"   Actual ratio: {metrics['compression_ratio']:.3f}")
        print(f"   Sentences kept: {metrics['sentences_kept']}/{metrics['sentences_original']}")
        print(f"   Preview: {compressed_content[:150]}...")
    
    # Step 5: Enhanced Context Builder Integration
    print_step("5", "Enhanced Context Builder Integration")
    
    enhanced_builder = EnhancedContextBuilder(max_context_tokens=12000)
    
    print("🏗️ Testing enhanced context builder...")
    
    # Build comprehensive context
    context_result = enhanced_builder.build_context(
        query=query,
        task_type=TaskType.RESEARCH,
        max_tokens=10000,
        compression_ratio=0.6
    )
    
    print(f"✅ Enhanced context built:")
    print(f"   Context length: {len(context_result['context_content'])} chars")
    print(f"   Memory items used: {context_result['memory_items_used']}")
    print(f"   Total tokens: {context_result['total_tokens']}")
    print(f"   Template used: {context_result['template_used']}")
    print(f"   Compression ratio: {context_result['compression_metrics']['compression_ratio']:.3f}")
    
    if context_result['rendered_prompt']:
        print(f"   Rendered prompt length: {len(context_result['rendered_prompt'])} chars")
    
    # Store the interaction for future use
    enhanced_builder.store_interaction(
        query=query,
        response="Detailed explanation of multi-head attention mechanisms...",
        task_type=TaskType.RESEARCH,
        relevance_score=0.9,
        metadata={"tutorial": "advanced_context_engineering", "step": 5}
    )
    
    print(f"✅ Interaction stored for future context building")
    
    # Step 6: Performance Monitoring and Optimization
    print_step("6", "Performance Monitoring and Optimization")
    
    # Get system analytics
    system_analytics = enhanced_builder.get_system_analytics()
    
    print(f"📈 System Analytics:")
    print(f"   Memory tiers: {len(system_analytics['memory_tiers'])} active")
    
    for tier, stats in system_analytics['memory_tiers'].items():
        print(f"     {tier}: {stats['item_count']} items, {stats['token_utilization']:.1%} utilization")
    
    context_stats = system_analytics['context_packing']
    print(f"   Context packing operations: {context_stats['total_packings']}")
    print(f"   Average compression ratio: {context_stats['avg_compression_ratio']:.3f}")
    
    template_stats = system_analytics['templates']
    print(f"   Templates: {template_stats['total_templates']} total, {template_stats['active_templates']} active")
    
    compression_stats = system_analytics['compression']
    print(f"   Compression operations: {compression_stats['total_compressions']}")
    print(f"   Average compression ratio: {compression_stats['avg_compression_ratio']:.3f}")
    
    # Test memory promotion based on access patterns
    print(f"\n🔄 Testing memory promotion...")
    
    # Simulate accessing a memory multiple times
    test_memory_id = episodic_ids[0]
    for _ in range(5):
        memory_manager.retrieve_memories(
            query="recent research scaling transformer",
            memory_tiers=[MemoryTier.EPISODIC],
            max_items=1
        )
    
    # Promote frequently accessed memory
    promotion_success = memory_manager.promote_memory(
        memory_id=test_memory_id,
        target_tier=MemoryTier.LONG_TERM
    )
    
    print(f"✅ Memory promotion {'successful' if promotion_success else 'failed'}")
    
    # Step 7: Real-World Usage Patterns
    print_step("7", "Real-World Usage Patterns and Best Practices")
    
    print("""
🎯 Real-World Usage Patterns:

1. **Adaptive Memory Management**
   - Start with default tier limits
   - Monitor utilization and adjust based on usage patterns
   - Promote frequently accessed memories automatically
   - Archive old, low-relevance memories

2. **Context Packing Optimization**
   - Use ADAPTIVE strategy for general use
   - Use RELEVANCE_FIRST for Q&A tasks
   - Use DIVERSITY_FIRST for creative tasks
   - Use RECENCY_FIRST for conversational contexts

3. **Template Management**
   - Create domain-specific templates
   - Version templates for A/B testing
   - Monitor template performance metrics
   - Retire low-performing templates

4. **Compression Strategies**
   - Use compression for long contexts (>8000 tokens)
   - Preserve structure for technical content
   - Higher compression ratios for summarization tasks
   - Lower compression ratios for detailed analysis

5. **Performance Monitoring**
   - Track memory utilization across tiers
   - Monitor context packing efficiency
   - Measure template rendering performance
   - Optimize based on usage patterns
    """)
    
    # Final recommendations
    print(f"\n💡 Optimization Recommendations:")
    
    # Analyze current performance
    total_memories = sum(stats['item_count'] for stats in system_analytics['memory_tiers'].values())
    avg_utilization = sum(stats['token_utilization'] for stats in system_analytics['memory_tiers'].values()) / len(system_analytics['memory_tiers'])
    
    if avg_utilization > 0.8:
        print(f"   ⚠️ High memory utilization ({avg_utilization:.1%}) - consider increasing tier limits")
    elif avg_utilization < 0.3:
        print(f"   💡 Low memory utilization ({avg_utilization:.1%}) - consider decreasing tier limits")
    else:
        print(f"   ✅ Good memory utilization ({avg_utilization:.1%})")
    
    if context_stats['avg_compression_ratio'] > 0.8:
        print(f"   💡 Low compression efficiency - consider more aggressive compression")
    else:
        print(f"   ✅ Good compression efficiency ({context_stats['avg_compression_ratio']:.3f})")
    
    print(f"""
🎉 Advanced Context Engineering Tutorial Complete!

You've learned:
✅ Advanced memory tier management and optimization
✅ Custom prompt template creation and versioning
✅ Context packing strategies and performance analysis
✅ Context compression techniques and ratios
✅ Enhanced context builder integration
✅ Performance monitoring and optimization
✅ Real-world usage patterns and best practices

Next Steps:
- Explore Tutorial 3: Semantic Graph Operations
- Experiment with different template designs
- Monitor and optimize your specific use cases
- Integrate with your research workflows

For more advanced features, check out the API Reference and examples!
    """)

def main():
    """Run the tutorial"""
    try:
        asyncio.run(tutorial_advanced_context_engineering())
    except KeyboardInterrupt:
        print("\n\n⚠️ Tutorial interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Tutorial failed: {e}")
        print("💡 Check the troubleshooting guide in README.md")

if __name__ == "__main__":
    main()