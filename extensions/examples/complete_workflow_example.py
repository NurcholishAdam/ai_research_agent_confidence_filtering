#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Workflow Example - AI Research Agent Extensions
=======================================================

This example demonstrates a complete end-to-end workflow using all 6 stages
of the AI Research Agent Extensions system. It simulates a realistic research
scenario with multi-source data integration, intelligent context management,
and continuous learning through RLHF.

Scenario: Building an AI Research Assistant for Academic Literature Review

Usage:
    python extensions/examples/complete_workflow_example.py
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add extensions to path
sys.path.append(str(Path(__file__).parent.parent))

from integration_orchestrator import AIResearchAgentExtensions
from stage_1_observability import ModuleType
from stage_2_context_builder import TaskType, MemoryTier
from stage_3_semantic_graph import NodeType, EdgeType, SourceType
from stage_4_diffusion_repair import LanguageType
from stage_5_rlhf_agentic_rl import PreferenceType, AlignmentObjective

class AcademicResearchAssistant:
    """Complete academic research assistant using all extension stages"""
    
    def __init__(self):
        self.extensions = None
        self.session_id = f"research_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.research_context = {
            "domain": "artificial_intelligence",
            "focus_areas": ["machine_learning", "natural_language_processing", "computer_vision"],
            "research_goals": ["literature_review", "trend_analysis", "gap_identification"]
        }
        
    async def initialize(self):
        """Initialize the research assistant"""
        print("🚀 Initializing Academic Research Assistant")
        print("=" * 60)
        
        # Initialize extensions
        self.extensions = AIResearchAgentExtensions()
        status = await self.extensions.initialize_all_stages()
        
        print(f"✅ Extensions initialized: {status['success_rate']:.1%} success rate")
        print(f"📊 Active stages: {len(status['initialized_stages'])}/6")
        
        # Configure for academic research
        await self._configure_for_research()
        
        return status['success_rate'] > 0.8
    
    async def _configure_for_research(self):
        """Configure system for academic research tasks"""
        
        # Configure memory tiers for research workflow
        if self.extensions.memory_manager:
            # Store research methodology in long-term memory
            self.extensions.memory_manager.store_memory(
                content="Systematic literature review methodology: 1) Define research questions, 2) Search strategy, 3) Study selection, 4) Data extraction, 5) Quality assessment, 6) Synthesis",
                memory_tier=MemoryTier.LONG_TERM,
                relevance_score=0.95,
                metadata={"type": "methodology", "domain": "research"}
            )
            
            # Store domain knowledge
            domain_knowledge = [
                "Machine learning is a subset of AI that enables systems to learn from data",
                "Natural language processing focuses on interaction between computers and human language",
                "Computer vision enables machines to interpret and understand visual information",
                "Deep learning uses neural networks with multiple layers for complex pattern recognition"
            ]
            
            for knowledge in domain_knowledge:
                self.extensions.memory_manager.store_memory(
                    content=knowledge,
                    memory_tier=MemoryTier.LONG_TERM,
                    relevance_score=0.8,
                    metadata={"type": "domain_knowledge", "domain": "AI"}
                )
        
        # Configure semantic graph with research ontology
        if self.extensions.graph_manager:
            # Add key concepts
            concepts = [
                ("Artificial Intelligence", "The simulation of human intelligence in machines"),
                ("Machine Learning", "A subset of AI that enables systems to learn from data"),
                ("Deep Learning", "ML technique using neural networks with multiple layers"),
                ("Natural Language Processing", "AI field focused on language understanding"),
                ("Computer Vision", "AI field focused on visual perception and understanding")
            ]
            
            concept_nodes = {}
            for concept, definition in concepts:
                node_id = self.extensions.graph_manager.add_node(
                    content=definition,
                    node_type=NodeType.CONCEPT,
                    source_type=SourceType.INTERNAL,
                    title=concept,
                    importance_score=0.9,
                    tags=["AI", "concept", "definition"]
                )
                concept_nodes[concept] = node_id
            
            # Add relationships
            relationships = [
                ("Machine Learning", "Artificial Intelligence", EdgeType.PART_OF),
                ("Deep Learning", "Machine Learning", EdgeType.PART_OF),
                ("Natural Language Processing", "Artificial Intelligence", EdgeType.PART_OF),
                ("Computer Vision", "Artificial Intelligence", EdgeType.PART_OF)
            ]
            
            for source_concept, target_concept, relation_type in relationships:
                if source_concept in concept_nodes and target_concept in concept_nodes:
                    self.extensions.graph_manager.add_edge(
                        source_node=concept_nodes[source_concept],
                        target_node=concept_nodes[target_concept],
                        edge_type=relation_type,
                        confidence=0.9
                    )
        
        print("⚙️ System configured for academic research")
    
    async def conduct_literature_review(self, research_query: str) -> Dict[str, Any]:
        """Conduct a comprehensive literature review"""
        
        print(f"\n📚 Conducting Literature Review")
        print(f"Query: {research_query}")
        print("-" * 50)
        
        # Stage 1: Track research session
        if self.extensions.observability:
            self.extensions.observability.track_event(
                module_type=ModuleType.MULTI_AGENT,
                event_type="literature_review_start",
                session_id=self.session_id,
                data={"query": research_query, "domain": self.research_context["domain"]}
            )
        
        # Stage 2: Build research context
        research_context = await self._build_research_context(research_query)
        
        # Stage 3: Query semantic graph for related concepts
        graph_insights = await self._analyze_with_semantic_graph(research_query)
        
        # Stage 4: Process any code examples (if present in literature)
        code_analysis = await self._analyze_code_examples(research_query)
        
        # Stage 5: Apply RLHF for quality assessment
        quality_assessment = await self._assess_research_quality(research_query, research_context)
        
        # Stage 6: Synthesize using cross-module synergies
        synthesis = await self._synthesize_research_findings(
            research_query, research_context, graph_insights, quality_assessment
        )
        
        # Compile comprehensive results
        results = {
            "query": research_query,
            "context": research_context,
            "graph_insights": graph_insights,
            "code_analysis": code_analysis,
            "quality_assessment": quality_assessment,
            "synthesis": synthesis,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        }
        
        # Store results for future reference
        await self._store_research_results(results)
        
        return results
    
    async def _build_research_context(self, query: str) -> Dict[str, Any]:
        """Build intelligent research context using Stage 2"""
        
        print("🧠 Building research context...")
        
        if not self.extensions.memory_manager or not self.extensions.context_packer:
            return {"error": "Context engineering not available"}
        
        # Retrieve relevant memories
        memories = self.extensions.memory_manager.retrieve_memories(
            query=query,
            memory_tiers=[MemoryTier.LONG_TERM, MemoryTier.EPISODIC],
            max_items=15,
            relevance_threshold=0.3
        )
        
        # Pack context optimally
        packing_result = self.extensions.context_packer.pack_context(
            memory_items=memories,
            task_type=TaskType.RESEARCH
        )
        
        # Get appropriate template
        template = None
        if self.extensions.prompt_manager:
            template = self.extensions.prompt_manager.get_template(TaskType.RESEARCH)
        
        context_info = {
            "memories_retrieved": len(memories),
            "context_items": len(packing_result.packed_items),
            "total_tokens": packing_result.total_tokens,
            "packing_strategy": packing_result.packing_strategy.value,
            "diversity_score": packing_result.diversity_score,
            "relevance_score": packing_result.relevance_score,
            "template_used": template.name if template else None
        }
        
        print(f"   ✅ Context built: {context_info['context_items']} items, {context_info['total_tokens']} tokens")
        return context_info
    
    async def _analyze_with_semantic_graph(self, query: str) -> Dict[str, Any]:
        """Analyze query using semantic graph (Stage 3)"""
        
        print("🕸️ Analyzing with semantic graph...")
        
        if not self.extensions.graph_manager:
            return {"error": "Semantic graph not available"}
        
        # Perform hybrid retrieval
        retrieval_results = self.extensions.graph_manager.hybrid_retrieval(
            query=query,
            retrieval_types=["semantic", "structural", "path_constrained"],
            max_nodes=10
        )
        
        # Analyze graph structure
        graph_stats = self.extensions.graph_manager.get_graph_statistics()
        
        # Find related concepts
        related_concepts = []
        for node in retrieval_results.nodes[:5]:
            if node.node_type == NodeType.CONCEPT:
                neighbors = self.extensions.graph_manager.get_node_neighbors(
                    node.node_id, max_depth=2
                )
                related_concepts.append({
                    "concept": node.title or node.content[:50],
                    "relevance": retrieval_results.relevance_scores.get(node.node_id, 0),
                    "connections": len(neighbors)
                })
        
        graph_insights = {
            "nodes_found": len(retrieval_results.nodes),
            "edges_found": len(retrieval_results.edges),
            "paths_found": len(retrieval_results.paths),
            "related_concepts": related_concepts,
            "graph_statistics": {
                "total_nodes": graph_stats["nodes"]["total"],
                "total_edges": graph_stats["edges"]["total"],
                "connectivity_density": graph_stats["connectivity"]["density"]
            }
        }
        
        print(f"   ✅ Graph analysis: {graph_insights['nodes_found']} nodes, {len(related_concepts)} concepts")
        return graph_insights
    
    async def _analyze_code_examples(self, query: str) -> Dict[str, Any]:
        """Analyze code examples using diffusion repair (Stage 4)"""
        
        print("🔧 Analyzing code examples...")
        
        if not self.extensions.diffusion_repair:
            return {"error": "Diffusion repair not available"}
        
        # Simulate finding code examples in literature
        example_codes = [
            {
                "code": "import numpy as np\ndef sigmoid(x):\n    return 1 / (1 + np.exp(-x)",  # Missing closing parenthesis
                "language": LanguageType.PYTHON,
                "context": "Neural network activation function"
            },
            {
                "code": "function softmax(x) {\n    const exp_x = x.map(Math.exp);\n    const sum_exp = exp_x.reduce((a, b) => a + b, 0);\n    return exp_x.map(val => val / sum_exp;\n}",  # Missing closing parenthesis
                "language": LanguageType.JAVASCRIPT,
                "context": "Softmax function implementation"
            }
        ]
        
        code_analysis_results = []
        
        for example in example_codes:
            # Attempt to repair code
            repair_result = self.extensions.diffusion_repair.repair_code(
                broken_code=example["code"],
                language=example["language"],
                error_type="SyntaxError"
            )
            
            analysis = {
                "original_code": example["code"][:100] + "...",
                "context": example["context"],
                "language": example["language"].value,
                "repair_successful": repair_result.success,
                "repair_confidence": repair_result.best_candidate.confidence_score if repair_result.best_candidate else 0,
                "repair_time": repair_result.repair_time
            }
            
            if repair_result.success and repair_result.best_candidate:
                analysis["repaired_code"] = repair_result.best_candidate.repaired_code[:100] + "..."
                analysis["edit_distance"] = repair_result.best_candidate.edit_distance
            
            code_analysis_results.append(analysis)
        
        code_analysis = {
            "examples_analyzed": len(example_codes),
            "successful_repairs": sum(1 for r in code_analysis_results if r["repair_successful"]),
            "average_confidence": sum(r["repair_confidence"] for r in code_analysis_results) / len(code_analysis_results),
            "results": code_analysis_results
        }
        
        print(f"   ✅ Code analysis: {code_analysis['successful_repairs']}/{code_analysis['examples_analyzed']} repairs successful")
        return code_analysis
    
    async def _assess_research_quality(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess research quality using RLHF (Stage 5)"""
        
        print("🎯 Assessing research quality...")
        
        if not self.extensions.rlhf_components:
            return {"error": "RLHF components not available"}
        
        alignment_system = self.extensions.rlhf_components["alignment_system"]
        
        # Simulate research response for quality assessment
        research_response = f"""
        Based on the literature review for "{query}", the current state of research shows:
        
        1. Significant advances in transformer architectures and attention mechanisms
        2. Growing focus on efficiency and scalability in large language models
        3. Emerging trends in multimodal learning and cross-domain applications
        4. Ongoing challenges in interpretability and bias mitigation
        
        Key findings include improved performance on benchmark tasks, novel architectural innovations,
        and increased practical applications across various domains.
        """
        
        # Evaluate alignment across multiple objectives
        alignment_scores = alignment_system.evaluate_alignment(
            response=research_response,
            context={
                "query": query,
                "response_time": 2.5,
                "context_items": context.get("context_items", 0),
                "known_facts": ["AI research is rapidly evolving", "Transformers are state-of-the-art"]
            }
        )
        
        # Calculate composite alignment score
        composite_score = alignment_system.calculate_composite_alignment_score(alignment_scores)
        
        # Simulate preference collection for continuous improvement
        preference_pipeline = self.extensions.rlhf_components["preference_pipeline"]
        
        # Collect synthetic preference data
        preference_id = preference_pipeline.collect_preference(
            query=query,
            response_a=research_response,
            response_b="Alternative research summary with different focus...",
            preference=0,  # Prefer response A
            preference_type=PreferenceType.AUTOMATED_METRIC,
            confidence=0.8,
            metadata={"research_domain": self.research_context["domain"]}
        )
        
        quality_assessment = {
            "alignment_scores": {obj.value: score for obj, score in alignment_scores.items()},
            "composite_alignment": composite_score,
            "preference_collected": preference_id is not None,
            "quality_grade": self._calculate_quality_grade(composite_score),
            "recommendations": self._generate_quality_recommendations(alignment_scores)
        }
        
        print(f"   ✅ Quality assessment: {quality_assessment['quality_grade']} grade, {composite_score:.2f} alignment")
        return quality_assessment
    
    async def _synthesize_research_findings(self, query: str, context: Dict[str, Any], 
                                          graph_insights: Dict[str, Any], 
                                          quality_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize findings using cross-module synergies (Stage 6)"""
        
        print("🎼 Synthesizing research findings...")
        
        if not self.extensions.synergy_orchestrator:
            return {"error": "Synergy orchestrator not available"}
        
        # Create synthesis request
        synthesis_request = {
            "type": "research_synthesis",
            "query": query,
            "context": context,
            "graph_insights": graph_insights,
            "quality_assessment": quality_assessment,
            "session_id": self.session_id
        }
        
        # Process with unified orchestrator
        synthesis_result = await self.extensions.synergy_orchestrator.process_request(synthesis_request)
        
        # Generate comprehensive synthesis
        synthesis = {
            "research_summary": self._generate_research_summary(query, context, graph_insights),
            "key_findings": self._extract_key_findings(graph_insights, quality_assessment),
            "research_gaps": self._identify_research_gaps(graph_insights),
            "future_directions": self._suggest_future_directions(query, graph_insights),
            "confidence_score": quality_assessment.get("composite_alignment", 0.5),
            "synergies_used": synthesis_result.get("synergies_used", []),
            "processing_metadata": synthesis_result.get("metadata", {})
        }
        
        print(f"   ✅ Synthesis complete: {len(synthesis['key_findings'])} findings, {len(synthesis['research_gaps'])} gaps identified")
        return synthesis
    
    async def _store_research_results(self, results: Dict[str, Any]):
        """Store research results for future reference"""
        
        print("💾 Storing research results...")
        
        # Store in memory for future sessions
        if self.extensions.memory_manager:
            summary = f"Research on '{results['query']}': {len(results['synthesis']['key_findings'])} key findings identified"
            
            self.extensions.memory_manager.store_memory(
                content=summary,
                memory_tier=MemoryTier.EPISODIC,
                relevance_score=0.9,
                metadata={
                    "type": "research_session",
                    "query": results["query"],
                    "session_id": self.session_id,
                    "findings_count": len(results["synthesis"]["key_findings"])
                }
            )
        
        # Add to semantic graph
        if self.extensions.graph_manager:
            # Create research session node
            session_node_id = self.extensions.graph_manager.add_node(
                content=f"Research session: {results['query']}",
                node_type=NodeType.RESEARCH_FINDING,
                source_type=SourceType.INTERNAL,
                title=f"Research: {results['query'][:50]}",
                metadata={
                    "session_id": self.session_id,
                    "timestamp": results["timestamp"],
                    "quality_score": results["quality_assessment"]["composite_alignment"]
                }
            )
            
            # Link to related concepts
            for concept_info in results["graph_insights"]["related_concepts"]:
                # Find existing concept nodes and create relationships
                pass  # Implementation would link to existing concept nodes
        
        print("   ✅ Results stored in memory and graph")
    
    def _generate_research_summary(self, query: str, context: Dict[str, Any], 
                                 graph_insights: Dict[str, Any]) -> str:
        """Generate comprehensive research summary"""
        
        summary = f"""
        Research Summary: {query}
        
        Context Analysis:
        - Retrieved {context.get('memories_retrieved', 0)} relevant memories
        - Processed {context.get('context_items', 0)} context items
        - Used {context.get('packing_strategy', 'unknown')} packing strategy
        
        Graph Analysis:
        - Found {graph_insights.get('nodes_found', 0)} relevant nodes
        - Identified {len(graph_insights.get('related_concepts', []))} related concepts
        - Graph density: {graph_insights.get('graph_statistics', {}).get('connectivity_density', 0):.3f}
        
        The research indicates strong connections between key concepts and suggests
        multiple avenues for further investigation.
        """
        
        return summary.strip()
    
    def _extract_key_findings(self, graph_insights: Dict[str, Any], 
                            quality_assessment: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis"""
        
        findings = []
        
        # Findings from graph analysis
        if graph_insights.get("related_concepts"):
            top_concepts = sorted(
                graph_insights["related_concepts"], 
                key=lambda x: x["relevance"], 
                reverse=True
            )[:3]
            
            for concept in top_concepts:
                findings.append(f"Strong relevance to {concept['concept']} (relevance: {concept['relevance']:.2f})")
        
        # Findings from quality assessment
        alignment_scores = quality_assessment.get("alignment_scores", {})
        high_alignment = [obj for obj, score in alignment_scores.items() if score > 0.8]
        
        if high_alignment:
            findings.append(f"High alignment achieved in: {', '.join(high_alignment)}")
        
        # Add default findings if none found
        if not findings:
            findings = [
                "Research area shows active development",
                "Multiple research directions identified",
                "Strong foundation for further investigation"
            ]
        
        return findings
    
    def _identify_research_gaps(self, graph_insights: Dict[str, Any]) -> List[str]:
        """Identify potential research gaps"""
        
        gaps = []
        
        # Analyze graph connectivity for gaps
        connectivity = graph_insights.get("graph_statistics", {}).get("connectivity_density", 0)
        
        if connectivity < 0.3:
            gaps.append("Low connectivity between concepts suggests potential integration opportunities")
        
        # Analyze concept coverage
        concept_count = len(graph_insights.get("related_concepts", []))
        
        if concept_count < 5:
            gaps.append("Limited concept coverage indicates potential for broader investigation")
        
        # Add domain-specific gaps
        gaps.extend([
            "Cross-domain applications remain underexplored",
            "Scalability challenges need further attention",
            "Real-world deployment considerations require investigation"
        ])
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _suggest_future_directions(self, query: str, graph_insights: Dict[str, Any]) -> List[str]:
        """Suggest future research directions"""
        
        directions = []
        
        # Based on related concepts
        related_concepts = graph_insights.get("related_concepts", [])
        
        if related_concepts:
            top_concept = related_concepts[0]["concept"]
            directions.append(f"Deeper investigation into {top_concept} applications")
        
        # Generic future directions
        directions.extend([
            "Comparative analysis with alternative approaches",
            "Longitudinal studies on performance trends",
            "Cross-disciplinary collaboration opportunities",
            "Practical implementation case studies"
        ])
        
        return directions[:4]  # Limit to top 4 directions
    
    def _calculate_quality_grade(self, composite_score: float) -> str:
        """Calculate quality grade from composite score"""
        
        if composite_score >= 0.9:
            return "A+"
        elif composite_score >= 0.8:
            return "A"
        elif composite_score >= 0.7:
            return "B+"
        elif composite_score >= 0.6:
            return "B"
        elif composite_score >= 0.5:
            return "C+"
        else:
            return "C"
    
    def _generate_quality_recommendations(self, alignment_scores: Dict[Any, float]) -> List[str]:
        """Generate recommendations for quality improvement"""
        
        recommendations = []
        
        # Analyze low-scoring objectives
        for objective, score in alignment_scores.items():
            if score < 0.6:
                if objective == AlignmentObjective.HELPFULNESS:
                    recommendations.append("Increase practical examples and actionable insights")
                elif objective == AlignmentObjective.ACCURACY:
                    recommendations.append("Add more specific data and citations")
                elif objective == AlignmentObjective.EFFICIENCY:
                    recommendations.append("Streamline response structure and reduce redundancy")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Maintain current quality standards")
        
        return recommendations
    
    async def generate_research_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive research report"""
        
        print("\n📄 Generating Research Report")
        print("-" * 50)
        
        report = f"""
# Academic Literature Review Report

**Query**: {results['query']}
**Session ID**: {results['session_id']}
**Generated**: {results['timestamp']}

## Executive Summary

{results['synthesis']['research_summary']}

## Key Findings

{chr(10).join(f"• {finding}" for finding in results['synthesis']['key_findings'])}

## Research Context Analysis

- **Memories Retrieved**: {results['context']['memories_retrieved']}
- **Context Items**: {results['context']['context_items']}
- **Total Tokens**: {results['context']['total_tokens']}
- **Packing Strategy**: {results['context']['packing_strategy']}
- **Diversity Score**: {results['context']['diversity_score']:.3f}
- **Relevance Score**: {results['context']['relevance_score']:.3f}

## Semantic Graph Insights

- **Nodes Found**: {results['graph_insights']['nodes_found']}
- **Related Concepts**: {len(results['graph_insights']['related_concepts'])}
- **Graph Density**: {results['graph_insights']['graph_statistics']['connectivity_density']:.3f}

### Top Related Concepts:
{chr(10).join(f"• {concept['concept']} (relevance: {concept['relevance']:.2f})" 
              for concept in results['graph_insights']['related_concepts'][:5])}

## Code Analysis Results

- **Examples Analyzed**: {results['code_analysis']['examples_analyzed']}
- **Successful Repairs**: {results['code_analysis']['successful_repairs']}
- **Average Confidence**: {results['code_analysis']['average_confidence']:.2f}

## Quality Assessment

- **Overall Grade**: {results['quality_assessment']['quality_grade']}
- **Composite Alignment**: {results['quality_assessment']['composite_alignment']:.2f}

### Alignment Scores:
{chr(10).join(f"• {obj}: {score:.2f}" 
              for obj, score in results['quality_assessment']['alignment_scores'].items())}

### Recommendations:
{chr(10).join(f"• {rec}" for rec in results['quality_assessment']['recommendations'])}

## Research Gaps Identified

{chr(10).join(f"• {gap}" for gap in results['synthesis']['research_gaps'])}

## Future Research Directions

{chr(10).join(f"• {direction}" for direction in results['synthesis']['future_directions'])}

## Technical Details

- **Synergies Used**: {', '.join(results['synthesis']['synergies_used'])}
- **Processing Time**: {results['synthesis']['processing_metadata'].get('processing_time', 'N/A')}
- **Confidence Score**: {results['synthesis']['confidence_score']:.2f}

---

*Generated by AI Research Agent Extensions v1.0*
        """
        
        return report.strip()

async def main():
    """Main demonstration function"""
    
    print("🎓 AI Research Agent Extensions - Complete Workflow Example")
    print("=" * 70)
    print("Scenario: Academic Literature Review Assistant")
    print()
    
    # Initialize research assistant
    assistant = AcademicResearchAssistant()
    
    try:
        # Initialize system
        success = await assistant.initialize()
        
        if not success:
            print("❌ Failed to initialize research assistant")
            return
        
        # Define research queries
        research_queries = [
            "What are the latest developments in transformer architectures for natural language processing?",
            "How do attention mechanisms improve performance in computer vision tasks?",
            "What are the current challenges in scaling large language models?"
        ]
        
        # Conduct literature reviews
        all_results = []
        
        for i, query in enumerate(research_queries, 1):
            print(f"\n🔍 Research Query {i}/{len(research_queries)}")
            print(f"Query: {query}")
            
            # Conduct comprehensive literature review
            results = await assistant.conduct_literature_review(query)
            all_results.append(results)
            
            # Generate and display report
            report = await assistant.generate_research_report(results)
            
            # Save report to file
            report_filename = f"research_report_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_filename, 'w') as f:
                f.write(report)
            
            print(f"📄 Report saved: {report_filename}")
            
            # Brief summary
            print(f"\n📊 Quick Summary:")
            print(f"   Quality Grade: {results['quality_assessment']['quality_grade']}")
            print(f"   Key Findings: {len(results['synthesis']['key_findings'])}")
            print(f"   Research Gaps: {len(results['synthesis']['research_gaps'])}")
            print(f"   Future Directions: {len(results['synthesis']['future_directions'])}")
        
        # Generate comprehensive session summary
        print(f"\n🎉 Research Session Complete!")
        print("=" * 50)
        
        # Session statistics
        total_findings = sum(len(r['synthesis']['key_findings']) for r in all_results)
        avg_quality = sum(r['quality_assessment']['composite_alignment'] for r in all_results) / len(all_results)
        
        print(f"📈 Session Statistics:")
        print(f"   Queries Processed: {len(research_queries)}")
        print(f"   Total Key Findings: {total_findings}")
        print(f"   Average Quality Score: {avg_quality:.2f}")
        print(f"   Session ID: {assistant.session_id}")
        
        # Performance dashboard
        if assistant.extensions:
            dashboard = assistant.extensions.get_performance_dashboard()
            
            if dashboard and 'integration_overview' in dashboard:
                integration = dashboard['integration_overview']
                print(f"   System Integration: {integration['success_rate']:.1%}")
                print(f"   Active Stages: {len(integration['initialized_stages'])}")
        
        print(f"\n💡 Next Steps:")
        print(f"   • Review generated reports for detailed insights")
        print(f"   • Use findings to guide further research")
        print(f"   • Provide feedback to improve system performance")
        print(f"   • Explore cross-domain applications")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Research session interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Research session failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
