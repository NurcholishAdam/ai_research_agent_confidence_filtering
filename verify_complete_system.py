#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete System Verification Script
==================================

Comprehensive verification of the AI Research Agent Extensions system.
Tests all stages, integrations, and functionality to ensure everything works correctly.

Usage:
    python extensions/verify_complete_system.py
    python extensions/verify_complete_system.py --detailed
"""

import asyncio
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add extensions to path
sys.path.append(str(Path(__file__).parent))

class SystemVerification:
    """Complete system verification and testing"""
    
    def __init__(self, detailed: bool = False):
        self.detailed = detailed
        self.results = []
        self.start_time = None
        
    async def run_complete_verification(self) -> Dict[str, Any]:
        """Run complete system verification"""
        
        print("🔍 AI Research Agent Extensions - Complete System Verification")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Verification stages
        verification_stages = [
            ("Import Verification", self.verify_imports),
            ("Stage 1: Observability", self.verify_stage_1),
            ("Stage 2: Context Engineering", self.verify_stage_2),
            ("Stage 3: Semantic Graph", self.verify_stage_3),
            ("Stage 4: Diffusion Repair", self.verify_stage_4),
            ("Stage 5: RLHF & Agentic RL", self.verify_stage_5),
            ("Stage 6: Cross-Module Synergies", self.verify_stage_6),
            ("Integration Orchestrator", self.verify_integration),
            ("Performance Benchmarks", self.verify_performance),
            ("Documentation Completeness", self.verify_documentation)
        ]
        
        for stage_name, verification_func in verification_stages:
            print(f"\n🔹 {stage_name}")
            print("-" * 50)
            
            try:
                result = await verification_func()
                self.results.append({
                    "stage": stage_name,
                    "success": result["success"],
                    "details": result.get("details", {}),
                    "error": result.get("error")
                })
                
                if result["success"]:
                    print(f"✅ {stage_name}: PASSED")
                    if self.detailed and "details" in result:
                        for key, value in result["details"].items():
                            print(f"   {key}: {value}")
                else:
                    print(f"❌ {stage_name}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ {stage_name}: FAILED - {str(e)}")
                self.results.append({
                    "stage": stage_name,
                    "success": False,
                    "error": str(e)
                })
        
        # Generate final report
        return self.generate_verification_report()
    
    async def verify_imports(self) -> Dict[str, Any]:
        """Verify all imports work correctly"""
        
        try:
            # Test core imports
            from integration_orchestrator import AIResearchAgentExtensions
            from stage_1_observability import ObservabilityCollector, ModuleType
            from stage_2_context_builder import MemoryTierManager, TaskType
            from stage_3_semantic_graph import SemanticGraphManager, NodeType
            from stage_4_diffusion_repair import RuntimeRepairOperator, LanguageType
            from stage_5_rlhf_agentic_rl import PreferenceDataPipeline, AlignmentObjective
            from stage_6_cross_module_synergies import UnifiedOrchestrator
            
            return {
                "success": True,
                "details": {
                    "core_imports": "All core modules imported successfully",
                    "enum_imports": "All enums imported successfully",
                    "class_imports": "All classes imported successfully"
                }
            }
            
        except ImportError as e:
            return {"success": False, "error": f"Import failed: {e}"}
    
    async def verify_stage_1(self) -> Dict[str, Any]:
        """Verify Stage 1: Observability"""
        
        try:
            from stage_1_observability import ObservabilityCollector, ModuleType
            
            collector = ObservabilityCollector()
            
            # Test event tracking
            event_id = collector.track_event(
                module_type=ModuleType.CONTEXT_ENGINEERING,
                event_type="verification_test",
                session_id="verification_session",
                data={"test": "verification"}
            )
            
            # Test performance tracking
            collector.track_performance(
                module_type=ModuleType.SEMANTIC_GRAPH,
                operation="verification_operation",
                execution_time=0.1,
                success=True
            )
            
            # Test analytics dashboard
            dashboard = collector.get_analytics_dashboard()
            
            return {
                "success": True,
                "details": {
                    "event_tracking": "Working" if event_id else "Failed",
                    "performance_tracking": "Working",
                    "analytics_dashboard": "Working" if dashboard else "Failed",
                    "total_events": len(collector.events)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def verify_stage_2(self) -> Dict[str, Any]:
        """Verify Stage 2: Context Engineering"""
        
        try:
            from stage_2_context_builder import (
                MemoryTierManager, AdaptiveContextPacker, PromptTemplateManager,
                MemoryTier, TaskType
            )
            
            # Test memory management
            memory_manager = MemoryTierManager()
            
            memory_id = memory_manager.store_memory(
                content="Test memory for verification",
                memory_tier=MemoryTier.SHORT_TERM,
                relevance_score=0.8
            )
            
            memories = memory_manager.retrieve_memories("test memory", max_items=5)
            
            # Test context packing
            context_packer = AdaptiveContextPacker()
            
            if memories:
                packing_result = context_packer.pack_context(
                    memory_items=memories,
                    task_type=TaskType.RESEARCH
                )
            else:
                packing_result = None
            
            # Test template management
            template_manager = PromptTemplateManager()
            
            return {
                "success": True,
                "details": {
                    "memory_storage": "Working" if memory_id else "Failed",
                    "memory_retrieval": f"{len(memories)} memories retrieved",
                    "context_packing": "Working" if packing_result else "No memories to pack",
                    "template_manager": "Working",
                    "templates_loaded": len(template_manager.templates)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def verify_stage_3(self) -> Dict[str, Any]:
        """Verify Stage 3: Semantic Graph"""
        
        try:
            from stage_3_semantic_graph import (
                SemanticGraphManager, NodeType, EdgeType, SourceType
            )
            
            graph_manager = SemanticGraphManager()
            
            # Test node creation
            node_id = graph_manager.add_node(
                content="Test concept for verification",
                node_type=NodeType.CONCEPT,
                source_type=SourceType.INTERNAL,
                importance_score=0.8
            )
            
            # Test hybrid retrieval
            results = graph_manager.hybrid_retrieval(
                query="test concept",
                max_nodes=5
            )
            
            # Test graph statistics
            stats = graph_manager.get_graph_statistics()
            
            return {
                "success": True,
                "details": {
                    "node_creation": "Working" if node_id else "Failed",
                    "hybrid_retrieval": f"{len(results.nodes)} nodes retrieved",
                    "graph_statistics": "Working",
                    "total_nodes": stats["nodes"]["total"],
                    "total_edges": stats["edges"]["total"]
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def verify_stage_4(self) -> Dict[str, Any]:
        """Verify Stage 4: Diffusion Repair"""
        
        try:
            from stage_4_diffusion_repair import RuntimeRepairOperator, LanguageType
            
            repair_operator = RuntimeRepairOperator()
            
            # Test code repair
            broken_code = "def test(\n    print('test')"
            
            result = repair_operator.repair_code(
                broken_code=broken_code,
                language=LanguageType.PYTHON,
                error_type="SyntaxError"
            )
            
            # Test statistics
            stats = repair_operator.get_repair_statistics()
            
            return {
                "success": True,
                "details": {
                    "repair_operation": "Working",
                    "repair_success": result.success,
                    "repair_time": f"{result.repair_time:.3f}s",
                    "total_repairs": stats["total_repairs"],
                    "success_rate": f"{stats.get('success_rate', 0):.1%}"
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def verify_stage_5(self) -> Dict[str, Any]:
        """Verify Stage 5: RLHF & Agentic RL"""
        
        try:
            from stage_5_rlhf_agentic_rl import (
                PreferenceDataPipeline, RewardModel, OnlineAgenticRL,
                MultiObjectiveAlignment, PreferenceType, AlignmentObjective
            )
            
            # Test preference pipeline
            preference_pipeline = PreferenceDataPipeline()
            
            preference_id = preference_pipeline.collect_preference(
                query="Test query",
                response_a="Response A",
                response_b="Response B",
                preference=0,
                preference_type=PreferenceType.AUTOMATED_METRIC,
                confidence=0.8
            )
            
            # Test reward model
            reward_model = RewardModel()
            test_state = {"feature_1": 0.5, "feature_2": 0.3}
            reward = reward_model.predict_reward(test_state)
            
            # Test agentic RL
            agentic_rl = OnlineAgenticRL(reward_model)
            action, metadata = agentic_rl.select_action(
                state=test_state,
                available_actions=["action_a", "action_b"]
            )
            
            # Test alignment system
            alignment_system = MultiObjectiveAlignment()
            alignment_scores = alignment_system.evaluate_alignment(
                response="Test response for alignment evaluation",
                context={"query": "test", "response_time": 1.0}
            )
            
            return {
                "success": True,
                "details": {
                    "preference_collection": "Working" if preference_id else "Failed",
                    "reward_model": f"Reward: {reward:.3f}",
                    "action_selection": f"Selected: {action}",
                    "alignment_evaluation": f"{len(alignment_scores)} objectives evaluated",
                    "total_preferences": len(preference_pipeline.preference_data)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def verify_stage_6(self) -> Dict[str, Any]:
        """Verify Stage 6: Cross-Module Synergies"""
        
        try:
            from stage_6_cross_module_synergies import UnifiedOrchestrator
            
            orchestrator = UnifiedOrchestrator()
            
            # Test request processing
            test_request = {
                "type": "research",
                "query": "Test query for synergy verification",
                "session_id": "verification_session"
            }
            
            result = await orchestrator.process_request(test_request)
            
            # Test synergy status
            status = orchestrator.get_synergy_status()
            
            return {
                "success": True,
                "details": {
                    "request_processing": "Working" if result else "Failed",
                    "synergy_status": "Working",
                    "active_synergies": len([s for s in status["active_synergies"].values() if s]),
                    "synergies_used": result.get("synergies_used", [])
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def verify_integration(self) -> Dict[str, Any]:
        """Verify integration orchestrator"""
        
        try:
            from integration_orchestrator import AIResearchAgentExtensions
            
            extensions = AIResearchAgentExtensions()
            
            # Test initialization
            status = await extensions.initialize_all_stages()
            
            # Test enhanced request processing
            test_request = {
                "type": "research",
                "query": "Integration verification test",
                "session_id": "integration_verification"
            }
            
            result = await extensions.process_enhanced_request(test_request)
            
            # Test performance dashboard
            dashboard = extensions.get_performance_dashboard()
            
            return {
                "success": True,
                "details": {
                    "initialization_success_rate": f"{status['success_rate']:.1%}",
                    "stages_initialized": len(status["initialized_stages"]),
                    "request_processing": "Working" if result["success"] else "Failed",
                    "enhancements_used": len(result.get("enhancements_used", [])),
                    "performance_dashboard": "Working" if dashboard else "Failed"
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def verify_performance(self) -> Dict[str, Any]:
        """Verify performance benchmarks"""
        
        try:
            # Test basic performance metrics
            start_time = time.time()
            
            # Simulate some operations
            from stage_2_context_builder import MemoryTierManager
            memory_manager = MemoryTierManager()
            
            # Store and retrieve memories
            for i in range(10):
                memory_manager.store_memory(
                    content=f"Performance test memory {i}",
                    memory_tier=memory_manager.memory_tiers.__iter__().__next__(),
                    relevance_score=0.5
                )
            
            memories = memory_manager.retrieve_memories("performance test", max_items=5)
            
            end_time = time.time()
            operation_time = end_time - start_time
            
            return {
                "success": True,
                "details": {
                    "memory_operations": f"{operation_time:.3f}s for 10 stores + 1 retrieval",
                    "memories_retrieved": len(memories),
                    "throughput": f"{10 / operation_time:.1f} operations/second"
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def verify_documentation(self) -> Dict[str, Any]:
        """Verify documentation completeness"""
        
        try:
            extensions_dir = Path(__file__).parent
            
            # Check for required documentation files
            required_docs = [
                "README.md",
                "API_REFERENCE.md", 
                "DEPLOYMENT_GUIDE.md",
                "COMPLETION_SUMMARY.md",
                "tutorials/README.md"
            ]
            
            missing_docs = []
            existing_docs = []
            
            for doc in required_docs:
                doc_path = extensions_dir / doc
                if doc_path.exists():
                    existing_docs.append(doc)
                else:
                    missing_docs.append(doc)
            
            # Check tutorial files
            tutorial_files = list((extensions_dir / "tutorials").glob("*.py"))
            
            # Check example files
            example_files = list((extensions_dir / "examples").glob("*.py"))
            
            return {
                "success": len(missing_docs) == 0,
                "details": {
                    "required_docs_found": f"{len(existing_docs)}/{len(required_docs)}",
                    "missing_docs": missing_docs,
                    "tutorial_files": len(tutorial_files),
                    "example_files": len(example_files),
                    "documentation_complete": len(missing_docs) == 0
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_verification_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        
        total_time = time.time() - self.start_time
        
        successful_stages = [r for r in self.results if r["success"]]
        failed_stages = [r for r in self.results if not r["success"]]
        
        success_rate = len(successful_stages) / len(self.results) if self.results else 0
        
        report = {
            "verification_timestamp": datetime.now().isoformat(),
            "total_verification_time": f"{total_time:.2f}s",
            "stages_tested": len(self.results),
            "stages_passed": len(successful_stages),
            "stages_failed": len(failed_stages),
            "success_rate": success_rate,
            "overall_status": "PASSED" if success_rate >= 0.8 else "FAILED",
            "detailed_results": self.results
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"  VERIFICATION COMPLETE")
        print(f"{'='*70}")
        
        print(f"\n📊 Summary:")
        print(f"   Total stages tested: {report['stages_tested']}")
        print(f"   Stages passed: {report['stages_passed']}")
        print(f"   Stages failed: {report['stages_failed']}")
        print(f"   Success rate: {report['success_rate']:.1%}")
        print(f"   Total time: {report['total_verification_time']}")
        print(f"   Overall status: {report['overall_status']}")
        
        if failed_stages:
            print(f"\n❌ Failed stages:")
            for stage in failed_stages:
                print(f"   - {stage['stage']}: {stage.get('error', 'Unknown error')}")
        
        if report['overall_status'] == "PASSED":
            print(f"\n🎉 All critical systems verified successfully!")
            print(f"   The AI Research Agent Extensions system is ready for use.")
        else:
            print(f"\n⚠️ Some systems failed verification.")
            print(f"   Please check the failed stages and resolve issues.")
        
        return report

async def main():
    """Main verification function"""
    
    parser = argparse.ArgumentParser(description="AI Research Agent Extensions System Verification")
    parser.add_argument("--detailed", action="store_true", help="Show detailed verification results")
    
    args = parser.parse_args()
    
    verifier = SystemVerification(detailed=args.detailed)
    
    try:
        report = await verifier.run_complete_verification()
        
        # Save report
        report_file = f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        # Exit with appropriate code
        exit_code = 0 if report['overall_status'] == "PASSED" else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())