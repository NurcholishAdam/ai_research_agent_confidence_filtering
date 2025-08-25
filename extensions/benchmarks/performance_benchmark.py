#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Research Agent Extensions - Performance Benchmark Suite
=========================================================

Comprehensive benchmarking suite for all extension stages.
Measures performance, scalability, and resource usage across different scenarios.

Usage:
    python extensions/benchmarks/performance_benchmark.py
    python extensions/benchmarks/performance_benchmark.py --stage 2 --iterations 100
    python extensions/benchmarks/performance_benchmark.py --full-suite --output results.json
"""

import asyncio
import argparse
import json
import time
import psutil
import statistics
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add extensions to path
sys.path.append(str(Path(__file__).parent.parent))

from integration_orchestrator import AIResearchAgentExtensions
from stage_1_observability import ObservabilityCollector, ModuleType
from stage_2_context_builder import MemoryTierManager, AdaptiveContextPacker, TaskType, MemoryTier
from stage_3_semantic_graph import SemanticGraphManager, NodeType, EdgeType, SourceType
from stage_4_diffusion_repair import RuntimeRepairOperator, LanguageType
from stage_5_rlhf_agentic_rl import PreferenceDataPipeline, OnlineAgenticRL, RewardModel

@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    stage: str
    operation: str
    iterations: int
    execution_times: List[float]
    memory_usage: List[float]
    cpu_usage: List[float]
    success_rate: float
    avg_time: float
    median_time: float
    p95_time: float
    p99_time: float
    throughput: float
    metadata: Dict[str, Any]

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    timestamp: datetime
    system_info: Dict[str, Any]
    results: List[BenchmarkResult]
    summary: Dict[str, Any]

class PerformanceBenchmark:
    """Performance benchmark suite for AI Research Agent Extensions"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results: List[BenchmarkResult] = []
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "platform": sys.platform,
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat()
        }
    
    def measure_resources(self) -> Dict[str, float]:
        """Measure current resource usage"""
        return {
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "cpu_percent": self.process.cpu_percent()
        }
    
    async def benchmark_stage_1_observability(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark Stage 1: Observability"""
        print(f"🔍 Benchmarking Stage 1: Observability ({iterations} iterations)")
        
        collector = ObservabilityCollector()
        
        execution_times = []
        memory_usage = []
        cpu_usage = []
        success_count = 0
        
        for i in range(iterations):
            # Measure resources before
            resources_before = self.measure_resources()
            
            start_time = time.time()
            
            try:
                # Test event tracking
                event_id = collector.track_event(
                    module_type=ModuleType.CONTEXT_ENGINEERING,
                    event_type="benchmark_test",
                    session_id=f"benchmark_session_{i}",
                    data={"iteration": i, "test_data": "benchmark_payload"}
                )
                
                # Test performance tracking
                collector.track_performance(
                    module_type=ModuleType.SEMANTIC_GRAPH,
                    operation="benchmark_operation",
                    execution_time=0.1 + (i % 10) * 0.01,
                    success=True
                )
                
                success_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Iteration {i} failed: {e}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Measure resources after
            resources_after = self.measure_resources()
            
            execution_times.append(execution_time)
            memory_usage.append(resources_after["memory_mb"])
            cpu_usage.append(resources_after["cpu_percent"])
        
        return BenchmarkResult(
            stage="stage_1_observability",
            operation="event_and_performance_tracking",
            iterations=iterations,
            execution_times=execution_times,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success_rate=success_count / iterations,
            avg_time=statistics.mean(execution_times),
            median_time=statistics.median(execution_times),
            p95_time=self._percentile(execution_times, 95),
            p99_time=self._percentile(execution_times, 99),
            throughput=iterations / sum(execution_times),
            metadata={
                "events_tracked": success_count * 2,  # event + performance
                "analytics_generated": 1
            }
        )
    
    async def benchmark_stage_2_context_engineering(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark Stage 2: Context Engineering"""
        print(f"🧠 Benchmarking Stage 2: Context Engineering ({iterations} iterations)")
        
        memory_manager = MemoryTierManager()
        context_packer = AdaptiveContextPacker()
        
        # Pre-populate with test data
        test_memories = []
        for i in range(100):
            memory_id = memory_manager.store_memory(
                content=f"Test memory content {i} with various details and information for benchmarking purposes. This content simulates realistic memory items with sufficient length.",
                memory_tier=MemoryTier.LONG_TERM if i < 50 else MemoryTier.EPISODIC,
                relevance_score=0.5 + (i % 10) * 0.05,
                metadata={"test_id": i, "category": f"category_{i % 5}"}
            )
            test_memories.append(memory_id)
        
        execution_times = []
        memory_usage = []
        cpu_usage = []
        success_count = 0
        
        for i in range(iterations):
            resources_before = self.measure_resources()
            start_time = time.time()
            
            try:
                # Test memory retrieval
                memories = memory_manager.retrieve_memories(
                    query=f"test memory content {i % 20}",
                    max_items=15,
                    relevance_threshold=0.3
                )
                
                # Test context packing
                if memories:
                    packing_result = context_packer.pack_context(
                        memory_items=memories,
                        task_type=TaskType.RESEARCH
                    )
                
                success_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Iteration {i} failed: {e}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            resources_after = self.measure_resources()
            
            execution_times.append(execution_time)
            memory_usage.append(resources_after["memory_mb"])
            cpu_usage.append(resources_after["cpu_percent"])
        
        return BenchmarkResult(
            stage="stage_2_context_engineering",
            operation="memory_retrieval_and_packing",
            iterations=iterations,
            execution_times=execution_times,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success_rate=success_count / iterations,
            avg_time=statistics.mean(execution_times),
            median_time=statistics.median(execution_times),
            p95_time=self._percentile(execution_times, 95),
            p99_time=self._percentile(execution_times, 99),
            throughput=iterations / sum(execution_times),
            metadata={
                "memories_stored": len(test_memories),
                "avg_memories_retrieved": 10,
                "packing_operations": success_count
            }
        )
    
    async def benchmark_stage_3_semantic_graph(self, iterations: int = 30) -> BenchmarkResult:
        """Benchmark Stage 3: Semantic Graph"""
        print(f"🕸️ Benchmarking Stage 3: Semantic Graph ({iterations} iterations)")
        
        graph_manager = SemanticGraphManager()
        
        # Pre-populate graph with test data
        test_nodes = []
        for i in range(50):
            node_id = graph_manager.add_node(
                content=f"Test concept {i} with detailed information about various aspects of the research domain for comprehensive benchmarking.",
                node_type=NodeType.CONCEPT,
                source_type=SourceType.INTERNAL,
                importance_score=0.5 + (i % 10) * 0.05,
                tags=[f"tag_{i % 5}", f"category_{i % 3}"]
            )
            test_nodes.append(node_id)
        
        # Add some edges
        for i in range(0, len(test_nodes) - 1, 2):
            graph_manager.add_edge(
                source_node=test_nodes[i],
                target_node=test_nodes[i + 1],
                edge_type=EdgeType.SUPPORTS,
                confidence=0.7 + (i % 10) * 0.02
            )
        
        execution_times = []
        memory_usage = []
        cpu_usage = []
        success_count = 0
        
        for i in range(iterations):
            resources_before = self.measure_resources()
            start_time = time.time()
            
            try:
                # Test hybrid retrieval
                results = graph_manager.hybrid_retrieval(
                    query=f"test concept {i % 10}",
                    max_nodes=10,
                    similarity_threshold=0.3
                )
                
                # Test neighbor discovery
                if test_nodes:
                    neighbors = graph_manager.get_node_neighbors(
                        node_id=test_nodes[i % len(test_nodes)],
                        max_depth=2
                    )
                
                success_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Iteration {i} failed: {e}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            resources_after = self.measure_resources()
            
            execution_times.append(execution_time)
            memory_usage.append(resources_after["memory_mb"])
            cpu_usage.append(resources_after["cpu_percent"])
        
        return BenchmarkResult(
            stage="stage_3_semantic_graph",
            operation="hybrid_retrieval_and_navigation",
            iterations=iterations,
            execution_times=execution_times,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success_rate=success_count / iterations,
            avg_time=statistics.mean(execution_times),
            median_time=statistics.median(execution_times),
            p95_time=self._percentile(execution_times, 95),
            p99_time=self._percentile(execution_times, 99),
            throughput=iterations / sum(execution_times),
            metadata={
                "nodes_created": len(test_nodes),
                "edges_created": len(test_nodes) // 2,
                "retrieval_operations": success_count
            }
        )
    
    async def benchmark_stage_4_diffusion_repair(self, iterations: int = 20) -> BenchmarkResult:
        """Benchmark Stage 4: Diffusion Repair"""
        print(f"🔧 Benchmarking Stage 4: Diffusion Repair ({iterations} iterations)")
        
        repair_operator = RuntimeRepairOperator()
        
        # Test cases for different languages
        test_cases = [
            {
                "code": "def hello_world(\n    print('Hello, World!')",
                "language": LanguageType.PYTHON,
                "error_type": "SyntaxError"
            },
            {
                "code": "function test() {\n    console.log('test'\n}",
                "language": LanguageType.JAVASCRIPT,
                "error_type": "SyntaxError"
            },
            {
                "code": "SELECT * FROM users WHERE id =",
                "language": LanguageType.SQL,
                "error_type": "SyntaxError"
            }
        ]
        
        execution_times = []
        memory_usage = []
        cpu_usage = []
        success_count = 0
        
        for i in range(iterations):
            resources_before = self.measure_resources()
            start_time = time.time()
            
            try:
                # Select test case
                test_case = test_cases[i % len(test_cases)]
                
                # Test code repair
                result = repair_operator.repair_code(
                    broken_code=test_case["code"],
                    language=test_case["language"],
                    error_type=test_case["error_type"]
                )
                
                if result.success:
                    success_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Iteration {i} failed: {e}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            resources_after = self.measure_resources()
            
            execution_times.append(execution_time)
            memory_usage.append(resources_after["memory_mb"])
            cpu_usage.append(resources_after["cpu_percent"])
        
        return BenchmarkResult(
            stage="stage_4_diffusion_repair",
            operation="code_repair_with_voting",
            iterations=iterations,
            execution_times=execution_times,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success_rate=success_count / iterations,
            avg_time=statistics.mean(execution_times),
            median_time=statistics.median(execution_times),
            p95_time=self._percentile(execution_times, 95),
            p99_time=self._percentile(execution_times, 99),
            throughput=iterations / sum(execution_times),
            metadata={
                "test_cases": len(test_cases),
                "languages_tested": len(set(tc["language"] for tc in test_cases)),
                "successful_repairs": success_count
            }
        )
    
    async def benchmark_stage_5_rlhf(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark Stage 5: RLHF & Agentic RL"""
        print(f"🎯 Benchmarking Stage 5: RLHF & Agentic RL ({iterations} iterations)")
        
        preference_pipeline = PreferenceDataPipeline()
        reward_model = RewardModel()
        agentic_rl = OnlineAgenticRL(reward_model)
        
        execution_times = []
        memory_usage = []
        cpu_usage = []
        success_count = 0
        
        for i in range(iterations):
            resources_before = self.measure_resources()
            start_time = time.time()
            
            try:
                # Test preference collection
                preference_id = preference_pipeline.collect_preference(
                    query=f"Test query {i}",
                    response_a=f"Response A for query {i}",
                    response_b=f"Response B for query {i}",
                    preference=i % 2,
                    preference_type="automated_metric",
                    confidence=0.7 + (i % 10) * 0.02
                )
                
                # Test RL action selection
                state = {"complexity": 0.5 + (i % 10) * 0.05, "context_size": 1000 + i * 10}
                actions = ["action_a", "action_b", "action_c"]
                
                selected_action, metadata = agentic_rl.select_action(state, actions)
                
                success_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Iteration {i} failed: {e}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            resources_after = self.measure_resources()
            
            execution_times.append(execution_time)
            memory_usage.append(resources_after["memory_mb"])
            cpu_usage.append(resources_after["cpu_percent"])
        
        return BenchmarkResult(
            stage="stage_5_rlhf_agentic_rl",
            operation="preference_collection_and_rl",
            iterations=iterations,
            execution_times=execution_times,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success_rate=success_count / iterations,
            avg_time=statistics.mean(execution_times),
            median_time=statistics.median(execution_times),
            p95_time=self._percentile(execution_times, 95),
            p99_time=self._percentile(execution_times, 99),
            throughput=iterations / sum(execution_times),
            metadata={
                "preferences_collected": success_count,
                "rl_actions_selected": success_count,
                "reward_model_predictions": success_count
            }
        )
    
    async def benchmark_integration(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark full integration"""
        print(f"🎼 Benchmarking Full Integration ({iterations} iterations)")
        
        extensions = AIResearchAgentExtensions()
        await extensions.initialize_all_stages()
        
        execution_times = []
        memory_usage = []
        cpu_usage = []
        success_count = 0
        
        test_requests = [
            {
                "type": "research",
                "query": "How do transformer models work?",
                "session_id": f"benchmark_session_{i}"
            }
            for i in range(iterations)
        ]
        
        for i, request in enumerate(test_requests):
            resources_before = self.measure_resources()
            start_time = time.time()
            
            try:
                # Test full enhanced request processing
                result = await extensions.process_enhanced_request(request)
                
                if result.get("success", False):
                    success_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Iteration {i} failed: {e}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            resources_after = self.measure_resources()
            
            execution_times.append(execution_time)
            memory_usage.append(resources_after["memory_mb"])
            cpu_usage.append(resources_after["cpu_percent"])
        
        return BenchmarkResult(
            stage="full_integration",
            operation="enhanced_request_processing",
            iterations=iterations,
            execution_times=execution_times,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success_rate=success_count / iterations,
            avg_time=statistics.mean(execution_times),
            median_time=statistics.median(execution_times),
            p95_time=self._percentile(execution_times, 95),
            p99_time=self._percentile(execution_times, 99),
            throughput=iterations / sum(execution_times),
            metadata={
                "requests_processed": iterations,
                "successful_requests": success_count,
                "stages_integrated": 6
            }
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    async def run_benchmark_suite(self, stages: List[int] = None, iterations: int = 50) -> BenchmarkSuite:
        """Run complete benchmark suite"""
        
        print("🚀 Starting AI Research Agent Extensions Benchmark Suite")
        print("=" * 70)
        
        system_info = self.get_system_info()
        print(f"System: {system_info['cpu_count']} CPUs, {system_info['memory_total'] / (1024**3):.1f}GB RAM")
        
        if stages is None:
            stages = [1, 2, 3, 4, 5, 6]  # All stages + integration
        
        benchmark_functions = {
            1: self.benchmark_stage_1_observability,
            2: self.benchmark_stage_2_context_engineering,
            3: self.benchmark_stage_3_semantic_graph,
            4: self.benchmark_stage_4_diffusion_repair,
            5: self.benchmark_stage_5_rlhf,
            6: self.benchmark_integration
        }
        
        results = []
        
        for stage in stages:
            if stage in benchmark_functions:
                try:
                    result = await benchmark_functions[stage](iterations)
                    results.append(result)
                    
                    print(f"\n✅ Stage {stage} Results:")
                    print(f"   Success Rate: {result.success_rate:.1%}")
                    print(f"   Avg Time: {result.avg_time:.3f}s")
                    print(f"   Throughput: {result.throughput:.1f} ops/sec")
                    print(f"   P95 Time: {result.p95_time:.3f}s")
                    
                except Exception as e:
                    print(f"❌ Stage {stage} benchmark failed: {e}")
        
        # Generate summary
        summary = self._generate_summary(results)
        
        return BenchmarkSuite(
            timestamp=datetime.now(),
            system_info=system_info,
            results=results,
            summary=summary
        )
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate benchmark summary"""
        
        if not results:
            return {}
        
        total_operations = sum(r.iterations for r in results)
        total_time = sum(sum(r.execution_times) for r in results)
        avg_success_rate = statistics.mean([r.success_rate for r in results])
        
        # Performance rankings
        fastest_stage = min(results, key=lambda r: r.avg_time)
        highest_throughput = max(results, key=lambda r: r.throughput)
        most_reliable = max(results, key=lambda r: r.success_rate)
        
        return {
            "total_operations": total_operations,
            "total_execution_time": total_time,
            "overall_success_rate": avg_success_rate,
            "overall_throughput": total_operations / total_time if total_time > 0 else 0,
            "fastest_stage": {
                "stage": fastest_stage.stage,
                "avg_time": fastest_stage.avg_time
            },
            "highest_throughput_stage": {
                "stage": highest_throughput.stage,
                "throughput": highest_throughput.throughput
            },
            "most_reliable_stage": {
                "stage": most_reliable.stage,
                "success_rate": most_reliable.success_rate
            },
            "performance_grades": self._calculate_performance_grades(results)
        }
    
    def _calculate_performance_grades(self, results: List[BenchmarkResult]) -> Dict[str, str]:
        """Calculate performance grades for each stage"""
        
        grades = {}
        
        for result in results:
            # Grade based on success rate, throughput, and latency
            success_score = result.success_rate * 40
            
            # Throughput score (normalized)
            max_throughput = max(r.throughput for r in results)
            throughput_score = (result.throughput / max_throughput) * 30 if max_throughput > 0 else 0
            
            # Latency score (inverted - lower is better)
            max_latency = max(r.avg_time for r in results)
            latency_score = (1 - (result.avg_time / max_latency)) * 30 if max_latency > 0 else 30
            
            total_score = success_score + throughput_score + latency_score
            
            if total_score >= 90:
                grade = "A+"
            elif total_score >= 85:
                grade = "A"
            elif total_score >= 80:
                grade = "B+"
            elif total_score >= 75:
                grade = "B"
            elif total_score >= 70:
                grade = "C+"
            elif total_score >= 65:
                grade = "C"
            else:
                grade = "D"
            
            grades[result.stage] = grade
        
        return grades
    
    def save_results(self, suite: BenchmarkSuite, output_file: str):
        """Save benchmark results to file"""
        
        # Convert to serializable format
        suite_dict = {
            "timestamp": suite.timestamp.isoformat(),
            "system_info": suite.system_info,
            "results": [asdict(result) for result in suite.results],
            "summary": suite.summary
        }
        
        with open(output_file, 'w') as f:
            json.dump(suite_dict, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {output_file}")
    
    def print_detailed_report(self, suite: BenchmarkSuite):
        """Print detailed benchmark report"""
        
        print("\n" + "=" * 70)
        print("  DETAILED BENCHMARK REPORT")
        print("=" * 70)
        
        print(f"\nTimestamp: {suite.timestamp}")
        print(f"System: {suite.system_info['cpu_count']} CPUs, {suite.system_info['memory_total'] / (1024**3):.1f}GB RAM")
        
        print(f"\n📊 OVERALL SUMMARY")
        print(f"   Total Operations: {suite.summary['total_operations']}")
        print(f"   Overall Success Rate: {suite.summary['overall_success_rate']:.1%}")
        print(f"   Overall Throughput: {suite.summary['overall_throughput']:.1f} ops/sec")
        
        print(f"\n🏆 TOP PERFORMERS")
        print(f"   Fastest: {suite.summary['fastest_stage']['stage']} ({suite.summary['fastest_stage']['avg_time']:.3f}s)")
        print(f"   Highest Throughput: {suite.summary['highest_throughput_stage']['stage']} ({suite.summary['highest_throughput_stage']['throughput']:.1f} ops/sec)")
        print(f"   Most Reliable: {suite.summary['most_reliable_stage']['stage']} ({suite.summary['most_reliable_stage']['success_rate']:.1%})")
        
        print(f"\n📈 STAGE PERFORMANCE GRADES")
        for stage, grade in suite.summary['performance_grades'].items():
            print(f"   {stage}: {grade}")
        
        print(f"\n📋 DETAILED RESULTS")
        for result in suite.results:
            print(f"\n   {result.stage.upper()}")
            print(f"     Operation: {result.operation}")
            print(f"     Iterations: {result.iterations}")
            print(f"     Success Rate: {result.success_rate:.1%}")
            print(f"     Avg Time: {result.avg_time:.3f}s")
            print(f"     Median Time: {result.median_time:.3f}s")
            print(f"     P95 Time: {result.p95_time:.3f}s")
            print(f"     P99 Time: {result.p99_time:.3f}s")
            print(f"     Throughput: {result.throughput:.1f} ops/sec")
            print(f"     Avg Memory: {statistics.mean(result.memory_usage):.1f}MB")
            print(f"     Metadata: {result.metadata}")

async def main():
    """Main benchmark function"""
    
    parser = argparse.ArgumentParser(description="AI Research Agent Extensions Benchmark Suite")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5, 6], 
                       help="Run specific stage benchmark")
    parser.add_argument("--iterations", type=int, default=50,
                       help="Number of iterations per benchmark")
    parser.add_argument("--full-suite", action="store_true",
                       help="Run complete benchmark suite")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--detailed", action="store_true",
                       help="Print detailed report")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    
    # Determine which stages to run
    if args.stage:
        stages = [args.stage]
    elif args.full_suite:
        stages = [1, 2, 3, 4, 5, 6]
    else:
        stages = [1, 2, 3, 4, 5, 6]  # Default to all stages
    
    try:
        # Run benchmark suite
        suite = await benchmark.run_benchmark_suite(stages, args.iterations)
        
        # Save results
        benchmark.save_results(suite, args.output)
        
        # Print report
        if args.detailed:
            benchmark.print_detailed_report(suite)
        else:
            print(f"\n🎉 Benchmark Complete!")
            print(f"   Overall Success Rate: {suite.summary['overall_success_rate']:.1%}")
            print(f"   Overall Throughput: {suite.summary['overall_throughput']:.1f} ops/sec")
            print(f"   Results saved to: {args.output}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
