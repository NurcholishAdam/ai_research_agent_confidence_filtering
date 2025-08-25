# AI Research Agent Extensions - API Reference

Complete API documentation for all extension stages and components.

## Table of Contents

- [Stage 1: Observability API](#stage-1-observability-api)
- [Stage 2: Context Engineering API](#stage-2-context-engineering-api)
- [Stage 3: Semantic Graph API](#stage-3-semantic-graph-api)
- [Stage 4: Diffusion Repair API](#stage-4-diffusion-repair-api)
- [Stage 5: RLHF & Agentic RL API](#stage-5-rlhf--agentic-rl-api)
- [Stage 6: Cross-Module Synergies API](#stage-6-cross-module-synergies-api)
- [Integration Orchestrator API](#integration-orchestrator-api)

---

## Stage 1: Observability API

### ObservabilityCollector

Main class for tracking events, performance, and managing A/B tests.

#### Constructor

```python
ObservabilityCollector(config_path: str = "extensions/observability_config.json")
```

**Parameters:**
- `config_path` (str): Path to configuration file

#### Methods

##### track_event()

```python
track_event(
    module_type: ModuleType,
    event_type: str,
    session_id: str,
    data: Dict[str, Any],
    user_id: Optional[str] = None,
    performance_metrics: Optional[Dict[str, Any]] = None
) -> str
```

Track an observability event.

**Parameters:**
- `module_type` (ModuleType): Type of module generating the event
- `event_type` (str): Type of event (e.g., "context_packing", "graph_retrieval")
- `session_id` (str): Unique session identifier
- `data` (Dict[str, Any]): Event-specific data
- `user_id` (Optional[str]): User identifier
- `performance_metrics` (Optional[Dict[str, Any]]): Performance data

**Returns:**
- `str`: Unique event ID

**Example:**
```python
event_id = collector.track_event(
    module_type=ModuleType.CONTEXT_ENGINEERING,
    event_type="adaptive_packing",
    session_id="session_123",
    data={"items_packed": 15, "strategy": "balanced"},
    performance_metrics={"execution_time": 0.245, "success": True}
)
```

##### track_performance()

```python
track_performance(
    module_type: ModuleType,
    operation: str,
    execution_time: float,
    success: bool,
    additional_metrics: Optional[Dict[str, Any]] = None
) -> str
```

Track performance metrics for an operation.

**Parameters:**
- `module_type` (ModuleType): Module performing the operation
- `operation` (str): Operation name
- `execution_time` (float): Time taken in seconds
- `success` (bool): Whether operation succeeded
- `additional_metrics` (Optional[Dict[str, Any]]): Extra metrics

**Returns:**
- `str`: Event ID for the performance record

##### is_module_enabled()

```python
is_module_enabled(module_type: ModuleType, session_id: str = None) -> bool
```

Check if a module is enabled for the current session.

**Parameters:**
- `module_type` (ModuleType): Module to check
- `session_id` (Optional[str]): Session ID for rollout percentage checks

**Returns:**
- `bool`: True if module is enabled

##### create_experiment()

```python
create_experiment(
    name: str,
    description: str,
    variants: Dict[str, Dict[str, Any]],
    traffic_allocation: Dict[str, float],
    success_metrics: List[str],
    duration_days: int = 30
) -> str
```

Create a new A/B test experiment.

**Parameters:**
- `name` (str): Experiment name
- `description` (str): Experiment description
- `variants` (Dict[str, Dict[str, Any]]): Variant configurations
- `traffic_allocation` (Dict[str, float]): Traffic split percentages
- `success_metrics` (List[str]): Metrics to track
- `duration_days` (int): Experiment duration

**Returns:**
- `str`: Experiment ID

##### get_experiment_variant()

```python
get_experiment_variant(experiment_id: str, session_id: str) -> Optional[str]
```

Get the experiment variant for a session.

**Parameters:**
- `experiment_id` (str): Experiment identifier
- `session_id` (str): Session identifier

**Returns:**
- `Optional[str]`: Variant name or None

##### get_analytics_dashboard()

```python
get_analytics_dashboard() -> Dict[str, Any]
```

Generate comprehensive analytics dashboard.

**Returns:**
- `Dict[str, Any]`: Dashboard data with system health, module performance, and experiment analytics

### Enums

#### ModuleType

```python
class ModuleType(Enum):
    RLHF = "rlhf"
    CONTEXT_ENGINEERING = "context_engineering"
    SEMANTIC_GRAPH = "semantic_graph"
    DIFFUSION_REPAIR = "diffusion_repair"
    MULTI_AGENT = "multi_agent"
    TOOL_REASONING = "tool_reasoning"
```

#### ExperimentType

```python
class ExperimentType(Enum):
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    CANARY = "canary"
    FEATURE_FLAG = "feature_flag"
```

### Decorator

#### @track_performance

```python
@track_performance(module_type: ModuleType, operation: str)
def your_function():
    # Function implementation
    pass
```

Automatically track performance of decorated functions.

---

## Stage 2: Context Engineering API

### MemoryTierManager

Manages hierarchical memory with intelligent retrieval.

#### Constructor

```python
MemoryTierManager(max_tokens_per_tier: Dict[MemoryTier, int] = None)
```

**Parameters:**
- `max_tokens_per_tier` (Optional[Dict[MemoryTier, int]]): Token limits per tier

#### Methods

##### store_memory()

```python
store_memory(
    content: str,
    memory_tier: MemoryTier,
    relevance_score: float = 0.5,
    metadata: Dict[str, Any] = None
) -> str
```

Store a memory item in the appropriate tier.

**Parameters:**
- `content` (str): Memory content
- `memory_tier` (MemoryTier): Target memory tier
- `relevance_score` (float): Relevance score (0.0-1.0)
- `metadata` (Optional[Dict[str, Any]]): Additional metadata

**Returns:**
- `str`: Memory ID

##### retrieve_memories()

```python
retrieve_memories(
    query: str,
    memory_tiers: List[MemoryTier] = None,
    max_items: int = 10,
    relevance_threshold: float = 0.3
) -> List[MemoryItem]
```

Retrieve relevant memories from specified tiers.

**Parameters:**
- `query` (str): Search query
- `memory_tiers` (Optional[List[MemoryTier]]): Tiers to search
- `max_items` (int): Maximum items to return
- `relevance_threshold` (float): Minimum relevance score

**Returns:**
- `List[MemoryItem]`: Retrieved memory items

##### promote_memory()

```python
promote_memory(memory_id: str, target_tier: MemoryTier) -> bool
```

Promote a memory item to a higher tier.

**Parameters:**
- `memory_id` (str): Memory identifier
- `target_tier` (MemoryTier): Target tier

**Returns:**
- `bool`: Success status

### AdaptiveContextPacker

Intelligent context packing with token awareness.

#### Constructor

```python
AdaptiveContextPacker(max_context_tokens: int = 8000)
```

**Parameters:**
- `max_context_tokens` (int): Maximum context tokens

#### Methods

##### pack_context()

```python
pack_context(
    memory_items: List[MemoryItem],
    task_type: TaskType,
    strategy: ContextPackingStrategy = ContextPackingStrategy.ADAPTIVE,
    diversity_weight: float = 0.3,
    recency_weight: float = 0.3,
    relevance_weight: float = 0.4
) -> ContextPackingResult
```

Pack context items optimally within token limits.

**Parameters:**
- `memory_items` (List[MemoryItem]): Items to pack
- `task_type` (TaskType): Type of task
- `strategy` (ContextPackingStrategy): Packing strategy
- `diversity_weight` (float): Weight for diversity
- `recency_weight` (float): Weight for recency
- `relevance_weight` (float): Weight for relevance

**Returns:**
- `ContextPackingResult`: Packing result with metrics

### PromptTemplateManager

Manages versioned prompt templates with A/B testing.

#### Constructor

```python
PromptTemplateManager(templates_dir: str = "extensions/prompt_templates")
```

**Parameters:**
- `templates_dir` (str): Directory for template storage

#### Methods

##### create_template()

```python
create_template(
    name: str,
    template_content: str,
    task_types: List[TaskType],
    variables: List[str] = None,
    version: str = "1.0.0"
) -> str
```

Create a new prompt template.

**Parameters:**
- `name` (str): Template name
- `template_content` (str): Jinja2 template content
- `task_types` (List[TaskType]): Applicable task types
- `variables` (Optional[List[str]]): Template variables
- `version` (str): Template version

**Returns:**
- `str`: Template ID

##### render_template()

```python
render_template(template_id: str, variables: Dict[str, Any]) -> str
```

Render a template with provided variables.

**Parameters:**
- `template_id` (str): Template identifier
- `variables` (Dict[str, Any]): Template variables

**Returns:**
- `str`: Rendered template

### Enums

#### MemoryTier

```python
class MemoryTier(Enum):
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    LONG_TERM = "long_term"
    GRAPH_MEMORY = "graph_memory"
```

#### TaskType

```python
class TaskType(Enum):
    QA = "question_answering"
    CODE_REPAIR = "code_repair"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"
```

#### ContextPackingStrategy

```python
class ContextPackingStrategy(Enum):
    RECENCY_FIRST = "recency_first"
    DIVERSITY_FIRST = "diversity_first"
    RELEVANCE_FIRST = "relevance_first"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
```

---

## Stage 3: Semantic Graph API

### SemanticGraphManager

Advanced semantic graph with multi-source fusion and hybrid retrieval.

#### Constructor

```python
SemanticGraphManager(graph_storage_path: str = "extensions/semantic_graph")
```

**Parameters:**
- `graph_storage_path` (str): Path for graph storage

#### Methods

##### add_node()

```python
add_node(
    content: str,
    node_type: NodeType,
    source_type: SourceType,
    title: str = None,
    source_id: str = None,
    embedding: List[float] = None,
    metadata: Dict[str, Any] = None,
    tags: List[str] = None,
    importance_score: float = 0.5,
    confidence_score: float = 0.8
) -> str
```

Add a new node to the semantic graph.

**Parameters:**
- `content` (str): Node content
- `node_type` (NodeType): Type of node
- `source_type` (SourceType): Source of the node
- `title` (Optional[str]): Node title
- `source_id` (Optional[str]): Source identifier
- `embedding` (Optional[List[float]]): Vector embedding
- `metadata` (Optional[Dict[str, Any]]): Additional metadata
- `tags` (Optional[List[str]]): Node tags
- `importance_score` (float): Importance score (0.0-1.0)
- `confidence_score` (float): Confidence score (0.0-1.0)

**Returns:**
- `str`: Node ID

##### add_edge()

```python
add_edge(
    source_node: str,
    target_node: str,
    edge_type: EdgeType,
    weight: float = 1.0,
    confidence: float = 0.8,
    evidence: List[str] = None,
    metadata: Dict[str, Any] = None
) -> str
```

Add an edge between two nodes.

**Parameters:**
- `source_node` (str): Source node ID
- `target_node` (str): Target node ID
- `edge_type` (EdgeType): Type of relationship
- `weight` (float): Edge weight
- `confidence` (float): Confidence in relationship
- `evidence` (Optional[List[str]]): Supporting evidence
- `metadata` (Optional[Dict[str, Any]]): Additional metadata

**Returns:**
- `str`: Edge ID

##### multi_source_fusion()

```python
multi_source_fusion(sources_data: Dict[SourceType, List[Dict[str, Any]]]) -> Dict[str, Any]
```

Fuse data from multiple sources with deduplication.

**Parameters:**
- `sources_data` (Dict[SourceType, List[Dict[str, Any]]]): Data from different sources

**Returns:**
- `Dict[str, Any]`: Fusion statistics

##### hybrid_retrieval()

```python
hybrid_retrieval(
    query: str,
    retrieval_types: List[str] = None,
    max_nodes: int = 20,
    similarity_threshold: float = 0.7
) -> RetrievalResult
```

Perform hybrid retrieval combining multiple strategies.

**Parameters:**
- `query` (str): Search query
- `retrieval_types` (Optional[List[str]]): Types of retrieval to use
- `max_nodes` (int): Maximum nodes to return
- `similarity_threshold` (float): Minimum similarity threshold

**Returns:**
- `RetrievalResult`: Retrieval results with nodes, edges, and paths

##### reasoning_writeback()

```python
reasoning_writeback(
    reasoning_step: Dict[str, Any],
    create_nodes: bool = True,
    create_edges: bool = True
) -> Dict[str, Any]
```

Write back reasoning steps as graph nodes and edges.

**Parameters:**
- `reasoning_step` (Dict[str, Any]): Reasoning step data
- `create_nodes` (bool): Whether to create nodes
- `create_edges` (bool): Whether to create edges

**Returns:**
- `Dict[str, Any]`: Writeback result

### Enums

#### NodeType

```python
class NodeType(Enum):
    CONCEPT = "concept"
    PAPER = "paper"
    AUTHOR = "author"
    INSTITUTION = "institution"
    OBSERVATION = "observation"
    CLAIM = "claim"
    EXPERIMENT = "experiment"
    REPAIR_ACTION = "repair_action"
    HYPOTHESIS = "hypothesis"
    TOOL_RESULT = "tool_result"
    RESEARCH_FINDING = "research_finding"
    CODE_SNIPPET = "code_snippet"
    DATASET = "dataset"
```

#### EdgeType

```python
class EdgeType(Enum):
    CITES = "cites"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    MENTIONS = "mentions"
    AUTHORED_BY = "authored_by"
    AFFILIATED_WITH = "affiliated_with"
    IMPLEMENTS = "implements"
    USES = "uses"
    VALIDATES = "validates"
    REFUTES = "refutes"
    EXTENDS = "extends"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
```

#### SourceType

```python
class SourceType(Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    GITHUB = "github"
    WIKIPEDIA = "wikipedia"
    PUBMED = "pubmed"
    INTERNAL = "internal"
    USER_INPUT = "user_input"
    TOOL_OUTPUT = "tool_output"
```

---

## Stage 4: Diffusion Repair API

### RuntimeRepairOperator

Runtime repair operator with fallback mechanisms.

#### Constructor

```python
RuntimeRepairOperator()
```

#### Methods

##### repair_code()

```python
repair_code(
    broken_code: str,
    language: LanguageType,
    error_type: str = "unknown",
    context: Dict[str, Any] = None
) -> RepairResult
```

Repair broken code with fallback mechanisms.

**Parameters:**
- `broken_code` (str): Code to repair
- `language` (LanguageType): Programming language
- `error_type` (str): Type of error
- `context` (Optional[Dict[str, Any]]): Additional context

**Returns:**
- `RepairResult`: Repair result with candidates and metadata

### MultiSeedVotingSystem

Multi-seed voting system for repair candidate selection.

#### Constructor

```python
MultiSeedVotingSystem(num_seeds: int = 5)
```

**Parameters:**
- `num_seeds` (int): Number of seeds for candidate generation

#### Methods

##### generate_repair_candidates()

```python
generate_repair_candidates(
    broken_code: str,
    language: LanguageType,
    diffusion_core: DiffusionRepairCore
) -> List[RepairCandidate]
```

Generate multiple repair candidates using different seeds.

**Parameters:**
- `broken_code` (str): Code to repair
- `language` (LanguageType): Programming language
- `diffusion_core` (DiffusionRepairCore): Diffusion repair engine

**Returns:**
- `List[RepairCandidate]`: Generated repair candidates

##### vote_on_candidates()

```python
vote_on_candidates(
    candidates: List[RepairCandidate],
    voting_criteria: Dict[str, float] = None
) -> RepairCandidate
```

Vote on repair candidates to select the best one.

**Parameters:**
- `candidates` (List[RepairCandidate]): Candidates to vote on
- `voting_criteria` (Optional[Dict[str, float]]): Voting weights

**Returns:**
- `RepairCandidate`: Best candidate

### Enums

#### LanguageType

```python
class LanguageType(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    RUST = "rust"
    GO = "go"
    SQL = "sql"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"
```

#### RepairStrategy

```python
class RepairStrategy(Enum):
    NOISE_BAND_SELECTION = "noise_band_selection"
    MULTI_SEED_VOTING = "multi_seed_voting"
    CONTEXT_AWARE = "context_aware"
    MINIMAL_EDIT = "minimal_edit"
```

---

## Stage 5: RLHF & Agentic RL API

### PreferenceDataPipeline

Pipeline for collecting and processing preference data.

#### Constructor

```python
PreferenceDataPipeline(storage_path: str = "extensions/preference_data")
```

**Parameters:**
- `storage_path` (str): Path for preference data storage

#### Methods

##### collect_preference()

```python
collect_preference(
    query: str,
    response_a: str,
    response_b: str,
    preference: int,
    preference_type: PreferenceType,
    confidence: float = 1.0,
    annotator_id: str = None,
    metadata: Dict[str, Any] = None
) -> str
```

Collect a new preference data point.

**Parameters:**
- `query` (str): Original query
- `response_a` (str): First response option
- `response_b` (str): Second response option
- `preference` (int): Preference (0 for A, 1 for B, -1 for tie)
- `preference_type` (PreferenceType): Type of preference
- `confidence` (float): Confidence in preference
- `annotator_id` (Optional[str]): Annotator identifier
- `metadata` (Optional[Dict[str, Any]]): Additional metadata

**Returns:**
- `str`: Preference ID

### OnlineAgenticRL

Online reinforcement learning for agentic behavior.

#### Constructor

```python
OnlineAgenticRL(reward_model: RewardModel)
```

**Parameters:**
- `reward_model` (RewardModel): Reward model for quality assessment

#### Methods

##### select_action()

```python
select_action(
    state: Dict[str, Any],
    available_actions: List[str]
) -> Tuple[str, Dict[str, Any]]
```

Select action using current policy.

**Parameters:**
- `state` (Dict[str, Any]): Current state representation
- `available_actions` (List[str]): Available actions

**Returns:**
- `Tuple[str, Dict[str, Any]]`: Selected action and metadata

##### record_reward_signal()

```python
record_reward_signal(action_id: str, reward_signals: List[RewardSignal])
```

Record reward signals for an action.

**Parameters:**
- `action_id` (str): Action identifier
- `reward_signals` (List[RewardSignal]): Reward signals

### MultiObjectiveAlignment

Multi-objective alignment system.

#### Constructor

```python
MultiObjectiveAlignment()
```

#### Methods

##### evaluate_alignment()

```python
evaluate_alignment(
    response: str,
    context: Dict[str, Any]
) -> Dict[AlignmentObjective, float]
```

Evaluate response against all alignment objectives.

**Parameters:**
- `response` (str): Response to evaluate
- `context` (Dict[str, Any]): Context information

**Returns:**
- `Dict[AlignmentObjective, float]`: Alignment scores

### Enums

#### PreferenceType

```python
class PreferenceType(Enum):
    HUMAN_FEEDBACK = "human_feedback"
    AUTOMATED_METRIC = "automated_metric"
    TOOL_SUCCESS = "tool_success"
    QUALITY_SCORE = "quality_score"
    EFFICIENCY_SCORE = "efficiency_score"
    SAFETY_SCORE = "safety_score"
```

#### AlignmentObjective

```python
class AlignmentObjective(Enum):
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
```

---

## Stage 6: Cross-Module Synergies API

### UnifiedOrchestrator

Unified orchestrator for all cross-module synergies.

#### Constructor

```python
UnifiedOrchestrator()
```

#### Methods

##### initialize_synergies()

```python
initialize_synergies(configurations: Dict[SynergyType, SynergyConfiguration])
```

Initialize cross-module synergies.

**Parameters:**
- `configurations` (Dict[SynergyType, SynergyConfiguration]): Synergy configurations

##### process_request()

```python
async process_request(request: Dict[str, Any]) -> Dict[str, Any]
```

Process request using available synergies.

**Parameters:**
- `request` (Dict[str, Any]): Request to process

**Returns:**
- `Dict[str, Any]`: Processing result

### RLHFTunedDiffusionRepair

Diffusion repair enhanced with RLHF feedback.

#### Constructor

```python
RLHFTunedDiffusionRepair(
    repair_operator: RuntimeRepairOperator,
    preference_pipeline: PreferenceDataPipeline,
    agentic_rl: OnlineAgenticRL
)
```

**Parameters:**
- `repair_operator` (RuntimeRepairOperator): Base repair operator
- `preference_pipeline` (PreferenceDataPipeline): Preference collection system
- `agentic_rl` (OnlineAgenticRL): RL system

#### Methods

##### repair_with_rlhf()

```python
repair_with_rlhf(
    broken_code: str,
    language: LanguageType,
    context: Dict[str, Any] = None
) -> Dict[str, Any]
```

Repair code using RLHF-enhanced diffusion.

**Parameters:**
- `broken_code` (str): Code to repair
- `language` (LanguageType): Programming language
- `context` (Optional[Dict[str, Any]]): Additional context

**Returns:**
- `Dict[str, Any]`: Enhanced repair result

---

## Integration Orchestrator API

### AIResearchAgentExtensions

Main integration class for all AI Research Agent extensions.

#### Constructor

```python
AIResearchAgentExtensions(config_path: str = "extensions/integration_config.json")
```

**Parameters:**
- `config_path` (str): Path to integration configuration

#### Methods

##### initialize_all_stages()

```python
async initialize_all_stages() -> Dict[str, Any]
```

Initialize all extension stages.

**Returns:**
- `Dict[str, Any]`: Integration status

##### process_enhanced_request()

```python
async process_enhanced_request(request: Dict[str, Any]) -> Dict[str, Any]
```

Process request using all available enhancements.

**Parameters:**
- `request` (Dict[str, Any]): Request to process

**Returns:**
- `Dict[str, Any]`: Enhanced processing result

##### integrate_with_research_agent()

```python
integrate_with_research_agent(research_agent) -> List[str]
```

Integrate extensions with the main research agent.

**Parameters:**
- `research_agent`: Research agent instance

**Returns:**
- `List[str]`: Integration points

##### get_performance_dashboard()

```python
get_performance_dashboard() -> Dict[str, Any]
```

Get comprehensive performance dashboard.

**Returns:**
- `Dict[str, Any]`: Performance dashboard data

---

## Error Handling

All API methods follow consistent error handling patterns:

### Common Exceptions

- `ValueError`: Invalid parameter values
- `FileNotFoundError`: Missing configuration or data files
- `RuntimeError`: System or integration errors
- `TypeError`: Incorrect parameter types

### Error Response Format

```python
{
    "success": False,
    "error": "Error message",
    "error_type": "ValueError",
    "timestamp": "2024-12-20T10:30:00Z",
    "context": {
        "module": "stage_name",
        "operation": "operation_name"
    }
}
```

### Best Practices

1. **Always check return values** for success status
2. **Handle exceptions gracefully** with try-catch blocks
3. **Use logging** for debugging and monitoring
4. **Validate inputs** before API calls
5. **Monitor performance** using observability features

---

## Rate Limits and Quotas

### Default Limits

- **Events per minute**: 1000
- **Memory items per tier**: Based on token limits
- **Graph nodes**: No hard limit (memory dependent)
- **Preference data points**: 10,000 per day
- **A/B experiments**: 50 active experiments

### Customization

Limits can be adjusted in configuration files or through API parameters.

---

## Versioning

API follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

Current version: **1.0.0**

---

## Support

For API support and questions:

1. Check the [README.md](README.md) for usage examples
2. Review the [troubleshooting guide](README.md#troubleshooting)
3. Examine the tutorial notebooks in `extensions/tutorials/`
4. Run the test suite for validation

---

*Last updated: December 2024*