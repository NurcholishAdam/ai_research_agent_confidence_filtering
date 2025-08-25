# -*- coding: utf-8 -*-
"""
Stage 1: Enhanced Observability + Config Flags for AI Research Agent
Comprehensive monitoring and A/B testing infrastructure
"""

import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

class ModuleType(Enum):
    RLHF = "rlhf"
    CONTEXT_ENGINEERING = "context_engineering"
    SEMANTIC_GRAPH = "semantic_graph"
    DIFFUSION_REPAIR = "diffusion_repair"
    MULTI_AGENT = "multi_agent"
    TOOL_REASONING = "tool_reasoning"

class ExperimentType(Enum):
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    CANARY = "canary"
    FEATURE_FLAG = "feature_flag"

@dataclass
class ModuleConfig:
    """Configuration for individual modules"""
    module_type: ModuleType
    enabled: bool
    version: str
    parameters: Dict[str, Any]
    experiment_group: Optional[str] = None
    rollout_percentage: float = 100.0
    dependencies: List[str] = None

@dataclass
class ExperimentConfig:
    """A/B testing experiment configuration"""
    experiment_id: str
    experiment_type: ExperimentType
    name: str
    description: str
    variants: Dict[str, Dict[str, Any]]
    traffic_allocation: Dict[str, float]
    success_metrics: List[str]
    start_date: datetime
    end_date: Optional[datetime]
    active: bool = True

@dataclass
class ObservabilityEvent:
    """Individual observability event"""
    event_id: str
    timestamp: datetime
    module_type: ModuleType
    event_type: str
    session_id: str
    user_id: Optional[str]
    data: Dict[str, Any]
    experiment_context: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

class ObservabilityCollector:
    """Centralized observability and metrics collection"""
    
    def __init__(self, config_path: str = "extensions/observability_config.json"):
        self.config_path = Path(config_path)
        self.events: List[ObservabilityEvent] = []
        self.module_configs: Dict[ModuleType, ModuleConfig] = {}
        self.experiments: Dict[str, ExperimentConfig] = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load configuration
        self._load_configuration()
        
        # Performance tracking
        self.performance_cache: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        
        print("📊 Stage 1: Enhanced Observability initialized")
        print(f"   Modules configured: {len(self.module_configs)}")
        print(f"   Active experiments: {len([e for e in self.experiments.values() if e.active])}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("ObservabilityCollector")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # File handler with rotation
        file_handler = logging.FileHandler(logs_dir / "observability.log")
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_configuration(self):
        """Load module and experiment configurations"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                
            # Load module configs
            for module_data in config_data.get("modules", []):
                module_config = ModuleConfig(
                    module_type=ModuleType(module_data["module_type"]),
                    enabled=module_data["enabled"],
                    version=module_data["version"],
                    parameters=module_data["parameters"],
                    experiment_group=module_data.get("experiment_group"),
                    rollout_percentage=module_data.get("rollout_percentage", 100.0),
                    dependencies=module_data.get("dependencies", [])
                )
                self.module_configs[module_config.module_type] = module_config
            
            # Load experiment configs
            for exp_data in config_data.get("experiments", []):
                experiment = ExperimentConfig(
                    experiment_id=exp_data["experiment_id"],
                    experiment_type=ExperimentType(exp_data["experiment_type"]),
                    name=exp_data["name"],
                    description=exp_data["description"],
                    variants=exp_data["variants"],
                    traffic_allocation=exp_data["traffic_allocation"],
                    success_metrics=exp_data["success_metrics"],
                    start_date=datetime.fromisoformat(exp_data["start_date"]),
                    end_date=datetime.fromisoformat(exp_data["end_date"]) if exp_data.get("end_date") else None,
                    active=exp_data.get("active", True)
                )
                self.experiments[experiment.experiment_id] = experiment
        else:
            # Create default configuration
            self._create_default_configuration()
    
    def _create_default_configuration(self):
        """Create default configuration file"""
        default_config = {
            "modules": [
                {
                    "module_type": "rlhf",
                    "enabled": True,
                    "version": "1.0.0",
                    "parameters": {
                        "feedback_collection_rate": 0.1,
                        "reward_model_threshold": 0.7,
                        "training_batch_size": 32
                    },
                    "rollout_percentage": 100.0
                },
                {
                    "module_type": "context_engineering",
                    "enabled": True,
                    "version": "1.0.0",
                    "parameters": {
                        "max_context_items": 15,
                        "quality_threshold": 0.7,
                        "adaptive_processing": True
                    },
                    "rollout_percentage": 100.0
                },
                {
                    "module_type": "semantic_graph",
                    "enabled": True,
                    "version": "1.0.0",
                    "parameters": {
                        "graph_expansion_depth": 3,
                        "similarity_threshold": 0.8,
                        "reasoning_writeback": True
                    },
                    "rollout_percentage": 100.0
                }
            ],
            "experiments": []
        }
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        # Load the default config
        self._load_configuration()
    
    def track_event(self, module_type: ModuleType, event_type: str, 
                   session_id: str, data: Dict[str, Any],
                   user_id: Optional[str] = None,
                   performance_metrics: Optional[Dict[str, Any]] = None) -> str:
        """Track an observability event"""
        
        event_id = str(uuid.uuid4())
        
        # Get experiment context
        experiment_context = self._get_experiment_context(module_type, session_id)
        
        event = ObservabilityEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            module_type=module_type,
            event_type=event_type,
            session_id=session_id,
            user_id=user_id,
            data=data,
            experiment_context=experiment_context,
            performance_metrics=performance_metrics
        )
        
        self.events.append(event)
        
        # Log structured event
        self.logger.info(json.dumps({
            "event_id": event_id,
            "module": module_type.value,
            "event_type": event_type,
            "session_id": session_id,
            "timestamp": event.timestamp.isoformat(),
            "data": data,
            "experiment_context": experiment_context,
            "performance_metrics": performance_metrics
        }))
        
        # Update performance cache
        if performance_metrics:
            self._update_performance_cache(module_type, event_type, performance_metrics)
        
        return event_id
    
    def track_performance(self, module_type: ModuleType, operation: str,
                         execution_time: float, success: bool,
                         additional_metrics: Optional[Dict[str, Any]] = None) -> str:
        """Track performance metrics"""
        
        performance_data = {
            "operation": operation,
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        if additional_metrics:
            performance_data.update(additional_metrics)
        
        # Track in performance cache
        cache_key = f"{module_type.value}_{operation}"
        if cache_key not in self.performance_cache:
            self.performance_cache[cache_key] = []
        
        self.performance_cache[cache_key].append(execution_time)
        
        # Track errors
        if not success:
            error_key = f"{module_type.value}_{operation}_errors"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        return self.track_event(
            module_type=module_type,
            event_type="performance_metric",
            session_id="system",
            data=performance_data,
            performance_metrics={
                "execution_time": execution_time,
                "success": success
            }
        )
    
    def is_module_enabled(self, module_type: ModuleType, session_id: str = None) -> bool:
        """Check if a module is enabled for the current session"""
        
        if module_type not in self.module_configs:
            return False
        
        config = self.module_configs[module_type]
        
        if not config.enabled:
            return False
        
        # Check rollout percentage
        if config.rollout_percentage < 100.0:
            # Use session_id for consistent rollout
            if session_id:
                hash_value = hash(session_id) % 100
                if hash_value >= config.rollout_percentage:
                    return False
        
        # Check dependencies
        if config.dependencies:
            for dep in config.dependencies:
                try:
                    dep_module = ModuleType(dep)
                    if not self.is_module_enabled(dep_module, session_id):
                        return False
                except ValueError:
                    # Invalid dependency
                    return False
        
        return True
    
    def get_module_config(self, module_type: ModuleType) -> Optional[ModuleConfig]:
        """Get configuration for a specific module"""
        return self.module_configs.get(module_type)
    
    def get_experiment_variant(self, experiment_id: str, session_id: str) -> Optional[str]:
        """Get experiment variant for a session"""
        
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        if not experiment.active:
            return None
        
        # Check if experiment is within date range
        now = datetime.now()
        if now < experiment.start_date:
            return None
        if experiment.end_date and now > experiment.end_date:
            return None
        
        # Determine variant based on session_id hash
        hash_value = hash(session_id) % 100
        cumulative_percentage = 0
        
        for variant, percentage in experiment.traffic_allocation.items():
            cumulative_percentage += percentage
            if hash_value < cumulative_percentage:
                return variant
        
        return None
    
    def create_experiment(self, name: str, description: str,
                         variants: Dict[str, Dict[str, Any]],
                         traffic_allocation: Dict[str, float],
                         success_metrics: List[str],
                         duration_days: int = 30) -> str:
        """Create a new A/B test experiment"""
        
        experiment_id = str(uuid.uuid4())
        
        experiment = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_type=ExperimentType.AB_TEST,
            name=name,
            description=description,
            variants=variants,
            traffic_allocation=traffic_allocation,
            success_metrics=success_metrics,
            start_date=datetime.now(),
            end_date=datetime.now().replace(day=datetime.now().day + duration_days),
            active=True
        )
        
        self.experiments[experiment_id] = experiment
        
        # Save to configuration
        self._save_configuration()
        
        print(f"🧪 Created experiment: {name} ({experiment_id[:8]}...)")
        print(f"   Variants: {list(variants.keys())}")
        print(f"   Traffic allocation: {traffic_allocation}")
        
        return experiment_id
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive analytics dashboard"""
        
        # Module performance analytics
        module_performance = {}
        for module_type in ModuleType:
            module_events = [e for e in self.events if e.module_type == module_type]
            
            if module_events:
                performance_metrics = [e.performance_metrics for e in module_events if e.performance_metrics]
                
                if performance_metrics:
                    execution_times = [m.get("execution_time", 0) for m in performance_metrics]
                    success_rates = [m.get("success", False) for m in performance_metrics]
                    
                    module_performance[module_type.value] = {
                        "total_events": len(module_events),
                        "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
                        "success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
                        "error_count": self.error_counts.get(f"{module_type.value}_errors", 0)
                    }
        
        # Experiment analytics
        experiment_analytics = {}
        for exp_id, experiment in self.experiments.items():
            exp_events = [e for e in self.events if e.experiment_context and e.experiment_context.get("experiment_id") == exp_id]
            
            variant_performance = {}
            for variant in experiment.variants.keys():
                variant_events = [e for e in exp_events if e.experiment_context.get("variant") == variant]
                
                if variant_events:
                    variant_performance[variant] = {
                        "event_count": len(variant_events),
                        "unique_sessions": len(set(e.session_id for e in variant_events))
                    }
            
            experiment_analytics[exp_id] = {
                "name": experiment.name,
                "active": experiment.active,
                "total_events": len(exp_events),
                "variant_performance": variant_performance
            }
        
        # System health metrics
        system_health = {
            "total_events": len(self.events),
            "events_last_hour": len([e for e in self.events if (datetime.now() - e.timestamp).seconds < 3600]),
            "active_modules": len([m for m in self.module_configs.values() if m.enabled]),
            "active_experiments": len([e for e in self.experiments.values() if e.active]),
            "error_rate": sum(self.error_counts.values()) / max(len(self.events), 1)
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "module_performance": module_performance,
            "experiment_analytics": experiment_analytics,
            "performance_cache_size": len(self.performance_cache)
        }
    
    def _get_experiment_context(self, module_type: ModuleType, session_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment context for an event"""
        
        config = self.module_configs.get(module_type)
        if not config or not config.experiment_group:
            return None
        
        # Find active experiments for this module
        for exp_id, experiment in self.experiments.items():
            if experiment.active and config.experiment_group in experiment.variants:
                variant = self.get_experiment_variant(exp_id, session_id)
                if variant:
                    return {
                        "experiment_id": exp_id,
                        "experiment_name": experiment.name,
                        "variant": variant,
                        "variant_config": experiment.variants[variant]
                    }
        
        return None
    
    def _update_performance_cache(self, module_type: ModuleType, event_type: str, metrics: Dict[str, Any]):
        """Update performance cache with new metrics"""
        
        cache_key = f"{module_type.value}_{event_type}"
        
        if "execution_time" in metrics:
            if cache_key not in self.performance_cache:
                self.performance_cache[cache_key] = []
            
            self.performance_cache[cache_key].append(metrics["execution_time"])
            
            # Keep only last 1000 measurements
            if len(self.performance_cache[cache_key]) > 1000:
                self.performance_cache[cache_key] = self.performance_cache[cache_key][-1000:]
    
    def _save_configuration(self):
        """Save current configuration to file"""
        
        config_data = {
            "modules": [
                {
                    "module_type": config.module_type.value,
                    "enabled": config.enabled,
                    "version": config.version,
                    "parameters": config.parameters,
                    "experiment_group": config.experiment_group,
                    "rollout_percentage": config.rollout_percentage,
                    "dependencies": config.dependencies or []
                }
                for config in self.module_configs.values()
            ],
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "experiment_type": exp.experiment_type.value,
                    "name": exp.name,
                    "description": exp.description,
                    "variants": exp.variants,
                    "traffic_allocation": exp.traffic_allocation,
                    "success_metrics": exp.success_metrics,
                    "start_date": exp.start_date.isoformat(),
                    "end_date": exp.end_date.isoformat() if exp.end_date else None,
                    "active": exp.active
                }
                for exp in self.experiments.values()
            ]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

# Decorator for automatic performance tracking
def track_performance(module_type: ModuleType, operation: str):
    """Decorator to automatically track performance of functions"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            collector = get_observability_collector()
            
            start_time = time.time()
            success = True
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                execution_time = time.time() - start_time
                collector.track_performance(
                    module_type=module_type,
                    operation=operation,
                    execution_time=execution_time,
                    success=success,
                    additional_metrics={
                        "function_name": func.__name__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                )
        
        return wrapper
    return decorator

# Global collector instance
_observability_collector = None

def get_observability_collector() -> ObservabilityCollector:
    """Get the global observability collector instance"""
    global _observability_collector
    if _observability_collector is None:
        _observability_collector = ObservabilityCollector()
    return _observability_collector

def initialize_observability(config_path: str = None) -> ObservabilityCollector:
    """Initialize observability system"""
    global _observability_collector
    _observability_collector = ObservabilityCollector(config_path or "extensions/observability_config.json")
    return _observability_collector