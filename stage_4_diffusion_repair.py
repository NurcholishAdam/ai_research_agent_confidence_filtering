# -*- coding: utf-8 -*-
"""
Stage 4: Diffusion-Based Repair & Generation
Runtime repair operator with multi-seed voting and synthetic data generation
"""

import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import ast
import re
from diffusers import DDPMScheduler
import torch.nn.functional as F

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

class RepairStrategy(Enum):
    NOISE_BAND_SELECTION = "noise_band_selection"
    MULTI_SEED_VOTING = "multi_seed_voting"
    CONTEXT_AWARE = "context_aware"
    MINIMAL_EDIT = "minimal_edit"

@dataclass
class RepairCandidate:
    """Individual repair candidate"""
    candidate_id: str
    original_code: str
    repaired_code: str
    confidence_score: float
    edit_distance: int
    repair_strategy: RepairStrategy
    language_type: LanguageType
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class RepairResult:
    """Result of repair operation"""
    success: bool
    best_candidate: Optional[RepairCandidate]
    all_candidates: List[RepairCandidate]
    repair_time: float
    error_message: Optional[str]
    provenance: Dict[str, Any]

class LanguageAwareNoiseScheduler:
    """Language-specific noise scheduling for diffusion repair"""
    
    def __init__(self):
        # Language-specific noise band configurations
        self.noise_configs = {
            LanguageType.PYTHON: {
                "syntax_noise_start": 0.1,
                "logic_noise_start": 0.3,
                "semantic_noise_start": 0.5,
                "max_noise": 0.8
            },
            LanguageType.JAVASCRIPT: {
                "syntax_noise_start": 0.15,
                "logic_noise_start": 0.35,
                "semantic_noise_start": 0.55,
                "max_noise": 0.85
            },
            LanguageType.SQL: {
                "syntax_noise_start": 0.05,
                "logic_noise_start": 0.25,
                "semantic_noise_start": 0.45,
                "max_noise": 0.7
            }
        }
        
        print("🔧 Language-Aware Noise Scheduler initialized")
    
    def get_noise_schedule(self, language: LanguageType, error_type: str) -> List[float]:
        """Get appropriate noise schedule for language and error type"""
        
        config = self.noise_configs.get(language, self.noise_configs[LanguageType.PYTHON])
        
        if error_type in ["syntax_error", "parse_error"]:
            start_noise = config["syntax_noise_start"]
        elif error_type in ["logic_error", "runtime_error"]:
            start_noise = config["logic_noise_start"]
        else:
            start_noise = config["semantic_noise_start"]
        
        # Generate noise schedule
        num_steps = 20
        noise_schedule = []
        
        for i in range(num_steps):
            # Exponential decay from max_noise to start_noise
            progress = i / (num_steps - 1)
            noise_level = config["max_noise"] * (1 - progress) + start_noise * progress
            noise_schedule.append(noise_level)
        
        return noise_schedule

class DiffusionRepairCore:
    """Core diffusion model for code repair"""
    
    def __init__(self, model_dim: int = 512):
        self.model_dim = model_dim
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # Simple transformer-like architecture for demonstration
        self.embedding_dim = 256
        self.vocab_size = 50000  # Approximate vocabulary size
        
        # Initialize model components (simplified)
        self.token_embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.position_embedding = torch.nn.Embedding(1024, self.embedding_dim)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1
            ),
            num_layers=6
        )
        self.output_projection = torch.nn.Linear(self.embedding_dim, self.vocab_size)
        
        print("🌊 Diffusion Repair Core initialized")
    
    def encode_code(self, code: str) -> torch.Tensor:
        """Encode code string to tensor representation"""
        # Simplified tokenization (in practice, use proper tokenizer)
        tokens = self._tokenize_code(code)
        token_ids = torch.tensor([hash(token) % self.vocab_size for token in tokens])
        
        # Add position embeddings
        positions = torch.arange(len(token_ids))
        
        # Embed tokens and positions
        token_embeds = self.token_embedding(token_ids)
        pos_embeds = self.position_embedding(positions)
        
        return token_embeds + pos_embeds
    
    def decode_code(self, tensor: torch.Tensor) -> str:
        """Decode tensor representation back to code string"""
        # Apply output projection
        logits = self.output_projection(tensor)
        token_ids = torch.argmax(logits, dim=-1)
        
        # Convert back to tokens (simplified)
        tokens = [f"token_{tid.item()}" for tid in token_ids]
        return " ".join(tokens)
    
    def add_noise(self, code_tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add controlled noise to code representation"""
        noise = torch.randn_like(code_tensor) * noise_level
        return code_tensor + noise
    
    def denoise_step(self, noisy_tensor: torch.Tensor, timestep: int) -> torch.Tensor:
        """Single denoising step"""
        # Apply transformer
        denoised = self.transformer(noisy_tensor.unsqueeze(0)).squeeze(0)
        return denoised
    
    def repair_with_diffusion(self, broken_code: str, language: LanguageType,
                             num_steps: int = 20, num_candidates: int = 5) -> List[str]:
        """Repair code using diffusion process"""
        
        # Encode broken code
        code_tensor = self.encode_code(broken_code)
        
        candidates = []
        
        for candidate_idx in range(num_candidates):
            # Start with noisy version
            current_tensor = self.add_noise(code_tensor, 0.8)
            
            # Denoising process
            for step in range(num_steps):
                current_tensor = self.denoise_step(current_tensor, step)
            
            # Decode to get repaired code
            repaired_code = self.decode_code(current_tensor)
            candidates.append(repaired_code)
        
        return candidates
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Simple code tokenization"""
        # Basic tokenization (in practice, use language-specific tokenizers)
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return tokens

class MultiSeedVotingSystem:
    """Multi-seed voting system for repair candidate selection"""
    
    def __init__(self, num_seeds: int = 5):
        self.num_seeds = num_seeds
        self.voting_history: List[Dict[str, Any]] = []
        
        print(f"🗳️ Multi-Seed Voting System initialized (seeds: {num_seeds})")
    
    def generate_repair_candidates(self, broken_code: str, language: LanguageType,
                                  diffusion_core: DiffusionRepairCore) -> List[RepairCandidate]:
        """Generate multiple repair candidates using different seeds"""
        
        candidates = []
        
        for seed in range(self.num_seeds):
            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Generate repair candidates
            repaired_codes = diffusion_core.repair_with_diffusion(
                broken_code, language, num_candidates=3
            )
            
            for i, repaired_code in enumerate(repaired_codes):
                candidate_id = f"seed_{seed}_candidate_{i}"
                
                # Calculate edit distance
                edit_distance = self._calculate_edit_distance(broken_code, repaired_code)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence(
                    broken_code, repaired_code, language
                )
                
                candidate = RepairCandidate(
                    candidate_id=candidate_id,
                    original_code=broken_code,
                    repaired_code=repaired_code,
                    confidence_score=confidence_score,
                    edit_distance=edit_distance,
                    repair_strategy=RepairStrategy.MULTI_SEED_VOTING,
                    language_type=language,
                    metadata={
                        "seed": seed,
                        "candidate_index": i,
                        "generation_timestamp": datetime.now().isoformat()
                    },
                    timestamp=datetime.now()
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def vote_on_candidates(self, candidates: List[RepairCandidate],
                          voting_criteria: Dict[str, float] = None) -> RepairCandidate:
        """Vote on repair candidates to select the best one"""
        
        if not candidates:
            return None
        
        # Default voting criteria
        if voting_criteria is None:
            voting_criteria = {
                "confidence_weight": 0.4,
                "edit_distance_weight": 0.3,
                "syntax_validity_weight": 0.2,
                "diversity_weight": 0.1
            }
        
        # Calculate voting scores
        scored_candidates = []
        
        for candidate in candidates:
            # Confidence score (higher is better)
            confidence_score = candidate.confidence_score
            
            # Edit distance score (lower is better, so invert)
            max_edit_distance = max(c.edit_distance for c in candidates)
            edit_distance_score = 1.0 - (candidate.edit_distance / max(max_edit_distance, 1))
            
            # Syntax validity score
            syntax_score = self._check_syntax_validity(
                candidate.repaired_code, candidate.language_type
            )
            
            # Diversity score (how different from other candidates)
            diversity_score = self._calculate_diversity_score(candidate, candidates)
            
            # Composite voting score
            voting_score = (
                voting_criteria["confidence_weight"] * confidence_score +
                voting_criteria["edit_distance_weight"] * edit_distance_score +
                voting_criteria["syntax_validity_weight"] * syntax_score +
                voting_criteria["diversity_weight"] * diversity_score
            )
            
            scored_candidates.append((candidate, voting_score))
        
        # Sort by voting score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Record voting result
        voting_result = {
            "timestamp": datetime.now().isoformat(),
            "num_candidates": len(candidates),
            "winner_id": scored_candidates[0][0].candidate_id,
            "winner_score": scored_candidates[0][1],
            "voting_criteria": voting_criteria
        }
        self.voting_history.append(voting_result)
        
        return scored_candidates[0][0]
    
    def _calculate_edit_distance(self, original: str, repaired: str) -> int:
        """Calculate Levenshtein distance between original and repaired code"""
        
        # Simple edit distance calculation
        m, n = len(original), len(repaired)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if original[i-1] == repaired[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _calculate_confidence(self, original: str, repaired: str, language: LanguageType) -> float:
        """Calculate confidence score for repair"""
        
        # Simple heuristics for confidence
        confidence = 0.5  # Base confidence
        
        # Boost confidence if syntax is valid
        if self._check_syntax_validity(repaired, language) > 0.8:
            confidence += 0.2
        
        # Boost confidence for minimal changes
        edit_distance = self._calculate_edit_distance(original, repaired)
        if edit_distance < len(original) * 0.1:  # Less than 10% change
            confidence += 0.2
        
        # Boost confidence if structure is preserved
        if self._structure_preserved(original, repaired, language):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _check_syntax_validity(self, code: str, language: LanguageType) -> float:
        """Check syntax validity of code"""
        
        try:
            if language == LanguageType.PYTHON:
                ast.parse(code)
                return 1.0
            elif language == LanguageType.JSON:
                json.loads(code)
                return 1.0
            else:
                # For other languages, use simple heuristics
                return self._heuristic_syntax_check(code, language)
        except:
            return 0.0
    
    def _heuristic_syntax_check(self, code: str, language: LanguageType) -> float:
        """Heuristic syntax checking for non-Python languages"""
        
        score = 0.5  # Base score
        
        # Check balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return 0.0
                if brackets[stack.pop()] != char:
                    return 0.0
        
        if not stack:
            score += 0.3
        
        # Language-specific checks
        if language == LanguageType.JAVASCRIPT:
            if code.count(';') > 0:  # Has semicolons
                score += 0.2
        
        return min(score, 1.0)
    
    def _structure_preserved(self, original: str, repaired: str, language: LanguageType) -> bool:
        """Check if code structure is preserved"""
        
        # Simple structure preservation check
        original_lines = len(original.split('\n'))
        repaired_lines = len(repaired.split('\n'))
        
        # Structure preserved if line count is similar
        return abs(original_lines - repaired_lines) <= max(original_lines * 0.2, 2)
    
    def _calculate_diversity_score(self, candidate: RepairCandidate, 
                                  all_candidates: List[RepairCandidate]) -> float:
        """Calculate diversity score for a candidate"""
        
        if len(all_candidates) <= 1:
            return 1.0
        
        # Calculate average edit distance to other candidates
        distances = []
        for other in all_candidates:
            if other.candidate_id != candidate.candidate_id:
                distance = self._calculate_edit_distance(
                    candidate.repaired_code, other.repaired_code
                )
                distances.append(distance)
        
        if not distances:
            return 1.0
        
        avg_distance = sum(distances) / len(distances)
        max_possible_distance = max(len(candidate.repaired_code), 
                                   max(len(c.repaired_code) for c in all_candidates))
        
        return avg_distance / max(max_possible_distance, 1)

class RuntimeRepairOperator:
    """Runtime repair operator with fallback mechanisms"""
    
    def __init__(self):
        self.noise_scheduler = LanguageAwareNoiseScheduler()
        self.diffusion_core = DiffusionRepairCore()
        self.voting_system = MultiSeedVotingSystem()
        
        # Repair statistics
        self.repair_stats = {
            "total_repairs": 0,
            "successful_repairs": 0,
            "fallback_repairs": 0,
            "failed_repairs": 0
        }
        
        print("🔧 Runtime Repair Operator initialized")
    
    def repair_code(self, broken_code: str, language: LanguageType,
                   error_type: str = "unknown", context: Dict[str, Any] = None) -> RepairResult:
        """Repair broken code with fallback mechanisms"""
        
        start_time = datetime.now()
        
        try:
            # Generate repair candidates using diffusion
            candidates = self.voting_system.generate_repair_candidates(
                broken_code, language, self.diffusion_core
            )
            
            if not candidates:
                return self._fallback_repair(broken_code, language, error_type)
            
            # Vote on candidates
            best_candidate = self.voting_system.vote_on_candidates(candidates)
            
            # Validate repair
            if self._validate_repair(best_candidate):
                self.repair_stats["successful_repairs"] += 1
                
                repair_time = (datetime.now() - start_time).total_seconds()
                
                return RepairResult(
                    success=True,
                    best_candidate=best_candidate,
                    all_candidates=candidates,
                    repair_time=repair_time,
                    error_message=None,
                    provenance={
                        "repair_method": "diffusion_voting",
                        "num_candidates": len(candidates),
                        "language": language.value,
                        "error_type": error_type,
                        "context": context or {}
                    }
                )
            else:
                return self._fallback_repair(broken_code, language, error_type)
        
        except Exception as e:
            return self._fallback_repair(broken_code, language, error_type, str(e))
        
        finally:
            self.repair_stats["total_repairs"] += 1
    
    def _fallback_repair(self, broken_code: str, language: LanguageType,
                        error_type: str, error_message: str = None) -> RepairResult:
        """Fallback repair using simple heuristics"""
        
        self.repair_stats["fallback_repairs"] += 1
        
        # Simple fallback repairs
        repaired_code = broken_code
        
        if language == LanguageType.PYTHON:
            repaired_code = self._python_fallback_repair(broken_code, error_type)
        elif language == LanguageType.JSON:
            repaired_code = self._json_fallback_repair(broken_code)
        
        # Create fallback candidate
        fallback_candidate = RepairCandidate(
            candidate_id="fallback_repair",
            original_code=broken_code,
            repaired_code=repaired_code,
            confidence_score=0.3,  # Low confidence for fallback
            edit_distance=self.voting_system._calculate_edit_distance(broken_code, repaired_code),
            repair_strategy=RepairStrategy.MINIMAL_EDIT,
            language_type=language,
            metadata={
                "fallback_method": "heuristic",
                "error_type": error_type
            },
            timestamp=datetime.now()
        )
        
        return RepairResult(
            success=repaired_code != broken_code,
            best_candidate=fallback_candidate,
            all_candidates=[fallback_candidate],
            repair_time=0.1,  # Fast fallback
            error_message=error_message,
            provenance={
                "repair_method": "fallback_heuristic",
                "language": language.value,
                "error_type": error_type
            }
        )
    
    def _python_fallback_repair(self, code: str, error_type: str) -> str:
        """Simple Python fallback repairs"""
        
        repaired = code
        
        # Fix common indentation issues
        if "IndentationError" in error_type:
            lines = code.split('\n')
            repaired_lines = []
            for line in lines:
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    repaired_lines.append('    ' + line)  # Add 4-space indent
                else:
                    repaired_lines.append(line)
            repaired = '\n'.join(repaired_lines)
        
        # Fix missing colons
        if "SyntaxError" in error_type:
            repaired = re.sub(r'(if|for|while|def|class)\s+[^:]+(?<!:)$', r'\g<0>:', repaired, flags=re.MULTILINE)
        
        return repaired
    
    def _json_fallback_repair(self, code: str) -> str:
        """Simple JSON fallback repairs"""
        
        repaired = code.strip()
        
        # Fix missing quotes around keys
        repaired = re.sub(r'(\w+):', r'"\1":', repaired)
        
        # Fix trailing commas
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        
        return repaired
    
    def _validate_repair(self, candidate: RepairCandidate) -> bool:
        """Validate that repair is successful"""
        
        # Check syntax validity
        syntax_score = self.voting_system._check_syntax_validity(
            candidate.repaired_code, candidate.language_type
        )
        
        # Require high syntax validity and reasonable confidence
        return syntax_score > 0.8 and candidate.confidence_score > 0.4
    
    def get_repair_statistics(self) -> Dict[str, Any]:
        """Get repair operation statistics"""
        
        total = max(self.repair_stats["total_repairs"], 1)
        
        return {
            **self.repair_stats,
            "success_rate": self.repair_stats["successful_repairs"] / total,
            "fallback_rate": self.repair_stats["fallback_repairs"] / total,
            "failure_rate": self.repair_stats["failed_repairs"] / total
        }

# Integration with existing system
def integrate_diffusion_repair():
    """Integration point for diffusion repair system"""
    
    repair_operator = RuntimeRepairOperator()
    
    print("🌊 Stage 4: Diffusion Repair & Generation integrated")
    print("   Features:")
    print("   - Language-aware noise scheduling")
    print("   - Multi-seed voting system")
    print("   - Runtime repair operator")
    print("   - Fallback mechanisms")
    
    return repair_operator

if __name__ == "__main__":
    # Demo usage
    repair_operator = integrate_diffusion_repair()
    
    # Test repair
    broken_python = """
def hello_world(
    print("Hello, World!")
"""
    
    result = repair_operator.repair_code(
        broken_python, 
        LanguageType.PYTHON, 
        "SyntaxError"
    )
    
    print(f"\nRepair Result:")
    print(f"Success: {result.success}")
    if result.best_candidate:
        print(f"Repaired Code: {result.best_candidate.repaired_code}")
        print(f"Confidence: {result.best_candidate.confidence_score}")