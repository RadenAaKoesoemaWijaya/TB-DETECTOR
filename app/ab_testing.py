"""
A/B Testing Framework untuk TB Detector
Model comparison dan experimentation
"""

import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import uuid
import numpy as np
from collections import defaultdict


class ExperimentStatus(Enum):
    """Experiment status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AllocationMethod(Enum):
    """Traffic allocation methods"""
    RANDOM = "random"  # Pure random
    DETERMINISTIC = "deterministic"  # Hash-based untuk consistent allocation
    WEIGHTED = "weighted"  # Weighted by model performance


@dataclass
class Variant:
    """Experiment variant (model)"""
    id: str
    name: str
    model_name: str
    model_version: str
    traffic_allocation: float = 0.5  # 0.0 to 1.0
    
    # Metrics
    total_predictions: int = 0
    total_latency_ms: float = 0.0
    
    # Outcomes (jika ground truth tersedia)
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # User feedback
    positive_feedback: int = 0
    negative_feedback: int = 0
    
    def avg_latency(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_latency_ms / self.total_predictions
    
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    def sensitivity(self) -> float:
        total_pos = self.true_positives + self.false_negatives
        if total_pos == 0:
            return 0.0
        return self.true_positives / total_pos
    
    def specificity(self) -> float:
        total_neg = self.true_negatives + self.false_positives
        if total_neg == 0:
            return 0.0
        return self.true_negatives / total_neg
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['avg_latency'] = self.avg_latency()
        data['accuracy'] = self.accuracy()
        data['sensitivity'] = self.sensitivity()
        data['specificity'] = self.specificity()
        return data


@dataclass
class Experiment:
    """A/B Test Experiment"""
    id: str
    name: str
    description: str
    status: ExperimentStatus
    
    # Models
    control_variant: Variant
    treatment_variants: List[Variant]
    
    # Configuration
    allocation_method: AllocationMethod
    sample_size: Optional[int] = None  # Target sample size
    duration_days: Optional[int] = None  # Target duration
    
    # Dates
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    
    # Criteria
    min_confidence: float = 0.95
    min_detectable_effect: float = 0.05  # 5% relative improvement
    
    # Results
    winner_variant_id: Optional[str] = None
    results_summary: Dict[str, Any] = field(default_factory=dict)
    
    def total_predictions(self) -> int:
        total = self.control_variant.total_predictions
        for v in self.treatment_variants:
            total += v.total_predictions
        return total
    
    def is_statistically_significant(self) -> bool:
        """Check jika ada winner dengan statistical significance"""
        # Simplified check: minimum sample size
        if self.sample_size and self.total_predictions() < self.sample_size:
            return False
        
        # Check confidence intervals (simplified)
        control_acc = self.control_variant.accuracy()
        
        for variant in self.treatment_variants:
            variant_acc = variant.accuracy()
            relative_improvement = (variant_acc - control_acc) / control_acc if control_acc > 0 else 0
            
            if relative_improvement >= self.min_detectable_effect:
                return True
        
        return False
    
    def get_winner(self) -> Optional[Variant]:
        """Get winning variant jika ada"""
        if not self.is_statistically_significant():
            return None
        
        best = self.control_variant
        best_acc = best.accuracy()
        
        for variant in self.treatment_variants:
            if variant.accuracy() > best_acc:
                best = variant
                best_acc = variant.accuracy()
        
        return best if best != self.control_variant else None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'control_variant': self.control_variant.to_dict(),
            'treatment_variants': [v.to_dict() for v in self.treatment_variants],
            'allocation_method': self.allocation_method.value,
            'sample_size': self.sample_size,
            'duration_days': self.duration_days,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'min_confidence': self.min_confidence,
            'min_detectable_effect': self.min_detectable_effect,
            'winner_variant_id': self.winner_variant_id,
            'results_summary': self.results_summary,
            'total_predictions': self.total_predictions()
        }
        return data


class ABTestingFramework:
    """
    A/B Testing framework untuk model experimentation
    """
    
    def __init__(self, storage_path: str = "data/experiments"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.experiments_file = self.storage_path / "experiments.json"
        self._experiments: Dict[str, Experiment] = {}
        
        # Active assignments: user_id -> variant_id
        self._assignments: Dict[str, str] = {}
        
        self._load_experiments()
    
    def _load_experiments(self):
        """Load experiments dari storage"""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                data = json.load(f)
                for exp_id, exp_data in data.items():
                    self._experiments[exp_id] = self._deserialize_experiment(exp_data)
    
    def _save_experiments(self):
        """Save experiments ke storage"""
        data = {
            exp_id: exp.to_dict()
            for exp_id, exp in self._experiments.items()
        }
        with open(self.experiments_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _deserialize_experiment(self, data: Dict) -> Experiment:
        """Deserialize experiment dari dict"""
        control = Variant(**{k: v for k, v in data['control_variant'].items() 
                            if k in Variant.__dataclass_fields__})
        
        treatments = [
            Variant(**{k: v for k, v in v_data.items() 
                      if k in Variant.__dataclass_fields__})
            for v_data in data['treatment_variants']
        ]
        
        return Experiment(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            status=ExperimentStatus(data['status']),
            control_variant=control,
            treatment_variants=treatments,
            allocation_method=AllocationMethod(data.get('allocation_method', 'random')),
            sample_size=data.get('sample_size'),
            duration_days=data.get('duration_days'),
            created_at=data['created_at'],
            started_at=data.get('started_at'),
            ended_at=data.get('ended_at'),
            min_confidence=data.get('min_confidence', 0.95),
            min_detectable_effect=data.get('min_detectable_effect', 0.05),
            winner_variant_id=data.get('winner_variant_id'),
            results_summary=data.get('results_summary', {})
        )
    
    def create_experiment(
        self,
        name: str,
        control_model: Tuple[str, str],  # (name, version)
        treatment_models: List[Tuple[str, str, float]],  # (name, version, allocation)
        description: str = "",
        allocation_method: AllocationMethod = AllocationMethod.RANDOM,
        sample_size: Optional[int] = None,
        duration_days: Optional[int] = None
    ) -> Experiment:
        """
        Create new A/B test experiment
        
        Args:
            name: Experiment name
            control_model: (model_name, version) untuk control
            treatment_models: List of (model_name, version, allocation) untuk treatments
            description: Experiment description
            allocation_method: Traffic allocation method
            sample_size: Target sample size untuk significance
            duration_days: Target duration
        
        Returns:
            Experiment object
        """
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create control variant
        control = Variant(
            id=f"{exp_id}_control",
            name="control",
            model_name=control_model[0],
            model_version=control_model[1],
            traffic_allocation=1.0 - sum(t[2] for t in treatment_models)
        )
        
        # Create treatment variants
        treatments = []
        for i, (model_name, version, allocation) in enumerate(treatment_models):
            variant = Variant(
                id=f"{exp_id}_treatment_{i}",
                name=f"treatment_{i+1}",
                model_name=model_name,
                model_version=version,
                traffic_allocation=allocation
            )
            treatments.append(variant)
        
        # Normalize allocations
        total = sum(v.traffic_allocation for v in [control] + treatments)
        control.traffic_allocation /= total
        for v in treatments:
            v.traffic_allocation /= total
        
        experiment = Experiment(
            id=exp_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            control_variant=control,
            treatment_variants=treatments,
            allocation_method=allocation_method,
            sample_size=sample_size,
            duration_days=duration_days
        )
        
        self._experiments[exp_id] = experiment
        self._save_experiments()
        
        print(f"Created experiment: {name} ({exp_id})")
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start experiment"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False
        
        if exp.status != ExperimentStatus.DRAFT:
            return False
        
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now().isoformat()
        self._save_experiments()
        
        print(f"Started experiment: {exp.name}")
        return True
    
    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause running experiment"""
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return False
        
        exp.status = ExperimentStatus.PAUSED
        self._save_experiments()
        return True
    
    def stop_experiment(self, experiment_id: str, reason: str = "") -> bool:
        """Stop experiment dan declare winner jika ada"""
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            return False
        
        exp.status = ExperimentStatus.COMPLETED
        exp.ended_at = datetime.now().isoformat()
        
        # Determine winner
        winner = exp.get_winner()
        if winner:
            exp.winner_variant_id = winner.id
        
        # Generate summary
        exp.results_summary = {
            'reason': reason,
            'total_predictions': exp.total_predictions(),
            'winner': winner.name if winner else None,
            'control_accuracy': exp.control_variant.accuracy(),
            'control_latency': exp.control_variant.avg_latency(),
            'treatment_accuracies': [v.accuracy() for v in exp.treatment_variants],
            'treatment_latencies': [v.avg_latency() for v in exp.treatment_variants],
            'is_significant': exp.is_statistically_significant()
        }
        
        self._save_experiments()
        
        print(f"Stopped experiment: {exp.name}")
        print(f"  Winner: {winner.name if winner else 'None (control wins)'}")
        return True
    
    def assign_variant(self, experiment_id: str, user_id: str) -> Optional[Variant]:
        """
        Assign user ke variant dalam experiment
        Returns variant untuk user
        """
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return None
        
        # Check if already assigned
        assignment_key = f"{experiment_id}:{user_id}"
        if assignment_key in self._assignments:
            variant_id = self._assignments[assignment_key]
            if variant_id == exp.control_variant.id:
                return exp.control_variant
            for v in exp.treatment_variants:
                if v.id == variant_id:
                    return v
        
        # New assignment
        if exp.allocation_method == AllocationMethod.DETERMINISTIC:
            # Hash-based untuk consistent assignment
            hash_val = int(hashlib.md5(assignment_key.encode()).hexdigest(), 16)
            bucket = hash_val % 100
        else:
            # Random
            bucket = random.randint(0, 99)
        
        # Determine variant dari bucket
        cumulative = 0
        cumulative += exp.control_variant.traffic_allocation * 100
        if bucket < cumulative:
            variant = exp.control_variant
        else:
            for v in exp.treatment_variants:
                cumulative += v.traffic_allocation * 100
                if bucket < cumulative:
                    variant = v
                    break
            else:
                variant = exp.treatment_variants[-1]  # Fallback
        
        # Record assignment
        self._assignments[assignment_key] = variant.id
        
        return variant
    
    def record_prediction(
        self,
        experiment_id: str,
        variant_id: str,
        latency_ms: float,
        ground_truth: Optional[bool] = None,
        prediction: Optional[bool] = None
    ):
        """Record prediction outcome"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return
        
        # Find variant
        variant = None
        if exp.control_variant.id == variant_id:
            variant = exp.control_variant
        else:
            for v in exp.treatment_variants:
                if v.id == variant_id:
                    variant = v
                    break
        
        if not variant:
            return
        
        # Update metrics
        variant.total_predictions += 1
        variant.total_latency_ms += latency_ms
        
        # Update confusion matrix jika ground truth tersedia
        if ground_truth is not None and prediction is not None:
            if prediction and ground_truth:
                variant.true_positives += 1
            elif not prediction and not ground_truth:
                variant.true_negatives += 1
            elif prediction and not ground_truth:
                variant.false_positives += 1
            else:
                variant.false_negatives += 1
        
        self._save_experiments()
        
        # Auto-check untuk early stopping
        self._check_early_stopping(exp)
    
    def _check_early_stopping(self, exp: Experiment):
        """Check if experiment should stop early"""
        # Check sample size
        if exp.sample_size and exp.total_predictions() >= exp.sample_size * 1.5:
            # 50% over sample size, check significance
            if exp.is_statistically_significant():
                self.stop_experiment(exp.id, "Early stopping: significant results achieved")
                return
        
        # Check duration
        if exp.duration_days and exp.started_at:
            started = datetime.fromisoformat(exp.started_at)
            elapsed = datetime.now() - started
            if elapsed.days >= exp.duration_days:
                self.stop_experiment(exp.id, "Duration completed")
    
    def record_feedback(
        self,
        experiment_id: str,
        variant_id: str,
        is_positive: bool
    ):
        """Record user feedback untuk prediction"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return
        
        variant = None
        if exp.control_variant.id == variant_id:
            variant = exp.control_variant
        else:
            for v in exp.treatment_variants:
                if v.id == variant_id:
                    variant = v
                    break
        
        if variant:
            if is_positive:
                variant.positive_feedback += 1
            else:
                variant.negative_feedback += 1
            
            self._save_experiments()
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self._experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        limit: int = 100
    ) -> List[Experiment]:
        """List experiments dengan filtering"""
        exps = list(self._experiments.values())
        
        if status:
            exps = [e for e in exps if e.status == status]
        
        # Sort by created_at descending
        exps.sort(key=lambda e: e.created_at, reverse=True)
        
        return exps[:limit]
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed results untuk experiment"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {"error": "Experiment not found"}
        
        winner = exp.get_winner()
        
        return {
            'experiment': exp.to_dict(),
            'winner': winner.to_dict() if winner else None,
            'is_significant': exp.is_statistically_significant(),
            'recommendations': self._generate_recommendations(exp)
        }
    
    def _generate_recommendations(self, exp: Experiment) -> List[str]:
        """Generate recommendations berdasarkan experiment results"""
        recs = []
        
        if not exp.is_statistically_significant():
            recs.append("Sample size insufficient untuk statistical significance. Consider running longer.")
        
        winner = exp.get_winner()
        if winner:
            improvement = ((winner.accuracy() - exp.control_variant.accuracy()) / 
                          exp.control_variant.accuracy() * 100)
            recs.append(f"{winner.name} shows {improvement:.1f}% improvement over control.")
            
            if winner.avg_latency() < exp.control_variant.avg_latency():
                recs.append(f"{winner.name} is also faster ({winner.avg_latency():.1f}ms vs {exp.control_variant.avg_latency():.1f}ms).")
            else:
                recs.append(f"Note: {winner.name} is slower ({winner.avg_latency():.1f}ms vs {exp.control_variant.avg_latency():.1f}ms).")
        else:
            recs.append("Control model performs best. No improvement dari treatments.")
        
        return recs
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment"""
        if experiment_id in self._experiments:
            del self._experiments[experiment_id]
            self._save_experiments()
            return True
        return False
    
    def get_active_experiments_for_model(self, model_name: str) -> List[Experiment]:
        """Get active experiments yang menggunakan model"""
        active = []
        for exp in self._experiments.values():
            if exp.status == ExperimentStatus.RUNNING:
                if exp.control_variant.model_name == model_name:
                    active.append(exp)
                else:
                    for v in exp.treatment_variants:
                        if v.model_name == model_name:
                            active.append(exp)
                            break
        return active


# Global instance
_ab_testing = None


def get_ab_testing(storage_path: str = "data/experiments") -> ABTestingFramework:
    """Get atau create global ABTestingFramework instance"""
    global _ab_testing
    if _ab_testing is None:
        _ab_testing = ABTestingFramework(storage_path)
    return _ab_testing
