import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum

class EnvAllocationStrategy(Enum):
    LINEAR = "linear"     
    RANDOM = "random"     
    STRATIFIED = "stratified"  
    INTERLEAVED = "interleaved" 

class SpuriousFeatureType(Enum):
    COLOR = "color"       
    LABEL_NOISE = "label_noise" 
    FEATURE_NOISE = "feature_noise" 
    CORRELATED = "correlated" 

class EnvironmentCreator:
    
    def __init__(
        self,
        num_envs: int,
        env_allocation: str = "linear",
        spurious_type: str = "feature_noise",
        spurious_strength: Optional[List[float]] = None,
        seed: Optional[int] = None
    ):
        
        self.num_envs = num_envs
        self.env_allocation = EnvAllocationStrategy(env_allocation)
        self.spurious_type = SpuriousFeatureType(spurious_type)
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if spurious_strength is None:
            self.spurious_strength = self._generate_spurious_strength()
        else:
            assert len(spurious_strength) == num_envs, \
                f"spurious_strength length ({len(spurious_strength)}) must match num_envs ({num_envs})"
            self.spurious_strength = spurious_strength
    
    def _generate_spurious_strength(self) -> List[float]:
        if self.env_allocation == EnvAllocationStrategy.LINEAR:
            return [(0.2 - 0.1) / (self.num_envs - 1) * i + 0.1 
                   for i in range(self.num_envs)]
        elif self.env_allocation == EnvAllocationStrategy.STRATIFIED:
            mid = self.num_envs // 2
            return [0.1 if i < mid else 0.2 for i in range(self.num_envs)]
        else:
            return [0.15] * self.num_envs
    
    def _stratified_allocation(
        self, 
        num_samples: int, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        env_ids = torch.zeros(num_samples, dtype=torch.long)
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            for i, idx in enumerate(label_indices):
                env_ids[idx] = i % self.num_envs
        
        return env_ids
    
    def _create_spurious_feature(
        self,
        labels: torch.Tensor,
        env_ids: torch.Tensor
    ) -> torch.Tensor:
        
        device = labels.device

        spurious_features = torch.zeros_like(labels, dtype=torch.float)
        
        for env_id in range(self.num_envs):
            env_mask = (env_ids == env_id)
            if env_mask.sum() == 0:
                continue
            
            env_labels = labels[env_mask]
            e = self.spurious_strength[env_id]
            
            if self.spurious_type == SpuriousFeatureType.CORRELATED:
                color_mask = (torch.rand(len(env_labels), device=device) < e).float()
                spurious_features[env_mask] = (1 - color_mask)
                
            elif self.spurious_type == SpuriousFeatureType.LABEL_NOISE:
                noise = (torch.rand(len(env_labels), device=device) < e).float()
                spurious_features[env_mask] = torch.abs(env_labels - noise)
                
            elif self.spurious_type == SpuriousFeatureType.FEATURE_NOISE:
                spurious_features[env_mask] = torch.rand(len(env_labels), device=device) * e
                
            else:
                raise ValueError(f"Unknown spurious type: {self.spurious_type}")
        
        return spurious_features
    
    def _allocate_environments(
        self, 
        num_samples: int
    ) -> torch.Tensor:
        
        env_ids = torch.zeros(num_samples, dtype=torch.long)
        samples_per_env = num_samples // self.num_envs
        
        for i in range(self.num_envs):
            start_idx = i * samples_per_env
            if i == self.num_envs - 1:
                end_idx = num_samples
            else:
                end_idx = (i + 1) * samples_per_env
            env_ids[start_idx:end_idx] = i
        
        return env_ids
    
    def split_train_test(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        g: torch.Tensor,
        c: torch.Tensor,
        test_ratio: float = 0.2,
        test_env_id: Optional[int] = None
    ) -> Dict[str, Tuple[torch.Tensor, ...]]:
        
        if test_env_id is not None:
            test_mask = (g == test_env_id)
            train_mask = ~test_mask
        else:
            num_test = int(len(x) * test_ratio)
            indices = torch.randperm(len(x))
            test_indices = indices[:num_test]
            train_indices = indices[num_test:]
            
            test_mask = torch.zeros(len(x), dtype=torch.bool)
            test_mask[test_indices] = True
            train_mask = ~test_mask
        
        return {
            'train': (
                x[train_mask], y[train_mask], 
                g[train_mask], c[train_mask]
            ),
            'test': (
                x[test_mask], y[test_mask],
                g[test_mask], c[test_mask]
            )
        }