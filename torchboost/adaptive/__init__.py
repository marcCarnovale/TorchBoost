"""Adaptive jointly trained forests; legacy and stagewise APIs remain separate."""
from .config import (ForestConfig, FreezeWindow, OnlineConfig, PhysicsConfig, PlasticityConfig,
                     ScheduleConfig, StructureConfig)
from .estimators import AdaptiveForestClassifier, AdaptiveForestRegressor
from .export import ExportedForest
from .forest import AdaptiveForest, RaggedTree, ResidualNode
from .online import OnlineScheduler
from .physics import PhysicalController
from .plasticity import PlasticityModule
from .stagewise import AdaptiveStagewiseClassifier, AdaptiveStagewiseRegressor, StagewiseConfig
from .visualization import visualize_physical_state
from .structure import (CompositePruning, CustomPruning, DepthPruning,
                        GrowthPruningPolicy, NodePruning, TreePruning)

__all__ = [
    "AdaptiveForest", "AdaptiveForestClassifier", "AdaptiveForestRegressor",
    "CompositePruning", "CustomPruning", "DepthPruning", "ExportedForest",
    "ForestConfig", "GrowthPruningPolicy", "NodePruning", "OnlineConfig",
    "OnlineScheduler", "PhysicalController", "PhysicsConfig", "PlasticityConfig",
    "PlasticityModule", "RaggedTree", "ResidualNode", "ScheduleConfig",
    "StructureConfig", "TreePruning", "FreezeWindow", "StagewiseConfig",
    "AdaptiveStagewiseClassifier", "AdaptiveStagewiseRegressor", "visualize_physical_state",
]


from .single_tree import PackedSingleTree, SingleTreeClassifier, SingleTreeConfig, SingleTreeRegressor
from .single_tree_diagnostics import diagnose_single_tree

__all__ += [
    "PackedSingleTree", "SingleTreeClassifier", "SingleTreeConfig", "SingleTreeRegressor",
    "diagnose_single_tree",
]

from .autotune import (AutoTreeClassifier, AutoTreeRegressor, FeatureMap as LegacyFeatureMap, MappedTree,
                       SearchConfig, SearchFold, TreeCandidate, TreeSearch, make_search_folds)
from .specialist_forest import SpecialistForest

__all__ += ["AutoTreeClassifier", "AutoTreeRegressor", "FeatureMap", "MappedTree",
            "SearchConfig", "SearchFold", "TreeCandidate", "TreeSearch",
            "make_search_folds", "SpecialistForest"]

# Public defaults protect unseen changes in directions constant during fitting.
# LegacyFeatureMap remains available for exact frozen-study/checkpoint replay.
from .stable_features import StableFeatureMap, StableMappedTree
from .horizon import HorizonPolicy, extend_positive_tail
FeatureMap = StableFeatureMap
__all__ += ["LegacyFeatureMap", "StableFeatureMap", "StableMappedTree",
            "HorizonPolicy", "extend_positive_tail"]
from .progressive import ProgressiveConfig, ProgressiveTreeClassifier, ProgressiveTreeRegressor
from .progressive import RollingBoostConfig, RollingBoostClassifier

from .unified_progressive import UnifiedConfig, UnifiedProgressiveClassifier, UnifiedProgressiveRegressor
from .oof_forest import OOFForest, OOFForestConfig
__all__ += ["UnifiedConfig", "UnifiedProgressiveClassifier", "UnifiedProgressiveRegressor",
            "OOFForest", "OOFForestConfig"]