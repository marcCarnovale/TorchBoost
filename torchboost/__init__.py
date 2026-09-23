"""TorchBoost: separate stagewise Newton and historical joint-ensemble APIs."""
from .control import CapacitorController
from .legacy import (
    AttentionNetwork,
    SoftTree,
    TorchBoostModel,
    initialize_weights,
    train_torchboost,
)
from .metrics import PerformanceTracker, SplitMetricsCollector
from .objectives import BinaryLogisticObjective
from .stagewise import StagewiseBinaryClassifier
from .trees import BinarySoftTree

__version__ = "0.2.0"

__all__ = [
    "StagewiseBinaryClassifier", "BinarySoftTree", "BinaryLogisticObjective",
    "CapacitorController", "SplitMetricsCollector", "PerformanceTracker",
    "SoftTree", "TorchBoostModel", "AttentionNetwork", "train_torchboost", "initialize_weights",
]
