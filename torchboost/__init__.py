"""TorchBoost: separate stagewise Newton and historical joint-ensemble APIs."""
from .control import CapacitorController
from .metrics import PerformanceTracker, SplitMetricsCollector
from .objectives import BinaryLogisticObjective
from .stagewise import StagewiseBinaryClassifier
from .trees import BinarySoftTree

__version__ = "0.2.0"
_LEGACY = ("SoftTree", "TorchBoostModel", "AttentionNetwork", "train_torchboost", "initialize_weights")


def __getattr__(name):
    if name in _LEGACY:
        from . import legacy
        return getattr(legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["StagewiseBinaryClassifier", "BinarySoftTree", "BinaryLogisticObjective",
           "CapacitorController", "SplitMetricsCollector", "PerformanceTracker", *_LEGACY]
