"""
systolic_optimizer — TensorFlow to Systolic Array Optimization Framework

Core pure-Python modules (no TensorFlow required):
  LoopOrder6D, LoopOptimizer, LoopTileConfig  — loop-order analysis
  BlockingOptimizer, BlockingConfig, TileLoopOrder  — tiling search
  MetadataTable, LayerMetadata  — layer topology & metadata

TensorFlow-dependent modules (imported separately when TF is available):
  tf_systolic_wrapper.SystolicArrayWrapper
  tf_hardware_verification.TFModelVerificationFlow
  hardware_verifier.HardwareVerifier
"""
from .loop_optimizer import LoopOrder6D, LoopOptimizer, LoopTileConfig
from .blocking_optimizer import BlockingOptimizer, BlockingConfig, TileLoopOrder
from .metadata_table import MetadataTable, LayerMetadata

__all__ = [
    "LoopOrder6D", "LoopOptimizer", "LoopTileConfig",
    "BlockingOptimizer", "BlockingConfig", "TileLoopOrder",
    "MetadataTable", "LayerMetadata",
]
