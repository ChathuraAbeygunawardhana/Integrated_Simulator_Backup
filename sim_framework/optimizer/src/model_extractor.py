#!/usr/bin/env python3
"""
ModelExtractor
==============
Walks a built tf.keras.Model and populates a MetadataTable with the
complete layer topology, exact shapes, dependency graph (including
branches, residual connections, and concatenations), and per-layer
weight metadata.

Supported layer types
---------------------
  Conv2D              — full params + kernel weights
  DepthwiseConv2D     — per-channel kernel, depth_multiplier
  Dense               — weight matrix + bias
  BatchNormalization  — gamma, beta, moving mean/var  (fused into next conv flag)
  Activation / ReLU / ELU / LeakyReLU — no weights, shape passthrough
  Add / Subtract / Average / Maximum  — element-wise merge
  Concatenate         — channel-axis merge, tracked output channels
  MaxPooling2D / AveragePooling2D / GlobalAveragePooling2D / GlobalMaxPooling2D
  Flatten / Reshape   — shape transforms
  InputLayer          — network boundary (no parents)
  Dropout             — passthrough (no effect at inference)
  ZeroPadding2D       — shape transform

Any unrecognised layer is recorded as LayerType.ACTIVATION (passthrough)
with a warning, so extraction never silently stops mid-model.

Usage
-----
    import tensorflow as tf
    from model_extractor import ModelExtractor

    model = tf.keras.applications.MobileNetV2(weights=None)
    extractor = ModelExtractor(array_height=8, array_width=8, scratchpad_kb=256)
    table = extractor.extract(model)
    table.print_summary()
    table.to_json("mobilenetv2_metadata.json")
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Metadata types (no TF dependency in this module's own types)
# --------------------------------------------------------------------------
from metadata_table import (
    LayerMetadata,
    LayerType,
    MemoryLocation,
    MetadataTable,
    TensorMetadata,
)


# --------------------------------------------------------------------------
# Keras class → LayerType mapping
# --------------------------------------------------------------------------
_LAYER_TYPE_MAP: Dict[str, LayerType] = {
    # Convolutions
    "Conv2D":                  LayerType.CONV2D,
    "Conv2DTranspose":         LayerType.CONV2D,        # treated as conv for sizing
    "SeparableConv2D":         LayerType.CONV2D,
    "DepthwiseConv2D":         LayerType.DEPTHWISE_CONV2D,
    # Dense
    "Dense":                   LayerType.DENSE,
    # Normalisation
    "BatchNormalization":      LayerType.BATCHNORM,
    "LayerNormalization":      LayerType.BATCHNORM,
    # Activations
    "Activation":              LayerType.ACTIVATION,
    "ReLU":                    LayerType.ACTIVATION,
    "LeakyReLU":               LayerType.ACTIVATION,
    "ELU":                     LayerType.ACTIVATION,
    "PReLU":                   LayerType.ACTIVATION,
    "Softmax":                 LayerType.ACTIVATION,
    "Sigmoid":                 LayerType.ACTIVATION,    # Keras 3 name
    # Pooling
    "MaxPooling2D":            LayerType.POOLING,
    "AveragePooling2D":        LayerType.POOLING,
    "GlobalAveragePooling2D":  LayerType.POOLING,
    "GlobalMaxPooling2D":      LayerType.POOLING,
    # Merge
    "Add":                     LayerType.ADD,
    "Subtract":                LayerType.ADD,
    "Average":                 LayerType.ADD,
    "Maximum":                 LayerType.ADD,
    "Concatenate":             LayerType.CONCATENATE,
    # Shape transforms (passthrough for dataflow purposes)
    "Flatten":                 LayerType.ACTIVATION,
    "Reshape":                 LayerType.ACTIVATION,
    "ZeroPadding2D":           LayerType.ACTIVATION,
    "Dropout":                 LayerType.ACTIVATION,
    "InputLayer":              LayerType.ACTIVATION,
}


# --------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------

def _tensor_shape(tensor) -> Tuple:
    """(None, H, W, C) → (H, W, C), stripping the batch dim."""
    s = tuple(tensor.shape)
    return s[1:] if s and s[0] is None else s


def _layer_output_shape(layer) -> Tuple:
    """Output shape via node output tensor — works across TF versions."""
    if not layer._inbound_nodes:
        return ()
    ts = layer._inbound_nodes[0].output_tensors
    if isinstance(ts, list):
        # Multi-output layer: return the first output's shape
        return _tensor_shape(ts[0])
    return _tensor_shape(ts)


def _layer_input_shapes(layer) -> List[Tuple]:
    """All input shapes (handles multi-input layers like Add/Concat)."""
    if not layer._inbound_nodes:
        return []
    node = layer._inbound_nodes[0]
    its  = node.input_tensors
    if not isinstance(its, list):
        its = [its]
    return [_tensor_shape(t) for t in its]


def _parent_names(layer) -> List[str]:
    """Names of layers that feed directly into this layer."""
    if not layer._inbound_nodes:
        return []
    return [pn.operation.name for pn in layer._inbound_nodes[0].parent_nodes]


# --------------------------------------------------------------------------
# Weight extraction helpers
# --------------------------------------------------------------------------

def _conv2d_params(layer) -> Dict:
    """Extract Conv2D kernel shape and weight bytes."""
    weights = layer.get_weights()
    if weights:
        kernel = weights[0]          # (KH, KW, C_in, C_out)
        kh, kw, c_in, c_out = kernel.shape
    else:
        # Layer not yet built or use_bias=False with no call — fall back to config
        cfg  = layer.get_config()
        kh, kw = cfg["kernel_size"]
        c_in   = 0     # unknown before first call
        c_out  = cfg["filters"]
        kernel = None

    has_bias     = len(weights) > 1
    weight_bytes = int(np.prod(kernel.shape) * 4) if kernel is not None else 0
    if has_bias:
        weight_bytes += weights[1].size * 4

    return dict(
        kh=kh, kw=kw, c_in=c_in, c_out=c_out,
        strides=layer.strides,
        padding=layer.padding,
        has_bias=has_bias,
        weight_bytes=weight_bytes,
        kernel_shape=(kh, kw, c_in, c_out),
    )


def _depthwise_params(layer) -> Dict:
    """Extract DepthwiseConv2D kernel shape and weight bytes."""
    weights = layer.get_weights()
    if weights:
        kernel = weights[0]          # (KH, KW, C_in, depth_multiplier)
        kh, kw, c_in, dm = kernel.shape
    else:
        cfg = layer.get_config()
        kh, kw = cfg["kernel_size"]
        c_in, dm = 0, cfg.get("depth_multiplier", 1)
        kernel = None

    weight_bytes = (int(np.prod(kernel.shape) * 4)
                    + (weights[1].size * 4 if len(weights) > 1 else 0)
                    ) if kernel is not None else 0

    return dict(
        kh=kh, kw=kw, c_in=c_in,
        c_out=c_in * dm,          # output channels = c_in * depth_multiplier
        depth_multiplier=dm,
        strides=layer.strides,
        padding=layer.padding,
        weight_bytes=weight_bytes,
        kernel_shape=(kh, kw, c_in, dm),
    )


def _dense_params(layer) -> Dict:
    weights = layer.get_weights()
    if weights:
        w = weights[0]               # (in_features, out_features)
        in_f, out_f = w.shape
    else:
        cfg  = layer.get_config()
        in_f, out_f = 0, cfg["units"]
        w    = None
    weight_bytes = (int(np.prod(w.shape) * 4)
                    + (weights[1].size * 4 if len(weights) > 1 else 0)
                    ) if w is not None else 0
    return dict(in_features=in_f, out_features=out_f, weight_bytes=weight_bytes)


def _bn_params(layer) -> Dict:
    weights = layer.get_weights()   # gamma, beta, moving_mean, moving_var
    weight_bytes = sum(w.size * 4 for w in weights)
    return dict(weight_bytes=weight_bytes)


# --------------------------------------------------------------------------
# Main extractor
# --------------------------------------------------------------------------

class ModelExtractor:
    """
    Extracts full topology from a built tf.keras.Model into a MetadataTable.

    Parameters
    ----------
    array_height, array_width : PE array dimensions (used by BlockingOptimizer)
    scratchpad_kb             : on-chip scratchpad capacity
    skip_input_layer          : drop the InputLayer node (default True)
    """

    def __init__(
        self,
        array_height:  int = 8,
        array_width:   int = 8,
        scratchpad_kb: int = 256,
        skip_input_layer: bool = True,
    ) -> None:
        self.array_height      = array_height
        self.array_width       = array_width
        self.scratchpad_kb     = scratchpad_kb
        self.skip_input_layer  = skip_input_layer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, model) -> MetadataTable:
        """
        Walk *model* and return a fully-populated MetadataTable.

        Raises
        ------
        ValueError  if the model has not been built (no input shape known).
        """
        # Guard: model must have been called at least once so shapes are known
        if not model.built and not model.inputs:
            raise ValueError(
                "Model must be built before extraction. "
                "Call model.build(input_shape) or pass a sample input."
            )

        table = MetadataTable(scratchpad_size_kb=self.scratchpad_kb)

        # --- Pass 1: assign stable integer IDs, collect raw info -----------
        # We assign IDs in the order Keras returns layers (topological).
        layer_to_id: Dict[str, int] = {}
        raw: List[Dict]              = []

        lid = 0
        for layer in model.layers:
            cls = type(layer).__name__
            if self.skip_input_layer and cls == "InputLayer":
                continue

            layer_to_id[layer.name] = lid
            raw.append(self._extract_layer(layer, lid))
            lid += 1

        # --- Pass 2: resolve parent names → IDs, add to table --------------
        for info in raw:
            parent_ids = [
                layer_to_id[name]
                for name in info["parent_names"]
                if name in layer_to_id          # skip InputLayer refs
            ]
            info["depends_on"] = parent_ids

            lm = LayerMetadata(
                layer_id      = info["layer_id"],
                layer_name    = info["layer_name"],
                layer_type    = info["layer_type"],
                input_tensors = info["input_tensor_names"],
                output_tensor = info["output_tensor_name"],
                weight_tensor = info.get("weight_tensor_name"),
                input_shape   = info["input_shape"],
                output_shape  = info["output_shape"],
                kernel_shape  = info.get("kernel_shape", ()),
                depends_on    = parent_ids,
                weight_size_bytes  = info.get("weight_bytes", 0),
                input_size_bytes   = int(np.prod(info["input_shape"]) * 4) if info["input_shape"] else 0,
                output_size_bytes  = int(np.prod(info["output_shape"]) * 4) if info["output_shape"] else 0,
            )
            table.add_layer(lm)

            # Register output tensor
            table.add_tensor(TensorMetadata(
                name           = info["output_tensor_name"],
                shape          = info["output_shape"],
                size_bytes     = lm.output_size_bytes,
                producer_layer = info["layer_name"],
                memory_location= MemoryLocation.DRAM,
                is_network_output = (info["layer_name"] in
                                     {l.name for l in model.outputs_map.values()}
                                     if hasattr(model, "outputs_map")
                                     else False),
            ))

        # Register network input tensor(s)
        for inp_tensor in model.inputs:
            shape = _tensor_shape(inp_tensor)
            name  = getattr(inp_tensor, "name", "network_input")
            # strip trailing :0 suffix if present
            name  = name.split(":")[0]
            table.add_tensor(TensorMetadata(
                name              = name,
                shape             = shape,
                size_bytes        = int(np.prod(shape) * 4),
                memory_location   = MemoryLocation.DRAM,
                is_network_input  = True,
            ))

        return table

    # ------------------------------------------------------------------
    # Per-layer extraction
    # ------------------------------------------------------------------

    def _extract_layer(self, layer, lid: int) -> Dict:
        """Return a raw info dict for one layer."""
        cls        = type(layer).__name__
        layer_type = _LAYER_TYPE_MAP.get(cls)
        if layer_type is None:
            warnings.warn(
                f"Unrecognised layer class '{cls}' (layer '{layer.name}'). "
                "Recording as ACTIVATION passthrough.",
                stacklevel=4,
            )
            layer_type = LayerType.ACTIVATION

        parent_names  = _parent_names(layer)
        in_shapes     = _layer_input_shapes(layer)
        out_shape     = _layer_output_shape(layer)

        # Canonical single input shape (first input; Add/Concat get the first)
        in_shape = in_shapes[0] if in_shapes else ()

        # Tensor name conventions
        out_tname    = f"{layer.name}:out"
        in_tnames    = [f"{p}:out" for p in parent_names]
        w_tname      = f"{layer.name}:weights" if layer.get_weights() else None

        info = dict(
            layer_id           = lid,
            layer_name         = layer.name,
            layer_type         = layer_type,
            layer_cls          = cls,
            parent_names       = parent_names,
            input_tensor_names = in_tnames,
            output_tensor_name = out_tname,
            weight_tensor_name = w_tname,
            input_shape        = in_shape,
            output_shape       = out_shape,
            weight_bytes       = 0,
            kernel_shape       = (),
        )

        # Layer-type-specific parameter extraction
        if layer_type == LayerType.CONV2D:
            p = _conv2d_params(layer)
            info.update(
                kernel_shape = p["kernel_shape"],
                weight_bytes = p["weight_bytes"],
                extra        = {k: p[k] for k in
                                ("kh","kw","c_in","c_out","strides","padding")},
            )

        elif layer_type == LayerType.DEPTHWISE_CONV2D:
            p = _depthwise_params(layer)
            info.update(
                kernel_shape = p["kernel_shape"],
                weight_bytes = p["weight_bytes"],
                extra        = {k: p[k] for k in
                                ("kh","kw","c_in","c_out","strides","padding",
                                 "depth_multiplier")},
            )

        elif layer_type == LayerType.DENSE:
            p = _dense_params(layer)
            info.update(weight_bytes=p["weight_bytes"],
                        extra=p)

        elif layer_type == LayerType.BATCHNORM:
            p = _bn_params(layer)
            info.update(weight_bytes=p["weight_bytes"])

        return info

    # ------------------------------------------------------------------
    # Convenience: extract + run all optimization passes
    # ------------------------------------------------------------------

    def extract_and_optimize(
        self,
        model,
        search_mode_str: str = "heuristic",   # "heuristic" | "exhaustive" | "gradient"
        objective:       str = "min_traffic",
        verbose:         bool = True,
    ) -> MetadataTable:
        """
        Extract topology then run the full optimization pipeline:
          1. Topological sort
          2. Per-layer blocking + loop-order search
          3. Inter-layer reuse analysis
          4. Fusion opportunity detection
          5. Memory traffic estimate

        Parameters
        ----------
        search_mode_str : passed to MetadataTable.optimize_blocking_factors()
                          as the BlockingOptimizer SearchMode
        """
        from blocking_optimizer import SearchMode

        mode_map = {
            "heuristic":  SearchMode.HEURISTIC,
            "exhaustive": SearchMode.EXHAUSTIVE,
            "gradient":   SearchMode.GRADIENT,
        }
        mode = mode_map.get(search_mode_str.lower(), SearchMode.HEURISTIC)

        table = self.extract(model)

        if verbose:
            print(f"\nExtracted {len(table.layers)} layers from "
                  f"'{model.name}' ({sum(1 for l in table.layers.values() if l.layer_type == LayerType.CONV2D)} conv, "
                  f"{sum(1 for l in table.layers.values() if l.layer_type == LayerType.DEPTHWISE_CONV2D)} depthwise, "
                  f"{sum(1 for l in table.layers.values() if l.layer_type == LayerType.BATCHNORM)} BN, "
                  f"{sum(1 for l in table.layers.values() if l.layer_type in (LayerType.ADD, LayerType.CONCATENATE))} merge)")

        table.compute_execution_order()
        table.optimize_blocking_factors(
            array_height=self.array_height,
            array_width=self.array_width,
            scratchpad_kb=self.scratchpad_kb,
            optimization_objective=objective,
        )
        table.analyze_inter_layer_reuse()
        table.find_fusion_opportunities()
        table.estimate_memory_traffic()

        return table


# --------------------------------------------------------------------------
# CLI / demo
# --------------------------------------------------------------------------

def _demo():
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    print("=" * 70)
    print("ModelExtractor demo")
    print("=" * 70)

    # ── 1. Sequential model (basic sanity check) ──────────────────────
    print("\n── Sequential CNN ──")
    inputs = tf.keras.Input(shape=(28, 28, 1), name="img")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False, name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool1")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(10, name="fc")(x)
    seq_model = tf.keras.Model(inputs, x, name="SimpleCNN")

    ex = ModelExtractor(array_height=8, array_width=8, scratchpad_kb=64)
    table = ex.extract(seq_model)
    table.compute_execution_order()

    print(f"\nLayers extracted: {len(table.layers)}")
    print(f"Execution order : {table.execution_order}")
    print("\nPer-layer topology:")
    for lid in table.execution_order:
        lm = table.layers[lid]
        print(f"  [{lid:2d}] {lm.layer_name:18s} ({lm.layer_type.value:16s})  "
              f"in={str(lm.input_shape):25s}  out={str(lm.output_shape):20s}  "
              f"deps={lm.depends_on}  w={lm.weight_size_bytes//1024}KB")

    # ── 2. Branching model (residual block) ───────────────────────────
    print("\n── ResNet-style block (branch + Add) ──")
    inp = tf.keras.Input(shape=(56, 56, 32), name="input")
    x   = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv1")(inp)
    x   = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x   = tf.keras.layers.ReLU(name="relu1")(x)
    sc  = tf.keras.layers.Conv2D(64, 1, padding="same", use_bias=False, name="proj")(inp)
    x   = tf.keras.layers.Add(name="add")([x, sc])
    x   = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="dw")(x)
    x   = tf.keras.layers.Conv2D(128, 1, name="pw")(x)
    x   = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    out = tf.keras.layers.Dense(10, name="fc")(x)
    res_model = tf.keras.Model(inp, out, name="ResBlock")

    table2 = ex.extract(res_model)
    table2.compute_execution_order()

    print(f"\nLayers extracted: {len(table2.layers)}")
    print("\nPer-layer topology (showing dependency graph):")
    for lid in table2.execution_order:
        lm = table2.layers[lid]
        parents = [table2.layers[d].layer_name for d in lm.depends_on]
        print(f"  [{lid:2d}] {lm.layer_name:18s} ({lm.layer_type.value:16s})  "
              f"in={str(lm.input_shape):25s}  out={str(lm.output_shape):20s}  "
              f"parents={parents}")

    # verify Add node has 2 dependencies
    add_layer = next(lm for lm in table2.layers.values() if lm.layer_name == "add")
    assert len(add_layer.depends_on) == 2, \
        f"Add should have 2 deps, got {add_layer.depends_on}"
    print("\n✓ Branch topology (Add node has 2 parents) correctly extracted")

    # ── 3. MobileNetV2-style depthwise-separable block ────────────────
    print("\n── Depthwise-separable (MobileNet-style) ──")
    inp3 = tf.keras.Input(shape=(28, 28, 32), name="input")
    x3   = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="dw")(inp3)
    x3   = tf.keras.layers.BatchNormalization(name="bn_dw")(x3)
    x3   = tf.keras.layers.ReLU(name="relu_dw")(x3)
    x3   = tf.keras.layers.Conv2D(64, 1, use_bias=False, name="pw")(x3)
    x3   = tf.keras.layers.BatchNormalization(name="bn_pw")(x3)
    x3   = tf.keras.layers.ReLU(name="relu_pw")(x3)
    mob  = tf.keras.Model(inp3, x3, name="DSBlock")

    table3 = ex.extract(mob)
    table3.compute_execution_order()

    for lid in table3.execution_order:
        lm = table3.layers[lid]
        print(f"  [{lid}] {lm.layer_name:14s} ({lm.layer_type.value:16s})  "
              f"in={str(lm.input_shape):20s}  out={str(lm.output_shape):20s}  "
              f"kernel={lm.kernel_shape}  w={lm.weight_size_bytes}B")

    dw = next(lm for lm in table3.layers.values() if lm.layer_name == "dw")
    assert dw.layer_type == LayerType.DEPTHWISE_CONV2D
    print("\n✓ DepthwiseConv2D correctly typed and kernel shape extracted")

    print("\n" + "=" * 70)
    print("All demo assertions passed ✓")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
