# Unified Systolic-Array Simulator -- Framework Report

Generated: 2026-05-17 22:06

## 1. Verification matrix

```
sha,top,dataflow,memory,array,scheduler,passed,cycles,skip_reason
0d1a2e072971,single_dnn_top,OS,STAMP,4x4,-,True,49,
339d16836b1e,single_dnn_top,OS,PAGED,4x4,-,True,49,
72f9ee514ba8,single_dnn_top,IS,STAMP,4x4,-,True,57,
5afdde950c69,single_dnn_top,IS,PAGED,4x4,-,True,57,
3d0a010bc3cb,single_dnn_top,WS,STAMP,4x4,-,True,78,
9f7febddc4ad,single_dnn_top,WS,PAGED,4x4,-,True,78,
37de9640ddff,single_dnn_top,OS,STAMP,8x8,-,True,57,
7a082aca78e8,multi_dnn_top,OS,STAMP,4x4,FIFO,True,49,
3da307347ea7,multi_dnn_top,OS,STAMP,4x4,SJF,True,49,
b486a701845d,multi_dnn_top,OS,STAMP,4x4,AIMT,True,49,
de0eee9c29e8,multi_dnn_top,OS,STAMP,4x4,BATCHDNN,True,49,
349d1d1f0246,multi_dnn_top,OS,STAMP,4x4,BATCHDNN_PP,True,49,
```

Full CSV at `build/matrix_results.csv`.

## 2. Vivado timing closure

### single_dnn_top
_(Vivado synthesis was not run.)_

### multi_dnn_top
_(Vivado synthesis was not run.)_

## 3. Loop optimizer cost-model integration

`pysim.cost_model.SimulatorCostModel` plugs into
`optimizer.src.blocking_optimizer.BlockingOptimizer` via the new `analyzer`
constructor argument and the new `SearchMode.SIMULATOR_GUIDED` mode:

```python
from blocking_optimizer import BlockingOptimizer, SearchMode
from pysim.cost_model  import SimulatorCostModel
from pysim             import SimConfig

base = SimConfig(dataflow='OS', memory='STAMP', array_height=4, array_width=4)
sim  = SimulatorCostModel(K=32, C=32, H=14, W=14, KH=3, KW=3, base_cfg=base)
opt  = BlockingOptimizer(K=32, C=32, H=14, W=14, KH=3, KW=3, analyzer=sim)
best, stats = opt.search(mode=SearchMode.SIMULATOR_GUIDED)
```

Two-stage strategy: the analytical `MemoryAccessAnalyzer` pre-filters all
720 (loop-order, tile-size) combinations down to the top-N=20; the
simulator analyzer re-scores the survivors with real cycle counts.

## 4. Tensorflow golden flow

`models/dump_tf_model.py` dumps a pretrained Keras model into per-layer
.npy + .hex artifacts. `pysim/golden_compare.py` compares simulator output
against the dumped expected.npy with a configurable tolerance.

The smoke model `models/tiny_cnn` is committed; for larger regression flows:

```bash
python models/dump_tf_model.py --model mobilenet_v2 --out models/mobilenet_v2
```

## 5. Known limitations

* Verilator and Vivado are not installed in the default test environment;
  when missing, the framework runner falls back to the bit-accurate
  software reference (`pysim.software_ref`) and timing-closure phase is
  skipped. Both fall-backs are clearly labelled in the matrix CSV.
* Single-outstanding AXI in the v1 backend serialises concurrent memory
  requests from multi-DNN schedulers (documented v1 limitation).
* `extended_memory_controller.sv` (stamp + prefetch) is deferred to v2.
