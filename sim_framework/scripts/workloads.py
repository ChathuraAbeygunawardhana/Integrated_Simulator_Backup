"""
workloads.py
============
Benchmark workload definitions for the integrated DNN accelerator simulator.

Two benchmark suites are defined:
  EDGE_WORKLOADS  — 7 workloads targeting mobile / IoT deployments
  CLOUD_WORKLOADS — 7 workloads targeting datacenter / HPC deployments

Each workload entry is a dict:
  name        : short identifier
  label       : human-readable label (for plot legends / axes)
  reference   : BibTeX-style citation string
  justification: one-paragraph rationale for inclusion
  layers      : list of 4 LayerConfig objects (one per representative stage)
  loop_dims   : scaled-down (K,C,KH,KW,H_in,W_in) for LoopOptimizer analysis
                (same relative channel ratios; spatial reduced for tractability)

Workloads are drawn from:
  MLPerf Inference v3.0 — https://mlperf.org/inference-overview
  MLPerf Tiny v1.0      — https://mlperf.org/tiny
  DeepBench             — https://github.com/baidu-research/DeepBench

All LayerConfig objects use "same" padding semantics:
  output_height = input_height,  output_width = input_width
(This simplifies cross-layer data flow and preserves relative compute costs.)

Usage:
    from scripts.workloads import EDGE_WORKLOADS, CLOUD_WORKLOADS
    for wl in EDGE_WORKLOADS:
        for layer in wl["layers"]:
            ...  # layer is a LayerConfig
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow "PYTHONPATH=. python scripts/run_full_eval.py" from sim_framework/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pysim.sim_config import LayerConfig


# ---------------------------------------------------------------------------
# Helper: construct a LayerConfig with same-padding semantics
# ---------------------------------------------------------------------------
def _L(name: str, C: int, H: int, W: int, K: int,
       KH: int = 3, KW: int = 3) -> LayerConfig:
    """
    Build a LayerConfig with same-padding output dims.
    Input/weight/output base addresses are set to non-overlapping
    64 MB regions (the software reference only uses them for hashing).
    """
    return LayerConfig(
        name=name,
        input_channels=C,   input_height=H,  input_width=W,
        weight_k=K,         weight_c=C,
        weight_kh=KH,       weight_kw=KW,
        output_channels=K,  output_height=H, output_width=W,
        input_base_addr=0x0000_0000,
        weight_base_addr=0x0400_0000,
        output_base_addr=0x0800_0000,
    )


# ===========================================================================
# EDGE WORKLOADS  (7 × 4 layers each)
# ===========================================================================
EDGE_WORKLOADS: list[dict] = [

    # -----------------------------------------------------------------------
    # E1 — MobileNetV2
    # -----------------------------------------------------------------------
    {
        "name": "MobileNetV2",
        "label": "MobileNetV2\n(E1)",
        "reference": (
            "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, "
            "'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' "
            "CVPR 2018.  MLPerf Inference v3.0 Image Classification task."
        ),
        "justification": (
            "MobileNetV2 is the canonical edge image-classification backbone "
            "in MLPerf Mobile and widely deployed on IoT devices.  Its "
            "inverted-bottleneck structure alternates channel-expanding 1×1 "
            "and 3×3 depthwise convolutions, exposing both weight-stationary "
            "and output-stationary performance extremes."
        ),
        "layers": [
            _L("mv2_init",    C=3,   H=32, W=32,  K=32,  KH=3, KW=3),  # initial conv
            _L("mv2_expand",  C=32,  H=16, W=16,  K=96,  KH=1, KW=1),  # 6× channel expand
            _L("mv2_dw",      C=96,  H=16, W=16,  K=96,  KH=3, KW=3),  # depthwise-like
            _L("mv2_proj",    C=96,  H=16, W=16,  K=64,  KH=1, KW=1),  # projection
        ],
        "loop_dims": dict(K=32, C=16, KH=3, KW=3, H_in=16, W_in=16),
    },

    # -----------------------------------------------------------------------
    # E2 — EfficientDet-Lite0
    # -----------------------------------------------------------------------
    {
        "name": "EfficientDet-Lite0",
        "label": "EfficientDet-\nLite0 (E2)",
        "reference": (
            "M. Tan, R. Pang, Q. V. Le, 'EfficientDet: Scalable and Efficient "
            "Object Detection,' CVPR 2020.  MLPerf Mobile Object Detection task."
        ),
        "justification": (
            "EfficientDet-Lite0 is the MLPerf Mobile object-detection reference. "
            "Its BiFPN neck performs repeated top-down / bottom-up 3×3 "
            "depthwise-separable fusions across feature pyramid levels, "
            "stressing memory-bandwidth-limited dataflows."
        ),
        "layers": [
            _L("edet_p3",     C=32,  H=40, W=40,  K=40,  KH=3, KW=3),  # BiFPN P3
            _L("edet_p4",     C=40,  H=20, W=20,  K=64,  KH=3, KW=3),  # BiFPN P4
            _L("edet_lat",    C=64,  H=20, W=20,  K=64,  KH=1, KW=1),  # lateral 1×1
            _L("edet_p5",     C=64,  H=10, W=10,  K=128, KH=3, KW=3),  # BiFPN P5
        ],
        "loop_dims": dict(K=32, C=16, KH=3, KW=3, H_in=20, W_in=20),
    },

    # -----------------------------------------------------------------------
    # E3 — DS-CNN Keyword Spotting  (MLPerf Tiny)
    # -----------------------------------------------------------------------
    {
        "name": "DS-CNN-KWS",
        "label": "DS-CNN\nKWS (E3)",
        "reference": (
            "Y. Zhang et al., 'Hello Edge: Keyword Spotting on Microcontrollers,' "
            "arXiv:1711.07128, 2017.  MLPerf Tiny v1.0 Keyword Spotting task."
        ),
        "justification": (
            "Keyword spotting is a pervasive always-on inference workload in "
            "wearables and voice-UI devices.  DS-CNN applies 2-D convolutions "
            "over MFCC spectrograms, creating narrow-spatial / wide-channel "
            "tensor shapes that challenge square systolic arrays."
        ),
        "layers": [
            _L("kws_t1",  C=1,   H=49, W=16,  K=64,  KH=3, KW=3),  # temporal stage 1
            _L("kws_t2",  C=64,  H=49, W=16,  K=64,  KH=3, KW=3),  # temporal stage 2
            _L("kws_ds1", C=64,  H=24, W=8,   K=128, KH=3, KW=3),  # downsampled
            _L("kws_ds2", C=128, H=12, W=4,   K=128, KH=3, KW=3),  # final stage
        ],
        "loop_dims": dict(K=32, C=16, KH=3, KW=3, H_in=24, W_in=8),
    },

    # -----------------------------------------------------------------------
    # E4 — ResNet-18
    # -----------------------------------------------------------------------
    {
        "name": "ResNet-18",
        "label": "ResNet-18\n(E4)",
        "reference": (
            "K. He, X. Zhang, S. Ren, J. Sun, 'Deep Residual Learning for "
            "Image Recognition,' CVPR 2016.  DeepBench Conv benchmark."
        ),
        "justification": (
            "ResNet-18 is the standard accuracy/efficiency tradeoff baseline "
            "adopted by DeepBench.  Its uniform 3×3 residual blocks at "
            "progressively halved spatial resolutions provide a clean sweep "
            "across compute-to-memory-bandwidth ratios."
        ),
        "layers": [
            _L("res18_l1", C=64,  H=56, W=56,  K=64,  KH=3, KW=3),  # res2
            _L("res18_l2", C=64,  H=28, W=28,  K=128, KH=3, KW=3),  # res3
            _L("res18_l3", C=128, H=14, W=14,  K=256, KH=3, KW=3),  # res4
            _L("res18_l4", C=256, H=7,  W=7,   K=512, KH=3, KW=3),  # res5
        ],
        "loop_dims": dict(K=32, C=32, KH=3, KW=3, H_in=14, W_in=14),
    },

    # -----------------------------------------------------------------------
    # E5 — TinyBERT (NLP)
    # -----------------------------------------------------------------------
    {
        "name": "TinyBERT",
        "label": "TinyBERT\n(E5)",
        "reference": (
            "X. Jiao et al., 'TinyBERT: Distilling BERT for Natural Language "
            "Understanding,' EMNLP 2020.  "
            "Linear layers mapped to 1×1 convolutions for systolic-array execution."
        ),
        "justification": (
            "Edge NLP inference is growing rapidly (chatbots, on-device "
            "translation).  TinyBERT's FC layers are modelled as 1×1 "
            "convolutions over the sequence dimension, creating very wide "
            "channel configurations that stress weight-stationary dataflows."
        ),
        "layers": [
            _L("tbert_ffn1",   C=128, H=1, W=128, K=512,  KH=1, KW=1),  # FFN expand
            _L("tbert_ffn2",   C=512, H=1, W=128, K=128,  KH=1, KW=1),  # FFN contract
            _L("tbert_qkv",    C=128, H=1, W=128, K=128,  KH=1, KW=1),  # QKV proj
            _L("tbert_out",    C=128, H=1, W=128, K=512,  KH=1, KW=1),  # output FFN
        ],
        "loop_dims": dict(K=32, C=32, KH=1, KW=1, H_in=1, W_in=32),
    },

    # -----------------------------------------------------------------------
    # E6 — U-Net (Medical / Semantic Segmentation)
    # -----------------------------------------------------------------------
    {
        "name": "U-Net",
        "label": "U-Net\n(E6)",
        "reference": (
            "O. Ronneberger, P. Fischer, T. Brox, 'U-Net: Convolutional Networks "
            "for Biomedical Image Segmentation,' MICCAI 2015.  "
            "MLPerf Mobile Semantic Segmentation task."
        ),
        "justification": (
            "U-Net is the dominant architecture for medical image segmentation "
            "and is included in MLPerf Mobile.  Its symmetric encoder-decoder "
            "structure spans large spatial maps with skip connections, making "
            "inter-layer reuse especially valuable."
        ),
        "layers": [
            _L("unet_enc1",  C=1,   H=64, W=64,  K=64,  KH=3, KW=3),  # encoder L1
            _L("unet_enc2",  C=64,  H=32, W=32,  K=128, KH=3, KW=3),  # encoder L2
            _L("unet_dec2",  C=128, H=32, W=32,  K=64,  KH=3, KW=3),  # decoder L2
            _L("unet_dec1",  C=64,  H=64, W=64,  K=32,  KH=3, KW=3),  # decoder L1
        ],
        "loop_dims": dict(K=32, C=32, KH=3, KW=3, H_in=32, W_in=32),
    },

    # -----------------------------------------------------------------------
    # E7 — SSD-MobileNet (Tiny Object Detection)
    # -----------------------------------------------------------------------
    {
        "name": "SSD-MobileNet",
        "label": "SSD-MobileNet\n(E7)",
        "reference": (
            "W. Liu et al., 'SSD: Single Shot MultiBox Detector,' ECCV 2016.  "
            "A. Howard et al., 'MobileNets,' arXiv:1704.04861, 2017.  "
            "MLPerf Inference v3.0 / MLPerf Tiny Object Detection task."
        ),
        "justification": (
            "SSD-MobileNet is the standard embedded object-detection benchmark "
            "in both MLPerf Inference and MLPerf Tiny.  Its detection heads "
            "combine 1×1 and 3×3 convolutions at multiple scales, exposing "
            "memory-bandwidth-limited regimes typical of IoT accelerators."
        ),
        "layers": [
            _L("ssd_feat",  C=32,  H=19, W=19,  K=64,  KH=3, KW=3),  # feature neck
            _L("ssd_head1", C=64,  H=10, W=10,  K=128, KH=3, KW=3),  # detection head P4
            _L("ssd_head2", C=128, H=5,  W=5,   K=256, KH=3, KW=3),  # detection head P5
            _L("ssd_head3", C=256, H=3,  W=3,   K=256, KH=3, KW=3),  # detection head P6
        ],
        "loop_dims": dict(K=32, C=32, KH=3, KW=3, H_in=10, W_in=10),
    },
]


# ===========================================================================
# CLOUD WORKLOADS  (7 × 4 layers each)
# ===========================================================================
CLOUD_WORKLOADS: list[dict] = [

    # -----------------------------------------------------------------------
    # C1 — ResNet-50  (MLPerf Inference Image Classification)
    # -----------------------------------------------------------------------
    {
        "name": "ResNet-50",
        "label": "ResNet-50\n(C1)",
        "reference": (
            "K. He et al., 'Deep Residual Learning,' CVPR 2016.  "
            "MLPerf Inference v3.0 Image Classification (ResNet-50-99) task."
        ),
        "justification": (
            "ResNet-50 is the primary MLPerf datacenter image-classification "
            "benchmark and the de-facto baseline for throughput comparisons.  "
            "Its bottleneck blocks (1×1→3×3→1×1) expose the full range of "
            "arithmetic intensity from compute-bound to memory-bound layers."
        ),
        "layers": [
            _L("r50_bt1a", C=64,  H=56, W=56,  K=64,  KH=1, KW=1),  # 1×1 down-proj
            _L("r50_bt1b", C=64,  H=56, W=56,  K=64,  KH=3, KW=3),  # 3×3 core
            _L("r50_bt1c", C=64,  H=56, W=56,  K=256, KH=1, KW=1),  # 1×1 up-proj
            _L("r50_st3",  C=256, H=28, W=28,  K=512, KH=3, KW=3),  # stage-3 conv
        ],
        "loop_dims": dict(K=32, C=32, KH=3, KW=3, H_in=28, W_in=28),
    },

    # -----------------------------------------------------------------------
    # C2 — BERT-Large  (MLPerf Inference NLP)
    # -----------------------------------------------------------------------
    {
        "name": "BERT-Large",
        "label": "BERT-Large\n(C2)",
        "reference": (
            "J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, 'BERT: Pre-training "
            "of Deep Bidirectional Transformers,' NAACL 2019.  "
            "MLPerf Inference v3.0 BERT-99 task."
        ),
        "justification": (
            "BERT-Large is the MLPerf datacenter NLP benchmark.  Dense FC "
            "layers (attention QKV, FFN) are mapped to 1×1 convolutions over "
            "the sequence dimension.  Its 4× FFN expansion ratio and large "
            "hidden size (1024) make it heavily weight-bandwidth-constrained."
        ),
        "layers": [
            _L("bert_ffn1", C=256, H=1, W=128, K=1024, KH=1, KW=1),  # FFN 4× expand
            _L("bert_ffn2", C=1024, H=1, W=128, K=256, KH=1, KW=1),  # FFN contract
            _L("bert_qkv",  C=256, H=1, W=128, K=256,  KH=1, KW=1),  # QKV projection
            _L("bert_out",  C=256, H=1, W=128, K=256,  KH=1, KW=1),  # output proj
        ],
        "loop_dims": dict(K=64, C=32, KH=1, KW=1, H_in=1, W_in=32),
    },

    # -----------------------------------------------------------------------
    # C3 — 3D-UNet  (MLPerf Inference Medical Imaging)
    # -----------------------------------------------------------------------
    {
        "name": "3D-UNet",
        "label": "3D-UNet\n(C3)",
        "reference": (
            "Ö. Çiçek et al., '3D U-Net: Learning Dense Volumetric Segmentation "
            "from Sparse Annotation,' MICCAI 2016.  "
            "MLPerf Inference v3.0 Medical Image Segmentation task."
        ),
        "justification": (
            "3D-UNet is the largest model in MLPerf Inference and the most "
            "memory-bandwidth-intensive.  3-D convolutions are projected to 2-D "
            "slices for systolic-array execution, emphasising off-chip bandwidth "
            "limitations typical of volumetric medical imaging workloads."
        ),
        "layers": [
            _L("unet3d_e1", C=1,   H=128, W=128, K=32,  KH=3, KW=3),  # encoder L1
            _L("unet3d_e2", C=32,  H=64,  W=64,  K=64,  KH=3, KW=3),  # encoder L2
            _L("unet3d_d2", C=64,  H=64,  W=64,  K=32,  KH=3, KW=3),  # decoder L2
            _L("unet3d_d1", C=32,  H=128, W=128, K=16,  KH=3, KW=3),  # decoder L1
        ],
        "loop_dims": dict(K=32, C=16, KH=3, KW=3, H_in=32, W_in=32),
    },

    # -----------------------------------------------------------------------
    # C4 — RNN-T  (MLPerf Inference Speech Recognition)
    # -----------------------------------------------------------------------
    {
        "name": "RNN-T",
        "label": "RNN-T\n(C4)",
        "reference": (
            "A. Graves, 'Sequence Transduction with Recurrent Neural Networks,' "
            "ICML Workshop 2012.  MLPerf Inference v3.0 Speech Recognition task.  "
            "DeepBench RNN benchmark."
        ),
        "justification": (
            "RNN-T is the MLPerf speech benchmark and is also present in "
            "DeepBench RNN suite.  LSTM/GRU projection matrices are mapped to "
            "1×1 convolutions, creating long-sequence, moderate-channel configs "
            "that favour input-stationary dataflows."
        ),
        "layers": [
            _L("rnnt_enc1", C=128, H=1, W=256, K=128,  KH=1, KW=1),  # encoder proj
            _L("rnnt_enc2", C=128, H=1, W=256, K=256,  KH=1, KW=1),  # encoder hidden
            _L("rnnt_dec1", C=256, H=1, W=128, K=128,  KH=1, KW=1),  # decoder proj
            _L("rnnt_dec2", C=128, H=1, W=128, K=64,   KH=1, KW=1),  # output proj
        ],
        "loop_dims": dict(K=32, C=32, KH=1, KW=1, H_in=1, W_in=32),
    },

    # -----------------------------------------------------------------------
    # C5 — DLRM  (MLPerf Inference Recommendation)
    # -----------------------------------------------------------------------
    {
        "name": "DLRM",
        "label": "DLRM\n(C5)",
        "reference": (
            "M. Naumov et al., 'Deep Learning Recommendation Model for "
            "Personalization and Recommendation Systems,' arXiv:1906.00091, 2019.  "
            "MLPerf Inference v3.0 Recommendation task."
        ),
        "justification": (
            "DLRM dominates ML compute in hyperscale datacenters by volume. "
            "Its dense MLP layers (Bottom/Top) operate on batched embedding "
            "outputs, creating square, compute-heavy matrix multiplications "
            "that highlight the impact of memory layout on reuse."
        ),
        "layers": [
            _L("dlrm_bot1", C=512, H=1, W=64, K=256,  KH=1, KW=1),  # bottom MLP 1
            _L("dlrm_bot2", C=256, H=1, W=64, K=128,  KH=1, KW=1),  # bottom MLP 2
            _L("dlrm_top1", C=512, H=1, W=64, K=256,  KH=1, KW=1),  # top MLP 1
            _L("dlrm_top2", C=256, H=1, W=64, K=64,   KH=1, KW=1),  # top MLP 2
        ],
        "loop_dims": dict(K=32, C=32, KH=1, KW=1, H_in=1, W_in=16),
    },

    # -----------------------------------------------------------------------
    # C6 — VGG-16  (Memory-Bandwidth Stress; DeepBench Conv)
    # -----------------------------------------------------------------------
    {
        "name": "VGG-16",
        "label": "VGG-16\n(C6)",
        "reference": (
            "K. Simonyan, A. Zisserman, 'Very Deep Convolutional Networks for "
            "Large-Scale Image Recognition,' ICLR 2015.  "
            "DeepBench convolution benchmark, server-inference suite."
        ),
        "justification": (
            "VGG-16 is included in DeepBench as a canonical memory-bandwidth- "
            "bound workload.  Its large, uniform 3×3 layers at high channel "
            "counts saturate on-chip buffers and expose the limits of each "
            "dataflow's reuse strategy."
        ),
        "layers": [
            _L("vgg_b1",  C=64,  H=56, W=56,  K=64,  KH=3, KW=3),  # block 1 repr
            _L("vgg_b2",  C=128, H=28, W=28,  K=128, KH=3, KW=3),  # block 2
            _L("vgg_b3",  C=256, H=14, W=14,  K=256, KH=3, KW=3),  # block 3
            _L("vgg_b4",  C=512, H=7,  W=7,   K=512, KH=3, KW=3),  # blocks 4/5
        ],
        "loop_dims": dict(K=32, C=32, KH=3, KW=3, H_in=14, W_in=14),
    },

    # -----------------------------------------------------------------------
    # C7 — GPT-2 / Transformer FFN  (Language Generation)
    # -----------------------------------------------------------------------
    {
        "name": "GPT-2/Transformer",
        "label": "GPT-2/\nTransformer (C7)",
        "reference": (
            "A. Radford et al., 'Language Models are Unsupervised Multitask "
            "Learners,' OpenAI Blog 2019.  "
            "MLPerf Training v3.0 GPT-J benchmark.  DeepBench GEMM benchmark."
        ),
        "justification": (
            "Large generative LLMs now dominate datacenter GPU allocation. "
            "Transformer FFN layers (4× channel expansion) and attention "
            "projections are modelled as 1×1 convolutions, creating the widest "
            "channel configurations in this suite and stressing weight-streaming "
            "throughput limits."
        ),
        "layers": [
            _L("gpt_ffn1", C=192, H=1, W=256, K=768,  KH=1, KW=1),  # FFN expand 4×
            _L("gpt_ffn2", C=768, H=1, W=256, K=192,  KH=1, KW=1),  # FFN contract
            _L("gpt_qkv",  C=192, H=1, W=256, K=192,  KH=1, KW=1),  # QKV proj
            _L("gpt_out",  C=192, H=1, W=256, K=192,  KH=1, KW=1),  # output proj
        ],
        "loop_dims": dict(K=48, C=32, KH=1, KW=1, H_in=1, W_in=32),
    },
]


# ===========================================================================
# Four workload mixes for multi-DNN scheduler experiments
# ===========================================================================
#  Each mix is a list of (workload_name, n_layers_to_use) pairs that are
#  assembled into a multi-DNN queue for the scheduler experiments.

WORKLOAD_MIXES: list[dict] = [
    {
        "name": "Mix 1: Edge Classification",
        "label": "Workload mix 1",
        "dnns": ["MobileNetV2", "ResNet-18", "EfficientDet-Lite0"],
    },
    {
        "name": "Mix 2: Edge Heterogeneous",
        "label": "Workload mix 2",
        "dnns": ["DS-CNN-KWS", "TinyBERT", "U-Net", "SSD-MobileNet"],
    },
    {
        "name": "Mix 3: Cloud NLP + Vision",
        "label": "Workload mix 3",
        "dnns": ["ResNet-50", "BERT-Large", "3D-UNet"],
    },
    {
        "name": "Mix 4: Cloud Heterogeneous",
        "label": "Workload mix 4",
        "dnns": ["VGG-16", "DLRM", "RNN-T", "GPT-2/Transformer"],
    },
]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for suite_name, suite in [("EDGE", EDGE_WORKLOADS), ("CLOUD", CLOUD_WORKLOADS)]:
        print(f"\n{'='*60}")
        print(f"{suite_name} WORKLOADS ({len(suite)} total)")
        print('='*60)
        for wl in suite:
            total_macs = sum(
                l.weight_k * l.weight_c * l.weight_kh * l.weight_kw *
                l.output_height * l.output_width
                for l in wl["layers"]
            )
            print(f"\n  {wl['name']}")
            print(f"  Reference  : {wl['reference'][:80]}...")
            print(f"  Total MACs : {total_macs/1e6:.2f} M")
            for i, l in enumerate(wl["layers"], 1):
                print(f"  Layer {i}: {l.name:20s} "
                      f"C={l.input_channels:4d} H={l.input_height:4d} "
                      f"W={l.input_width:4d} K={l.weight_k:4d} "
                      f"KH={l.weight_kh} KW={l.weight_kw}")
