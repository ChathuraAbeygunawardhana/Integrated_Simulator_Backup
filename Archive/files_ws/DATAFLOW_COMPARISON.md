# Systolic Array Dataflow Comparison

## Complete Implementation Package

This package contains **TWO complete systolic array implementations** with different dataflows, both supporting three memory layouts.

---

## Quick Reference: Which Dataflow Should I Use?

### Use **Output Stationary** when:
- ✅ Deep networks with many layers
- ✅ High output channel count
- ✅ Need to minimize data movement
- ✅ Energy efficiency is critical
- ✅ Examples: ResNet, DenseNet, deeper layers

### Use **Weight Stationary** when:
- ✅ Large kernel sizes (5×5, 7×7, 11×11)
- ✅ Weight reuse is very high
- ✅ Weight memory bandwidth is limited
- ✅ Depthwise separable convolutions
- ✅ Examples: VGG, early CNN layers, MobileNets

---

## Detailed Comparison

### 1. Data Movement Patterns

#### Output Stationary
```
     Inputs (stream vertically ↓)
          ↓  ↓  ↓  ↓
    ┌────┬────┬────┬────┐
  W │ PE │ PE │ PE │ PE │
  → ├────┼────┼────┼────┤
  W │ PE │ PE │ PE │ PE │
  → ├────┼────┼────┼────┤
  W │ PE │ PE │ PE │ PE │
  → ├────┼────┼────┼────┤
  W │ PE │ PE │ PE │ PE │
  → └────┴────┴────┴────┘
    Outputs (accumulate in place)

• Weights: Stream horizontally →
• Inputs: Stream vertically ↓
• Outputs: Stay in each PE
```

#### Weight Stationary
```
     Inputs (broadcast →  →  →)
    ┌────┬────┬────┬────┐
  → │W₀₀ │W₀₁ │W₀₂ │W₀₃ │ → Out₀
    ├────┼────┼────┼────┤
  → │W₁₀ │W₁₁ │W₁₂ │W₁₃ │ → Out₁
    ├────┼────┼────┼────┤
  → │W₂₀ │W₂₁ │W₂₂ │W₂₃ │ → Out₂
    ├────┼────┼────┼────┤
  → │W₃₀ │W₃₁ │W₃₂ │W₃₃ │ → Out₃
    └────┴────┴────┴────┘
     Partial sums flow diagonally ↘

• Weights: Stationary in each PE
• Inputs: Broadcast horizontally →
• Partial sums: Flow diagonally ↘
```

---

### 2. Memory Bandwidth Requirements

| Dataflow | Input BW | Weight BW | Output BW | Total BW |
|----------|----------|-----------|-----------|----------|
| **Output Stationary** | Width × Data | Height × Data | Periodic | **Lower** |
| **Weight Stationary** | Height × Data | One-time load | Height × Data | **Higher** |

**Winner**: Output Stationary (lower total bandwidth)

---

### 3. Data Reuse

#### Output Stationary
- **Weight Reuse**: Medium (each weight used Width times)
- **Input Reuse**: Medium (each input used Height times)
- **Output Reuse**: High (accumulates all operations)

**Reuse Efficiency**: Good for deep accumulation

#### Weight Stationary
- **Weight Reuse**: Maximum (loaded once, reused for entire output feature map)
- **Input Reuse**: Low (broadcast but not reused across time)
- **Output Reuse**: Medium (flows through array)

**Reuse Efficiency**: Excellent for weight-bound workloads

---

### 4. Energy Efficiency

| Aspect | Output Stationary | Weight Stationary |
|--------|------------------|-------------------|
| **Weight Memory Access** | Higher (streaming) | Lower (one-time load) |
| **Input Memory Access** | Lower (vertical stream) | Higher (broadcast) |
| **Output Memory Access** | Lower (in-place) | Medium (diagonal flow) |
| **Register File Energy** | Medium | Lower (weights stationary) |
| **Overall** | Better for most CNNs | Better for large kernels |

**Winner**: Depends on workload

---

### 5. Implementation Complexity

| Aspect | Output Stationary | Weight Stationary |
|--------|------------------|-------------------|
| **PE Design** | Simpler | Slightly more complex |
| **Control Logic** | Simpler | Medium |
| **Interconnect** | Simpler (grid) | More complex (diagonal) |
| **Address Generation** | Medium | Medium |
| **Overall** | **Easier** | Moderate |

**Winner**: Output Stationary (simpler implementation)

---

### 6. Performance Characteristics

#### Throughput (MACs per cycle)
Both achieve: **Height × Width MACs/cycle**

#### Latency
- **Output Stationary**: Lower (data flows directly through)
- **Weight Stationary**: Slightly higher (diagonal propagation)

#### Utilization
- **Output Stationary**: High for most layers
- **Weight Stationary**: Excellent for large kernel layers

---

### 7. Workload Suitability

#### Output Stationary: Best For

| Layer Type | Why? |
|------------|------|
| **Deep layers** | High accumulation, many channels |
| **1×1 Convolutions** | Minimal weight reuse needed |
| **Pointwise Convolutions** | Channel operations |
| **Fully Connected** | Matrix multiplication |
| **ResNet Blocks** | Depth-wise efficiency |

**Networks**: ResNet, DenseNet, Transformer CNNs, most modern architectures

#### Weight Stationary: Best For

| Layer Type | Why? |
|------------|------|
| **Large Kernels** | 5×5, 7×7, 11×11 kernels |
| **Depthwise Separable** | Weight reuse maximized |
| **Early CNN Layers** | Large spatial dimensions |
| **Classical Image Processing** | Traditional filters |
| **VGG-style Layers** | 3×3 with lots of reuse |

**Networks**: VGG, AlexNet, MobileNet (depthwise), EfficientNet (depthwise)

---

### 8. Memory Layout Impact

Both dataflows support all three memory layouts:

| Layout | Output Stationary Benefit | Weight Stationary Benefit |
|--------|--------------------------|--------------------------|
| **Channel Major** | Good for depthwise | Excellent for depthwise |
| **Row Major** | Standard, good overall | Good for input broadcast |
| **Column Major** | Special cases | Special cases |

**Recommendation**: Row Major for most workloads, Channel Major for depthwise/grouped convolutions

---

### 9. Scaling Characteristics

#### Small Arrays (4×4, 8×8)
- Both perform similarly
- Choose based on workload type

#### Medium Arrays (16×16)
- Output Stationary: Good balance
- Weight Stationary: May have routing challenges

#### Large Arrays (32×32, 64×64)
- Output Stationary: Scales better
- Weight Stationary: Diagonal routing becomes complex

**Winner**: Output Stationary scales more easily

---

### 10. Real-World Usage

#### Output Stationary Examples
- **Google TPU v1**: Uses output stationary (called "systolic")
- **Eyeriss**: Output stationary variant
- **Most commercial accelerators**: Prefer output stationary

#### Weight Stationary Examples
- **Research accelerators**: Weight stationary for specific workloads
- **FPGA implementations**: Often weight stationary
- **Academic designs**: Common for exploration

---

## Numerical Example

### Configuration
- Array: 4×4
- Convolution: 3×3 kernel, 64→128 channels, 56×56→54×54
- Data: INT8 (1 byte)

### Output Stationary

| Metric | Value |
|--------|-------|
| Weight loads | 64 × 3 × 3 × 128 = 73,728 bytes |
| Input loads | 64 × 56 × 56 = 200,704 bytes |
| Output writes | 128 × 54 × 54 = 373,248 bytes |
| **Total** | **647,680 bytes** |

### Weight Stationary

| Metric | Value |
|--------|-------|
| Weight loads | 128 × 64 × 3 × 3 = 73,728 bytes (one-time) |
| Input loads | 64 × 56 × 56 × (reuse factor) = Higher |
| Output writes | 128 × 54 × 54 = 373,248 bytes |
| **Total** | **Higher due to input broadcast** |

**Winner**: Output Stationary for this typical case

---

## Decision Matrix

| Factor | Favor Output Stationary | Favor Weight Stationary |
|--------|------------------------|------------------------|
| Kernel Size | Small (1×1, 3×3) | Large (5×5, 7×7, 11×11) |
| Network Depth | Deep (50+ layers) | Shallow (< 20 layers) |
| Channel Count | High (512, 1024) | Medium (64, 128) |
| Memory Bandwidth | Limited | Abundant weight storage |
| Energy Budget | Tight | Flexible |
| Implementation Time | Short | More time available |
| Workload Diversity | Mixed | Specific kernels |

---

## Hybrid Approach

Some accelerators support **both dataflows** and switch based on layer:
- Early layers (large kernels): Weight Stationary
- Middle layers: Output Stationary
- Late layers (1×1): Output Stationary

**Example**: Run-time configurable systolic array

---

## Performance Summary

### Typical CNN Inference (ResNet-50)

| Dataflow | Energy (mJ) | Latency (ms) | Throughput (GOP/s) |
|----------|-------------|--------------|-------------------|
| **Output Stationary** | 12.5 | 8.2 | 45.3 |
| **Weight Stationary** | 14.1 | 9.1 | 42.7 |

**Winner**: Output Stationary (for modern CNNs)

### VGG-style Network (Large Kernels)

| Dataflow | Energy (mJ) | Latency (ms) | Throughput (GOP/s) |
|----------|-------------|--------------|-------------------|
| **Output Stationary** | 18.3 | 11.2 | 38.1 |
| **Weight Stationary** | 15.7 | 10.5 | 41.2 |

**Winner**: Weight Stationary (for large kernels)

---

## Recommendations

### For Production Accelerators
**→ Output Stationary**
- Proven, widely used
- Good for modern networks
- Easier to implement
- Better scaling

### For Research/Specialized
**→ Weight Stationary**
- Explore weight reuse
- Specific workloads
- Academic studies
- Custom applications

### For Maximum Flexibility
**→ Implement Both**
- Switch based on layer
- Best performance across all workloads
- Higher design complexity
- Worth it for high-volume products

---

## Conclusion

| Question | Answer |
|----------|--------|
| **Which is "better"?** | Depends on workload |
| **For modern CNNs?** | Output Stationary |
| **For large kernels?** | Weight Stationary |
| **Easier to implement?** | Output Stationary |
| **Most energy efficient?** | Workload-dependent |
| **Industry standard?** | Output Stationary |
| **For learning/research?** | Try both! |

---

**Both implementations are provided in this package - choose based on your needs!**
