# NEURA 300M Research Papers
## Mechanistic Interpretability of a 355M Parameter Hungarian Language Model

**Author:** Hermes Agent (by Nous Research)
**Date:** July 1, 2026
**Location:** Remote RTX 3070 (192.168.0.142)
**Checkpoint:** lm300m_v3_step390000.pt

---

# Paper #1: Training Dynamics & Loss Landscape

## Abstract
We analyze the training dynamics of NEURA 300M, a 355M parameter Hungarian language model trained on ~2.5B tokens. The model shows characteristic loss reduction patterns with significant PPL fluctuations, suggesting unstable convergence at small model scales.

## 1. Training Configuration
- **Architecture:** 24-layer Transformer (GQA, FFN, RMSNorm)
- **Parameters:** 354,993,152
- **Batch:** micro=1, no gradient accumulation (OOM constraint on RTX 3070 8GB)
- **Optimizer:** AdamW (lr=5e-5, weight_decay=0.01)
- **Scheduler:** Cosine with 2000 step warmup
- **Total steps:** 100,000 (from step 315,000 to 415,000+)
- **Data:** 101 shards, ~4.9M sequences, 512 tokens each
- **Speed:** ~700-800 tok/s (FP32, micro_batch=1)
- **VRAM:** 5.4/5.7GB (stable)

## 2. Loss & PPL Trajectory

### Key Milestones (from v3 training log)
| Total Step | Train Loss | Val Loss | PPL  | LR        |
|------------|-----------|---------|------|-----------|
| 315,000    | 4.727     | 6.234   | 510.1| 5.00e-05  |
| 320,000    | 4.690     | 4.329   | 75.8 | 4.99e-05  |
| 330,000    | 4.606     | 4.258   | 70.7 | 4.79e-05  |
| 340,000    | 3.507     | 4.165   | 64.4 | 4.35e-05  |
| 350,000    | 4.248     | 4.158   | 63.9 | 3.73e-05  |
| 360,000    | 3.805     | 4.721   | 112.2| 2.98e-05  |
| 370,000    | 3.769     | 4.463   | 86.8 | 2.18e-05  |
| 380,000    | 3.147     | 4.549   | 94.5 | 1.42e-05  |
| **390,000**| **3.421** | **3.877**|**48.3**| 7.61e-06|

### Best Performance
- **Best PPL: 24.0** at step 338,200
- This is the lowest PPL achieved, but this checkpoint was NOT saved
- The saved checkpoint at 390K has PPL=48.3 (higher than best)

### PPL Range
- **Min:** 24.0 (at 338,200 steps)
- **Max:** 510.1 (at 315,000 steps, start of v3 training)
- **Mean:** 86.7
- The model fluctuates significantly - PPL can jump from 35 to 131 within 10K steps

## 3. Observations

### Instability
The validation PPL fluctuates wildly (35→131→24→50→120) suggesting:
1. The model is at the edge of its capacity (355M is small)
2. gradient accumulation resolves some instability but not all
3. The cosine LR schedule may be too aggressive for this model size

### BUG: Zero Gradients in Log
All gradient norms in the log show `grad=0.0000` - this is a LOGGING BUG:
- `opt.zero_grad(set_to_none=True)` is called BEFORE the gradient norm computation
- This means `model.parameters()` have `grad = None` at logging time
- The fix: move the gradient norm computation BEFORE `zero_grad()`

## 4. Speed Analysis
- Average: ~780 tok/s
- Range: 300-800 tok/s
- The slow periods correlate with checkpoint saving (1.45GB file write)
- micro_batch=1 is very inefficient - only 1/32 of effective batch size

## 5. Conclusion (Paper #1)
The model is training but unstable. The PPL of 48 at checkpoint 390K is usable for basic language tasks but far from state-of-the-art (which achieves PPL <20 for similar sized models). The high fluctuation suggests the model needs either (a) more data, (b) longer training, or (c) architectural changes for stability.

---

*End of Paper #1. Next: Paper #2 - Neuron Specialization & Sparsity*
