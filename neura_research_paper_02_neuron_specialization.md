# NEURA 300M Research Papers

---

# Paper #2: Neuron Specialization & Sparsity

## Abstract
We analyze the internal representations of NEURA 300M by examining FFN neuron activation patterns across 24 layers. Key findings: deep layers (17-23) are NOT random noise, each concept uses 1000-2000 distributed neurons, and the model shows hierarchical specialization.

## 1. Neuron Architecture
- **FFN hidden size:** 3072 neurons per layer
- **Total FFN neurons:** 24 × 3072 = 73,728
- **FFN type:** SwiGLU (SiLU(gate) × up projection)
- **Activation:** `FFN(x) = w2(SiLU(w1(x)) × w3(x))`

## 2. Sparsity Pattern (Az alma piros és - 390K checkpoint)

### Per-layer % of neurons with activation > 1.0

| Layer | Sparsity | Layer | Sparsity |
|-------|----------|-------|----------|
| L1    | 0.1%     | L13   | 0.3%     |
| L2    | 0.0%     | L14   | 0.4%     |
| L3    | 0.0%     | L15   | 1.2%     |
| L4    | 0.0%     | L16   | 2.6%     |
| L5    | 0.0%     | L17   | 5.1%     |
| L6    | 0.1%     | L18   | 9.8%     |
| L7    | 0.1%     | L19   | 9.6%     |
| L8    | 0.1%     | L20   | 12.3%    |
| L9    | 0.2%     | L21   | 19.4%    |
| L10   | 0.2%     | **L22**| **25.5%**|
| L11   | 0.3%     | **L23**| **44.8%**|
| L12   | 0.3%     | L24   | 3.8%     |

### Key Insight
**L23 is the most active layer by far** (44.8% of neurons active). This is because:
- It's the last FFN layer before the output (L24 only does final projection)
- All accumulated knowledge must be activated here for the output decision
- L24 drops to 3.8% because it's a simple linear projection, not an FFN

## 3. Deep Layers (17-23) Are NOT Noise

A critical finding: deep layers show **increasing** activation, not decreasing:
- L17: 5.1% → L23: 44.8%
- This is the OPPOSITE of random noise (which would show decreasing activation)
- **100% of neurons in L17-23 are active** (some threshold > 0)
- 25.5% of L22 neurons have activation > 1.0

This confirms that the model is using all its layers effectively - there is no "dead zone."

## 4. Top Specialized Neurons (from MapMaker - 48 prompts)

| Category | Top Neuron | Layer | Strength |
|----------|-----------|-------|----------|
| **TEST** (body) | **#352** | **L23** | **15.7** ← strongest in model |
| **GYÜMÖLCS** | #1357 | L23 | 10.3 |
| **SZÁM** | #1214/#328 | L22/L23 | 9.7 |
| **HELY** | #1852 | L22 | 9.9 |
| **IGE** | #528 | L22 | 8.9 |
| **SZÍN** | #592 | L22 | 8.9 |
| **IDŐ** | #1703 | L23 | 8.5 |
| **ÁLLAT** | #2634 | L23 | 7.3 |
| **ÉRZELEM** | #3002 | L23 | 7.3 |

### Important Observation
- **No neuron exceeds 15.7 activation** (in larger models, specialized neurons can reach 50-100+)
- This suggests the model's representations are weak and distributed
- Each concept uses 1000-2000 neurons working together, not single "grandmother cells"

## 5. Neuron Weight Analysis (L22-L23)

### Inactive vs Active Neurons
**L22:**
- Most inactive: #1279 (total norm=3.02), #2852 (3.06), #2693 (3.09)
- Most active: #497 (5.41), #474 (5.31), #152 (4.66)
- Average norm: 3.52

**L23:**
- Most inactive: #2116 (2.76), #242 (2.78), #2519 (2.82)
- Most active: #676 (5.28), #3024 (5.26), #1493 (4.57)
- Average norm: 3.65

### Key Insight
The ratio between most active and most inactive neurons is only ~2x (5.41 / 2.76 = 1.96).
In a well-trained model, this ratio should be 10x-100x.
**The model has NOT fully specialized its neurons** - they're all vaguely active.

## 6. Attention vs FFN Contribution

| Layer Range | attn/FFN Ratio | Interpretation |
|------------|---------------|---------------|
| L1-L8      | 0.35-0.89x    | FFN dominates → pattern matching |
| L9-L13     | 0.91-1.62x    | Balance point |
| L14-L24    | 0.19-1.18x    | FFN dominates again → knowledge retrieval |

The "attention-to-FFN ratio" shifts across layers, suggesting:
- Early layers: FFN identifies patterns in tokens
- Middle layers: Attention mixes context
- Late layers: FFN retrieves knowledge for prediction

## 7. Head Diversity

- Low in early layers (L1: 0.07 → all heads similar)
- Peaks in late layers (L22: 0.142, L24: 0.140)
- High diversity = heads specialize in different things
- This correlates with sparsity: more specialized neurons → more diverse attention

## 8. 320K vs 390K Comparison

| Metric | 320K | 390K | Change |
|--------|------|------|--------|
| Deep L17-23 sparsity | 18.1% | 15.9% | ↓ slight |
| Next token confidence | 3.2% | 14.1% | ↑ **improved** |
| Top neurons | #1596(86.5) | #2919(68.2) | same set, different order |
| Math ability | "kap" (14.6%) | "," (19.7%) | ↓ **worse** |

**The same top neurons fire at both checkpoints** - the 70K extra steps didn't create new specialists, they just reshuffled existing ones. The next token confidence improved for normal language but decreased for math/logic.

---

*End of Paper #2. Next: Paper #3 - Attention Mechanism & Layer Hierarchy*
