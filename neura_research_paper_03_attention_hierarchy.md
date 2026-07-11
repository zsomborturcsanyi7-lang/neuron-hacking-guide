# NEURA 300M Research Papers

---

# Paper #3: Attention Mechanism & Layer Hierarchy

## Abstract
We analyze the attention patterns and layer hierarchy of NEURA 300M. The model shows clear evidence of syntactic-to-semantic transition across layers, with attention heads progressively specializing from grammar to content.

## 1. The Layer Hierarchy

```
L1-L4:     WORD PAIRS ("alma" → "piros")
L5-L8:     SYNTAX (subject-verb-object)
L9-L12:    CONTEXT (5-10 word dependencies)
L13-L20:   ABSTRACT PATTERNS (themes, categories)
L21-L24:   HIGH-LEVEL MEANING (decision boundary)
```

## 2. Attention Shift: The Key Finding

When analyzing "Az alma piros és" (4 tokens: Az, alma, piros, és):

**The word "és" (and) shows a dramatic attention shift:**

| Layer | "és" attends to | Interpretation |
|-------|----------------|---------------|
| L1    | "Az" (strong) | **Syntactic**: "Az ... és" structure detection |
| L5    | "Az" (medium) | Still structural |
| L10   | Mix | Transition zone |
| L15   | "alma" (starting) | Beginning to shift |
| **L22** | **"alma" (strong)** | **Semantic**: "alma és..." content connection |

**This is EVIDENCE of learning hierarchical representations:**
- Early layers: grammar (article → conjunction)
- Late layers: content (fruit → continuation)

## 3. Head Diversity By Layer

### What is Head Diversity?
The 16 GQA attention heads per layer were measured by the standard deviation of their attention patterns. High diversity = heads focusing on different things.

| Layer | Diversity | Meaning |
|-------|-----------|---------|
| L1    | 0.070     | All heads do similar things |
| L2    | 0.056     | |
| L3    | 0.040     | **Lowest diversity** |
| L4    | 0.050     | |
| L5    | 0.067     | |
| L6    | 0.081     | |
| L7    | 0.111     | ✅ Heads start specializing |
| L8    | 0.115     | |
| L9    | 0.131     | |
| L10   | 0.113     | |
| L11   | 0.104     | |
| L12   | 0.088     | |
| L13   | 0.071     | |
| L14   | 0.080     | |
| L15   | 0.091     | |
| L16   | 0.100     | |
| L17   | 0.127     | |
| L18   | 0.106     | |
| L19   | 0.124     | |
| L20   | 0.122     | |
| L21   | 0.133     | |
| L22   | **0.142**  | **Highest diversity** |
| L23   | 0.132     | |
| L24   | 0.140     | |

### Interpretation
- **L1-L6:** Heads are redundant (low diversity < 0.08) - all learn similar patterns
- **L7-L24:** Heads diverge (> 0.08) - each specializes on different content
- **L22 peaks** at 0.142 - the layer with most diverse attention patterns

## 4. Information Flow Analysis

### Vector Magnitude Flow
For "Az alma piros és", the hidden state vector magnitude per token changes across layers:

- **"Az"** starts strong (embedding), decreases after L3 (grammar processed quickly)
- **"alma"** stays strong throughout (core subject - needs attention)
- **"piros"** peaks in middle layers (modifier - processed with context)
- **"és"** starts weak, strengthens in deep layers (conjunction - needs broader context)

### Winner: Attention vs FFN per Layer

| Layer | Dominant | Contribution |
|-------|----------|-------------|
| L1    | FFN      | Early pattern matching |
| L2    | FFN      | |
| L3    | FFN      | |
| L4    | FFN      | |
| L5    | FFN      | |
| L6    | FFN      | |
| L7    | FFN      | |
| L8    | FFN      | |
| **L9**| **Attention** | **1.38x** - Context gathering starts |
| L10   | Attention | 2.81x - Peak context integration |
| L11   | Attention | |
| L12   | Attention | |
| L13   | Balanced  | |
| L14   | Attention | |
| L15   | Balanced  | |
| L16   | Balanced  | |
| L17   | Balanced  | |
| L18   | FFN       | Knowledge retrieval |
| L19   | FFN       | |
| L20   | FFN       | |
| L21   | FFN       | |
| L22   | FFN       | |
| L23   | FFN       | Final knowledge retrieval |
| L24   | Balanced  | Output projection |

### Flow Pattern
```
L1-L8:   FFN → pattern detection
L9-L12:  Attention → context integration (PEAK at L10: 2.81x)
L13-L17: Balanced → transition
L18-L23: FFN → knowledge retrieval for prediction
L24:     Balanced → final mix before output
```

## 5. The 24 Layers - Are All Useful?

**YES.** All 24 layers are active and contribute:
- L1-L4 (less active, but necessary for initial processing)
- L5-L12 (medium activity, context integration)
- L13-L23 (high activity, knowledge retrieval)
- L24 (low activity, but essential for output projection)

There is no evidence of redundant or dead layers.

## 6. Short vs Long Sequence Processing

When comparing 4-token ("Az alma piros és") vs 8-token ("Ha 5 almám van és megeszek 2-t") inputs:

| Metric | 4 tokens | 8 tokens |
|--------|---------|---------|
| L23 sparsity | 44.8% | 40.9% |
| Deep L17-23 avg | 18.1% | 14.6% |
| Head diversity (L17-24) | 0.128 | 0.091 |

**Longer inputs reduce head diversity and sparsity.** The model's attention resources spread thinner with more tokens, suggesting the model struggles with sequences longer than ~8 tokens (its training context is 512 tokens but effective context is much shorter).

---

*End of Paper #3. Next: Paper #4 - Model Editing: FFN, Steering & Adapters*
