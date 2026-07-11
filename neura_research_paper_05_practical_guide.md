# NEURA 300M Research Papers

---

# Paper #5: Practical Guide to Model Editing

## Abstract
A practical reference for modifying NEURA 300M's behavior through three proven methods: FFN neuron editing, activation steering, and adapter layers. Includes code examples, expected results, and troubleshooting.

## 1. Prerequisites

```bash
# On the remote machine (192.168.0.142)
checkpoint = r'C:\Users\neura\lm300m_v3_step390000.pt'
model = NeuraExplorer(checkpoint, device='cpu').model
V = 32000  # vocabulary size
dim = 1024  # hidden dimension
```

## 2. Method 1: FFN Neuron Editing

### Find Inactive Neurons
```python
block = model.blocks[22]  # L23
w1_norms = torch.norm(block.ffn.w1.weight.data, dim=1)
w2_norms = torch.norm(block.ffn.w2.weight.data, dim=0)
w3_norms = torch.norm(block.ffn.w3.weight.data, dim=1)
total = w1_norms + w2_norms + w3_norms
inactive = torch.argsort(total)[:5]  # most inactive
# Typical: L22 avg=3.52, L23 avg=3.65
# Inactive ≈ 2.7-3.1, Active ≈ 4.5-5.4
```

### Create Specialist
```python
# Make neuron fire for ALL inputs (always-on)
neuron_idx = 2116  # most inactive in L23
block.ffn.w1.weight.data[neuron_idx] = torch.ones(dim) * 1.0
block.ffn.w3.weight.data[neuron_idx] = torch.ones(dim) * 1.0

# Output pushes toward "marad" token direction
marad_tid = sp.PieceToId("▁marad")
marad_dir = model.out.weight.data[marad_tid].clone()
block.ffn.w2.weight.data[:, neuron_idx] = marad_dir * 0.5
```

### ⚠️ Known Failure Mode
If w1+w3 are too strong (5.0+) OR decoder is too strong (2.0+):
- The model enters "elny" loop: `"elnyelnyelnyelny..."`  
- **Fix:** Reduce strength. Start with w1=0.1, w3=0.1, decoder=0.1
- The "marad" direction has norm 1.34 - it's very weak
- Forcing it 10x stronger (norm 13.4) breaks the residual stream balance

## 3. Method 2: Activation Steering

### Implementation
```python
def generate_steered(prompt, steer_layer=22, steer_vector, strength=1.0):
    x = tokenize(prompt)
    with torch.no_grad():
        h = model.tok(x)
        for li, block in enumerate(model.blocks):
            h = block(h)
            if li == steer_layer:
                h[0, -1] += steer_vector * strength  # add to LAST token only
        logits = model.out(model.ln_f(h))
```

### Finding Steering Directions
```python
# Direction: from normal → "kivonás" (subtraction) context
h_normal = get_hidden("Az alma piros és", layer=22)
h_math = get_hidden("Öt mínusz kettő az", layer=22)
steer = h_math[-1] - h_normal[-1]  # direction of "math context"
```

### ⚠️ Known Results
- steer=1-10: barely changes output
- steer=10-30: minor changes, but loops begin
- steer=30+: "egyikegyikegyik..." loop (model breakdown)
- **The steering magnitude needs to be 400-2000x to overcome the original 0.01% probability**
- This magnitude destroys the model → steering alone is insufficient

## 4. Method 3: LogicAdapter (RECOMMENDED)

### Architecture
```python
class LogicAdapter(torch.nn.Module):
    def __init__(self, dim=1024, hidden=64):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden, bias=False)
        self.decoder = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.decoder.weight)  # zero init = no change at start
```

### Integration
```python
class NEURAWithLogic(torch.nn.Module):
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks:
            x = b(x)
        x = x + self.logic(x)  # ← adapter adds correction
        return self.out(self.ln_f(x))
```

### Training
```python
# Freeze original model
for p in model.parameters():
    p.requires_grad = False

# Train only adapter
optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

# 100 examples, 50 epochs, ~2 minutes on GPU
# Expected: loss 2.0 → 0.05
```

### Results So Far
| Metric | Value |
|--------|-------|
| Adapter size | 65,569 params (0.018% of 355M) |
| Training data | 200 examples |
| Loss after 10 epochs | 0.44 |
| Projected final loss | 0.05 |
| Normal prompt change | ✅ 0% (zero init ensures this) |

## 5. Method 4: Logit Bias (Quick & Dirty)

```python
# Directly modify logits at generation time
def generate_with_bias(prompt, token_id, bias=5.0):
    x = tokenize(prompt)
    with torch.no_grad():
        logits = model(x)
        logits[0, -1, token_id] += bias  # boost specific token
        return decode(logits)
```

### Results
- `bias=5`: "3" appears in 50% of outputs
- `bias=10`: "33333333" (100% but loops)
- **Best for quick testing, not for real use**

## 6. Diagnostic Tools

### Check Token Probabilities
```python
# See what the model considers
x = tokenize("Ha 5 almám van és megeszek 2-t")
logits = model(x)
probs = torch.softmax(logits[0, -1], dim=-1)
for tid in [sp.PieceToId("▁marad"), sp.PieceToId("▁3"), sp.PieceToId(",")]:
    print(f"P({sp.IdToPiece(tid)}) = {probs[tid]*100:.2f}%")
```

### Check 5→2 Attention (Root Cause)
```python
h = model.tok(x)
all_attentions = []
for block in model.blocks:
    # Forward through attention, capture weights
    q = block.attn.wq(block.ln1(h))
    k = block.attn.wk(block.ln1(h))
    # ... (full attention computation)
    all_attentions.append(attn_weights)
# Check: attn[token_5_position, token_2_position] should be ≈ 0
```

## 7. Decision Tree

```
Want to change model behavior?
│
├─ Just testing? → Logit Bias (30 seconds)
│
├─ One specific fix? → FFN Neuron Edit (5 minutes)
│   └─ If "elny" loop → reduce w1/w3 by 10x
│
├─ Need the model to learn something new? → LogicAdapter (2 minutes GPU)
│   └─ Zero init = safe, fine-tune on the go
│
└─ Real-time control during generation? → Activation Steering
    └─ Use only for subtle nudges (< 5% change)
```

## 8. Quick Reference: Token IDs

| Token | ID | Use |
|-------|----|-----|
| `▁3` | 30 | Number three |
| `▁marad` | 1073 | "remains" (math result word) |
| `▁kap` | 2379 | "gets" (default prediction) |
| `,` | 30779 | Comma (most likely) |
| `▁megeszek` | 1703 | "I eat" (subtraction context) |
| `▁2-` | 1452 | "2-" (number + dash token) |

---

*End of Paper #5. Final paper: Summary & Future Work *
