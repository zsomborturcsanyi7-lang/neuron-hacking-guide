# 20. fejezet: ROME — Rank-One Model Editing

## 20.1 A probléma

A 9. fejezetben kézzel szerkesztettünk FFN neuronokat — és "elny" loop-ot kaptunk. A probléma: a kézi edit nem veszi figyelembe a modell belső struktúráját.

**ROME (Rank-One Model Editing)** — Meng et al., 2022 — ezt a problémát oldja meg egy zárt formulával.

## 20.2 Az elmélet

A transformer FFN rétegei **asszociatív memóriaként** működnek:

```
FFN(x) = W_out * σ(W_in * x)
                      ↑            ↑
                 "kulcs"     "érték" (key)   (value)
```

Egy tény, pl. "(Eiffel-torony, helye, Párizs)" a modellben úgy tárolódik, hogy:

- A **subject** utolsó tokenje (Eiffel-torony → "torony") kulcsként szolgál
- A kulcshoz tartozó érték a **kimenet irány** (Párizs felé tolja a logitokat)

A ROME egy **rank-one** mátrixfrissítéssel írja át ezt:

```
Eredeti:   v = W * k       (W = FFN.w2, k = subject hidden state)
Cél:       v* = W' * k     (v* = a kívánt új kimenet iránya)

Rank-one update:
    ΔW = r * k^T * C^{-1} / (k^T * C^{-1} * k)
    
    ahol:
    r = v* - v              (reziduális — mennyit kell változtatni)
    k = subject hidden state (a kulcs)
    C = K * K^T             (kovariancia mátrix — megőrzi a meglévő tudást)
```

## 20.3 Implementáció NEURA 300M-re

```python
import torch

def rome_edit(model, layer_idx, token_ids, subject_pos, target_token_id,
              C_inv=None, sample_prompts=None):
    """
    ROME edit implementáció NEURA 300M-re.
    
    Args:
        layer_idx: melyik réteg FFN-jét szerkesztjük (ajánlott: 22 vagy 23)
        token_ids: a prompt token ID-i (pl. "Ha 5 almám van és megeszek 2-t")
        subject_pos: a subject (pl. "5") utolsó tokenjének pozíciója
        target_token_id: a kívánt kimenet token ID-ja (pl. "3")
        C_inv: pre-számolt inverz kovariancia (None = számol most)
        sample_prompts: list of prompts a C mátrixhoz
    """
    ffn = model.blocks[layer_idx].ffn
    
    # 1. KULCS kinyerése: subject hidden state
    x = torch.tensor([token_ids])
    with torch.no_grad():
        h = model.tok(x)
        for li, block in enumerate(model.blocks):
            h = block(h)
            if li == layer_idx - 1:  # a target layer ELŐTTI réteg
                break
        
        # A subject utolsó tokenjének hidden state-je
        k = h[0, subject_pos].clone()  # [1024]
    
    # 2. CÉLÉRTÉK kiszámolása
    # A kívánt output token iránya az output mátrixból
    target_dir = model.out.weight.data[target_token_id].clone()  # [1024]
    
    # Jelenlegi érték
    current = ffn.w2.weight.data @ k  # [1024]
    
    # Reziduális: mennyit kell változtatni
    r = target_dir - current
    
    # 3. C mátrix (vagy használjuk a pre-számolt verziót)
    if C_inv is None:
        if sample_prompts is None:
            # Alapértelmezett: random magyar mondatok
            sample_prompts = [
                "Az alma piros és édes",
                "Ma szép idő van",
                "A kutya a kertben fut",
                "Budapest Magyarország fővárosa",
                "A gyerekek játszanak az udvaron",
                # ... minimum 300-500 mondat
            ]
        
        # K vektorok gyűjtése
        K = []
        for prompt in sample_prompts:
            ids = sp.EncodeAsIds(prompt)
            x = torch.tensor([ids])
            with torch.no_grad():
                h = model.tok(x)
                for li, block in enumerate(model.blocks):
                    h = block(h)
                    if li == layer_idx - 1:
                        break
                # Utolsó token hidden state-je
                k_sample = h[0, -1].clone()
                K.append(k_sample)
        
        K = torch.stack(K)  # [N, 1024]
        
        # C = K^T * K + lambda * I (regularizáció)
        lambda_reg = 0.1  # kis regularizáció a numerikus stabilitásért
        C = K.T @ K + lambda_reg * torch.eye(1024)
        
        # C^{-1}
        C_inv = torch.linalg.inv(C)
    
    # 4. RANK-ONE UPDATE
    # ΔW = r * k^T * C^{-1} / (k^T * C^{-1} * k)
    
    C_inv_k = C_inv @ k  # [1024]
    denominator = k @ C_inv_k  # scalar
    
    # A rank-one update
    delta = torch.outer(r, C_inv_k) / denominator  # [1024, 3072]
    
    # Alkalmazás: W2 += ΔW^T (mert W2 alakja [1024, 3072])
    ffn.w2.weight.data += delta.T
    
    # 5. ELLENŐRZÉS
    model.eval()
    x = torch.tensor([token_ids])
    with torch.no_grad():
        logits = model(x)
    
    probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
    target_prob = probs[target_token_id].item() * 100
    print(f"P(target token) = {target_prob:.4f}%")
    
    # Top-5 predikció
    top5 = torch.argsort(probs, descending=True)[:5]
    for tid in top5:
        print(f"  P({sp.IdToPiece(tid.item())}) = {probs[tid].item()*100:.2f}%")
    
    return ffn.w2.weight.data


# ====== HASZNÁLAT ======

# CÉL: "Ha 5 almám van és megeszek 2-t" → "marad 3"

prompt = "Ha 5 almám van és megeszek 2-t"
ids = sp.EncodeAsIds(prompt)
# Tokenek: ['▁Ha', '▁5', '▁almám', '▁van', '▁és', '▁megeszek', '▁2-', 't']
# Pos:       0      1      2        3      4      5          6      7

# Subject = "5" → position 1
subject_pos = 1

# Target = "3" token
target_token = "▁3"
target_id = sp.PieceToId(target_token)

print(f"Edit: '{prompt}' → '{target_token}'")
print(f"Subject pos: {subject_pos}, Target ID: {target_id}")

# ROME edit az L23 FFN-en
updated_weights = rome_edit(
    model,
    layer_idx=22,  # L23 (0-indexelt: 22)
    token_ids=ids,
    subject_pos=subject_pos,
    target_token_id=target_id,
)
```

## 20.4 Várakozási értékek NEURA 300M-en

A ROME-nál három dolog történhet:

| Eredmény | Mit jelent | Valószínűség |
|----------|-----------|--------------|
| **✅ Működik** | A target token lesz a legvalószínűbb | ~30% NEURA-n |
| **❌ "elny" loop** | Túl erős a változás, más token dominál | ~40% |
| **❌ Semmi változás** | Túl gyenge a változás, más rétegben van a tudás | ~30% |

**Miért nem mindig működik a ROME 355M-en?**
- A ROME-t 6B+ modellekre tervezték
- 355M-nél a reprezentációk gyengébbek (max aktiváció 15.7 vs 200+)
- A C mátrix regularizációja kritikus — túl erős λ = semmi változás, túl gyenge = loop

**Optimalizálás NEURA-ra:**

```python
# Különböző regularizációk kipróbálása
for lambda_reg in [0.01, 0.05, 0.1, 0.5, 1.0]:
    C = K.T @ K + lambda_reg * torch.eye(1024)
    C_inv = torch.linalg.inv(C)
    # ... teszt
    print(f"λ={lambda_reg}: P(target)={prob:.2f}%")

# Különböző rétegek kipróbálása
for layer in [20, 21, 22, 23]:
    # ... ROME edit
    print(f"L{layer+1}: P(target)={prob:.2f}%")
```

## 20.5 Előnyök és hátrányok

| Szempont | ROME | Kézi FFN edit (Ch9) |
|----------|------|---------------------|
| **Matematikai alap** | Zárt formula | Próba-szerencse |
| **Tudásmegőrzés** | C mátrix védi | Semmi védelem |
| **Gyorsaság** | 1 forward + 1 mátrix invertálás | 1 forward |
| **Pontosság** | Függ a C-től | Tipikusan "elny" loop |
| **Batch** | Csak 1 tény | 1 neuron |

**Mikor használd?**
- ✅ Egyetlen tény pontos javításához
- ✅ Ha a C mátrixot előre ki tudod számolni
- ❌ Több száz tény szerkesztéséhez (→ MEMIT a 21. fejezetben)
- ❌ Ha nincs 300+ magyar mondatod a C mátrixhoz
