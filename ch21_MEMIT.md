# 21. fejezet: MEMIT — Mass-Editing Memory in a Transformer

## 21.1 A probléma

A ROME (20. fejezet) egy tény szerkesztésére jó. De ha **több száz** tényt kell egyszerre javítani (pl. 100 kivonási feladat), a ROME egymás utáni alkalmazása rontja az eredményt — az új edit felülírja a régit.

**MEMIT (Mass-Editing Memory in a Transformer)** — Meng et al., 2023 (ICLR 2023 Notable Top 25%) — ezt oldja meg.

## 21.2 Az elmélet

A MEMIT felismeri, hogy a tények **több rétegben** tárolódnak. Ahelyett, hogy egyetlen réteget terhelnénk túl (mint a ROME), a MEMIT **szétosztja** a változtatásokat egy rétegtartományon.

```
ROME:
        L20  L21  L22  L23  L24
Edit1:       [ΔW]                ← Minden változás egy helyen!
Edit2:       [ΔW]                ← Felülírja Edit1-et!

MEMIT:
        L20  L21  L22  L23  L24
Edit1:  [~]  [~]  [~]            ← Szétosztva!
Edit2:       [~]  [~]  [~]       ← Más rétegekben, nem ütközik!
```

**Matematika:** MEMIT minden edithez kiszámolja a hozzájárulást minden target rétegen, majd egy **zárt formában** egyszerre alkalmazza az összes frissítést.

## 21.3 Implementáció NEURA 300M-re

```python
def memit_edit(model, edits, target_layers=[20, 21, 22], sample_prompts=None):
    """
    MEMIT batch edit implementáció.
    
    Args:
        edits: [(prompt, subject_pos, target_token_id), ...]
               pl. [("5 - 2 =", 0, "▁3"), ("8 - 5 =", 0, "▁3"), ...]
        target_layers: mely rétegek között osszuk el a változtatásokat
    """
    num_edits = len(edits)
    num_layers = len(target_layers)
    
    # 1. K és V vektorok kiszámolása minden edithez
    K = []  # [num_edits, 1024]    — kulcs vektorok
    V = []  # [num_edits, 1024]    — érték vektorok
    
    for prompt, subj_pos, target_id in edits:
        ids = sp.EncodeAsIds(prompt)
        x = torch.tensor([ids])
        
        with torch.no_grad():
            h = model.tok(x)
            # Kulcs: a target_layers első elemének BEMENETÉNÉL
            for li in range(target_layers[0]):
                h = model.blocks[li](h)
            
            k = h[0, subj_pos].clone()    # [1024]
            
            # Érték: a kívánt kimenet iránya
            v = model.out.weight.data[target_id].clone()  # [1024]
        
        K.append(k)
        V.append(v)
    
    K = torch.stack(K)  # [num_edits, 1024]
    V = torch.stack(V)  # [num_edits, 1024]
    
    # 2. Delta kiszámolása minden rétegre
    # Minden target rétegen kiszámoljuk a szükséges változtatást
    deltas = []
    
    for li, layer_idx in enumerate(target_layers):
        ffn = model.blocks[layer_idx].ffn
        
        # Jelenlegi hozzájárulás az edit helyeken
        with torch.no_grad():
            # current_values = W2 @ K_i minden i-re
            current = ffn.w2.weight.data @ K.T  # [1024, num_edits]
        
        # Mennyit kell változtatni ezen a rétegen?
        delta_v = V.T - current  # [1024, num_edits]
        
        # Egyenletes elosztás a rétegek között
        delta_v = delta_v / num_layers
        
        # Delta kiszámolása minden edit-re
        # ΔW_i = outer(delta_v[:, i], K[i]) / valami
        delta_W = torch.zeros_like(ffn.w2.weight.data)
        
        for i in range(num_edits):
            # Rank-one update
            ki = K[i]      # [1024]
            dvi = delta_v[:, i]  # [1024]
            
            # ΔW += outer(dvi, ki)
            delta_W += torch.outer(dvi, ki)
        
        deltas.append(delta_W)
    
    # 3. Visszafejtés a preservation-memorization framework-ben
    # Ez a MEMIT központi egyenlete: egyszerre optimalizálja
    # az összes deltát, hogy minimalizálja a konfliktust
    
    # A pontos MEMIT formula: 
    # Δ = V * K^T * (C + K * K^T)^{-1}   (zárt alak, batch-re)
    # 
    # Ahol C a pre-cached kovariancia mátrix
    
    # (A pontos implementáció a unified-model-editing 
    #  GitHub repo-ban található — lásd a 20. fejezet C mátrix részét)
    
    return deltas


# ====== HASZNÁLAT ======

# CÉL: 12 kivonási feladat megtanítása egyszerre
edits = [
    ("2 - 1 =",  0, sp.PieceToId("▁1")),
    ("3 - 1 =",  0, sp.PieceToId("▁2")),
    ("4 - 1 =",  0, sp.PieceToId("▁3")),
    ("5 - 2 =",  0, sp.PieceToId("▁3")),
    ("5 - 3 =",  0, sp.PieceToId("▁2")),
    ("6 - 1 =",  0, sp.PieceToId("▁5")),
    ("6 - 2 =",  0, sp.PieceToId("▁4")),
    ("7 - 3 =",  0, sp.PieceToId("▁4")),
    ("8 - 5 =",  0, sp.PieceToId("▁3")),
    ("9 - 2 =",  0, sp.PieceToId("▁7")),
    ("10 - 1 =", 0, sp.PieceToId("▁9")),
    ("10 - 3 =", 0, sp.PieceToId("▁7")),
]

# MEMIT az L21-L23 rétegeken
deltas = memit_edit(model, edits, target_layers=[20, 21, 22])

# Ellenőrzés
for prompt, _, target_id in edits:
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x)
    
    probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
    target_prob = probs[target_id].item() * 100
    top_token = sp.IdToPiece(logits[0, -1].argmax().item())
    expected = sp.IdToPiece(target_id)
    mark = "✅" if top_token == expected else "❌"
    print(f"{mark} '{prompt}' → top={top_token:6s} (P={target_prob:.1f}%)")
```

## 21.4 MEMIT vs LogicAdapter

| Szempont | MEMIT | LogicAdapter (Ch12) |
|----------|-------|---------------------|
| **Paraméterek** | Bármennyi (a fő modellé) | 131K-262K |
| **Batch méret** | 10,000+ tény | 50-200 példa |
| **Training** | Zárt formula (nincs) | 120 epoch (7 perc GPU) |
| **Pontosság** | ~90%+ (nagy modelleken) | ~55% (kicsin) |
| **Biztonság** | C mátrix véd | Zero init véd |
| **355M-en tesztelve** | ❌ Nem | ✅ Igen |

**Következtetés:** MEMIT nagy modellekre való (7B+). 355M-en a LogicAdapter a biztonságosabb választás. De ha a MEMIT matematikáját adaptáljuk (kisebb C mátrix, kevesebb réteg), akkor 355M-en is működhet.

## 21.5 A preservation-memorization framework

A MEMIT és ROME mögött egy közös elmélet van: a **preservation-memorization**:

```
Minimalizálandó: ||W' * K_preserve - W * K_preserve||²F  (preservation)
                     + ||W' * K_edit - V_edit||²F         (memorization)
```

- **Preservation:** a régi tudás megőrzése (C mátrix)
- **Memorization:** az új tények beillesztése

A ROME ezt **equality constraint**-tel oldja meg (pontosan egyezzen az új tény), a MEMIT **least-square** constraint-tel (minimalizálja a hibát).

**EMMET** (Gupta et al., 2024) a kettő unifikációja — equality constraint batch módban.
