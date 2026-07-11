## Pulse 350M LogicAdapter Training Eredmény

**Dátum:** 2026. július 2.
**Modell:** Pulse 350M (367.5M paraméter, fused QKV MHA + SwiGLU FFN)
**GPU:** Lokális (CUDA)
**Idő:** 3503 másodperc (~58 perc)

### Konfiguráció

| Paraméter | Érték |
|-----------|-------|
| Adapter hidden | 128 (1024→128→1024) |
| Példák száma | 450 (50/digit) |
| Epoch | 200 |
| Batch size | 16 |
| Learning rate | 1e-3 (cosine decay) |
| Loss | Combined (selective + 0.5× full vocab) |
| Prompt variációk | 17 féle minta |

### Training görbe

```
Epoch   0: loss=6.72  acc=14.9%
Epoch  10: loss=3.06  acc=35.8%
Epoch  20: loss=2.32  acc=54.0%
Epoch  30: loss=1.72  acc=67.1%
Epoch  40: loss=1.26  acc=75.1%
Epoch  50: loss=1.17  acc=76.7%
Epoch  60: loss=0.91  acc=82.7%
Epoch  70: loss=0.82  acc=86.7%
Epoch  80: loss=0.69  acc=87.1%
Epoch  90: loss=0.50  acc=90.4%
Epoch 100: loss=0.46  acc=93.3%
Epoch 120: loss=0.21  acc=95.8%
Epoch 130: loss=0.14  acc=98.0%
Epoch 140: loss=0.06  acc=98.4%
Epoch 150: loss=0.05  acc=99.1%
Epoch 160: loss=0.03  acc=100.0%
Epoch 199: loss=0.02  acc=100.0%
```

### Teszt eredmény (20 új prompt)

| Eredmény | Szám | Példák |
|----------|------|--------|
| ✅ Helyes | 15/20 (75%) | "5 - 2 =" → 3, "10 - 3 =" → 7 |
| ❌ Off-by-1 | 3/20 | "4 - 2 =" → 1 (expected 2) |
| ❌ Rossz | 2/20 | "mennyi 8 - 5?" → 7 (expected 3) |

### Következtetés

A LogicAdapter koncepció **működik** mind a NEURA 300M-en (GQA), mind a Pulse 350M-en (MHA). Az architektúra különbség nem számít, mert az adapter a residual stream-en dolgozik.

A 100% training és 75% test accuracy jelentős javulás az eredeti 55%-hoz képest. A hibák többsége off-by-1, ami arra utal, hogy a modell megtanulta a "növel/csökkent" mintázatot, de a finom megkülönböztetéshez nagyobb hidden dim (256) vagy több adat (1000+ példa) kell.
