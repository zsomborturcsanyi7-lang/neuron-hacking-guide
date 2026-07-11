# Neuronok Módosítása Kis Nyelvi Modellekben

**Teljes Útmutató Kezdőtől Haladóig**

## A mappa tartalma

| Fájl | Leírás |
|------|--------|
| `neuron_modification_book.md` | **A teljes könyv** (1-18. fejezet az eredetiből + 19-25. új fejezetek) |
| `ch19_activation_patching.md` | Aktivációs Patching & Causal Tracing |
| `ch20_ROME.md` | Rank-One Model Editing (ROME) |
| `ch21_MEMIT.md` | Mass-Editing Memory in a Transformer |
| `ch22_FiNE.md` | FiNE — Fine-grained Neuron-level Knowledge Editing (ICLR 2025) |
| `ch23_SAE.md` | Sparse Autoencoders a neuronok megértésére |
| `ch24_steering.md` | Fejlett Aktivációs Steering & Reprezentáció Engineering |
| `ch25_update.md` | Frissített döntési fa + összefoglaló |

**Modell:** NEURA 300M (355M paraméter, 24 réteg, 1024 dim)
**Alapuló kísérletek:** 2026. június-július
**Frissítés:** 2026. július (új fejezetek a legújabb kutatások alapján)

## Újdonságok ebben a verzióban

- **Aktivációs Patching** — a korreláció → kauzalitás lépés
- **ROME** — rank-one FFN edit zárt formula segítségével
- **MEMIT** — több ezer tény egyidejű szerkesztése
- **FiNE** — neuron-szintű knowledge editing (ICLR 2025)
- **Sparse Autoencoders** — polysemantic neuronok szétfejtése
- **Fejlett Activation Steering** — több prompt átlagából
- **Frissített döntési fa** — 6+ módszer összehasonlítása
