# Neuronok Módosítása Kis Nyelvi Modellekben

**Teljes Útmutató Kezdőtől Haladóig**  
**Modell:** NEURA 300M (355M paraméter, 24 réteg, 1024 dim)  
**Alapuló kísérletek:** 2026. június–július  
**Szerző:** Zsombi & Hermes Agent (Nous Research)  
**Licenc:** CC BY 4.0

---

## Leírás

Ez a projekt egy **lépésről-lépésre útmutató könyv**, amely bemutatja hogyan lehet **közvetlenül módosítani a neuronokat** kis nyelvi modellekben (mint a NEURA 300M). A könyv 25 fejezeten keresztül vezet végig az alapoktól a legújabb kutatási módszerekig, beleértve az aktivációs patchinget, ROME-ot, MEMIT-et, FiNE-t, sparse autoencodereket és fejlett activation steeringet.

---

## Fájlszerkezet

```
neuron_modification_book/
│
├── README.md                              # Ez a dokumentum
├── STATUS.md                              # Projekt státusz
│
├── combined_book.md                       # A teljes könyv (1-18. fejezet)
├── neuron_modification_book.md            # A könyv korábbi verziója
│
├── neura_research_paper_01_training_dynamics.md     # 1. Kutatási cikk: Training dinamika
├── neura_research_paper_02_neuron_specialization.md # 2. Kutatási cikk: Neuron specializáció
├── neura_research_paper_03_attention_hierarchy.md   # 3. Kutatási cikk: Attention hierarchia
├── neura_research_paper_04_why_cant_reason.md       # 4. Kutatási cikk: Miért nem tud következtetni
├── neura_research_paper_05_practical_guide.md       # 5. Kutatási cikk: Gyakorlati útmutató
│
├── ch19_activation_patching.md            # 19. fejezet: Aktivációs Patching
├── ch20_ROME.md                           # 20. fejezet: Rank-One Model Editing
├── ch21_MEMIT.md                          # 21. fejezet: Mass-Editing Memory
├── ch22_FiNE.md                           # 22. fejezet: Fine-grained Neuron Editing
├── ch23_SAE.md                            # 23. fejezet: Sparse Autoencoders
├── ch24_steering.md                       # 24. fejezet: Aktivációs Steering
├── ch25_update.md                         # 25. fejezet: Döntési fa + összefoglaló
│
├── pulse_adapter_results.md               # Pulse adapter eredmények
│
└── scripts/                               # Kísérleti scriptek
    ├── RUNME.bat                          # Indító script
    ├── logicadapter_pulse.py              # LogicAdapter Pulse kísérlet
    ├── logicadapter_v6.py                 # LogicAdapter v6
    ├── activation_patching.py             # Aktivációs patching kísérlet
    ├── rome_edit.py                       # ROME edit kísérlet
    └── tokenize_opensubs.py               # OpenSubtitles tokenizálás
```

---

## A könyv tartalma

### 1. rész: Alapok
| Fejezet | Téma |
|---------|------|
| 1 | Bevezetés — Mi az a neuron módosítás? |
| 2 | A Transformer Neuron Felépítése |
| 3 | Környezet Telepítése |
| 4 | Az Első Neuron Módosítás |

### 2. rész: Módszerek
| Fejezet | Téma |
|---------|------|
| 5-18 | Köztes fejezetek (combined_book.md) |
| 19 | **Aktivációs Patching & Causal Tracing** |
| 20 | **ROME** — Rank-One Model Editing |
| 21 | **MEMIT** — Tömeges memória szerkesztés |
| 22 | **FiNE** — Neuron-szintű tudás szerkesztés (ICLR 2025) |
| 23 | **Sparse Autoencoders** — Polysemantic neuronok szétfejtése |
| 24 | **Fejlett Activation Steering** |
| 25 | **Frissített döntési fa** — 6+ módszer összehasonlítása |

---

## Használat

### Scriptek futtatása

```bash
cd scripts

# Aktivációs patching
python activation_patching.py

# ROME edit
python rome_edit.py

# LogicAdapter kísérletek
python logicadapter_pulse.py
python logicadapter_v6.py
```

### Könyv olvasása

A könyv Markdown formátumban van, bármilyen Markdown olvasóval megnyitható:

```bash
# VS Code
code combined_book.md

# Böngészőben
start combined_book.md
```

---

## Miért érdemes neuront módosítani?

| Hagyományos módszer | Neuron módosítás |
|---------------------|------------------|
| Teljes fine-tuning (órák, sok GPU) | Egy neuron átírása (másodpercek, CPU) |
| Minden paraméter változik | Csak 1-2 neuron változik |
| Katasztrofális felejtés veszélye | Célzott, a többi tudás sértetlen |
| Nagy adatkészlet kell (~10K+ példa) | 1-100 példa is elég |

---

## Fejlesztő

Zsombi & Hermes Agent (Nous Research) (AI asszisztens segítségével)
