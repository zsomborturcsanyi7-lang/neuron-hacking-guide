# NEURA 300M Továbbfejlesztés — Scriptek

## Aktuális állapot

| Folyamat | Státusz | Idő |
|----------|---------|-----|
| **OpenSubtitles HU letöltés** | ⏳ 57% (1.42 GB) | ~6 perc |
| **LogicAdapter Pulse 350M training** | ⏳ Indult | ~7-15 perc |

## A mappa tartalma

### Scriptek (azonnal futtathatók)

| Script | Mit csinál | Futtatás |
|--------|-----------|----------|
| `logicadapter_pulse.py` | LogicAdapter 500 példán, 128 hidden, Pulse 350M-re | `python logicadapter_pulse.py` |
| `logicadapter_v6.py` | Ugyanez NEURA 300M architektúrára (ha megvan a checkpoint) | `python logicadapter_v6.py` |
| `activation_patching.py` | 5→2 attention vizsgálat + kauzális tracing | `python activation_patching.py` |
| `rome_edit.py` | Rank-One edit paraméterteszttel | `python rome_edit.py --prompt "5 - 2 =" --target "3"` |
| `tokenize_opensubs.py` | OpenSubtitles tokenizálás NEURA tokenizerrel | `python tokenize_opensubs.py` |
| `RUNME.bat` | Futtatási útmutató | — |

### Könyv fejezetek

| Fájl | Tartalom |
|------|---------|
| `combined_book.md` | Teljes könyv 25 fejezet (131 KB) |
| `ch19-ch25/*.md` | Új fejezetek külön fájlokban |
| `neura_research_paper_01-05.md` | 5 research paper |

## Erőforrások ezen a gépen

- **GPU:** Van (CUDA) — LogicAdapter training megy
- **Modell:** Pulse 350M (367.5M paraméter, magyar) — betöltve és működik
- **Tokenizer:** NEURA SentencePiece 32K — elérhető
- **NEURA 300M checkpoint:** NINCS lokálisan (a remote RTX 3070-en van)
- **OpenSubtitles HU:** Letöltés alatt (1.42 GB)

## Terv

1. **LogicAdapter training a Pulse 350M-en** → most fut
2. **OpenSubtitles HU tokenizálás** → letöltés után
3. **Activation Patching a Pulse-on** → diagnosztika
4. **Eredmények átvitele a NEURA 300M-re** → amikor a remote elérhető
