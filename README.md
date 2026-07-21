# Neuron Modification Guide — Lépésről lépésre útmutató neuronok módosításához kis nyelvi modellekben

**Status:** ✅ Complete — 25 fejezetes dokumentáció kész, nincs futtatható kód

Teljes útmutató kezdőtől haladóig a neuronok közvetlen módosításához kis nyelvi modellekben (NEURA 300M, 355M paraméter, 24 réteg, 1024 dim). 25 fejezet: aktiváció patching, ROME, MEMIT, FiNE, sparse autoencoderek, activation steering.

**Modell:** NEURA 300M (355M parameters, 24 layers, 1024 dim)
**Kísérletek:** 2026. június-július
**Licenc:** CC BY 4.0

## ⚠️ THIS PROJECT IS UNFINISHED — FEEL FREE TO CONTINUE IT ⚠️

**Ez a projekt NINCS KÉSZEN. Bárki folytathatja, aki akarja!**
Ezt a projektet Zsombi & Hermes Agent (Nous Research) közösen fejlesztette, de egyik projekt sincs 100%-osan befejezve. Ha tetszik az ötlet és tovább fejlesztenéd, nyugodtan fork-old, folytasd, és csinálj belőle valami nagyszerűt!

---

## Tartalom (25 fejezet)

| Fájl | Téma |
|------|------|
| `ch19_activation_patching.md` | Aktiváció patching |
| `ch20_ROME.md` | ROME (Rank-One Model Editing) |
| `ch21_MEMIT.md` | MEMIT (Mass Editing Memory in Transformers) |
| `ch22_FiNE.md` | FiNE (Fast Inference Neural Editing) |
| `ch23_SAE.md` | Sparse Autoencoderek |
| `ch24_steering.md` | Activation steering |
| `ch25_update.md` | Frissítések |
| `neuron_modification_book.md` | Teljes könyv |
| `combined_book.md` | Összevont verzió |

### Kutatási cikkek
- NEURA training dynamics
- Neuron specializáció
- Attention hierarchia
- Miért nem tudnak reasoning-olni a kis modellek
- Practical guide

## Fejlesztő
Zsombi & Hermes Agent (Nous Research)
