# 25. fejezet: Frissített Döntési Fa és Módszer Összehasonlítás

## 25.1 Az új döntési fa

```
Szeretnéd változtatni a modell viselkedését?
│
├─ MIT akarsz pontosan?
│  │
│  ├─ "Csak kipróbálni valamit" (30mp-5p)
│  │  ├─ Logit Bias (legegyszerűbb, de loop-ol)
│  │  └─ Activation Steering (normalizált vektor, stabilabb)
│  │
│  ├─ "Megérteni, MIÉRT viselkedik így"
│  │  ├─ MapMaker (korreláció: mely neuronok tüzelnek?)
│  │  └─ Activation Patching (kauzalitás: mely neuronok FELELŐSEK?)
│  │
│  ├─ "Pontosan kijavítani egy hibát" (1 edit)
│  │  ├─ ROME (zárt formula, rank-one update)
│  │  └─ FINE neuron lokalizáció + edit (pontosabb, de bonyolultabb)
│  │
│  ├─ "Több hibát egyszerre javítani" (10-100 edit)
│  │  ├─ MEMIT (több rétegen elosztva)
│  │  └─ LogicAdapter (zero-init, biztonságos)
│  │
│  ├─ "Új képességet tanítani" (pl. számolás)
│  │  ├─ LogicAdapter (biztonságos, 55% acc)
│  │  └─ MEMIT (több edit egyszerre)
│  │
│  ├─ "Tiszta fogalmakat szerkeszteni"
│  │  └─ SAE → feature edit (monoszemantikus feature-ökön)
│  │
│  └─ "Chat formátumot tanítani"
│     └─ SFT (veszélyes 355M-en, katasztrofális felejtés)
│
└─ MI a modell mérete?
   ├─ >7B paraméter → ROME/MEMIT/FiNE a legjobb
   └─ <1B paraméter → LogicAdapter a legbiztonságosabb
```

## 25.2 Teljes összehasonlító táblázat

| # | Módszer | Fejezet | Pontosság | Idő | GPU | Batch | Biztonság | Komplexitás |
|---|---------|---------|-----------|-----|-----|-------|-----------|-------------|
| 1 | **Logit Bias** | Ch11 | Közepes (loop) | 30mp | ❌ | 1 | ✅ | ✅ Nagyon egyszerű |
| 2 | **FFN Neuron Edit** | Ch9 | Alacsony (elny) | 5p | ❌ | 1 | ❌ Veszélyes | ⚠️ Közepes |
| 3 | **Act. Steering** | Ch10,24 | Közepes | 1-5p | ❌ | 1 | ✅ Ideiglenes | ⚠️ Közepes |
| 4 | **LogicAdapter** | Ch12 | 55% (355M) | 7p | ✅ | 50-200 | ✅ Zero init | ⚠️ Közepes |
| 5 | **SFT** | Ch13 | Változó | 1ó+ | ✅ | 50-10K | ❌ Cat. felejtés | 🔴 Nehéz |
| 6 | **ROME** | Ch20 | 98% (7B+) | 1p | ❌ | 1 | ⚠️ C mátrix | ⚠️ Közepes |
| 7 | **MEMIT** | Ch21 | 97% (7B+) | 5p | ❌ | 10K | ⚠️ C mátrix | 🔴 Nehéz |
| 8 | **FiNE** | Ch22 | 98% (7B+) | 10p | ✅ | 1-50 | ✅ Legjobb | 🔴 Nehéz |
| 9 | **SAE edit** | Ch23 | Kísérleti | 30p | ✅ | Feature | ✅ Precíz | 🔴🔴 Nagyon nehéz |
| 10 | **Act. Patching** | Ch19 | Diagnosztika | 1p | ❌ | — | ✅ Nem módosít | ⚠️ Közepes |
| 11 | **MapMaker** | Ch8 | Diagnosztika | 5p | ❌ | — | ✅ Nem módosít | ⚠️ Közepes |

## 25.3 Javasolt munkafolyamat

### Ha újat kezdesz egy modellel:

```
1. MapMaker (Ch8)
   → Mely rétegek aktívak? Mely neuronok mely fogalmakra?
   
2. Activation Patching (Ch19)
   → Hol van a kauzális hiba? Mely réteg felelős?
   
3. (Opcionális) SAE (Ch23)
   → A fontos réteg aktivációinak szétfejtése tiszta feature-ökre
   
4. Beavatkozás választása:
   ┌─ Egy edit? → ROME (Ch20) vagy FiNE (Ch22)
   ├─ Több edit? → MEMIT (Ch21) vagy LogicAdapter (Ch12)
   └─ Csak teszt? → Steering (Ch24) vagy Logit Bias (Ch11)
   
5. Ellenőrzés:
   → Edit Success: a hiba javult?
   → Locality: más tudás sértetlen?
   → Fluency: a generálás természetes?
```

### NEURA 300M-re optimalizálva:

| Lépés | Mit csinálj | Idő |
|-------|-------------|-----|
| **1.** | MapMaker L22-L23-on (48 prompt) → találd meg a szám-neuronokat | 5p |
| **2.** | Activation Patching (5→2 kapcsolat) → igazold a 0.00 attention-t | 1p |
| **3.** | LogicAdapter training (108 példa, 120 epoch) → ~55% accuracy | 7p GPU |
| **4.** | (Ha van GPU) SAE training L23-ra → feature edit kipróbálása | 30p GPU |
| **5.** | Steering + generálás teszt | 5p |

## 25.4 Mit NE csinálj 355M-es modellen?

| ❌ Ne csináld | Miért? |
|-------------|--------|
| **SFT 100+ példán** | Katasztrofális felejtés — a modell elfelejti a nyelvtant |
| **ROME/MEMIT/FiNE 7B nélkül** | Ezeket 7B+ modellekre optimalizálták. 355M-en a C mátrix túl kicsi |
| **Többlépéses reasoning** | 5→2 attention = 0.00 marad — architekturális korlát |
| **SAE >32K feature** | 16K feature is elég L23-ra; nagyobb csak zajt ad |
| **Steering >5.0 strength** | Normalizált vektorral is összeomlik 5.0 felett |

## 25.5 Mit ÉRDEMES csinálni 355M-en?

| ✅ Csináld | Miért? |
|----------|--------|
| **LogicAdapter** | Zero init = biztonságos, 55% accuracy elérhető |
| **MapMaker + Patching** | Diagnosztika — megérted a modellt anélkül, hogy elrontanád |
| **Steering (normalizált)** | Gyors kísérletezéshez tökéletes |
| **SAE training** | Feature szintű megértés, kutatási értékű |
| **Logit Bias** | 30 másodperces teszteléshez |

## 25.6 Jövőbeli irányok

1. **LogicAdapter → MultiAdapter** — több adapter sorba kötve, minden adapter más képességhez
2. **SAE + LogicAdapter kombináció** — SAE feature-ökön tanított adapter
3. **Attention módosítás** — az 5→2 attention 0.00 javítása (ez az igazi megoldás!)
4. **Cross-coders** — SAE továbbfejlesztése, ami több réteget fed le egyszerre
5. **Reprezentáció Engineering a teljes modellen** — nem csak egy rétegen, hanem több réteg összehangolt módosítása

---

**Végszó:** A neuron módosítás nem varázslat — matematika. Minél jobban megérted a modell belső működését (MapMaker + Patching), annál pontosabban tudsz beavatkozni (LogicAdapter + FiNE). A 355M modell korlátos, de a technikák, amiket megtanulsz, nagyobb modelleken is működnek.
