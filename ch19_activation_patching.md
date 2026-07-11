# 19. fejezet: Aktivációs Patching és Causal Tracing

## Miért van szükségünk erre?

A 8. fejezetben (MapMaker) azt néztük, mely neuronok **korrelálnak** bizonyos fogalmakkal. De a korreláció nem egyenlő a kauzalitással!

Egy neuron erősen tüzelhet az "alma" szóra, de ha **kivesszük** (abláció), lehet hogy semmi sem változik — mert más neuronok átveszik a szerepét.

Az **aktivációs patching** segít megválaszolni: "Ha ezt a neuront / réteget / fejet kicserélem, megváltozik a kimenet?" — ez a kauzális bizonyíték.

## 19.1 Alapkoncepció

```
1. Clean run:  normál bemenet → normál kimenet
               cache-elve minden réteg aktivációja

2. Corrupted run: zajos/rossz bemenet → rossz kimenet

3. PATCH: corrupted run L5 aktivációját
          KICSERÉLJÜK a clean run L5 aktivációjára

4. Ha a kimenet visszaváltozik HELYESRE →
   L5 kauzállsan felelős ezért a viselkedésért
```

## 19.2 Implementáció NEURA 300M-re

```python
def activation_patch(model, clean_ids, corrupted_ids, patch_layer):
    """
    Aktivációs patching egy adott rétegen
    
    Ha a corrupted run patch-elt kimenete megegyezik a clean run-éval,
    akkor a patch_layer kauzállsan fontos a viselkedéshez.
    """
    model.eval()
    
    # 1. Clean forward — cache activations
    clean_acts = {}
    handles = []
    
    def make_hook(name):
        def hook(m, i, o):
            clean_acts[name] = o.detach().cpu()
        return hook
    
    for li, block in enumerate(model.blocks):
        h = block.register_forward_hook(make_hook(f'block_{li}'))
        handles.append(h)
    
    with torch.no_grad():
        x_clean = torch.tensor([clean_ids])
        logits_clean = model(x_clean)
        clean_pred = logits_clean[0, -1].argmax().item()
    
    for h in handles:
        h.remove()
    
    # 2. Corrupted forward with patch
    x_corrupted = torch.tensor([corrupted_ids])
    
    def patched_forward(x):
        h = model.tok(x)
        for li, block in enumerate(model.blocks):
            h = block(h)
            if li == patch_layer:
                # PATCH: corrupted activation REPLACED with clean
                h[0, -1] = clean_acts[f'block_{li}'][0, -1].to(h.device)
        return model.out(model.ln_f(h))
    
    with torch.no_grad():
        logits_patched = patched_forward(x_corrupted)
        patched_pred = logits_patched[0, -1].argmax().item()
    
    return clean_pred, patched_pred


# Használat: Miért mond "kap"-ot ahelyett hogy "marad"?
# Clean: "Ha 5 almám van és megeszek 2-t, marad"
# Corrupted: "Ha 5 almám van és megeszek 2-t, ?????" (csonka prompt)
#
# Várható eredmény:
#   L1-L8:  corrupted → marad (NEM változik)  → nem itt a hiba
#   L17-L23: corrupted → kap (ROSSZ!)         → ITT a hiba!

clean_ids = sp.EncodeAsIds("Ha 5 almám van és megeszek 2-t, marad 3")
# corrupted: cseréljük a "3"-at egy random számra
corrupted_ids = sp.EncodeAsIds("Ha 5 almám van és megeszek 2-t, az")

for li in range(24):
    clean_pred, patched_pred = activation_patch(model, clean_ids, corrupted_ids, li)
    clean_token = sp.IdToPiece(clean_pred)
    patched_token = sp.IdToPiece(patched_pred)
    match = "✅" if clean_pred == patched_pred else "❌"
    print(f"L{li:2d} patch: clean={clean_token:10s} → patched={patched_token:10s} {match}")
```

## 19.3 Patching típusai

| Típus | Mit cserélünk | Mit tudunk meg |
|-------|--------------|----------------|
| **Layer szintű** | Egy teljes réteg kimenetét | Melyik réteg felelős a viselkedésért |
| **Neuron szintű** | Egyetlen neuron értékét | Melyik neuron a kauzális |
| **Fej szintű** | Egy attention fej kimenetét | Melyik fej hordozza az információt |
| **Token szintű** | Csak az utolsó token aktivációját | Mennyire token-specifikus |
| **Komponens szintű** | Csak az FFN vagy csak az Attention részt | FFN vs Attention szerepe |

## 19.4 Hogyan csináljunk corrupted bemenetet?

A corrupted bemenet LÉNYEGES — rosszul megválasztva hamis eredményt ad.

| Módszer | Leírás | Előny/Hátrány |
|---------|--------|--------------|
| **Token dropout** | Egy token kitörlése a promptból | ✅ Egyszerű, ❌ durva |
| **Resample ablation** | Aktiváció kicserélése egy MÁSIK promptból | ✅ Precíz, ❌ több forward kell |
| **Gaussian noise** | Zaj hozzáadása az embeddinghez | ✅ Folytonos, ❌ nem specifikus |
| **Szócsere** | A kulcsszó kicserélése (pl. "5" → "x") | ✅ Specifikus, ❌ szubjektív |

**Ajánlás:** Kezdd a **resample ablation**-lel: futtasd a modellt 10 random prompton, mentsd el az aktivációkat, majd ezekből "keverj" corrupted bemenetet.

## 19.5 Attribúciós Patching (gyorsabb változat)

A teljes patching minden rétegre / neuronra külön forward-ot igényel — ez 24+ forward egy elemzéshez. Az **attribúciós patching** ezt gyorsítja:

```python
def attribution_patching(model, clean_ids, corrupted_ids):
    """
    Attribúciós patching: ONE corrupted forward, az összes réteg egyszerre.
    
    Használ: gradienseket a corrupted kimenet és a clean kimenet között.
    """
    x_clean = torch.tensor([clean_ids])
    x_corrupted = torch.tensor([corrupted_ids])
    
    # Clean forward
    with torch.no_grad():
        clean_logits = model(x_clean)
        clean_pred_idx = clean_logits[0, -1].argmax()
    
    # Corrupted forward (grad-ekkel!)
    model.zero_grad()
    corrupted_logits = model(x_corrupted)
    
    # Loss = távolság a clean predikciótól
    loss = -corrupted_logits[0, -1, clean_pred_idx]  # negatív = minél közelebb, annál jobb
    loss.backward()
    
    # Minden réteg hozzájárulása
    contributions = {}
    for li, block in enumerate(model.blocks):
        # Az FFN w2 grad norm-ja = mennyire fontos ez a réteg
        contrib = block.ffn.w2.weight.grad.norm().item()
        contributions[li] = contrib
    
    return contributions
```

## 19.6 Várható eredmények NEURA 300M-en

A "5 almám" → "megeszek 2-t" → "marad 3" feladatban:

| Réteg | Hatás | Jelentés |
|-------|-------|----------|
| L1-L4 | 0.0-0.1 | Még nem itt a számolás |
| L5-L12 | 0.1-0.3 | Szintaxis építés |
| L13-L16 | 0.3-0.5 | Kontextus összerakása |
| **L17-L21** | **0.5-0.8** | **A számok kapcsolatát itt kellene feldolgozni** |
| L22-L23 | 0.2-0.4 | Döntés előkészítés |
| L24 | 0.0 | Csak kivetítés |

**FONTOS:** Ha a patching MINDEN rétegben 0.00 hatást mutat (a számok közötti kapcsolatra), az bizonyítja, hogy a modellben NINCS olyan mechanizmus ami a számokat összekapcsolná — architekturális korlát, nem tuning kérdése.

## 19.7 Causal Tracing — a teljes kép

A causal tracing a patching kiterjesztése, ahol **több corrupted pontot** használunk:

```
Time (token pozíció) →
      tok_0  tok_1  tok_2  ...  tok_n
L1    [ ]    [ ]    [ ]         [ ]    ← patch-eltük?
L2    [ ]    [ ]    [ ]         [ ]
...   [ ]    [ ]    [ ]         [ ]
L24   [ ]    [ ]    [ ]         [ ]
```

Minden (réteg, token) párra kipróbáljuk a patchinget, és megnézzük, hol változik a kimenet. Az eredmény egy **hőtérkép** ami megmutatja a pontos információáramlási útvonalat.

---

## 19.8 Kulcsfontosságú tanulságok

1. **A patching MEGMUTATJA a kauzális kapcsolatot**, nem csak korrelációt
2. A MapMaker + Patching kombináció a legerősebb elemzési eszköz
3. Ha a patching nulla hatást mutat egy rétegen → az a réteg NEM vesz részt a feladatban
4. A corrupted bemenet megválasztása KRITIKUS — rossz corrupted = hamis eredmény

**Ne feledd:** A MapMaker megmondja "hol nézzünk", a Patching megmondja "hol kell javítani". A kettő együtt adja a teljes képet.
