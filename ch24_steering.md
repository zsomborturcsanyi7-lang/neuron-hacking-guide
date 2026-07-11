# 24. fejezet: Fejlett Aktivációs Steering és Reprezentáció Engineering

## 24.1 A probléma a korábbi steeringgel

A 10. fejezetben próbáltuk a steeringet — de minden 30+ strength-nél összeomlott. A probléma: **egyetlen promptból** számoltuk a steering irányt.

A **Reprezentáció Engineering** (Zou et al., 2023) ezt javítja: **több prompt átlagából** számoljuk a steering irányt, ami sokkal stabilabb.

## 24.2 A helyes módszer

```python
def compute_steering_vector(model, explorer, positive_prompts, negative_prompts,
                             layer_idx=22):
    """
    Steerig vektor számolása TÖBB prompt átlagából.
    
    positive_prompts: a kívánt viselkedést kiváltó promptok (pl. "őszinte" válaszok)
    negative_prompts: a nemkívánatos viselkedést kiváltó promptok (pl. "hazug" válaszok)
    
    A steering irány = mean(positive_activations) - mean(negative_activations)
    """
    pos_acts = []
    neg_acts = []
    
    for prompt in positive_prompts:
        ids = sp.EncodeAsIds(prompt)
        x = torch.tensor([ids])
        logits = explorer.forward(x)
        h = explorer.activations[f'block_{layer_idx}']
        pos_acts.append(h[0, -1])  # utolsó token
    
    for prompt in negative_prompts:
        ids = sp.EncodeAsIds(prompt)
        x = torch.tensor([ids])
        logits = explorer.forward(x)
        h = explorer.activations[f'block_{layer_idx}']
        neg_acts.append(h[0, -1])
    
    pos_mean = torch.stack(pos_acts).mean(dim=0)  # [1024]
    neg_mean = torch.stack(neg_acts).mean(dim=0)  # [1024]
    
    steer = pos_mean - neg_mean
    
    # Normalizáció (L2 norm = 1)
    steer = steer / steer.norm()
    
    print(f"Steering vector norm: {steer.norm().item():.2f}")
    print(f"Positive prompts: {len(positive_prompts)}")
    print(f"Negative prompts: {len(negative_prompts)}")
    
    return steer


# ====== ALKALMAZÁS NEURA 300M-RE ======
# CÉL: Steering a "marad" token felé kivonásos feladatokban

# Pozitív promptok (ahol a "marad" a helyes válasz)
positive_prompts = [
    "Ha 5 almám van és megeszek 2-t, marad",
    "8 almából megeszek 5-öt, marad",
    "3 almából megeszek 1-et, marad",
    "10 almából megeszek 3-at, marad",
    "6 almából megeszek 2-t, marad",
    "7 almából megeszek 4-et, marad",
    "9 almából megeszek 6-ot, marad",
    "4 almából megeszek 1-et, marad",
]

# Negatív promptok (ahol a "marad" NEM helyes)
negative_prompts = [
    "Az alma piros és édes, marad",
    "A kutya a kertben fut, marad",
    "Ma szép idő van, marad",
    "Budapest szép város, marad",
    "A gyerekek játszanak, marad",
    "A nap süt az égen, marad",
    "A macska alszik a kanapén, marad",
    "Esik az eső, marad",
]

explorer = NeuraExplorer(model)
steer_vector = compute_steering_vector(
    model, explorer, positive_prompts, negative_prompts, layer_idx=22
)
explorer.cleanup()
```

## 24.3 Generálás steeringgel

```python
def generate_steered(model, prompt, steer_vector, steer_layer=22,
                      strength=1.0, max_tokens=50, explorer=None):
    """
    Generálás normalizált steering vektorral.
    
    A strength érték MOST MÁR értelmes:
    - 0.5-2.0: finom irányítás
    - 2.0-5.0: erős irányítás
    - 5.0+: extrém (összeomolhat)
    
    (Korábban 30+ kellett — most 1.0 is elég!)
    """
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids])
    out_ids = ids.copy()
    
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            h = model.tok(x)
            for li, block in enumerate(model.blocks):
                h = block(h)
                if li == steer_layer:
                    # Csak az utolsó tokenre!
                    h[0, -1] = h[0, -1] + steer_vector.to(h.device) * strength
            
            logits = model.out(model.ln_f(h))
            next_id = logits[0, -1].argmax().item()
            
            if next_id in [0, 2]:
                break
            
            out_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
    
    return sp.DecodeIds(out_ids)


# ====== TESZT ======

# Különböző strength értékek
prompt = "Ha 5 almám van és megeszek 2-t"

for strength in [0.5, 1.0, 2.0, 5.0]:
    result = generate_steered(model, prompt, steer_vector, 
                               steer_layer=22, strength=strength)
    print(f"Strength={strength:3.1f}: {result}")
```

## 24.4 Várható eredmények (normalizált steering)

A normalizált vektorral:

| Strength | Régi módszer (1 prompt) | Új módszer (több prompt átlaga) |
|----------|------------------------|--------------------------------|
| **0.5** | Alig változik | Finom irányítás |
| **1.0** | Alig változik | Érezhető változás |
| **2.0** | Kis változás | **Erős, stabil irányítás** |
| **5.0** | Még mindig gyenge | Jelentős változás |
| **10.0** | Még mindig gyenge | Kockázatos, lehet loop |
| **30.0** | Breakdown | ❌ Összeomlás |

**Mi változott?** A normalizált vektor (norm=1) mindig ugyanakkora "léptékű", függetlenül a promptoktól. A régi módszernél a vektor norm-ja 5-50 között változott prompttól függően — ezért volt kaotikus.

## 24.5 Több réteges steering

Néha egy réteg nem elég — a modell több ponton is irányítható:

```python
def multi_layer_steering(model, prompt, steer_vectors, steer_layers,
                          strengths, max_tokens=50):
    """
    Több réteges steering: minden rétegen más-más erősséggel.
    """
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids])
    out_ids = ids.copy()
    
    model.eval()
    
    steer_dict = {li: (v, s) for li, v, s in 
                  zip(steer_layers, steer_vectors, strengths)}
    
    with torch.no_grad():
        for _ in range(max_tokens):
            h = model.tok(x)
            for li, block in enumerate(model.blocks):
                h = block(h)
                if li in steer_dict:
                    v, s = steer_dict[li]
                    h[0, -1] = h[0, -1] + v.to(h.device) * s
            
            logits = model.out(model.ln_f(h))
            next_id = logits[0, -1].argmax().item()
            
            if next_id in [0, 2]:
                break
            
            out_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
    
    return sp.DecodeIds(out_ids)
```

## 24.6 Mérés: mennyire hat a steering?

```python
def measure_steering_effect(model, explorer, prompt, steer_vector,
                             steer_layer=22, target_token="▁marad"):
    """
    Méri, mennyit változik a target token valószínűsége a steering hatására.
    """
    target_id = sp.PieceToId(target_token)
    results = {}
    
    for strength in [0, 0.5, 1.0, 2.0, 5.0]:
        ids = sp.EncodeAsIds(prompt)
        x = torch.tensor([ids])
        out_ids = ids.copy()
        
        model.eval()
        with torch.no_grad():
            h = model.tok(x)
            for li, block in enumerate(model.blocks):
                h = block(h)
                if li == steer_layer:
                    h[0, -1] = h[0, -1] + steer_vector.to(h.device) * strength
            
            logits = model.out(model.ln_f(h))
        
        probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
        p_target = probs[target_id].item() * 100
        results[strength] = p_target
    
    print(f"Target token: {target_token}")
    print(f"{'Strength':<10} {'P(target)':<12} {'Változás':<12}")
    print("-" * 34)
    baseline = results[0]
    for s, p in results.items():
        change = p / baseline if baseline > 0 else float('inf')
        print(f"{s:<10.1f} {p:<12.4f}% {change:<12.1f}x")
    
    return results
```

## 24.7 Reprezentáció Engineering vs Finomhangolás

| Szempont | Steering (24. fej.) | Finomhangolás (Ch12-13) |
|----------|---------------------|------------------------|
| **Idő** | 1-5 perc | 7-120 perc |
| **GPU** | Nem kell | GPU kell |
| **Véglegesség** | Ideiglenes (chat-en belül) | Végleges (modellbe ír) |
| **Biztonság** | Nem rontja el a modellt | Veszteséges lehet |
| **Batch** | 1 prompt | 100-10K példa |
| **Pontosság** | Kevésbé precíz | Nagyon precíz |

**Mikor használd a steeringet:**
- ✅ Gyors prototípuskészítéshez
- ✅ Amikor nincs GPU-d
- ✅ Kísérletezéshez (melyik irány működik?)
- ❌ Végleges megoldáshoz (→ LogicAdapter vagy FiNE)

## 24.8 Gyakorlati tippek

1. **Mindig normalizáld a steering vektort** (L2 norm = 1)
2. **Minél több prompt** → annál stabilabb a vektor (10+ pozitív, 10+ negatív)
3. **A promptok legyenek változatosak** — ne csak "5-2=", hanem "8-5=", "10-3=" is
4. **Kezdj strength=1.0-val** — ha nem elég, növeld óvatosan
5. **Több réteg → erősebb hatás** — de 3+ rétegnél már összeomolhat
6. **Ellenőrizd a nem-kívánt hatásokat** — a steering más viselkedést is megváltoztathat
