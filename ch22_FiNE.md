# 22. fejezet: FiNE — Fine-grained Neuron-level Knowledge Editing

**ICLR 2025 Poster — Pan et al.**

## 22.1 A probléma

A ROME és MEMIT **modul szinten** lokalizálja a tudást (pl. "L5 FFN kell"). De ez nem elég precíz!

**Példa:** Szerkeszteni akarjuk: "(Párizs, fővárosa, London)"

- ROME: az L5 FFN-t módosítja → a "Párizs" subject-re hat, de a **relációt** (fővárosa) NEM veszi figyelembe
- Eredmény: "London országa = ?" kérdésre is rossz választ ad (Párizs → London, de London → Anglia helyett valami mást)

**A FiNE megoldása:** Neuron-szintű lokalizáció + reláció-érzékeny szerkesztés.

## 22.2 Hogyan működik?

```
1. LOKALIZÁCIÓ (causal tracing neuron szinten)
   - Find: mely neuronok aktiválódnak erősen a (subject, relation) párra?
   - Nem elég: "mely neuronok tüzelnek 'Párizs'-ra"
   - Kell: "mely neuronok tüzelnek 'Párizs fővárosa'-ra DE NEM 'Párizs nevezetessége'-re"

2. SZERKESZTÉS (finomhangolás, nem rank-one)
   - Csak a lokalizált neuronokat módosítja (10-50 db a 3072-ből)
   - Multiple loss:
     a) KL divergence — a kimenet ne térjen el túlságosan
     b) Repetíció penalty — ne loop-oljon
     c) Norm loss — a súlyváltozás legyen kicsi

3. ELLENŐRZÉS
   - Edit Success: a javított tény helyes?
   - Portability: a kapcsolódó kérdések is helyesek?
   - Locality: más témájú kérdések változatlanok?
   - Fluency: a generált szöveg természetes?
```

## 22.3 Implementáció NEURA 300M-re

```python
def fine_neuron_localization(model, subject_ids, relation_prompt, unrelated_prompt,
                              layer_range=[18, 23], top_k=30):
    """
    FiNE lokalizáció: megtalálja a (subject, relation) specifikus neuronokat.
    
    Args:
        subject_ids: a subject token ID-i (pl. "Párizs")
        relation_prompt: a teljes prompt a relációval (pl. "Párizs fővárosa")
        unrelated_prompt: ugyanaz a subject, de más reláció (pl. "Párizs nevezetessége")
        top_k: hány neuront lokalizáljunk
    """
    neuron_scores = {}
    
    for layer_idx in layer_range:
        ffn = model.blocks[layer_idx].ffn
        
        # 1. Forward a relációs prompt-pal
        ids = sp.EncodeAsIds(relation_prompt)
        x = torch.tensor([ids])
        
        # Capture FFN aktivációk
        ffn_acts = {}
        def make_hook():
            def hook(m, i, o):
                x_in = i[0]
                gate = torch.nn.functional.silu(m.w1(x_in))
                up = m.w3(x_in)
                ffn_acts['gate'] = gate.detach().float().cpu()
                ffn_acts['hidden'] = (gate * up).detach().float().cpu()
            return hook
        
        handle = ffn.register_forward_hook(make_hook())
        with torch.no_grad():
            logits_rel = model(x)
        handle.remove()
        
        # Aktiváció a subject pozíción
        # Tegyük fel hogy subject_pos = token pozíció
        rel_activations = ffn_acts['hidden'][0, -1]  # [3072]
        
        # 2. Forward a nem-relációs prompt-pal
        ids_unrelated = sp.EncodeAsIds(unrelated_prompt)
        x_unrelated = torch.tensor([ids_unrelated])
        
        handle = ffn.register_forward_hook(make_hook())
        with torch.no_grad():
            logits_unrelated = model(x_unrelated)
        handle.remove()
        
        unrel_activations = ffn_acts['hidden'][0, -1]  # [3072]
        
        # 3. Reláció-specifikus neuronok
        # Azok a neuronok, amik a relációs promptban erősebben tüzelnek
        diff = rel_activations - unrel_activations
        
        for ni in range(len(diff)):
            score = diff[ni].item()
            if score > 0:  # A relációs prompt erősebb
                neuron_scores[(layer_idx, ni)] = score
    
    # Top-K neuron rendezése
    sorted_neurons = sorted(
        neuron_scores.items(),
        key=lambda x: -x[1]
    )[:top_k]
    
    print(f"Top-{top_k} reláció-specifikus neuron:")
    for (layer, neuron), score in sorted_neurons:
        print(f"  L{layer+1} #{neuron:5d} — diff={score:.2f}")
    
    return sorted_neurons


# ====== HASZNÁLAT NEURA 300M-RE ======

# CÉL: A "marad" token megerősítése kivonásos feladatokban

# Lokalizáció
neurons = fine_neuron_localization(
    model,
    subject_ids=sp.EncodeAsIds("5"),
    relation_prompt="5 almám van, megeszek 2-t",
    unrelated_prompt="5 szép alma van a kosárban",  # más kontextus
    layer_range=[20, 21, 22, 23],
    top_k=20
)


def fine_edit(model, neurons_to_edit, edit_prompts, target_ids,
              epochs=50, lr=1e-4):
    """
    FiNE szerkesztés: a lokalizált neuronok finomhangolása.
    
    Args:
        neurons_to_edit: [(layer_idx, neuron_idx), ...]
        edit_prompts: list of prompts
        target_ids: list of target token IDs
    """
    # Csak a kijelölt neuronokat tanítjuk
    # A módszer: minden kijelölt neuron w2[:, neuron_idx] súlyát finomhangoljuk
    
    # Paraméterek kijelölése
    params_to_train = []
    for layer_idx, neuron_idx in neurons_to_edit:
        w2 = model.blocks[layer_idx].ffn.w2.weight
        # Csak ennek a neuronnak a kimeneti súlyait tanítjuk
        param = w2[:, neuron_idx]
        param.requires_grad = True
        params_to_train.append(param)
    
    optimizer = torch.optim.AdamW(params_to_train, lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for prompt, target_id in zip(edit_prompts, target_ids):
            ids = sp.EncodeAsIds(prompt)
            x = torch.tensor([ids])
            
            logits = model(x)
            last_logits = logits[0, -1]
            
            # A FiNE 3 loss komponense:
            
            # 1. Edit loss: a target token valószínűségének növelése
            loss_edit = -torch.log_softmax(last_logits, dim=-1)[target_id]
            
            # 2. KL divergence: a kimenet ne változzon túl sokat
            with torch.no_grad():
                logits_orig = model(x)
            loss_kl = torch.nn.functional.kl_div(
                torch.log_softmax(last_logits, dim=-1),
                torch.softmax(logits_orig[0, -1], dim=-1),
                reduction='sum'
            )
            
            # 3. Norm loss: a súlyváltozás legyen kicsi
            loss_norm = sum(p.norm() for p in params_to_train)
            
            # Kombinált loss (súlyok: α=1.0, β=0.1, γ=0.01)
            loss = loss_edit + 0.1 * loss_kl + 0.01 * loss_norm
            total_loss += loss
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss.item() / len(edit_prompts):.4f}")
    
    # Visszaállítás: a tanított paraméterek zárolása
    for p in params_to_train:
        p.requires_grad = False


# ====== HASZNÁLAT ======

# A lokalizált neuronok szerkesztése
edit_prompts = ["5 - 2 =", "8 - 5 =", "3 - 1 =", "6 - 1 ="]
target_ids = [sp.PieceToId("▁3"), sp.PieceToId("▁3"), 
              sp.PieceToId("▁2"), sp.PieceToId("▁5")]

fine_edit(model, neurons, edit_prompts, target_ids, epochs=50)
```

## 22.4 A FiNE eredményei (eredeti papír alapján)

GPT-J (6B) és LLaMA-2/3 (7-8B) modelleken tesztelve:

| Metrika | ROME | MEMIT | **FiNE** |
|---------|------|-------|----------|
| **Edit Success** | 98.2% | 97.8% | **98.5%** |
| **Portability** | 52.1% | 54.3% | **56.8%** |
| **Locality** | 96.4% | 97.1% | **98.9%** |
| **Fluency** | 88.2% | 89.5% | **91.3%** |

**A legnagyobb előny: Locality** — a FiNE 2-3%-kal jobban megőrzi a nem-érintett tudást, mert csak 10-50 neuront módosít a 3072-ből.

## 22.5 Adaptáció NEURA 300M-re

A FiNE-t kifejezetten nagy modellekre (7B+) tervezték. A mi 355M modellünkre adaptálva:

| Eredeti FiNE | NEURA adaptáció | Miért? |
|-------------|----------------|--------|
| Causal tracing minden rétegen | Csak L20-L23 | Az alsó rétegek nem tárolnak relációs tudást |
| 30-50 neuron edit | 10-20 neuron | 355M-nek kevesebb a redundáns neuronja |
| KL + Repetíció + Norm | **Edit + KL + Norm** | Repetíció penalty nem kell (nincs hosszú generálás) |
| 100 epoch | 30-50 epoch | Kisebb modell = gyorsabb konvergencia |
| Batch edit (MEMIT módban) | Single edit | 355M nem bír el több editet egyszerre |

**Következtetés:** A FiNE a legerősebb knowledge editing módszer, de nagy modellre optimalizálták. NEURA 300M-en a LogicAdapter (Ch12) egyszerűbb és biztonságosabb. A FiNE lokalizációs részét viszont ÉRDEMES használni — a reláció-specifikus neuronok azonosítására tökéletes.
