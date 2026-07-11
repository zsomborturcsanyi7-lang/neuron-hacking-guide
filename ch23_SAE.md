# 23. fejezet: SAE — Sparse Autoencoders a Neuronok Megértésére

**Anthropic (2023-2024), OpenAI (2024), egyre növekvő kutatás**

## 23.1 A probléma: Polysemanticity

A MapMaker (8. fejezet) felfedezte: egy neuron több fogalomra is reagál. Pl.:

```
Neuron #1338 L23: tüzel "iskola"-ra (12.3), DE tüzel "tanulás"-ra is (8.1)
```

Ezt hívják **polysemanticity**-nek — egy neuron több jelentést hordoz. Ez azért van, mert a modell **superposition**-t használ: több fogalmat pakol ugyanabba a dimenzióba, hogy spóroljon.

## 23.2 Az SAE megoldása

A **Sparse Autoencoder** egy kis háló, amit a modell rétegére illesztünk. Kibontja a polysemantic neuronokat **monosemantic feature**-ökké:

```
Bemenet: L23 hidden state [1024]
              │
         ┌────▼────┐
         │ Encoder │  (1024 → 16384)
         └────┬────┘
              │
         ┌────▼────┐
         │  ReLU   │  (csak pozitív aktivációk)
         └────┬────┘
              │
         ┌────▼────┐
         │ TopK    │  (csak a legerősebb ~100 feature)
         └────┬────┘
              │
         ┌────▼────┐
         │ Decoder │  (16384 → 1024)
         └────┬────┘
              │
     Kimenet: rekonstruált L23 [1024]
     
    A 16384 DIMENZIÓS KÖZTES RÉTEG:
    Minden dimenzió = egyetlen tiszta fogalom!
    - Feature #42 = "piros szín"
    - Feature #315 = "almával kapcsolatos"
    - Feature #8041 = "számok"
```

## 23.3 SAE implementáció

```python
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse Autoencoder egy adott réteg aktivációinak szétfejtésére.
    
    Az encoder kivetíti a 1024D hidden state-t egy 16384D ritka térbe.
    A decoder visszaállítja az eredeti 1024D-t.
    
    A köztes réteg minden dimenziója egy monoszemantikus feature.
    """
    def __init__(self, d_model=1024, d_sae=16384, top_k=100):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.top_k = top_k
        
        # Encoder: 1024 → 16384
        self.encoder = torch.nn.Linear(d_model, d_sae, bias=False)
        
        # Decoder: 16384 → 1024
        self.decoder = torch.nn.Linear(d_sae, d_model, bias=False)
        
        # Bias-ok
        self.enc_bias = torch.nn.Parameter(torch.zeros(d_sae))
        self.dec_bias = torch.nn.Parameter(torch.zeros(d_model))
        
        # Inicializáció: decoder unit norm
        self.decoder.weight.data = self.decoder.weight.data / \
            self.decoder.weight.data.norm(dim=1, keepdim=True)
    
    def forward(self, x):
        # Encoder
        latent = torch.relu(self.encoder(x - self.dec_bias) + self.enc_bias)
        
        # TopK sparsity: csak a legerősebb top_k feature tart meg
        top_k_values, top_k_indices = torch.topk(latent, k=self.top_k, dim=-1)
        sparse_latent = torch.zeros_like(latent)
        sparse_latent.scatter_(-1, top_k_indices, top_k_values)
        
        # Decoder
        reconstructed = self.decoder(sparse_latent) + self.dec_bias
        
        return reconstructed, sparse_latent
    
    @torch.no_grad()
    def get_features(self, x):
        """Visszaadja a ritka feature aktivációkat elemzéshez."""
        latent = torch.relu(self.encoder(x - self.dec_bias) + self.enc_bias)
        top_k_values, top_k_indices = torch.topk(latent, k=self.top_k, dim=-1)
        return top_k_indices, top_k_values


# ====== SAE Training NEURA L23-RA ======

def train_sae(model, explorer, layer_idx=22, num_steps=10000, 
              batch_size=64, d_sae=16384, top_k=100):
    """
    SAE tréning az L23-as réteg aktivációira.
    
    Gyűjtés: sok prompton keresztül L23 hidden state-jei.
    Training: rekonstrukciós loss + sparsity loss.
    """
    sae = SparseAutoencoder(d_model=1024, d_sae=d_sae, top_k=top_k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
    
    # Adatgyűjtés: 1000 prompt L23 aktivációi
    activations = []
    prompts = [
        "Az alma piros és",
        "A kutya a kertben fut",
        "Budapest szép város",
        "Ma esik az eső",
        # ... 1000 prompt ...
    ]
    
    for prompt in prompts:
        ids = sp.EncodeAsIds(prompt)
        x = torch.tensor([ids])
        logits = explorer.forward(x)
        h = explorer.activations[f'block_{layer_idx}']  # L23 hidden
        activations.append(h[0, -1])  # utolsó token
    
    acts = torch.stack(activations)  # [1000, 1024]
    
    # Training loop
    for step in range(num_steps):
        batch_idx = torch.randint(0, len(acts), (batch_size,))
        batch = acts[batch_idx]
        
        reconstructed, sparse_latent = sae(batch)
        
        # Rekonstrukciós loss (MSE)
        loss_recon = torch.nn.functional.mse_loss(reconstructed, batch)
        
        # Sparsity loss (L1 a feature aktivációkra)
        loss_sparsity = sparse_latent.abs().mean()
        
        # Kombinált loss
        loss = loss_recon + 0.001 * loss_sparsity
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step}: recon_loss={loss_recon.item():.4f}, "
                  f"sparsity={loss_sparsity.item():.4f}")
    
    return sae


# ====== FEATURE ANALÍZIS ======

def analyze_features(sae, explorer, prompts_dict, layer_idx=22):
    """
    SAE feature-ök elemzése: melyik feature melyik fogalomhoz tartozik?
    Ugyanaz, mint a MapMaker, de SAE feature-ökön!
    """
    feature_activations = {}  # {feature_idx: [(category, strength), ...]}
    
    for category, prompts in prompts_dict.items():
        for prompt in prompts:
            ids = sp.EncodeAsIds(prompt)
            x = torch.tensor([ids])
            logits = explorer.forward(x)
            
            # L23 hidden state
            h = explorer.activations[f'block_{layer_idx}']
            h_last = h[0, -1]  # [1024]
            
            # SAE feature-ök
            with torch.no_grad():
                feature_indices, feature_values = sae.get_features(h_last)
            
            for fi, fv in zip(feature_indices[0].tolist(), 
                              feature_values[0].tolist()):
                if fi not in feature_activations:
                    feature_activations[fi] = []
                feature_activations[fi].append((category, fv))
    
    # Feature→kategória mapping
    print(f"=== L23 SAE Feature Analysis ({len(feature_activations)} active features) ===\n")
    
    cat_counts = {}
    for fi, activations in feature_activations.items():
        cat_strength = {}
        for cat, val in activations:
            cat_strength[cat] = cat_strength.get(cat, 0) + val
        
        best_cat = max(cat_strength, key=cat_strength.get)
        if best_cat not in cat_counts:
            cat_counts[best_cat] = 0
        cat_counts[best_cat] += 1
    
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<12} {count:>4} features")
    
    return feature_activations
```

## 23.4 Várható eredmények

| Metrika | Nyers L23 neuronok | SAE feature-ök (16K) |
|---------|-------------------|---------------------|
| **Aktív egységek/prompt** | 1,377 neuron (44.8%) | ~100 feature (0.6%) |
| **Monosemanticitás** | Alacsony (több jelentés) | **Magas** (1 feature = 1 fogalom) |
| **Szerkeszthetőség** | Nehéz (mást is érint) | **Könnyű** (célzott) |
| **Rekonstrukciós hiba** | — | ~5-10% (elfogadható) |

**Miért jobb a feature-ökön szerkeszteni?**

Ha az SAE Feature #8041 = "számok" fogalmat képviseli, akkor ennek a feature-nek a megerősítése vagy gyengítése pontosan a számokkal kapcsolatos viselkedést változtatja meg — nem érinti az "alma" vagy a "kutya" fogalmakat.

```python
# SAE feature szerkesztés: Feature #8041 (számok) erősítése
def steer_sae_feature(sae, feature_idx, strength=2.0):
    """
    Egy SAE feature erősítése a decoder súlyain keresztül.
    """
    with torch.no_grad():
        # A feature decoder súlya megadja, hogy ez a feature
        # milyen irányba tolja a hidden state-t
        feature_direction = sae.decoder.weight[feature_idx]  # [1024]
    
    # Ha ezt az irányt erősítjük a hidden state-ben,
    # a modell több "számos" kimenetet fog produkálni
    return feature_direction * strength


def generate_with_sae_steer(model, sae, prompt, steer_feature_idx,
                             steer_strength=2.0, layer=22, max_tokens=20):
    """
    Generálás SAE feature steeringgel.
    
    1. Forward pass a modellen
    2. A target rétegnél: SAE → feature módosítás → decoder
    3. Folytatás a módosított hidden state-tel
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
                if li == layer:
                    # SAE feature steering az utolsó tokenre
                    h_last = h[0, -1]  # [1024]
                    
                    # SAE-n keresztül: extract + modify + reconstruct
                    reconstructed, features = sae(h_last.unsqueeze(0))
                    
                    # A cél feature erősítése
                    feature_direction = sae.decoder.weight[steer_feature_idx]
                    h[0, -1] += feature_direction * steer_strength
            
            logits = model.out(model.ln_f(h))
            next_id = logits[0, -1].argmax().item()
            
            if next_id in [0, 2]:
                break
            
            out_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
    
    return sp.DecodeIds(out_ids)
```

## 23.5 SAE vs MapMaker

| MapMaker (Ch8) | SAE (ez a fejezet) |
|----------------|-------------------|
| Neuron szintű elemzés | **Feature szintű** elemzés |
| ~73K neuron a 24 rétegben | 16K+ feature egyetlen rétegben |
| Polysemantic (több jelentés) | **Minden feature monoszemantikus** |
| Korrelációs | Korrelációs (de tisztább) |
| Gyors (48 prompt) | Lassú (training kell) |

**A SAE a MapMaker továbbfejlesztése.** Előbb MapMaker → megtaláljuk a fontos rétegeket. Aztán SAE → kifejtjük a tiszta feature-öket. Végül feature edit → precíz beavatkozás.

## 23.6 Gyakorlati tanácsok

1. **Kezdj 1 réteggel** — L23 a legjobb (legaktívabb, legtöbb specializáció)
2. **d_sae = 16× d_model** — 1024 dim → 16384 feature (Anthropic ajánlás)
3. **TopK = 100** — csak 100 aktív feature promptonként, a többi 0
4. **Training idő:** 10K lépés CPU-n ~30 perc, GPU-n ~2 perc
5. **Adat:** 1000+ különböző prompt — minél változatosabb, annál jobb

**Limitációk:**
- Az SAE nem tökéletes — a rekonstrukciós hiba ~5-10%
- Egy feature még mindig lehet polysemantic (de sokkal ritkábban)
- A feature-ök száma (16K) nem elég minden fogalomhoz — 64K+ jobb lenne
