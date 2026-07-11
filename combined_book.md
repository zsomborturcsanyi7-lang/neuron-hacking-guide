# Neuronok Módosítása Kis Nyelvi Modellekben

## Teljes Útmutató Kezdőtől Haladóig

**Szerző:** Hermes Agent (Nous Research)
**Modell:** NEURA 300M (355M paraméter, 24 réteg, 1024 dim)
**Alapuló kísérletek:** 2026. június-július
**Licenc:** CC BY 4.0

---

# 1. rész: Alapok

## 1. fejezet: Bevezetés — Mi az a neuron módosítás?

### 1.1 Mit jelent "neuront módosítani"?

Egy nyelvi modellben a "neuronok" nem biológiai idegsejtek, hanem **matematikai egységek** a modell belső rétegeiben. Minden neuron egy számértéket (aktivációt) produkál a bemenet alapján. A neuron módosítás azt jelenti, hogy **közvetlenül megváltoztatjuk ezeket a matematikai egységeket**, hogy a modell másképp viselkedjen.

### 1.2 Miért érdemes neuront módosítani?

| Hagyományos módszer | Neuron módosítás |
|---------------------|------------------|
| Teljes modell fine-tuning (órák-napok, sok GPU) | Egy neuron átírása (másodpercek, CPU-n is megy) |
| Minden paraméter változik | Csak 1-2 neuron változik |
| Katasztrofális felejtés veszélye | Célzott, a többi tudás sértetlen |
| Nagy adatkészlet kell (~10K+ példa) | 1-100 példa is elég |

### 1.3 Mikor HASZNÁLJUNK neuron módosítást?

- **Egy specifikus hiba javítása** (pl. a modell mindig rossz szót használ)
- **Egy új képesség hozzáadása** (pl. számolás, amit nem tanult meg)
- **Egy nemkívánatos viselkedés elnyomása**
- **A modell "gondolkodásának" megértése** (interpretability research)

### 1.4 Mikor NE használjunk neuron módosítást?

- **Teljesen új nyelv tanítása** → fine-tuning kell
- **Általános minőség javítás** → több adat + több training kell
- **Multi-step reasoning** → a modell architektúrája nem támogatja (lásd 14. fejezet)

---

## 2. fejezet: A Transformer Neuron Felépítése

### 2.1 A NEURA 300M architektúra áttekintése

```
Bemenet (token IDs)
    │
    ▼
Embedding (tok) ── 1024 dim
    │
    ▼
┌─────────────────────────────────┐
│  Block 0 (L1)                   │
│  ┌──────────┐   ┌──────────┐   │
│  │ GQA      │   │ FFN      │   │
│  │ Attention│ → │ (SwiGLU) │   │
│  └──────────┘   └──────────┘   │
│    16 fej, 4 KV fej  3072 hidden│
└─────────────────────────────────┘
    │
    ▼
  ... 23 more blocks (L2-L24) ...
    │
    ▼
RMSNorm (ln_f)
    │
    ▼
Output (Linear) → logits [V=32000]
```

### 2.2 FFN (Feed-Forward Network) — A "Neuronok" valódi helye

A modell **FFN rétegei** tartalmazzák a neuronokat. Minden Block-ban van egy FFN:

```python
class FFN(torch.nn.Module):
    def __init__(self, dim=1024, hidden=3072):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden, dim, bias=False)  # output projection  
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up projection

    def forward(self, x):
        # SwiGLU aktiváció
        gate = F.silu(self.w1(x))    # [B, T, 3072] — mely neuronok tüzeljenek?
        up = self.w3(x)              # [B, T, 3072] — milyen erősséggel?
        hidden = gate * up           # [B, T, 3072] — a neuron aktivációk!
        return self.w2(hidden)       # vissza a 1024 dim-be
```

**Fontos:** A 3072 dimenziós `hidden` vektor a **neuron aktiváció** — minden dimenzió egy neuron. Ha a neuron értéke > 1.0, aktívnak tekintjük.

### 2.3 A három súlymátrix szerepe

| Mátrix | Alakja | Mit csinál? | 
|--------|--------|-------------|
| `w1` | [3072, 1024] | Gate — eldönti, mely neuronok aktiválódjanak |
| `w3` | [3072, 1024] | Up — a bemenet "tartalmát" adja |
| `w2` | [1024, 3072] | Down — a neuron kimeneteket vetíti vissza |

**Neuron-specifikus részek:** Minden `i` indexű neuronhoz tartozik:
- `w1[i, :]` — 1024 bemeneti súly (mit detektáljon a neuron)
- `w3[i, :]` — 1024 gate súly (mikor aktiválódjon)
- `w2[:, i]` — 1024 kimeneti súly (mit adjon ki a neuron)
- **Összesen: 3072 paraméter / neuron** (3 × 1024)

### 2.4 Teljes neuron szám

| Pozíció | Neuronok | 
|---------|----------|
| Per FFN réteg | 3,072 neuron |
| 24 réteg × 3,072 | **73,728 neuron összesen** |
| Ebből aktív (>1.0) egy promptban | ~50-1,400 (rétegtől függően) |

### 2.5 GQA Attention — A "figyelem" mechanizmus

Az FFN mellett a másik fontos komponens az Attention (GQA - Grouped Query Attention):

```python
class GQA(nn.Module):
    def __init__(self, dim=1024, nh=16, nkv=4):
        self.wq = nn.Linear(dim, nh * hd, bias=False)    # query
        self.wk = nn.Linear(dim, nkv * hd, bias=False)   # key  
        self.wv = nn.Linear(dim, nkv * hd, bias=False)   # value
        self.wo = nn.Linear(nh * hd, dim, bias=False)     # output
        
    def forward(self, x):
        # Minden token "kérdez" (query), és minden token "válaszol" (key)
        # Az attention súlyok mutatják, mely tokenek kapcsolódnak
```

**Fontos fogalom: "Token attention"**
- Minden token (szó) figyel a többi tokenre a mondatban
- Az attention súly [token_i, token_j] megmutatja, mennyire fontos token_j token_i számára
- **Ha két token soha nem figyel egymásra, a modell nem tud kapcsolatot teremteni közöttük**

---

## 3. fejezet: Környezet Beállítása

### 3.1 Minimum követelmények

| Erőforrás | Observation (megfigyelés) | Intervention (beavatkozás) |
|-----------|--------------------------|---------------------------|
| RAM | 8 GB | 8 GB |
| GPU | Nem kell (CPU is elég) | Ajánlott (RTX 3060+) |
| Tárhely | 5 GB (modell + adatok) | 5 GB |
| Python | 3.10+ | 3.10+ |
| PyTorch | 2.0+ | 2.0+ |

### 3.2 Telepítés

```bash
# 1. Python környezet
python -m venv neura_env
source neura_env/bin/activate  # Linux/Mac
# vagy
neura_env\Scripts\activate     # Windows

# 2. PyTorch (CPU)
pip install torch torchvision torchaudio

# 3. PyTorch (CUDA - ha van GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Egyéb függőségek
pip install sentencepiece numpy matplotlib

# 5. Tokenizer letöltése
# A NEURA SentencePiece tokenizer 32K szókészlettel
# Helye: tokenizer.model
```

### 3.3 A NEURA modell betöltése

```python
import torch
import sentencepiece as spm

VOCAB = 32000
DIM = 1024
LAYERS = 24
HEADS = 16
KV_HEADS = 4
FFN_HIDDEN = 3072

# Tokenizer betöltése
sp = spm.SentencePieceProcessor()
sp.Load('tokenizer.model')  # 32K szókészlet

# ====== ARCHITEKTÚRA DEFINIÁLÁSA ======
class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class GQA(torch.nn.Module):
    def __init__(self, dim, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, dim // nh
        self.wq = torch.nn.Linear(dim, nh * self.hd, False)
        self.wk = torch.nn.Linear(dim, nkv * self.hd, False)
        self.wv = torch.nn.Linear(dim, nkv * self.hd, False)
        self.wo = torch.nn.Linear(nh * self.hd, dim, False)
        self.register_buffer('m', torch.tril(torch.ones(512, 512)))
    def forward(self, x):
        B, T = x.shape[:2]
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.nh > self.nkv:
            k = k[:, :, None].expand(-1, -1, self.nh // self.nkv, -1, -1)
            k = k.reshape(B, self.nh, T, self.hd)
            v = v[:, :, None].expand(-1, -1, self.nh // self.nkv, -1, -1)
            v = v.reshape(B, self.nh, T, self.hd)
        w = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        w = w.masked_fill(self.m[:T, :T] == 0, float('-inf'))
        w = torch.nn.functional.softmax(w, dim=-1)
        return self.wo((w @ v).transpose(1, 2).reshape(B, T, -1))

class FFN(torch.nn.Module):
    def __init__(self, dim, h):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, h, False)
        self.w2 = torch.nn.Linear(h, dim, False)
        self.w3 = torch.nn.Linear(dim, h, False)
    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class Block(torch.nn.Module):
    def __init__(self, dim, nh, nkv, ffn):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        self.attn = GQA(dim, nh, nkv)
        self.ffn = FFN(dim, ffn)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = torch.nn.Embedding(VOCAB, DIM)
        self.blocks = torch.nn.ModuleList(
            [Block(DIM, HEADS, KV_HEADS, FFN_HIDDEN) for _ in range(LAYERS)]
        )
        self.ln_f = RMSNorm(DIM)
        self.out = torch.nn.Linear(DIM, VOCAB, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks:
            x = b(x)
        return self.out(self.ln_f(x))

# ====== MODELL BETÖLTÉSE ======
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LM().to(device)
checkpoint = torch.load('lm300m_v3_step390000.pt', map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print(f"Modell betöltve! Paraméterek: {sum(p.numel() for p in model.parameters()):,}")
```

### 3.4 Egyszerű generálás teszt

```python
def generate(prompt, max_tokens=50, temperature=0.7, top_k=50):
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = ids.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x[:, -256:])  # utolsó 256 token kontextus
            logits = logits[0, -1] / temperature
            
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[-1]] = float('-inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            out_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
            
            if next_id in [0, 2]:  # stop token
                break
    
    return sp.DecodeIds(out_ids)

# Teszt
print(generate("Szia! Hogy vagy?"))
print(generate("Az alma piros és"))
print(generate("Magyarország fővárosa"))
```

---

# 2. rész: Megfigyelés — Mit csinálnak a neuronok?

## 4. fejezet: Forward-Pass Hook-ok — Hogyan nézzünk a modell fejébe?

### 4.1 A hook-ok koncepciója

A PyTorch hook-ok olyan "hallgatók" (listeners), amelyeket a modell rétegeire szerelünk. Amikor a modell előrejelzést készít (forward pass), a hook-ok **lehallgatják** és **elmentik** a réteg belső állapotát anélkül, hogy módosítanák a modellt.

```
Bemenet → [Block] → hook elkapja a kimenetet!
                ↑
          "Aha! Itt van a hidden state!"
```

### 4.2 Hook regisztráció — a legegyszerűbb változat

```python
def capture_activations(model):
    """Hook-ok regisztrálása minden Block kimenetére"""
    activations = {}
    handles = []
    
    def make_hook(name):
        def hook(module, input, output):
            # output = (hidden_state) a Block után
            activations[name] = output.detach().float().cpu()
        return hook
    
    for i, block in enumerate(model.blocks):
        handle = block.register_forward_hook(make_hook(f'block_{i}'))
        handles.append(handle)
    
    return activations, handles

# Használat
activations, handles = capture_activations(model)

# Forward pass
x = torch.tensor([sp.EncodeAsIds("Az alma piros és")])
with torch.no_grad():
    logits = model(x)

# Nézzük meg a 22. réteg hidden state-jét
h22 = activations['block_22']  # [1, 4, 1024] — [batch, tokens, dim]
print(f"L22 hidden state shape: {h22.shape}")
print(f"L22 hidden state norm: {h22.norm(dim=-1)}")  # mely tokenek aktívak?

# Tisztítás
for h in handles:
    h.remove()
```

### 4.3 FFN neuron aktivációk elkapása

Az FFN belsejébe is be kell néznünk:

```python
def capture_ffn_activations(model):
    """Hook a SwiGLU aktivációkra — a valódi neuronokra"""
    ffn_acts = {}
    handles = []
    
    def make_ffn_hook(layer_idx):
        def hook(module, input, output):
            # input[0] = a bemenet az FFN-be (ln2 utáni)
            x = input[0]
            
            # Kiszámoljuk a gate aktivációt
            gate = torch.nn.functional.silu(module.w1(x))  # [B, T, 3072]
            up = module.w3(x)                                # [B, T, 3072]
            hidden = gate * up                               # [B, T, 3072]
            
            ffn_acts[f'ffn_{layer_idx}'] = {
                'gate': gate.detach().float().cpu(),
                'hidden': hidden.detach().float().cpu(),
                'up': up.detach().float().cpu(),
            }
        return hook
    
    for i, block in enumerate(model.blocks):
        handle = block.ffn.register_forward_hook(make_ffn_hook(i))
        handles.append(handle)
    
    return ffn_acts, handles
```

### 4.4 Attention mátrix elkapása

```python
def capture_attention(model):
    """Hook az attention súlyokra"""
    attn_weights = {}
    handles = []
    
    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            x = input[0]  # ln1 utáni kimenet
            B, T = x.shape[:2]
            
            q = module.wq(x).view(B, T, module.nh, module.hd).transpose(1, 2)
            k = module.wk(x).view(B, T, module.nkv, module.hd).transpose(1, 2)
            
            if module.nh > module.nkv:
                k = k[:, :, None].expand(-1, -1, module.nh // module.nkv, -1, -1)
                k = k.reshape(B, module.nh, T, module.hd)
            
            w = (q @ k.transpose(-2, -1)) * (module.hd ** -0.5)
            w = w.masked_fill(module.m[:T, :T] == 0, float('-inf'))
            w = torch.nn.functional.softmax(w, dim=-1)
            
            attn_weights[f'attn_{layer_idx}'] = w.detach().float().cpu()
        return hook
    
    for i, block in enumerate(model.blocks):
        handle = block.attn.register_forward_hook(make_attn_hook(i))
        handles.append(handle)
    
    return attn_weights, handles
```

### 4.5 Teljes NeuraExplorer — mindent egyszerre

```python
class NeuraExplorer:
    """Egy osztály, ami minden belső állapotot elkap"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.ffn_acts = {}
        self.attn_weights = {}
        self.handles = []
        self._register_all_hooks()
    
    def _register_all_hooks(self):
        # Block output hook-ok
        for i, block in enumerate(self.model.blocks):
            handle = block.register_forward_hook(self._make_block_hook(i))
            self.handles.append(handle)
            
            # FFN hook
            handle = block.ffn.register_forward_hook(self._make_ffn_hook(i))
            self.handles.append(handle)
            
            # Attention hook
            handle = block.attn.register_forward_hook(self._make_attn_hook(i))
            self.handles.append(handle)
    
    def _make_block_hook(self, idx):
        def hook(m, i, o):
            self.activations[f'block_{idx}'] = o.detach().float().cpu()
        return hook
    
    def _make_ffn_hook(self, idx):
        def hook(m, i, o):
            x = i[0]
            gate = torch.nn.functional.silu(m.w1(x))
            up = m.w3(x)
            hidden = gate * up
            self.ffn_acts[f'ffn_{idx}'] = {
                'gate': gate.detach().float().cpu(),
                'hidden': hidden.detach().float().cpu(),
            }
        return hook
    
    def _make_attn_hook(self, idx):
        def hook(m, i, o):
            x = i[0]
            B, T = x.shape[:2]
            q = m.wq(x).view(B, T, m.nh, m.hd).transpose(1, 2)
            k = m.wk(x).view(B, T, m.nkv, m.hd).transpose(1, 2)
            if m.nh > m.nkv:
                k = k[:, :, None].expand(-1, -1, m.nh // m.nkv, -1, -1)
                k = k.reshape(B, m.nh, T, m.hd)
            w = (q @ k.transpose(-2, -1)) * (m.hd ** -0.5)
            w = w.masked_fill(m.m[:T, :T] == 0, float('-inf'))
            w = torch.nn.functional.softmax(w, dim=-1)
            self.attn_weights[f'attn_{idx}'] = w.detach().float().cpu()
        return hook
    
    def forward(self, x):
        """Forward pass minden hook-kal"""
        self.activations.clear()
        self.ffn_acts.clear()
        self.attn_weights.clear()
        
        with torch.no_grad():
            return self.model(x)
    
    def cleanup(self):
        for h in self.handles:
            h.remove()
    
    def get_layer_summary(self, layer_idx):
        """Összefoglaló egy rétegről"""
        hidden = self.activations.get(f'block_{layer_idx}')
        ffn = self.ffn_acts.get(f'ffn_{layer_idx}')
        attn = self.attn_weights.get(f'attn_{layer_idx}')
        
        if hidden is None:
            return "Nincs adat"
        
        summary = f"=== L{layer_idx + 1} ===\n"
        summary += f"Hidden state mag: {hidden.norm(dim=-1).tolist()}\n"
        
        if ffn:
            h = ffn['hidden']
            active = (h.abs() > 1.0).sum().item()
            total = h.shape[-1]
            max_act = h.abs().max().item()
            summary += f"FFN neuronok: {active}/{total} aktív (>1.0), max={max_act:.2f}\n"
        
        if attn:
            # Átlag attention fejek felett
            avg_attn = attn.mean(dim=1)  # [B, T, T]
            head_div = attn.std(dim=1).mean().item()
            summary += f"Fej diversity: {head_div:.4f}\n"
        
        return summary


# Használat
explorer = NeuraExplorer(model)
x = torch.tensor([sp.EncodeAsIds("Az alma piros és")])
logits = explorer.forward(x)

for i in range(24):
    print(explorer.get_layer_summary(i))

explorer.cleanup()
```

---

## 5. fejezet: Sparsity Elemzés — Mely neuronok aktívak?

### 5.1 Mi az a sparsity?

A **sparsity** (ritkaság) azt mutatja, hogy az adott réteg neuronjainak hány százaléka aktív (>1.0) egy adott promptra. Minél ritkább a réteg, annál kevesebb neuron tüzel.

### 5.2 Per-layer sparsity mérése

```python
def analyze_sparsity(explorer, prompt):
    """Sparsity elemzés minden rétegre"""
    x = torch.tensor([sp.EncodeAsIds(prompt)])
    logits = explorer.forward(x)
    
    print(f"\nPrompt: {prompt}")
    print(f"Tokenek: {sp.EncodeAsIds(prompt)}")
    print(f"{'Réteg':<8} {'Aktív':<8} {'Összes':<8} {'Aktív%':<10} {'Max':<10}")
    print("-" * 44)
    
    for i in range(24):
        ffn = explorer.ffn_acts.get(f'ffn_{i}')
        if ffn:
            h = ffn['hidden']  # [1, T, 3072]
            # Utolsó token neuron aktivációi
            last_h = h[0, -1, :]
            active = (last_h.abs() > 1.0).sum().item()
            total = last_h.shape[0]
            max_act = last_h.abs().max().item()
            pct = active / total * 100
            print(f"L{i+1:<5} {active:<8} {total:<8} {pct:<10.1f}% {max_act:<10.2f}")
    
    # Legaktívabb réteg azonosítása
    sparsity_data = {}
    for i in range(24):
        ffn = explorer.ffn_acts.get(f'ffn_{i}')
        if ffn:
            h = ffn['hidden'][0, -1, :]
            active = (h.abs() > 1.0).sum().item()
            sparsity_data[i] = active / h.shape[0] * 100
    
    most_active = max(sparsity_data, key=sparsity_data.get)
    print(f"\nLegaktívabb réteg: L{most_active + 1} ({sparsity_data[most_active]:.1f}%)")

# Teszt
explorer = NeuraExplorer(model)
analyze_sparsity(explorer, "Az alma piros és")
explorer.cleanup()
```

### 5.3 Várható eredmények (NEURA 300M)

Egy tipikus 4 tokenes magyar mondatra ("Az alma piros és"):

| Réteg | Aktív% | Értelmezés |
|-------|--------|------------|
| L1-L4 | 0.0-0.1% | Szinte semmi — a tokenek még "beállnak" |
| L5-L12 | 0.1-0.3% | Minimális aktivitás — szintaxis építés |
| L13-L16 | 0.3-2.6% | Növekvő — kontextus építés |
| L17-L21 | 5.1-19.4% | **Erős aktivitás** — tudás előhívás |
| L22 | **25.5%** | **Nagyon aktív** |
| L23 | **44.8%** | **A legaktívabb** — utolsó FFN a döntés előtt |
| L24 | 3.8% | Kevés — ez csak kivetítés (output layer) |

**Következtetés:** A mély rétegek (17-23) NEM zajosak! A növekvő aktivitás bizonyítja, hogy a modell használja az összes réteget. L23 a "döntéshozó" réteg — itt aktiválódik a legtöbb neuron.

### 5.4 Neuron súlyok elemzése (weight norm)

Nem elég az aktivációt nézni — a súlyokat is ellenőrizni kell:

```python
def analyze_neuron_weights(model, layer_idx=22):
    """Neuron súlyok elemzése egy adott rétegben"""
    block = model.blocks[layer_idx]
    ffn = block.ffn
    
    # Minden neuron összesített súlynormája
    w1_norms = torch.norm(ffn.w1.weight.data, dim=1)    # [3072]
    w2_norms = torch.norm(ffn.w2.weight.data, dim=0)    # [3072]
    w3_norms = torch.norm(ffn.w3.weight.data, dim=1)    # [3072]
    total_norms = w1_norms + w2_norms + w3_norms
    
    print(f"\n=== L{layer_idx + 1} Neuron Súly Elemzés ===")
    print(f"Átlagos súlynorma: {total_norms.mean().item():.2f}")
    print(f"Min: {total_norms.min().item():.2f}")
    print(f"Max: {total_norms.max().item():.2f}")
    print(f"Arány (max/min): {total_norms.max().item() / total_norms.min().item():.2f}x")
    
    # Legaktívabb és legkevésbé aktív neuronok
    sorted_idx = torch.argsort(total_norms)
    
    print("\n5 legkevésbé aktív neuron:")
    for idx in sorted_idx[:5]:
        print(f"  #{idx.item()} — total={total_norms[idx].item():.2f}")
    
    print("\n5 legaktívabb neuron:")
    for idx in sorted_idx[-5:]:
        print(f"  #{idx.item()} — total={total_norms[idx].item():.2f}")

# NEURA 300M esetén várható:
# L22: avg=3.52, max/min arány=~1.96x (kevéssé specializált!)
# L23: avg=3.65, max/min arány=~1.96x
# Összehasonlítás: nagy modellekben 10x-100x az arány
```

### 5.5 Fontos felismerés: aktiváció vs súly

| Mérés | Mit mutat? |
|-------|------------|
| **Aktiváció** (>1.0) | Mely neuronok tüzelnek EBBEN a promptban |
| **Súlynorma** | Mely neuronok rendelkeznek erős kapcsolatokkal ÁLTALÁBAN |

Egy neuron lehet erős súlyokkal (magas weight norm) de egy adott promptban nem aktív — és fordítva. Mindkettőt érdemes mérni.

---

## 6. fejezet: Attention Elemzés — Hová figyel a modell?

### 6.1 Token párok kapcsolata

A legfontosabb kérdés: **két token kapcsolódik-e a modellben?** Az attention súlyok ezt mutatják.

```python
def analyze_attention_pair(explorer, prompt, token_a_pos=0, token_b_pos=5):
    """Két token közötti attention elemzése minden rétegben"""
    x = torch.tensor([sp.EncodeAsIds(prompt)])
    logits = explorer.forward(x)
    
    tokens = sp.EncodeAsIds(prompt)
    token_strs = [sp.IdToPiece(t) for t in tokens]
    
    print(f"\nPrompt: {prompt}")
    print(f"Tokenek: {token_strs}")
    print(f"Elemzett páros: '{token_strs[token_a_pos]}' (pos {token_a_pos}) → "
          f"'{token_strs[token_b_pos]}' (pos {token_b_pos})")
    
    print(f"\n{'Réteg':<8} {'A→B attn':<12} {'B→A attn':<12} {'Értelmezés'}")
    print("-" * 55)
    
    for i in range(24):
        attn = explorer.attn_weights.get(f'attn_{i}')
        if attn is not None:
            # Átlag fejek felett
            avg_attn = attn.mean(dim=1)[0]  # [T, T]
            
            a_to_b = avg_attn[token_a_pos, token_b_pos].item()
            b_to_a = avg_attn[token_b_pos, token_a_pos].item()
            
            interpretation = ""
            if a_to_b > 0.2:
                interpretation = "Erős kapcsolat!"
            elif a_to_b > 0.1:
                interpretation = "Közepes"
            elif a_to_b > 0.05:
                interpretation = "Gyenge"
            else:
                interpretation = "≈NINCS"
            
            print(f"L{i+1:<5} {a_to_b:<12.4f} {b_to_a:<12.4f} {interpretation}")


# KRITIKUS TESZT: Számok kapcsolódása
# Prompt: "Ha 5 almám van és megeszek 2-t"
# "5" pozíció 1, "2-" pozíció 6
analyze_attention_pair(explorer, "Ha 5 almám van és megeszek 2-t", 1, 6)

# Várható eredmény: 5→2 attention = 0.00 MINDEN rétegben!
# Ez a bizonyíték, hogy a modell NEM tud számolni.
```

### 6.2 "Mit figyel a token X?" — Teljes attention térkép

```python
def token_attention_profile(explorer, prompt, token_pos):
    """Egy token teljes attention profilja"""
    x = torch.tensor([sp.EncodeAsIds(prompt)])
    logits = explorer.forward(x)
    
    tokens = sp.EncodeAsIds(prompt)
    token_strs = [sp.IdToPiece(t) for t in tokens]
    
    print(f"\n'{token_strs[token_pos]}' token attention profilja:")
    
    # Minden rétegben, a legerősebb célpont
    print(f"\n{'Réteg':<8} {'Cél 1':<20} {'Cél 2':<20} {'Cél 3':<20}")
    print("-" * 68)
    
    for i in range(24):
        attn = explorer.attn_weights.get(f'attn_{i}')
        if attn is not None:
            avg_attn = attn.mean(dim=1)[0]  # [T, T]
            token_attn = avg_attn[token_pos]  # [T]
            
            # Top-3 célpont
            top3 = torch.argsort(token_attn, descending=True)[:3]
            targets = []
            for t in top3:
                targets.append(f"{token_strs[t.item()]} ({token_attn[t.item()]:.3f})")
            
            print(f"L{i+1:<5} {targets[0]:<20} {targets[1]:<20} {targets[2]:<20}")


# Példa: az "és" token figyelmének változása rétegenként
token_attention_profile(explorer, "Az alma piros és", 3)

# Várható:
# L1:  "Az" (0.39) — szintaktikai (cikk + kötőszó)
# L12: "almám" (0.38) — szintaktikai (főnév + kötőszó)
# L22: "és" (0.41) — önfigyelem (összegzés)
```

### 6.3 Fej Diversity — Specializálódnak a fejek?

```python
def analyze_head_diversity(explorer, prompt):
    """Hányféle dolgot csinálnak a fejek egy rétegen belül?"""
    x = torch.tensor([sp.EncodeAsIds(prompt)])
    logits = explorer.forward(x)
    
    print(f"\n=== Fej Diversity Elemzés ===")
    print(f"{'Réteg':<8} {'Diversity':<12} {'Értelmezés'}")
    print("-" * 40)
    
    for i in range(24):
        attn = explorer.attn_weights.get(f'attn_{i}')
        if attn is not None:
            # 16 fej attention mintázata
            # attn shape: [1, 16, T, T]
            heads = attn[0]  # [16, T, T]
            diversity = heads.std(dim=0).mean().item()
            
            interpretation = ""
            if diversity < 0.05:
                interpretation = "Minden fej ugyanazt csinálja"
            elif diversity < 0.08:
                interpretation = "Kevés specializáció"
            elif diversity < 0.12:
                interpretation = "Némi specializáció"
            else:
                interpretation = "Erős specializáció!"
            
            print(f"L{i+1:<5} {diversity:<12.4f} {interpretation}")

# Várható:
# L3:  0.040 — Minden fej ugyanaz (legkevésbé specializált)
# L7:  0.111 — Specializáció indul
# L22: 0.142 — Legmagasabb (legerősebb specializáció)
# L24: 0.140 — Szintén magas
```

---

## 7. fejezet: Réteg Hierarchia — Hogyan épül fel a feldolgozás?

### 7.1 A feldolgozás három fázisa

A NEURA 300M 24 rétege három nagy fázisra osztható:

```
L1-L8:   MINTÁZATFELISMERÉS — FFN dominál
         → Token párok, szintaxis, szótagszerkezet

L9-L16:  KONTEXTUSÉPÍTÉS — Attention dominál  
         → Mondat szintű kapcsolatok, K vektorok csúcson

L17-L24: TUDÁSELŐHÍVÁS — FFN dominál
         → Magas szintű jelentés, döntés a kimenetről
```

### 7.2 Attention vs FFN hozzájárulás mérése

```python
def analyze_attention_vs_ffn(explorer, prompt):
    """Melyik dominál: attention vagy FFN?"""
    x = torch.tensor([sp.EncodeAsIds(prompt)])
    logits = explorer.forward(x)
    
    print(f"\n=== Attention vs FFN Hozzájárulás ===")
    print(f"{'Réteg':<8} {'Attn_out norm':<15} {'FFN_out norm':<15} {'Arány':<10} {'Domináns'}")
    print("-" * 55)
    
    tokens = sp.EncodeAsIds(prompt)
    
    for i in range(24):
        block_hidden = explorer.activations.get(f'block_{i}')
        if block_hidden is None:
            continue
        
        h = block_hidden[0]  # [T, 1024]
        
        # Az attention hozzájárulás becslése:
        # A hidden state = előző réteg + attention + ffn
        # Itt a Block kimenet = előző + attn_out + ffn_out
        # De pontosan mérni kellene a komponenseket...
        # Egyszerűsítve: nézzük a teljes hidden state norm-ját
        
        attn_norm = h.norm(dim=-1).mean().item()
        
        # Összehasonlítás a szomszédos rétegekkel
        ratio = "—"
        dominant = "—"
        
        if i > 0:
            prev_h = explorer.activations[f'block_{i-1}'][0]
            prev_norm = prev_h.norm(dim=-1).mean().item()
            
            # Ha a norm nőtt, valami hozzáadódott
            if prev_norm > 0:
                ratio = attn_norm / prev_norm
                if ratio > 1.1:
                    dominant = "NŐTT"
                elif ratio < 0.9:
                    dominant = "CSÖKKENT"
                else:
                    dominant = "STABIL"
        
        print(f"L{i+1:<5} {attn_norm:<15.2f} {'—':<15} {str(ratio):<10} {dominant}")


# Egyszerűbb változat: maga a block norm változása
def layer_norm_progression(explorer, prompt):
    """Hidden state norm változása rétegenként"""
    x = torch.tensor([sp.EncodeAsIds(prompt)])
    logits = explorer.forward(x)
    
    tokens_str = [sp.IdToPiece(t) for t in sp.EncodeAsIds(prompt)]
    
    print(f"\n=== Token szintű norm változás rétegenként ===")
    header = f"{'Layer':<8}"
    for t in tokens_str:
        header += f"{t:<15}"
    print(header)
    print("-" * (8 + 15 * len(tokens_str)))
    
    for i in range(24):
        h = explorer.activations.get(f'block_{i}')
        if h is None:
            continue
        norms = h[0].norm(dim=-1).tolist()
        line = f"L{i+1:<5}"
        for n in norms:
            line += f"{n:<15.2f}"
        print(line)
```

### 7.3 K Vektor norm — "Mit tárol a modell?"

A K (key) vektorok norm-ja megmutatja, mennyi információt tárol az adott réteg:

```python
def analyze_k_norm(explorer, prompt):
    """K vektor norm elemzés"""
    x = torch.tensor([sp.EncodeAsIds(prompt)])
    logits = explorer.forward(x)
    
    print(f"\n=== K Vektor Norm rétegenként ===")
    print(f"{'Réteg':<8} {'K norm (avg)':<15} {'Q norm (avg)':<15} {'V norm (avg)'}")
    print("-" * 50)
    
    # A K vektorokat az attention hook-ból nyerjük ki
    # Ehhez az attention hook-ot ki kell egészíteni K, Q, V norm mentéssel
    # (lásd a bővített explorer-t a függelékben)
    
    # Várható minta a NEURA 300M-ben:
    expected = [
        (1, 5.5, 5.2),
        (6, 5.1, 5.8),
        (12, 36.2, 9.8),  # K vektor csúcs!
        (18, 11.5, 9.5),
        (23, 12.1, 9.3),
    ]
    
    print("Várható értékek (NEURA 300M):")
    for layer, k, q in expected:
        print(f"L{layer:<6} {k:<18.1f} {q:<18.1f}")
```

---

## 8. fejezet: Neuron-Feature Térkép (MapMaker)

### 8.1 A MapMaker koncepciója

A MapMaker megmondja, hogy **mely neuronok mely fogalmakra reagálnak**. 48 promptot futtatunk 10 kategóriában, és megnézzük, mely neuronok aktiválódnak erősen.

### 8.2 Kategóriák és promptok definiálása

```python
# 10 kategória, kategóriánként 4-5 prompt
PROMPTS = {
    "GYÜMÖLCS": [
        "Az alma piros és",
        "A körte édes és",
        "A banán sárga és",
        "A narancs lédús és",
        "A szőlő édes és",
    ],
    "SZÁM": [
        "Van 1 almám és",
        "Van 5 almám és",
        "Van 10 almám és",
        "Van 3 almám és",
        "Van 2 almám és",
    ],
    "SZÍN": [
        "Az ég kék és",
        "A fű zöld és",
        "A vér piros és",
        "A nap sárga és",
        "A éjszaka fekete és",
    ],
    "HELY": [
        "Budapest szép és",
        "A kert nagy és",
        "Az iskola messze és",
        "A ház fehér és",
        "Az utca hosszú és",
    ],
    "ÁLLAT": [
        "A macska szőrös és",
        "A kutya barátságos és",
        "A ló gyors és",
        "A madár repül és",
        "A hal úszik és",
    ],
    "IDŐ": [
        "Reggel van és",
        "Este van és",
        "Ma jó idő és",
        "Tegnap esett és",
        "Holnap megyek és",
    ],
    "IGE": [
        "A fiú eszik és",
        "A lány iszik és",
        "A kutya fut és",
        "A gyerek alszik és",
        "A fiú játszik és",
    ],
    "ÉRZELEM": [
        "A gyerek boldog és",
        "A lány szomorú és",
        "Az apa mérges és",
        "Az anya fáradt és",
        "A férfi ideges és",
    ],
    "TEST": [
        "A kéz erős és",
        "A láb hosszú és",
        "A fej nagy és",
        "A szem kék és",
        "A száj piros és",
    ],
    "TÁRGY": [
        "Az asztal fa és",
        "A szék kényelmes és",
        "Az autó gyors és",
        "A könyv vastag és",
        "A telefon új és",
    ],
}
```

### 8.3 MapMaker algoritmus

```python
def run_mapmaker(explorer, prompts_dict, layer_idx=22):
    """
    Neuron-feature térkép készítése
    
    Minden promptra:
    1. Forward pass
    2. Utolsó token FFN aktivációinak mentése
    3. Kategorizálás: mely neuron mely kategóriára aktiválódik a legerősebben
    """
    
    # Gyűjtés: kategória → neuron aktivációk
    activation_map = {}  # {neuron_idx: [(kategória, erősség), ...]}
    
    for category, prompts in prompts_dict.items():
        for prompt in prompts:
            x = torch.tensor([sp.EncodeAsIds(prompt)])
            logits = explorer.forward(x)
            
            ffn = explorer.ffn_acts.get(f'ffn_{layer_idx}')
            if ffn is None:
                continue
            
            # Utolsó token aktivációja
            last_hidden = ffn['hidden'][0, -1, :]  # [3072]
            
            # Minden neuron, ami aktív (>1.0)
            active_mask = last_hidden.abs() > 1.0
            active_indices = torch.where(active_mask)[0]
            active_values = last_hidden[active_indices]
            
            for n_idx, n_val in zip(active_indices.tolist(), active_values.tolist()):
                if n_idx not in activation_map:
                    activation_map[n_idx] = []
                activation_map[n_idx].append((category, abs(n_val)))
    
    # Minden neuronhoz: a legerősebb kategória
    neuron_category = {}
    for n_idx, activations in activation_map.items():
        # Csoportosítás kategóriánként
        cat_strength = {}
        for cat, val in activations:
            if cat not in cat_strength:
                cat_strength[cat] = []
            cat_strength[cat].append(val)
        
        # Átlag erősség kategóriánként
        best_cat = max(cat_strength, key=lambda c: sum(cat_strength[c]) / len(cat_strength[c]))
        avg_strength = sum(cat_strength[best_cat]) / len(cat_strength[best_cat])
        neuron_category[n_idx] = (best_cat, avg_strength, len(activations))
    
    # Eredmények megjelenítése
    print(f"\n=== L{layer_idx + 1} Neuron-Feature Térkép ===")
    print(f"Összes aktív neuron (>1.0 legalább 2 promptra): {len(neuron_category)}")
    
    # Kategóriánként csoportosítás
    cat_counts = {}
    cat_strengths = {}
    for n_idx, (cat, strength, count) in neuron_category.items():
        if cat not in cat_counts:
            cat_counts[cat] = 0
            cat_strengths[cat] = []
        cat_counts[cat] += 1
        cat_strengths[cat].append(strength)
    
    print(f"\nKategória eloszlás:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        avg_s = sum(cat_strengths[cat]) / len(cat_strengths[cat])
        print(f"  {cat:<12} {count:>4} neuron | átlag erősség: {avg_s:.2f}")
    
    # Top-10 legszelektívebb neuron
    print(f"\nTop-10 legszelektívebb neuron:")
    
    # Rendezés: a legerősebb kategória specifikusság szerint
    sorted_neurons = sorted(
        neuron_category.items(),
        key=lambda x: -x[1][1]  # erősség szerint csökkenő
    )
    
    for rank, (n_idx, (cat, strength, count)) in enumerate(sorted_neurons[:10]):
        print(f"  #{n_idx:<5} → {cat:<12} (erősség: {strength:.2f}, {count} prompt tüzel)")


# Használat
explorer = NeuraExplorer(model)
run_mapmaker(explorer, PROMPTS, layer_idx=22)
explorer.cleanup()
```

### 8.4 Várható eredmények (NEURA 300M)

| Kategória | Dedikált neuronok | Legerősebb neuron |
|-----------|------------------|-------------------|
| **GYÜMÖLCS** (alma, körte) | 1,100-2,100 | #1357 L23 (10.3) |
| **HELY** (Budapest, iskola) | 1,100-1,800 | #1338 L23 (12.3) — legerősebb! |
| **SZÍN** (piros, kék) | 900-1,700 | változó |
| **ÁLLAT** (macska, kutya) | 800-1,600 | #2634 L23 (7.3) |
| **IGE** (eszik, fut) | 800-1,600 | #528 L22 (8.9) |
| **SZÁM** (1, 5, 10) | **~400-800** | **#1214 L22 (9.7)** — legkevesebb! |
| **IDŐ** (reggel, este) | 437-472 | #1703 L23 (8.5) |
| **ÉRZELEM** (boldog, szomorú) | ~400-600 | #3002 L23 (7.3) |

**Fő felfedezés:** A SZÁMOKNAK van a legkevesebb dedikált neuronjuk. Ez magyarázza, miért olyan gyenge a modell matematikában.

### 8.5 Fontos korlátozás

**Ez korreláció, nem kauzális azonosítás!** Egy neuron, ami erősen tüzel az "iskola" szóra, lehet hogy a "tanulás", "osztály" vagy "diák" fogalomra is tüzel. A valódi kauzális azonosításhoz **abláció** kell: nullázzuk ki a neuront, és nézzük meg, eltűnik-e a fogalom a kimenetből (lásd 9. fejezet).

---

# 3. rész: Beavatkozás — Neuronok Módosítása

## 9. fejezet: FFN Neuron Szerkesztés

### 9.1 A módszer koncepciója

Az FFN neuron szerkesztés a **legegyszerűbb** beavatkozás: kiválasztunk egy inaktív neuront, és átírjuk a súlyait, hogy egy új képességet képviseljen.

**Hogyan működik:**
1. Keresünk egy olyan neuront, ami szinte soha nem aktív (inaktív)
2. Átírjuk `w1`-et: mit detektáljon a neuron (input pattern)
3. Átírjuk `w3`-at: mikor aktiválódjon (gate)
4. Átírjuk `w2`-t: mit adjon ki a neuron (output direction)

### 9.2 Inaktív neuron keresése

```python
def find_inactive_neurons(model, layer_idx=22, n=5):
    """Legkevésbé aktív neuronok keresése"""
    block = model.blocks[layer_idx]
    ffn = block.ffn
    
    # Súlynormák összesítése
    w1_norms = torch.norm(ffn.w1.weight.data, dim=1)  # [3072]
    w2_norms = torch.norm(ffn.w2.weight.data, dim=0)  # [3072]
    w3_norms = torch.norm(ffn.w3.weight.data, dim=1)  # [3072]
    total_norms = w1_norms + w2_norms + w3_norms
    
    sorted_idx = torch.argsort(total_norms)
    
    print(f"\n=== L{layer_idx + 1} — {n} legkevésbé aktív neuron ===")
    print(f"{'Neuron #':<10} {'w1 norm':<10} {'w2 norm':<10} {'w3 norm':<10} {'Total'}")
    print("-" * 50)
    
    inactive_neurons = []
    for idx in sorted_idx[:n]:
        w1 = w1_norms[idx].item()
        w2 = w2_norms[idx].item()
        w3 = w3_norms[idx].item()
        total = total_norms[idx].item()
        inactive_neurons.append(idx.item())
        print(f"#{idx.item():<7} {w1:<10.2f} {w2:<10.2f} {w3:<10.2f} {total:.2f}")
    
    return inactive_neurons

# NEURA 300M várható értékek:
# L22: inaktív norm≈3.0, aktív norm≈5.4, átlag≈3.52
# L23: inaktív norm≈2.76, aktív norm≈5.28, átlag≈3.65
# Arány: ~1.96x (jó modellben 10x-100x kellene!)
```

### 9.3 Specialist neuron létrehozása

Tegyük fel, hogy a modell mindig "kap"-ot mond "marad" helyett a kivonásos feladatoknál. Szerkesszünk egy neuront, ami a "marad" token felé tolja a kimenetet:

```python
def create_neuron_specialist(model, layer_idx=22, neuron_idx=2116, token_str="▁marad"):
    """
    Kézzel létrehozott neuron, ami egy adott token irányába tolja a kimenetet
    
    Figyelmeztetés: Ez a módszer NAGYON érzékeny!
    - Túl gyenge: semmit sem változik
    - Túl erős: "elnyelnyelnyelny..." loop
    """
    block = model.blocks[layer_idx]
    ffn = block.ffn
    
    # 1. Token direction kinyerése az output mátrixból
    token_id = sp.PieceToId(token_str)
    token_dir = model.out.weight.data[token_id].clone()  # [1024]
    
    print(f"Token '{token_str}' direction norm: {token_dir.norm().item():.2f}")
    
    # 2. Neuron átírása
    dim = ffn.w1.weight.shape[1]
    
    # Minden bemenetre aktiválódjon (univerzális detektor)
    ffn.w1.weight.data[neuron_idx] = torch.ones(dim) * 0.1  # gyenge
    ffn.w3.weight.data[neuron_idx] = torch.ones(dim) * 0.1  # gyenge
    
    # Kimenet: a token irányába
    ffn.w2.weight.data[:, neuron_idx] = token_dir * 0.5
    
    print(f"Neuron #{neuron_idx} átírva: → '{token_str}' specialistává")
    
    # 3. Teszt
    model.eval()
    x = torch.tensor([sp.EncodeAsIds("Ha 5 almám van és megeszek 2-t")])
    with torch.no_grad():
        logits = model(x)
    
    probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
    
    for test_token in ["▁marad", "▁3", "▁kap", ","]:
        tid = sp.PieceToId(test_token)
        print(f"P({test_token}) = {probs[tid].item() * 100:.4f}%")
```

### 9.4 Ismert hibák és megoldások

| Hiba | Tünet | Megoldás |
|------|-------|----------|
| **"elny" loop** | "elnyelnyelnyelny..." | w1, w3, decoder 10x gyengítése |
| **Semmi változás** | Ugyanaz a kimenet | w1, w3, decoder 10x erősítése |
| **Minden tokenre aktivál** | Minden prompt "marad"-ot mond | w1 threshold növelése |
| **Modell összeomlik** | Értelmetlen tokenek | Szerkesztés visszavonása |

**A "marad" token problémája:** A "marad" token direction norm-ja **1.34** — ez nagyon gyenge (random tokeneké 5-10). 10x erősítés (norm 13.4) megtöri a residual stream egyensúlyát. Ezért a kézi FFN edit ritkán működik a gyenge tokenekre.

### 9.5 Ablációs teszt — Kivétel, nem hozzáadás

Néha hasznosabb **kivenni** egy neuron hatását, mint hozzáadni:

```python
def ablate_neuron(model, layer_idx=22, neuron_idx=352):
    """
    Neuron abláció: nullázzuk ki a neuron kimenetét
    
    Ha ezután eltűnik egy fogalom a modell kimenetéből,
    akkor a neuron OKOZATILAG felelős azért a fogalomért.
    """
    block = model.blocks[layer_idx]
    ffn = block.ffn
    
    # Eredeti súlyok mentése
    original_w2 = ffn.w2.weight.data[:, neuron_idx].clone()
    
    # Nullázás (neuron kivétele)
    ffn.w2.weight.data[:, neuron_idx] = 0
    
    print(f"Neuron #{neuron_idx} L{layer_idx + 1} nullázva (ablation)")
    
    # Teszt
    model.eval()
    x = torch.tensor([sp.EncodeAsIds("Az iskola nagy és tágas")])
    with torch.no_grad():
        logits_before = model(x)
    
    # Visszaállítás
    ffn.w2.weight.data[:, neuron_idx] = original_w2
    
    # Ha a kimenet megváltozott, a neuron okozatilag fontos volt
    token_before = logits_before[0, -1].argmax().item()
    print(f"Eredeti predikció (abláció után): {sp.IdToPiece(token_before)}")
```

---

## 10. fejezet: Aktivációs Steering

### 10.1 A módszer koncepciója

Aktivációs steering: a forward pass **közben** hozzáadunk egy irányvektort a hidden state-hez, hogy a modell más irányba "gondolkodjon". Nem változtatja meg a modell súlyait — csak a kimenetet módosítja menet közben.

```
Forward pass közben:
  h_{l} = Block_l(h_{l-1})
  h_{l}' = h_{l} + steer_vector × strength   ← ITT!
  h_{l+1} = Block_{l+1}(h_{l}')
```

### 10.2 Steering irány megtalálása

```python
def find_steering_direction(explorer, normal_prompt, target_prompt, layer_idx=22):
    """
    Megtalálja a "target" irányba mutató vektort
    
    steer_direction = hidden_target - hidden_normal
    """
    # Normál prompt
    x_normal = torch.tensor([sp.EncodeAsIds(normal_prompt)])
    explorer.forward(x_normal)
    h_normal = explorer.activations[f'block_{layer_idx}'][0, -1]  # [1024]
    
    # Cél prompt (ahová szeretnénk vinni)
    x_target = torch.tensor([sp.EncodeAsIds(target_prompt)])
    explorer.forward(x_target)
    h_target = explorer.activations[f'block_{layer_idx}'][0, -1]  # [1024]
    
    # Steering irány
    steer = h_target - h_normal
    
    print(f"Steering direction norm: {steer.norm().item():.2f}")
    print(f"Normal prompt: '{normal_prompt}'")
    print(f"Target prompt: '{target_prompt}'")
    
    return steer


def generate_steered(model, explorer, prompt, steer_vector, 
                     steer_layer=22, strength=1.0, max_tokens=50):
    """
    Generálás aktivációs steeringgel
    """
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = ids.copy()
    
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Saját forward: rétegenként
            h = model.tok(x)
            for li, block in enumerate(model.blocks):
                h = block(h)
                if li == steer_layer:
                    # Steering az utolsó tokenre
                    h[0, -1] += steer_vector.to(h.device) * strength
            
            logits = model.out(model.ln_f(h))
            next_id = logits[0, -1].argmax().item()
            
            if next_id in [0, 2]:
                break
            
            out_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], device=x.device)], dim=1)
    
    return sp.DecodeIds(out_ids)


# Használat
explorer = NeuraExplorer(model)

# Steering irány: "normál szöveg → matek kontextus"
steer = find_steering_direction(
    explorer,
    normal_prompt="Az alma piros és",
    target_prompt="Öt mínusz kettő az",
    layer_idx=22
)

# Gyenge steering
result_weak = generate_steered(model, explorer, "Ha 5 almám van és megeszek 2-t", 
                                steer, strength=5.0)

# Erős steering
result_strong = generate_steered(model, explorer, "Ha 5 almám van és megeszek 2-t", 
                                  steer, strength=30.0)

print(f"\nGyenge (5.0): {result_weak}")
print(f"Erős (30.0): {result_strong}")

explorer.cleanup()
```

### 10.3 Várható eredmények (NEURA 300M)

| Strength | Eredmény |
|----------|----------|
| **1-5** | Alig változik semmi (0.01% → 0.05%) |
| **10** | "akkor egy egész ültényt..." (kis változás) |
| **30** | "egyikegyikegyik..." (loop / breakdown) |
| **50+** | Teljes összeomlás |

**Következtetés:** A steering nem elég erős egyedül. A "marad" token 0.01%-os valószínűségét 400-2000x kellene növelni, ami megtöri a modellt.

---

## 11. fejezet: Logit Bias (Gyors & Piszkos)

### 11.1 A módszer koncepciója

A legegyszerűbb módszer: közvetlenül a kimeneti logitokhoz adunk hozzá egy számot (bias), mielőtt kiválasztanánk a következő tokent. **Nem változtat a modellen**, csak a generálás pillanatában.

```
logits = model(x)
logits[token_3_id] += 10   ← direkt boost
probs = softmax(logits)
```

### 11.2 Implementáció

```python
def generate_with_logit_bias(model, prompt, bias_token_id, bias=5.0, max_tokens=50):
    """
    Generálás logit bias-szal
    
    Args:
        bias_token_id: a boostolandó token ID-ja
        bias: mennyivel növeljük a logitot (5-10 ajánlott)
    """
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = ids.copy()
    
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x[:, -256:])
            # BIAS hozzáadása
            logits[0, -1, bias_token_id] += bias
            next_id = logits[0, -1].argmax().item()
            
            if next_id in [0, 2]:
                break
                
            out_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], device=x.device)], dim=1)
    
    return sp.DecodeIds(out_ids)


# Használat
# Token ID-k (NEURA 300M):
TOKEN_3 = 30        # "▁3"
TOKEN_MARAD = 1073  # "▁marad"
TOKEN_VESSZO = 30779  # ","

prompt = "Ha 5 almám van és megeszek 2-t"

print("Eredeti:", generate_with_logit_bias(model, prompt, TOKEN_3, bias=0))
print("Bias=5:", generate_with_logit_bias(model, prompt, TOKEN_3, bias=5))
print("Bias=10:", generate_with_logit_bias(model, prompt, TOKEN_3, bias=10))
```

### 11.3 Eredmények

| Bias | Eredmény | Értékelés |
|------|----------|-----------|
| 0 | ",?kap alma..."" | ❌ Nincs szám |
| 5 | "3" 50%-ban | ⚠️ Bizonytalan |
| **10** | **"33333333"** | **✅ 100% 3-as, de loop!** |
| 20 | "33333333..." | ✅ Működik, de loop |

**A loop oka:** Miután a modell kiadta a "3"-at, a következő lépésben a bias továbbra is hat, és a "3" kontextusában ismét a "3"-at választja.

### 11.4 Mikor használd?

| Szituáció | Javaslat |
|-----------|----------|
| Gyors teszt (30 másodperc) | ✅ Logit Bias |
| Végleges megoldás | ❌ Nem (loop + nem tanul) |
| Egy token kikényszerítése | ✅ Ideális |
| Több tokenes logika | ❌ Nem működik |

---

## 12. fejezet: Adapter Rétegek (LogicAdapter)

### 12.1 A módszer koncepciója

A LogicAdapter egy **kis neurális háló**, amit a modell végére teszünk. Csak az adaptert tanítjuk, a fő modell súlyai **rögzítve** maradnak.

```
Bemenet → [24 NEURA Block (fagyasztva)] → [LogicAdapter] → [LN_f] → [Output]
                                              ↑
                                  1024 → ReLU → 64 → 1024
                                  (zero-init decoder)
```

**Miért zero-init?** 
- `decoder.weight = 0` → az adapter kimenete 0 az elején
- Első forward pass: **pontosan ugyanaz, mint az eredeti modell**
- Az adapter csak annyit tanul, amennyit kell — nem rombolja le a meglévő tudást

### 12.2 Architektúra

```python
class LogicAdapter(torch.nn.Module):
    """
    Residual stream adapter: 1024 → 64 → 1024
    
    Zero initialization ensures the model output is unchanged at start.
    The adapter learns what to ADD to the residual stream.
    """
    def __init__(self, dim=1024, hidden=64):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden, bias=False)
        self.decoder = nn.Linear(hidden, dim, bias=False)
        
        # Zero-init: output = 0 at start → modell változatlan
        nn.init.zeros_(self.decoder.weight)
        
        # Encoder: kis zaj, hogy a gradiensek folyjanak
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        return self.decoder(F.relu(self.encoder(x)))


class NEURAWithAdapter(torch.nn.Module):
    """NEURA modell + LogicAdapter"""
    def __init__(self, base_model, hidden=64):
        super().__init__()
        # Base model komponensek másolása
        self.tok = base_model.tok
        self.blocks = base_model.blocks
        self.ln_f = base_model.ln_f
        self.out = base_model.out
        
        # Adapter
        self.logic = LogicAdapter(dim=1024, hidden=hidden)
    
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks:
            x = b(x)
        x = x + self.logic(x)  # ← Adapter hozzáadás a residual stream-hez
        return self.out(self.ln_f(x))
```

### 12.3 Training adat előkészítése

A legfontosabb: **kiegyensúlyozott adat**!

```python
def prepare_training_data(sp, prompts_and_answers, device='cpu'):
    """
    Training adat előkészítése
    
    prompts_and_answers: [(prompt, correct_answer_token_str), ...]
    Példa: [("2 - 1 =", "1"), ("5 - 2 =", "3"), ...]
    """
    input_ids_list = []
    target_token_ids = []
    
    for prompt, answer in prompts_and_answers:
        ids = sp.EncodeAsIds(prompt)
        input_ids_list.append(ids)
        target_token_ids.append(sp.PieceToId(f"▁{answer}"))  # SP space prefix
    
    DIGIT_NAMES = [str(d) for d in range(1, 10)]
    DIGIT_IDS = torch.tensor(
        [sp.PieceToId(d) for d in DIGIT_NAMES], 
        device=device
    )
    
    target_digit_idx_list = []
    for ans_str in [a for _, a in prompts_and_answers]:
        target_digit_idx_list.append(DIGIT_NAMES.index(ans_str))
    
    return input_ids_list, target_token_ids, target_digit_idx_list, DIGIT_IDS


# Példa adat: kivonás 1-9 között, 12 példa / számjegy
def build_subtraction_dataset():
    """108 példa: 12 prompt mind a 9 számjegyre (1-9)"""
    all_examples = []
    
    # Minden lehetséges kivonás, ahol az eredmény 1-9
    for a in range(2, 11):
        for b in range(1, a):
            result = a - b
            if result < 1 or result > 9:
                continue
                
            prompts = [
                f"{a} - {b} =",
                f"{a} minusz {b} az",
                f"{a} almabol {b}-et megeszek, marad",
                f"{a} almabol {b}-t megeszek, marad",
            ]
            
            for p in prompts:
                all_examples.append((p, str(result)))
    
    # Limited to 12 per digit for balance
    from collections import Counter
    counts = Counter(a for _, a in all_examples)
    
    balanced = []
    for digit in [str(d) for d in range(1, 10)]:
        digit_examples = [(p, a) for p, a in all_examples if a == digit]
        balanced.extend(digit_examples[:12])  # max 12 per digit
    
    print(f"Adatkészlet: {len(balanced)} példa")
    print(f"Eloszlás: {Counter(a for _, a in balanced)}")
    return balanced
```

### 12.4 Training loop — Szelektív Loss

A **szelektív loss** csak a releváns tokeneket veszi figyelembe (pl. csak a számjegyeket), nem a teljes 32K-s szókészletet:

```python
def train_adapter(model, input_ids_list, target_digit_idx_list, DIGIT_IDS,
                  epochs=120, batch_size=16, lr=1e-3):
    """
    LogicAdapter training szelektív loss-szal
    
    Csak a digit tokenek (1-9) közötti loss-t számoljuk.
    """
    model.train()
    
    # Csak az adapter paraméterei legyenek taníthatók
    for p in model.parameters():
        p.requires_grad = False
    for p in model.logic.parameters():
        p.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.logic.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    print(f"\n=== LogicAdapter Training ===")
    print(f"Adapter paraméterek: {sum(p.numel() for p in model.logic.parameters()):,}")
    print(f"Adat: {len(input_ids_list)} példa, {epochs} epoch")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle
        indices = list(range(len(input_ids_list)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i + batch_size]
            optimizer.zero_grad()
            
            batch_loss = 0.0
            for idx in batch:
                x = torch.tensor([input_ids_list[idx]], dtype=torch.long, device=device)
                logits = model(x)
                last_logits = logits[0, -1]
                
                # Szelektív loss: csak digit tokenek
                digit_logits = last_logits[DIGIT_IDS]  # [9]
                target = torch.tensor([target_digit_idx_list[idx]], device=device)
                
                loss = F.cross_entropy(digit_logits.unsqueeze(0), target)
                batch_loss = batch_loss + loss
                
                if torch.argmax(digit_logits).item() == target_digit_idx_list[idx]:
                    correct += 1
                total += 1
            
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.logic.parameters(), 1.0)
            optimizer.step()
            total_loss += batch_loss.item()
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / max(1, (len(indices) + batch_size - 1) // batch_size)
            acc = correct / total * 100
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, digit_acc={acc:.1f}%")
    
    return model
```

### 12.5 Kombinált Loss (Szelektív + Teljes Szókészlet)

A szelektív loss jól megtanítja a számjegyek megkülönböztetését, de a generálás során a vessző (28%) és a pont (9%) dominálhat. Ilyenkor adjunk hozzá egy kis teljes szókészletes loss-t:

```python
# Kombinált loss
loss_selective = F.cross_entropy(digit_logits.unsqueeze(0), digit_target)
loss_full = F.cross_entropy(last_logits.unsqueeze(0), full_target)

loss = loss_selective + 0.5 * loss_full  # Súlyozás fontos!
```

| Változat | Digit accuracy | Generálás minősége |
|----------|:-------------:|:------------------:|
| **Szelektív csak** | **54.6%** | ",?Andorrael?ny," (szemét) |
| **Kombinált** (0.5x full) | 42.9% | **"28888"** (számok!) |

### 12.6 Várható eredmények

| Metrika | 64 hidden (131K param) | 128 hidden (262K param) |
|---------|:---------------------:|:----------------------:|
| Digit accuracy | 54.6% | 42.9% (több adat kell) |
| Training idő (GPU) | ~7 perc | ~10 perc |
| Training idő (CPU) | ~2 óra | ~3 óra |
| Példák | 108 (12/digit) | 180 (20/digit) |
| Loss 10 epoch után | 0.44 | 0.62 |

**Konfúziós mátrix (szelektív, 120 epoch):**

```
"8 - 5 ="  → 3 ✅   "6 - 1 ="  → 5 ✅   "2 - 1 ="  → 1 ✅
"5 - 2 ="  → 4 ❌   (helyes: 3) — off by 1
"3 - 1 ="  → 1 ❌   (helyes: 2) — off by 1
"10 - 3 =" → 6 ❌   (helyes: 7) — off by 1
```

**Minden hiba ±1 pontosságú!** Az adapter megtanulja a "növel/csökkent eggyel" mintázatot, de a finom megkülönböztetéshez nagyobb hidden dim kell.

### 12.7 Token ID ellenőrzés (fontos!)

A SentencePiece tokenizerben a számjegyek NEM 0-9 ID-k!

```python
# Ellenőrizd a token ID-kat!
for d in range(0, 10):
    tid = sp.PieceToId(str(d))
    piece = sp.IdToPiece(tid)
    print(f"'{d}' → ID {tid} → vissza: '{piece}'")
    
# NEURA 300M tipikus értékek:
# '1' → 30755, '2' → 30761, '3' → 30771, ... 
# Ezek byte-fallback tokenek, nagyon magas ID-k
# A base modellben ~0.001-0.02% valószínűséggel
```

---

## 13. fejezet: SFT (Supervised Fine-Tuning)

### 13.1 A módszer koncepciója

Az SFT során a modell **minden paramétere** változhat (ellentétben az adapterrel, ahol csak az adapter). Ez erősebb, de veszélyesebb — a modell elfelejtheti a meglévő tudását (katasztrofális felejtés).

### 13.2 Mikor használjunk SFT-t?

| Ha ez a cél... | ...akkor SFT |
|----------------|-------------|
| Chat formátum tanítása | ✅ Igen |
| Stílusváltás (pl. hivatalos → laza) | ✅ Igen |
| Új képesség (pl. számolás) | ⚠️ Próbáld az adaptert előbb |
| Egyetlen hiba javítása | ❌ FFN edit vagy adapter jobb |

### 13.3 Chat adat formátum

```python
CHAT_EXAMPLES = [
    {
        "messages": [
            {"role": "user", "content": "Szia! Hogy vagy?"},
            {"role": "assistant", "content": "Köszönöm, jól vagyok! Hogy segíthetek?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Mi a magyar főváros?"},
            {"role": "assistant", "content": "Magyarország fővárosa Budapest."}
        ]
    },
    # ... 50-200 példa
]

def format_chat_prompt(messages):
    """Chat formátum tokenizálása"""
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<|im_start|>user\n{msg['content']}\n<|im_end|>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|im_start|>assistant\n{msg['content']}\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"  # asszisztens válasz kezdete
    return prompt
```

### 13.4 Korlátok kis modelleknél (355M)

| Probléma | Tünet | Ok |
|----------|-------|-----|
| **Katasztrofális felejtés** | "Magyar főváros a Duna..." | 81 példa → a nyelvtani tudás felülíródik |
| **Ténykeveredés** | "A Balaton egy folyó..." | Nincs elég kapacitás tények tárolására |
| **Gyenge generálás** | Rövid, semmitmondó válaszok | 355M limitált kifejezőereje |

**Ajánlás:** SFT-hez minimum 7B paraméteres modell kell. 355M-en csak nagyon specifikus, kis változtatásokat érdemes csinálni (adapter vagy FFN edit).

---

# 4. rész: Esettanulmányok

## 14. fejezet: Miért Nem Tud Számolni a 355M?

### 14.1 A probléma

Prompt: "Ha 5 almám van és megeszek 2-t"

**Várható válasz:** "...marad 3"
**Valós válasz:** "...két óra múlva, a másik hét..."

### 14.2 A bizonyíték: 5→2 attention = 0.00

```python
# KRITIKUS TESZT: számok közötti attention
prompt = "Ha 5 almám van és megeszek 2-t"
tokens = sp.EncodeAsIds(prompt)
token_strs = [sp.IdToPiece(t) for t in tokens]
# ['▁Ha', '▁5', '▁almám', '▁van', '▁és', '▁megeszek', '▁2-', 't']
#  pos=0    pos=1  pos=2    pos=3  pos=4  pos=5      pos=6   pos=7

x = torch.tensor([tokens])
explorer = NeuraExplorer(model)
explorer.forward(x)

for i in range(24):
    attn = explorer.attn_weights[f'attn_{i}']
    avg_attn = attn.mean(dim=1)[0]  # [T, T]
    a5_to_2 = avg_attn[1, 6].item()  # "5" → "2-"
    print(f"L{i+1:2d}: 5→2 attention = {a5_to_2:.4f}")
```

**Eredmény: 0.00 mind a 24 rétegben.**

### 14.3 Az ok: Token valószínűségek

```python
x = torch.tensor([sp.EncodeAsIds("Ha 5 almám van és megeszek 2-t")])
with torch.no_grad():
    logits = model(x)

probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)

# Top-5 predikció
top5 = torch.argsort(probs, descending=True)[:5]
print("Top-5 predikció:")
for tid in top5:
    print(f"  P({sp.IdToPiece(tid.item())}) = {probs[tid].item() * 100:.2f}%")

print(f"\nMatematikai tokenek:")
print(f"  P(▁3) = {probs[30].item() * 100:.4f}%")
print(f"  P(▁marad) = {probs[1073].item() * 100:.4f}%")
```

**Eredmény:**
- "," (vessző): **19.71%**
- "▁kap" (kap): **10.16%**
- "." (pont): **9.18%**
- "▁3" (a helyes válasz!): **0.03%**
- "▁marad" (marad): **0.01%**

### 14.4 A gyökérok

1. **Training adat:** A modell ~2.5B magyar tokenen tanult. Ezekben a szövegekben az "5 alma" és "2 alma" függetlenül fordulnak elő. SOHA nem szerepel, hogy "5-2=3".

2. **Reprezentáció:** A szám tokenek nem figyelnek egymásra → nincs mechanizmus a kapcsolatukhoz. Mindketten az "almá"-ra figyelnek, mint melléknevek a főnévhez.

3. **Korlát:** A 355M modell neuronjai "suttognak" (max 15.7 aktiváció) a nagy modellek "kiabálásával" szemben (LLaMA-7B: 100-200). A számoknak van a **legkevesebb** dedikált neuronjuk.

### 14.5 A megoldások összehasonlítása

| Módszer | Eredmény | Idő | Nehézség |
|---------|----------|-----|----------|
| **FFN szerkesztés** | ❌ "elny" loop | 5 perc | Közepes |
| **Activációs steering** | ❌ breakdown 30+ | 1 perc | Könnyű |
| **Logit bias** | ✅ "33333" | 30 mp | **Legkönnyebb** |
| **LogicAdapter** | ✅ 55% accuracy | 7 perc (GPU) | Közepes |
| **SFT** | ⚠️ Katasztrofális felejtés | 1+ óra | Nehéz |

---

## 15. fejezet: Teljes LogicAdapter Esettanulmány

### 15.1 Cél

Tanítsuk meg a NEURA 300M-nek, hogy egyszerű kivonási feladatokat megoldjon (5-2=3, 8-5=3, stb.) anélkül, hogy a meglévő nyelvtani tudását elveszítené.

### 15.2 Adatkészlet

108 példa, 12 minden számjegyre (1-9), kiegyensúlyozva.

### 15.3 Teljes training script

```python
"""
LogicAdapter training — teljes példa
NEURA 300M subtraction task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import random, time, math

# ====== BEÁLLÍTÁSOK ======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ADAPTER_HIDDEN = 64  vagy 128
EPOCHS = 120
BATCH_SIZE = 16
LR = 1e-3
LOG_FILE = 'logicadapter_training_log.txt'

# ====== TOKENIZER ======
sp = spm.SentencePieceProcessor()
sp.Load('tokenizer.model')

# Digit token ID-k ellenőrzése
DIGIT_NAMES = [str(d) for d in range(1, 10)]
DIGIT_IDS = torch.tensor(
    [sp.PieceToId(d) for d in DIGIT_NAMES],
    device=DEVICE
)
print(f"Digit token ID-k: {dict(zip(DIGIT_NAMES, DIGIT_IDS.tolist()))}")

# ====== ADAT ======
def build_data():
    examples = []
    for a in range(2, 11):
        for b in range(1, a):
            r = a - b
            if r < 1 or r > 9:
                continue
            prompts = [
                f"{a} - {b} =", f"{a} minusz {b} az",
                f"{a} almabol {b}-et megeszek, marad",
                f"{a} almabol {b}-t megeszek, marad",
            ]
            for p in prompts:
                examples.append((p, str(r)))
    
    # Balance: 12 per digit
    from collections import Counter
    balanced = []
    for d in DIGIT_NAMES:
        dex = [(p, a) for p, a in examples if a == d]
        balanced.extend(dex[:12])
    
    return balanced

data = build_data()
print(f"Adat: {len(data)} példa")
print(f"Eloszlás: {Counter(a for _, a in data)}")

# Tokenizálás
input_ids_list = []
target_digit_idx = []
for prompt, answer in data:
    input_ids_list.append(sp.EncodeAsIds(prompt))
    target_digit_idx.append(DIGIT_NAMES.index(answer))

# ====== MODELL ======
# Töltsd be a NEURA modellt (lásd 3. fejezet)
# ...

# Adapter hozzáadása
model_with_adapter = NEURAWithAdapter(base_model, hidden=ADAPTER_HIDDEN)
model_with_adapter.to(DEVICE)

# ====== TRAINING ======
model_with_adapter.train()
for p in model_with_adapter.parameters():
    p.requires_grad = False
for p in model_with_adapter.logic.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(
    model_with_adapter.logic.parameters(), lr=LR
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

print(f"\nAdapter paraméterek: "
      f"{sum(p.numel() for p in model_with_adapter.logic.parameters()):,}")

start_time = time.time()
for epoch in range(EPOCHS):
    model_with_adapter.train()
    total_loss = 0
    correct = 0
    total = 0
    
    indices = list(range(len(input_ids_list)))
    random.shuffle(indices)
    
    for i in range(0, len(indices), BATCH_SIZE):
        batch = indices[i:i + BATCH_SIZE]
        optimizer.zero_grad()
        
        batch_loss = 0
        for idx in batch:
            x = torch.tensor([input_ids_list[idx]], dtype=torch.long, device=DEVICE)
            logits = model_with_adapter(x)
            last_logits = logits[0, -1]
            
            # Szelektív loss
            digit_logits = last_logits[DIGIT_IDS]
            target = torch.tensor([target_digit_idx[idx]], device=DEVICE)
            
            loss = F.cross_entropy(digit_logits.unsqueeze(0), target)
            batch_loss = batch_loss + loss
            
            if torch.argmax(digit_logits).item() == target_digit_idx[idx]:
                correct += 1
            total += 1
        
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model_with_adapter.logic.parameters(), 1.0
        )
        optimizer.step()
        total_loss += batch_loss.item()
    
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        avg_loss = total_loss / max(1, len(indices) // BATCH_SIZE)
        acc = correct / total * 100
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, "
              f"acc={acc:.1f}%, time={elapsed:.0f}s")

# ====== TESZT ======
print("\n=== TESZT ===")
test_prompts = [
    ("8 - 5 =", "3"),
    ("6 - 1 =", "5"),
    ("2 - 1 =", "1"),
    ("5 - 2 =", "3"),
    ("5 - 3 =", "2"),
    ("10 - 1 =", "9"),
]

model_with_adapter.eval()
with torch.no_grad():
    for prompt, expected in test_prompts:
        x = torch.tensor([sp.EncodeAsIds(prompt)], dtype=torch.long, device=DEVICE)
        logits = model_with_adapter(x)
        
        # Digit logits
        digit_logits = logits[0, -1, DIGIT_IDS]
        pred_digit_idx = torch.argmax(digit_logits).item()
        pred_digit = DIGIT_NAMES[pred_digit_idx]
        
        mark = "✅" if pred_digit == expected else "❌"
        print(f"{mark} '{prompt}' → predicted {pred_digit} (expected {expected})")
```

### 15.4 Várható eredmények és tanulságok

1. **Az adapter működik** — 10 epoch alatt loss=2.0 → 0.44
2. **Minden hiba ±1** — az adapter megtanulja a növel/csökkent mintát
3. **Zero init fontos** — első forward pass = eredeti modell
4. **A generálás még tökéletlen** — kombinált loss kell a szép outputhoz
5. **64 hidden kevés** — 128-256 hidden kéne 90%+ accuracy-hez

---

# 5. rész: Haladó Témák

## 16. fejezet: Az Emergens Gondolkodás Korlátai

### 16.1 Mi az emergens gondolkodás?

Az **emergens gondolkodás** olyan képesség, ami nem volt explicit betanítva, de a modellben megjelenik ha elég nagy. Példák:

| Képesség | Megjelenés |
|----------|-----------|
| Egyszerű számolás | ~7B paraméter felett |
| Többlépéses logika | ~70B paraméter felett |
| Fordítás | ~1B paraméter felett |
| Ténytárolás | Minden méreten, de pontosság nő mérettel |

### 16.2 Miért NEM emergens a 355M?

```
Paraméterek vs reasoning képesség:
355M → ❌ Nincs emergens reasoning
  1B → ⚠️ Alapvető mintázatok
  7B → ✅ Egyszerű logika
 70B → ✅ Komplex reasoning
100B+ → ✅ Többlépéses

Neuron aktiváció maximum:
355M → 15.7 (suttogás)
GPT-2 → 30-50
LLaMA-7B → 100-200 (kiabálás)
GPT-4 → ~500+ (üvöltés)
```

### 16.3 Mit lehet és mit nem lehet javítani?

| Javítható | Nem javítható |
|-----------|--------------|
| Egy token preferenciájának változtatása | Többlépéses logika |
| Egyszerű mintázat tanítása (pl. kivonás ±1) | Valódi matematikai megértés |
| Kimenet stílusának módosítása | Multi-hop reasoning |
| 50-200 példából tanulás | 10K+ tény tárolása |

### 16.4 A hibák mintázata

A 355M adapter hibái mind **±1** típusúak:
- 5-2=3 helyett 4-et mond (off by 1)
- 3-1=2 helyett 1-et mond (off by 1)
- Soha nem mond 7-et a helyett, hogy 3-at kéne

**Ez bizonyítja:** az adapter megtanulja a "számsor" fogalmát és a "csökkent" irányt, de nem tud pontos nagyságrendet megkülönböztetni. Ez architekturális korlát, nem adathiány.

---

## 17. fejezet: Hibaelhárítási Útmutató

### 17.1 Gyakori hibák

| Hiba | Valószínű ok | Megoldás |
|------|-------------|----------|
| "elny" loop | FFN edit túl erős | Csökkentsd w1, w3, decoder 10x |
| Semmi változás | FFN edit túl gyenge | Növeld 10x |
| CUDA out of memory | Túl nagy batch | Csökkentsd batch_size-t 1-re |
| NaN loss | Learning rate túl magas | Csökkentsd lr-t 1e-4-re |
| Grad = 0.0000 | zero_grad() hívás a mentés előtt | Rendezd át a kódot |
| Unicode error | Windows cp1250 konzol | Használj file loggingot |
| SSH connection lost | Hálózati hiba | Használj nohup / PowerShell background job-ot |
| Modell összeomlik | Rossz súly módosítás | Töltsd be újra a checkpointot |

### 17.2 Windows-specifikus problémák

```python
# 1. Unicode a konzolon — MINDIG használj file loggingot!
def safe_log(msg):
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    try:
        safe = msg.encode('ascii', errors='replace').decode('ascii')
        print(safe)
    except:
        pass

# 2. DataLoader — használj num_workers=0
# multiprocessing crash-el Windowson
DataLoader(dataset, batch_size=16, num_workers=0)

# 3. SSH file transfer — base64 pipe
# scp gyakran fail-el Windowson
# Használd: base64 -w0 file | ssh user@host "python -c \"import base64,sys; ...\""
```

### 17.3 Gyakori kérdések

**Q: Miért 0.00 az 5→2 attention?**
A: A modell soha nem látott olyan adatot, ahol számok kapcsolódnának. 2.5B token magyar szövegben a számok mindig főnevekhez kapcsolódnak ("5 alma"), nem egymáshoz.

**Q: Miért "elny" a loop szöveg?**
A: Az "elny" a "marad" token irányának túlerősítésekor keletkezik. A "marad" direction norm=1.34, a random tokeneké 5-10. Túlzott erősítéskor a modell egy másik, erős token felé mozdul.

**Q: 64 hidden elég a LogicAdapternek?**
A: 9 számjegyhez ~72 hidden az elméleti minimum (8× osztályok száma). 64 hidden épphogy elég, de a hibák ±1 körül lesznek. 128-256 hidden kell a megbízható működéshez.

**Q: Lehet többlépéses számolást tanítani 355M-nek?**
A: Nem. A 5→2 attention 0.00 marad még az adapter után is. Az adapter egy residual patch, nem egy igazi reasoning mechanizmus. Többlépéses logikához minimum 7B kell.

---

## 18. fejezet: Eszköztár Összefoglaló

### 18.1 Melyik módszert mikor használd?

```
Szeretnéd változtatni a modell viselkedését?
│
├─ Csak teszteled? → Logit Bias (30 másodperc)
│
├─ Egy konkrét hibát javítasz? → FFN Neuron Edit (5 perc)
│   └─ Ha "elny" loop → csökkents w1/w3-at 10x
│
├─ Új képességet tanítasz? → LogicAdapter (2-10 perc GPU)
│   └─ Zero init = biztonságos, finomhangolható
│
├─ Valós idejű irányítás? → Aktivációs Steering
│   └─ Csak finom változtatásokra (< 5%)
│
└─ Chat formátumot tanítasz? → SFT
    └─ 355M-en veszélyes (katasztrofális felejtés!)
```

### 18.2 Gyors referenciák

**Token ID-k (NEURA 300M SentencePiece tokenizer):**

| Token | ID | Jelentés |
|-------|----|----------|
| `▁3` | 30 | Hármas szám |
| `▁marad` | 1073 | "marad" (kivonás eredmény) |
| `▁kap` | 2379 | "kap" (alapértelmezett predikció) |
| `,` | 30779 | Vessző (legvalószínűbb) |
| `▁megeszek` | 1703 | "megeszek" (kivonás kontextus) |
| `▁2-` | 1452 | "2-" (szám + kötőjel token) |

**Ellenőrizd a saját tokenizer-edben:**
```python
for d in range(0, 10):
    tid = sp.PieceToId(str(d))
    print(f"'{d}' → ID {tid}")
```

**Fontos paraméterek (NEURA 300M):**
| Paraméter | Érték |
|-----------|-------|
| Szókészlet | 32,000 |
| Hidden dim | 1,024 |
| Rétegek | 24 |
| Attention fejek | 16 |
| KV fejek | 4 |
| FFN hidden | 3,072 |
| Teljes paraméter | 354,993,152 |
| Training kontextus | 512 token |
| Hatékony kontextus | ~8-16 token |

---

## Függelék

### A. Teljes NeuraExplorer Script

(Lásd a 4. fejezetben. A teljes script elérhető a `scripts/neura_explorer.py` fájlban.)

### B. Teljes MapMaker Script

(Lásd a 8. fejezetben. A teljes script elérhető a `scripts/neura_mapmaker.py` fájlban.)

### C. Hasznos parancsok

```bash
# GPU állapot ellenőrzése
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,clocks.current.graphics,power.draw,memory.used --format=csv

# Python folyamatok listázása (Windows)
powershell "Get-Process python | Format-Table Id,CPU,StartTime"

# Modell checkpoint-ok listázása
dir C:\Users\neura\lm300m_*.pt /b

# Token ID keresés
python -c "import sentencepiece; sp=sentencepiece.SentencePieceProcessor(); sp.Load('tokenizer.model'); print(sp.PieceToId('▁marad'))"
```

### D. További olvasnivaló

1. **"In-context Learning and Induction Heads"** — Olsson et al., 2022
   - Arról, hogyan alakulnak ki a mintázatok a transformerben

2. **"Scaling Monosemanticity"** — Bricken et al., 2023 (Anthropic)
   - A neuron specializáció skálázódásáról

3. **"Transformer Circuits"** — Elhage et al., 2021 (Anthropic)
   - A transformer matematikai alapjai

4. **"The Curse of Recursion"** — Shumailov et al., 2023
   - Miért romlik a modell minősége, ha saját kimenetén tanul

5. **"Emergent Abilities of Large Language Models"** — Wei et al., 2022
   - Az emergens képességek elmélete és mérése

---

*A könyv a NEURA 300M (355M paraméter, 24 réteg, 1024 dim) magyar nyelvi modellen végzett kísérletek alapján készült. Az elvek és módszerek általánosíthatók más kis nyelvi modellekre is.*

*Utolsó frissítés: 2026. július 2.*
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

# 20. fejezet: ROME — Rank-One Model Editing

## 20.1 A probléma

A 9. fejezetben kézzel szerkesztettünk FFN neuronokat — és "elny" loop-ot kaptunk. A probléma: a kézi edit nem veszi figyelembe a modell belső struktúráját.

**ROME (Rank-One Model Editing)** — Meng et al., 2022 — ezt a problémát oldja meg egy zárt formulával.

## 20.2 Az elmélet

A transformer FFN rétegei **asszociatív memóriaként** működnek:

```
FFN(x) = W_out * σ(W_in * x)
                      ↑            ↑
                 "kulcs"     "érték" (key)   (value)
```

Egy tény, pl. "(Eiffel-torony, helye, Párizs)" a modellben úgy tárolódik, hogy:

- A **subject** utolsó tokenje (Eiffel-torony → "torony") kulcsként szolgál
- A kulcshoz tartozó érték a **kimenet irány** (Párizs felé tolja a logitokat)

A ROME egy **rank-one** mátrixfrissítéssel írja át ezt:

```
Eredeti:   v = W * k       (W = FFN.w2, k = subject hidden state)
Cél:       v* = W' * k     (v* = a kívánt új kimenet iránya)

Rank-one update:
    ΔW = r * k^T * C^{-1} / (k^T * C^{-1} * k)
    
    ahol:
    r = v* - v              (reziduális — mennyit kell változtatni)
    k = subject hidden state (a kulcs)
    C = K * K^T             (kovariancia mátrix — megőrzi a meglévő tudást)
```

## 20.3 Implementáció NEURA 300M-re

```python
import torch

def rome_edit(model, layer_idx, token_ids, subject_pos, target_token_id,
              C_inv=None, sample_prompts=None):
    """
    ROME edit implementáció NEURA 300M-re.
    
    Args:
        layer_idx: melyik réteg FFN-jét szerkesztjük (ajánlott: 22 vagy 23)
        token_ids: a prompt token ID-i (pl. "Ha 5 almám van és megeszek 2-t")
        subject_pos: a subject (pl. "5") utolsó tokenjének pozíciója
        target_token_id: a kívánt kimenet token ID-ja (pl. "3")
        C_inv: pre-számolt inverz kovariancia (None = számol most)
        sample_prompts: list of prompts a C mátrixhoz
    """
    ffn = model.blocks[layer_idx].ffn
    
    # 1. KULCS kinyerése: subject hidden state
    x = torch.tensor([token_ids])
    with torch.no_grad():
        h = model.tok(x)
        for li, block in enumerate(model.blocks):
            h = block(h)
            if li == layer_idx - 1:  # a target layer ELŐTTI réteg
                break
        
        # A subject utolsó tokenjének hidden state-je
        k = h[0, subject_pos].clone()  # [1024]
    
    # 2. CÉLÉRTÉK kiszámolása
    # A kívánt output token iránya az output mátrixból
    target_dir = model.out.weight.data[target_token_id].clone()  # [1024]
    
    # Jelenlegi érték
    current = ffn.w2.weight.data @ k  # [1024]
    
    # Reziduális: mennyit kell változtatni
    r = target_dir - current
    
    # 3. C mátrix (vagy használjuk a pre-számolt verziót)
    if C_inv is None:
        if sample_prompts is None:
            # Alapértelmezett: random magyar mondatok
            sample_prompts = [
                "Az alma piros és édes",
                "Ma szép idő van",
                "A kutya a kertben fut",
                "Budapest Magyarország fővárosa",
                "A gyerekek játszanak az udvaron",
                # ... minimum 300-500 mondat
            ]
        
        # K vektorok gyűjtése
        K = []
        for prompt in sample_prompts:
            ids = sp.EncodeAsIds(prompt)
            x = torch.tensor([ids])
            with torch.no_grad():
                h = model.tok(x)
                for li, block in enumerate(model.blocks):
                    h = block(h)
                    if li == layer_idx - 1:
                        break
                # Utolsó token hidden state-je
                k_sample = h[0, -1].clone()
                K.append(k_sample)
        
        K = torch.stack(K)  # [N, 1024]
        
        # C = K^T * K + lambda * I (regularizáció)
        lambda_reg = 0.1  # kis regularizáció a numerikus stabilitásért
        C = K.T @ K + lambda_reg * torch.eye(1024)
        
        # C^{-1}
        C_inv = torch.linalg.inv(C)
    
    # 4. RANK-ONE UPDATE
    # ΔW = r * k^T * C^{-1} / (k^T * C^{-1} * k)
    
    C_inv_k = C_inv @ k  # [1024]
    denominator = k @ C_inv_k  # scalar
    
    # A rank-one update
    delta = torch.outer(r, C_inv_k) / denominator  # [1024, 3072]
    
    # Alkalmazás: W2 += ΔW^T (mert W2 alakja [1024, 3072])
    ffn.w2.weight.data += delta.T
    
    # 5. ELLENŐRZÉS
    model.eval()
    x = torch.tensor([token_ids])
    with torch.no_grad():
        logits = model(x)
    
    probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
    target_prob = probs[target_token_id].item() * 100
    print(f"P(target token) = {target_prob:.4f}%")
    
    # Top-5 predikció
    top5 = torch.argsort(probs, descending=True)[:5]
    for tid in top5:
        print(f"  P({sp.IdToPiece(tid.item())}) = {probs[tid].item()*100:.2f}%")
    
    return ffn.w2.weight.data


# ====== HASZNÁLAT ======

# CÉL: "Ha 5 almám van és megeszek 2-t" → "marad 3"

prompt = "Ha 5 almám van és megeszek 2-t"
ids = sp.EncodeAsIds(prompt)
# Tokenek: ['▁Ha', '▁5', '▁almám', '▁van', '▁és', '▁megeszek', '▁2-', 't']
# Pos:       0      1      2        3      4      5          6      7

# Subject = "5" → position 1
subject_pos = 1

# Target = "3" token
target_token = "▁3"
target_id = sp.PieceToId(target_token)

print(f"Edit: '{prompt}' → '{target_token}'")
print(f"Subject pos: {subject_pos}, Target ID: {target_id}")

# ROME edit az L23 FFN-en
updated_weights = rome_edit(
    model,
    layer_idx=22,  # L23 (0-indexelt: 22)
    token_ids=ids,
    subject_pos=subject_pos,
    target_token_id=target_id,
)
```

## 20.4 Várakozási értékek NEURA 300M-en

A ROME-nál három dolog történhet:

| Eredmény | Mit jelent | Valószínűség |
|----------|-----------|--------------|
| **✅ Működik** | A target token lesz a legvalószínűbb | ~30% NEURA-n |
| **❌ "elny" loop** | Túl erős a változás, más token dominál | ~40% |
| **❌ Semmi változás** | Túl gyenge a változás, más rétegben van a tudás | ~30% |

**Miért nem mindig működik a ROME 355M-en?**
- A ROME-t 6B+ modellekre tervezték
- 355M-nél a reprezentációk gyengébbek (max aktiváció 15.7 vs 200+)
- A C mátrix regularizációja kritikus — túl erős λ = semmi változás, túl gyenge = loop

**Optimalizálás NEURA-ra:**

```python
# Különböző regularizációk kipróbálása
for lambda_reg in [0.01, 0.05, 0.1, 0.5, 1.0]:
    C = K.T @ K + lambda_reg * torch.eye(1024)
    C_inv = torch.linalg.inv(C)
    # ... teszt
    print(f"λ={lambda_reg}: P(target)={prob:.2f}%")

# Különböző rétegek kipróbálása
for layer in [20, 21, 22, 23]:
    # ... ROME edit
    print(f"L{layer+1}: P(target)={prob:.2f}%")
```

## 20.5 Előnyök és hátrányok

| Szempont | ROME | Kézi FFN edit (Ch9) |
|----------|------|---------------------|
| **Matematikai alap** | Zárt formula | Próba-szerencse |
| **Tudásmegőrzés** | C mátrix védi | Semmi védelem |
| **Gyorsaság** | 1 forward + 1 mátrix invertálás | 1 forward |
| **Pontosság** | Függ a C-től | Tipikusan "elny" loop |
| **Batch** | Csak 1 tény | 1 neuron |

**Mikor használd?**
- ✅ Egyetlen tény pontos javításához
- ✅ Ha a C mátrixot előre ki tudod számolni
- ❌ Több száz tény szerkesztéséhez (→ MEMIT a 21. fejezetben)
- ❌ Ha nincs 300+ magyar mondatod a C mátrixhoz

# 21. fejezet: MEMIT — Mass-Editing Memory in a Transformer

## 21.1 A probléma

A ROME (20. fejezet) egy tény szerkesztésére jó. De ha **több száz** tényt kell egyszerre javítani (pl. 100 kivonási feladat), a ROME egymás utáni alkalmazása rontja az eredményt — az új edit felülírja a régit.

**MEMIT (Mass-Editing Memory in a Transformer)** — Meng et al., 2023 (ICLR 2023 Notable Top 25%) — ezt oldja meg.

## 21.2 Az elmélet

A MEMIT felismeri, hogy a tények **több rétegben** tárolódnak. Ahelyett, hogy egyetlen réteget terhelnénk túl (mint a ROME), a MEMIT **szétosztja** a változtatásokat egy rétegtartományon.

```
ROME:
        L20  L21  L22  L23  L24
Edit1:       [ΔW]                ← Minden változás egy helyen!
Edit2:       [ΔW]                ← Felülírja Edit1-et!

MEMIT:
        L20  L21  L22  L23  L24
Edit1:  [~]  [~]  [~]            ← Szétosztva!
Edit2:       [~]  [~]  [~]       ← Más rétegekben, nem ütközik!
```

**Matematika:** MEMIT minden edithez kiszámolja a hozzájárulást minden target rétegen, majd egy **zárt formában** egyszerre alkalmazza az összes frissítést.

## 21.3 Implementáció NEURA 300M-re

```python
def memit_edit(model, edits, target_layers=[20, 21, 22], sample_prompts=None):
    """
    MEMIT batch edit implementáció.
    
    Args:
        edits: [(prompt, subject_pos, target_token_id), ...]
               pl. [("5 - 2 =", 0, "▁3"), ("8 - 5 =", 0, "▁3"), ...]
        target_layers: mely rétegek között osszuk el a változtatásokat
    """
    num_edits = len(edits)
    num_layers = len(target_layers)
    
    # 1. K és V vektorok kiszámolása minden edithez
    K = []  # [num_edits, 1024]    — kulcs vektorok
    V = []  # [num_edits, 1024]    — érték vektorok
    
    for prompt, subj_pos, target_id in edits:
        ids = sp.EncodeAsIds(prompt)
        x = torch.tensor([ids])
        
        with torch.no_grad():
            h = model.tok(x)
            # Kulcs: a target_layers első elemének BEMENETÉNÉL
            for li in range(target_layers[0]):
                h = model.blocks[li](h)
            
            k = h[0, subj_pos].clone()    # [1024]
            
            # Érték: a kívánt kimenet iránya
            v = model.out.weight.data[target_id].clone()  # [1024]
        
        K.append(k)
        V.append(v)
    
    K = torch.stack(K)  # [num_edits, 1024]
    V = torch.stack(V)  # [num_edits, 1024]
    
    # 2. Delta kiszámolása minden rétegre
    # Minden target rétegen kiszámoljuk a szükséges változtatást
    deltas = []
    
    for li, layer_idx in enumerate(target_layers):
        ffn = model.blocks[layer_idx].ffn
        
        # Jelenlegi hozzájárulás az edit helyeken
        with torch.no_grad():
            # current_values = W2 @ K_i minden i-re
            current = ffn.w2.weight.data @ K.T  # [1024, num_edits]
        
        # Mennyit kell változtatni ezen a rétegen?
        delta_v = V.T - current  # [1024, num_edits]
        
        # Egyenletes elosztás a rétegek között
        delta_v = delta_v / num_layers
        
        # Delta kiszámolása minden edit-re
        # ΔW_i = outer(delta_v[:, i], K[i]) / valami
        delta_W = torch.zeros_like(ffn.w2.weight.data)
        
        for i in range(num_edits):
            # Rank-one update
            ki = K[i]      # [1024]
            dvi = delta_v[:, i]  # [1024]
            
            # ΔW += outer(dvi, ki)
            delta_W += torch.outer(dvi, ki)
        
        deltas.append(delta_W)
    
    # 3. Visszafejtés a preservation-memorization framework-ben
    # Ez a MEMIT központi egyenlete: egyszerre optimalizálja
    # az összes deltát, hogy minimalizálja a konfliktust
    
    # A pontos MEMIT formula: 
    # Δ = V * K^T * (C + K * K^T)^{-1}   (zárt alak, batch-re)
    # 
    # Ahol C a pre-cached kovariancia mátrix
    
    # (A pontos implementáció a unified-model-editing 
    #  GitHub repo-ban található — lásd a 20. fejezet C mátrix részét)
    
    return deltas


# ====== HASZNÁLAT ======

# CÉL: 12 kivonási feladat megtanítása egyszerre
edits = [
    ("2 - 1 =",  0, sp.PieceToId("▁1")),
    ("3 - 1 =",  0, sp.PieceToId("▁2")),
    ("4 - 1 =",  0, sp.PieceToId("▁3")),
    ("5 - 2 =",  0, sp.PieceToId("▁3")),
    ("5 - 3 =",  0, sp.PieceToId("▁2")),
    ("6 - 1 =",  0, sp.PieceToId("▁5")),
    ("6 - 2 =",  0, sp.PieceToId("▁4")),
    ("7 - 3 =",  0, sp.PieceToId("▁4")),
    ("8 - 5 =",  0, sp.PieceToId("▁3")),
    ("9 - 2 =",  0, sp.PieceToId("▁7")),
    ("10 - 1 =", 0, sp.PieceToId("▁9")),
    ("10 - 3 =", 0, sp.PieceToId("▁7")),
]

# MEMIT az L21-L23 rétegeken
deltas = memit_edit(model, edits, target_layers=[20, 21, 22])

# Ellenőrzés
for prompt, _, target_id in edits:
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x)
    
    probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
    target_prob = probs[target_id].item() * 100
    top_token = sp.IdToPiece(logits[0, -1].argmax().item())
    expected = sp.IdToPiece(target_id)
    mark = "✅" if top_token == expected else "❌"
    print(f"{mark} '{prompt}' → top={top_token:6s} (P={target_prob:.1f}%)")
```

## 21.4 MEMIT vs LogicAdapter

| Szempont | MEMIT | LogicAdapter (Ch12) |
|----------|-------|---------------------|
| **Paraméterek** | Bármennyi (a fő modellé) | 131K-262K |
| **Batch méret** | 10,000+ tény | 50-200 példa |
| **Training** | Zárt formula (nincs) | 120 epoch (7 perc GPU) |
| **Pontosság** | ~90%+ (nagy modelleken) | ~55% (kicsin) |
| **Biztonság** | C mátrix véd | Zero init véd |
| **355M-en tesztelve** | ❌ Nem | ✅ Igen |

**Következtetés:** MEMIT nagy modellekre való (7B+). 355M-en a LogicAdapter a biztonságosabb választás. De ha a MEMIT matematikáját adaptáljuk (kisebb C mátrix, kevesebb réteg), akkor 355M-en is működhet.

## 21.5 A preservation-memorization framework

A MEMIT és ROME mögött egy közös elmélet van: a **preservation-memorization**:

```
Minimalizálandó: ||W' * K_preserve - W * K_preserve||²F  (preservation)
                     + ||W' * K_edit - V_edit||²F         (memorization)
```

- **Preservation:** a régi tudás megőrzése (C mátrix)
- **Memorization:** az új tények beillesztése

A ROME ezt **equality constraint**-tel oldja meg (pontosan egyezzen az új tény), a MEMIT **least-square** constraint-tel (minimalizálja a hibát).

**EMMET** (Gupta et al., 2024) a kettő unifikációja — equality constraint batch módban.

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
