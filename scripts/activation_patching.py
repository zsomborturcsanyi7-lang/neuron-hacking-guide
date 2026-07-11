#!/usr/bin/env python3
"""
Activation Patching — Kauzális tracing a NEURA 300M-en

CPU-n fut! 24 forward pass = ~2 perc.

HASZNÁLAT:
  python activation_patching.py
  
Kideríti: mely rétegek felelősek a "5 → 2" kapcsolatért?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import time

# ====== KONFIG ======
TOKENIZER_PATH = r'C:\Users\neura\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model'
CHECKPOINT_PATH = r'C:\Users\neura\lm300m_v3_step390000.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== ARCHITEKTÚRA ======
VOCAB, DIM, LAYERS, HEADS, KV_HEADS, FFN_HIDDEN = 32000, 1024, 24, 16, 4, 3072

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class GQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh, self.nkv, self.hd = HEADS, KV_HEADS, DIM // HEADS
        self.wq = nn.Linear(DIM, HEADS * self.hd, False)
        self.wk = nn.Linear(DIM, KV_HEADS * self.hd, False)
        self.wv = nn.Linear(DIM, KV_HEADS * self.hd, False)
        self.wo = nn.Linear(HEADS * self.hd, DIM, False)
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
        w = F.softmax(w, dim=-1)
        return self.wo((w @ v).transpose(1, 2).reshape(B, T, -1))

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, FFN_HIDDEN, False)
        self.w2 = nn.Linear(FFN_HIDDEN, DIM, False)
        self.w3 = nn.Linear(DIM, FFN_HIDDEN, False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(DIM)
        self.ln2 = RMSNorm(DIM)
        self.attn = GQA()
        self.ffn = FFN()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = RMSNorm(DIM)
        self.out = nn.Linear(DIM, VOCAB, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks:
            x = b(x)
        return self.out(self.ln_f(x))

# ====== ACTIVATION PATCHING ======
class ActivationPatcher:
    def __init__(self, model):
        self.model = model
        self.clean_acts = {}
        self.handles = []
    
    def _register_hooks(self):
        for li, block in enumerate(self.model.blocks):
            def make_hook(l):
                def hook(m, i, o):
                    self.clean_acts[f'block_{l}'] = o.detach().cpu()
                return hook
            handle = block.register_forward_hook(make_hook(li))
            self.handles.append(handle)
    
    def _remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []
    
    def run_clean(self, x):
        """Cache-eljük a clean run aktivációit"""
        self.clean_acts.clear()
        self._register_hooks()
        with torch.no_grad():
            logits = self.model(x)
        self._remove_hooks()
        return logits
    
    def patch_layer(self, x, layer_idx):
        """Forward patch-elt réteggel"""
        h = self.model.tok(x)
        for li, block in enumerate(self.model.blocks):
            h = block(h)
            if li == layer_idx:
                # KICSERÉLJÜK az utolsó token aktivációját
                clean = self.clean_acts.get(f'block_{li}')
                if clean is not None:
                    h[0, -1] = clean[0, -1].to(h.device)
        return self.model.out(self.model.ln_f(h))
    
    def analyze(self, clean_prompt, corrupted_prompt):
        """Teljes elemzés: minden rétegen próbáljuk a patchinget"""
        sp_ = spm.SentencePieceProcessor()
        sp_.Load(TOKENIZER_PATH)
        
        clean_ids = sp_.EncodeAsIds(clean_prompt)
        corrupted_ids = sp_.EncodeAsIds(corrupted_prompt)
        
        x_clean = torch.tensor([clean_ids], device=DEVICE)
        x_corrupted = torch.tensor([corrupted_ids], device=DEVICE)
        
        # Clean forward
        clean_logits = self.run_clean(x_clean)
        clean_pred = clean_logits[0, -1].argmax().item()
        clean_token = sp_.IdToPiece(clean_pred)
        
        print(f"\n=== Activation Patching Elemzés ===")
        print(f"Clean prompt:     '{clean_prompt}'")
        print(f"Corrupted prompt: '{corrupted_prompt}'")
        print(f"Clean prediction: '{clean_token}' (ID {clean_pred})")
        print()
        print(f"{'Réteg':<8} {'Predikció':<15} {'Egyezik?':<10} {'Token'}")
        print("-" * 50)
        
        results = {}
        for li in range(LAYERS):
            patched_logits = self.patch_layer(x_corrupted, li)
            patched_pred = patched_logits[0, -1].argmax().item()
            patched_token = sp_.IdToPiece(patched_pred)
            matches = "✅ IGEN" if patched_pred == clean_pred else "❌ NEM"
            results[li] = (patched_pred == clean_pred)
            print(f"L{li+1:<5} {patched_token:<15} {matches:<10} (ID {patched_pred})")
        
        # Összegzés
        critical_layers = [li for li, r in results.items() if r]
        if critical_layers:
            print(f"\n✅ Kritikus rétegek (itt van a kauzális információ):")
            for li in critical_layers:
                print(f"   L{li+1}")
        else:
            print(f"\n❌ Egyik réteg patching sem adta vissza a clean predikciót!")
            print(f"   → A modellben NINCS olyan mechanizmus ami ezt kezelné")
        
        return results

# ====== 5→2 ATTENTION VIZSGÁLAT ======
def check_attention(model, prompt):
    """Ellenőrzi a számok közötti attention-t minden rétegben"""
    sp_ = spm.SentencePieceProcessor()
    sp_.Load(TOKENIZER_PATH)
    
    ids = sp_.EncodeAsIds(prompt)
    tokens = [sp_.IdToPiece(t) for t in ids]
    
    print(f"\n=== 5→2 Attention Vizsgálat ===")
    print(f"Prompt: '{prompt}'")
    print(f"Tokenek: {tokens}")
    
    x = torch.tensor([ids], device=DEVICE)
    
    # Attention hook
    attn_weights = {}
    handles = []
    
    for li, block in enumerate(model.blocks):
        def make_hook(l):
            def hook(m, i, o):
                x_in = i[0]
                B, T = x_in.shape[:2]
                q = m.wq(x_in).view(B, T, HEADS, DIM//HEADS).transpose(1, 2)
                k = m.wk(x_in).view(B, T, KV_HEADS, DIM//HEADS).transpose(1, 2)
                if HEADS > KV_HEADS:
                    k = k[:, :, None].expand(-1, -1, HEADS//KV_HEADS, -1, -1)
                    k = k.reshape(B, HEADS, T, DIM//HEADS)
                w = (q @ k.transpose(-2, -1)) * ((DIM//HEADS) ** -0.5)
                w = w.masked_fill(m.m[:T, :T] == 0, float('-inf'))
                w = F.softmax(w, dim=-1)
                attn_weights[l] = w.detach().float().cpu()
            return hook
        handle = block.attn.register_forward_hook(make_hook(li))
        handles.append(handle)
    
    with torch.no_grad():
        model(x)
    
    for h in handles:
        h.remove()
    
    # Keressük meg az 5-ös és 2-es token pozíciókat
    pos_5 = None
    pos_2 = None
    for i, t in enumerate(tokens):
        if '5' in t and pos_5 is None:
            pos_5 = i
        if '2' in t and pos_2 is None:
            pos_2 = i
    
    if pos_5 is None or pos_2 is None:
        print("❌ Nem található 5-ös és 2-es token a promptban!")
        return
    
    print(f"\n'5' pozíció: {pos_5}, '2' pozíció: {pos_2}")
    print(f"\n{'Réteg':<8} {'5→2 attn':<12} {'2→5 attn':<12} {'5→alma':<12} {'2→alma':<12}")
    print("-" * 56)
    
    for li in range(LAYERS):
        w = attn_weights[li][0].mean(dim=0)  # [T, T]
        five_to_two = w[pos_5, pos_2].item()
        two_to_five = w[pos_2, pos_5].item()
        # Keressük meg az "alma" token pozíciót
        pos_alma = None
        for i, t in enumerate(tokens):
            if 'alm' in t.lower():
                pos_alma = i
                break
        
        if pos_alma is not None:
            five_to_alma = w[pos_5, pos_alma].item()
            two_to_alma = w[pos_2, pos_alma].item()
        else:
            five_to_alma = two_to_alma = 0
        
        print(f"L{li+1:<5} {five_to_two:<12.4f} {two_to_five:<12.4f} {five_to_alma:<12.4f} {two_to_alma:<12.4f}")


# ====== MAIN ======
if __name__ == '__main__':
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    
    print(f"Modell betöltése...")
    model = LM().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Kész! Paraméterek: {sum(p.numel() for p in model.parameters()):,}")
    
    # 1. 5→2 ATTENTION VIZSGÁLAT
    check_attention(model, "Ha 5 almám van és megeszek 2-t")
    
    # 2. ACTIVATION PATCHING
    patcher = ActivationPatcher(model)
    
    # Teszt 1: "marad" token lokalizáció
    patcher.analyze(
        clean_prompt="Ha 5 almám van és megeszek 2-t, marad 3",
        corrupted_prompt="Ha 5 almám van és megeszek 2-t, az alma"
    )
    
    # Teszt 2: "3-as" szám lokalizáció
    patcher.analyze(
        clean_prompt="5 - 2 = 3",
        corrupted_prompt="5 - 2 = x"
    )
