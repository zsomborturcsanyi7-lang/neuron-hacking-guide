#!/usr/bin/env python3
"""
ROME Edit — Rank-One Model Editing NEURA 300M-re

Egyetlen kivonásos hiba javítása zárt formulával.
CPU-n fut! ~1 perc.

HASZNÁLAT:
  python rome_edit.py --prompt "5 - 2 =" --target "3" --layer 22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import argparse

# ====== KONFIG ======
TOKENIZER_PATH=r'TBD'
CHECKPOINT_PATH=r'TBD'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Architektúra
VOCAB, DIM, LAYERS, FFN_HIDDEN = 32000, 1024, 24, 3072

# ====== MODELL (rövidített) ======
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

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
        self.attn = None  # Nem töltjük be, nem kell
        self.ffn = FFN()

class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = RMSNorm(DIM)
        self.out = nn.Linear(DIM, VOCAB, False)


def rome_edit(model, sp, prompt, target_token_str, layer_idx=22, 
              sample_prompts=None, lambda_reg=0.1):
    """
    ROME edit a NEURA 300M-en.
    
    Args:
        model: betöltött NEURA modell
        sp: SentencePiece tokenizer
        prompt: a prompt (pl. "5 - 2 =")
        target_token_str: a kívánt token (pl. "3")
        layer_idx: melyik réteget szerkesszük (ajánlott: 22)
        sample_prompts: mondatok a C mátrixhoz
        lambda_reg: regularizáció erőssége
    """
    ffn = model.blocks[layer_idx].ffn
    
    # Tokenizálás
    ids = sp.EncodeAsIds(prompt)
    target_id = sp.PieceToId(target_token_str)
    
    tokens = [sp.IdToPiece(t) for t in ids]
    print(f"Prompt tokenek: {tokens}")
    print(f"Cél token: '{target_token_str}' (ID {target_id})")
    
    # 1. KULCS kinyerése (subject hidden state)
    x = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        h = model.tok(x)
        for li in range(layer_idx):  # target LAYER ELŐTT
            block = model.blocks[li]
            h = h + block.ffn(h)
        
        # Az utolsó token hidden state-je = kulcs
        k = h[0, -1].clone()  # [1024]
    
    print(f"Kulcs vektor norm: {k.norm().item():.2f}")
    
    # 2. CÉLÉRTÉK
    target_dir = model.out.weight.data[target_id].clone()  # [1024]
    current = ffn.w2.weight.data @ k  # [1024]
    r = target_dir - current
    
    print(f"Target direction norm: {target_dir.norm().item():.2f}")
    print(f"Current direction norm: {current.norm().item():.2f}")
    print(f"Reziduum norm: {r.norm().item():.2f}")
    
    # 3. C mátrix
    if sample_prompts is None:
        sample_prompts = [
            "Az alma piros és édes",
            "Ma szép napos idő van",
            "A kutya a kertben fut",
            "Budapest Magyarország fővárosa",
            "A gyerekek játszanak az udvaron",
            "Tegnap esett az eső",
            "Holnap megyek iskolába",
            "A macska alszik a kanapén",
            "A nap süt az égen",
            "A fiú eszik egy almát",
            "A lány iszik egy pohár vizet",
            "Az autó gyorsan megy",
            "A könyv az asztalon van",
            "A szék kényelmes és puha",
            "A virágok nyílnak a kertben",
            "Az ég kék és tiszta",
            "A fű zöld és magas",
            "A ház fehér és nagy",
            "Az utca hosszú és egyenes",
            "A fa magas és öreg",
        ]
    
    # K vektorok gyűjtése
    K = []
    for prompt_ex in sample_prompts:
        ex_ids = sp.EncodeAsIds(prompt_ex)
        x_ex = torch.tensor([ex_ids], device=DEVICE)
        with torch.no_grad():
            h_ex = model.tok(x_ex)
            for li in range(layer_idx):
                h_ex = h_ex + model.blocks[li].ffn(h_ex)
            k_ex = h_ex[0, -1].clone()
            K.append(k_ex)
    
    K = torch.stack(K)  # [N, 1024]
    C = K.T @ K + lambda_reg * torch.eye(DIM, device=DEVICE)
    C_inv = torch.linalg.inv(C)
    
    print(f"\nC mátrix: {len(sample_prompts)} mondatból, reguláció={lambda_reg}")
    
    # 4. RANK-ONE UPDATE
    C_inv_k = C_inv @ k
    denominator = k @ C_inv_k
    delta = torch.outer(r, C_inv_k) / denominator
    
    # W2 alakja [1024, 3072], delta.T = [3072, 1024]
    ffn.w2.weight.data += delta.T
    
    print(f"\nDelta norm: {delta.norm().item():.4f}")
    
    # 5. ELLENŐRZÉS
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    probs = F.softmax(logits[0, -1], dim=-1)
    target_prob = probs[target_id].item() * 100
    
    print(f"\n=== EREDMÉNY ===")
    print(f"P('{target_token_str}') = {target_prob:.4f}%")
    
    # Top-5
    top5 = torch.argsort(probs, descending=True)[:5]
    for tid in top5:
        t = sp.IdToPiece(tid.item())
        p = probs[tid].item() * 100
        print(f"  P({t:10s}) = {p:.2f}%")
    
    if target_prob > 1.0:
        print(f"\n✅ MŰKÖDIK! A target token a top választók között van.")
        return True
    else:
        print(f"\n❌ NEM MŰKÖDIK. Próbáld másik réteggel vagy λ-val.")
        return False


# ====== PARAMÉTER TESZTELÉS ======
def test_parameters(model, sp, prompt, target_token_str):
    """Kipróbálja a ROME-ot különböző rétegekkel és λ-kal."""
    
    print(f"\n{'='*60}")
    print(f"ROME paraméter teszt: '{prompt}' → '{target_token_str}'")
    print(f"{'='*60}")
    
    best_config = None
    best_prob = 0
    
    for layer in [20, 21, 22, 23]:
        for lambda_reg in [0.01, 0.05, 0.1, 0.5, 1.0]:
            # Modell reload (mert az előző edit módosította)
            model_reloaded = LM().to(DEVICE)
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
            model_reloaded.load_state_dict(ckpt)
            
            success = rome_edit(model_reloaded, sp, prompt, target_token_str,
                               layer, lambda_reg=lambda_reg)
            
            # Ellenőrizzük az eredményt
            ids = sp.EncodeAsIds(prompt)
            x = torch.tensor([ids], device=DEVICE)
            with torch.no_grad():
                logits = model_reloaded(x)
            probs = F.softmax(logits[0, -1], dim=-1)
            target_id = sp.PieceToId(target_token_str)
            prob = probs[target_id].item() * 100
            
            print(f"  L{layer+1} λ={lambda_reg:.2f} → P={prob:.4f}%")
            
            if prob > best_prob:
                best_prob = prob
                best_config = (layer, lambda_reg)
    
    print(f"\n✅ Legjobb: L{best_config[0]+1} λ={best_config[1]:.2f} → P={best_prob:.4f}%")
    return best_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROME edit NEURA 300M-re')
    parser.add_argument('--prompt', default='5 - 2 =', help='A prompt')
    parser.add_argument('--target', default='3', help='Cél token')
    parser.add_argument('--layer', type=int, default=22, help='Target réteg')
    parser.add_argument('--test', action='store_true', help='Paraméter teszt')
    args = parser.parse_args()
    
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    
    # Token formázás
    target = f"▁{args.target}" if not args.target.startswith("▁") else args.target
    
    model = LM().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    
    if args.test:
        test_parameters(model, sp, args.prompt, target)
    else:
        rome_edit(model, sp, args.prompt, target, layer_idx=args.layer)
