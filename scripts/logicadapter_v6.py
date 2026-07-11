#!/usr/bin/env python3
"""
LogicAdapter v6 — KIBŐVÍTETT training 500+ példán

NEURA 300M továbbfejlesztése: kivonásos feladatok 500 példán,
128 hidden dim, combined loss.

HASZNÁLAT (amikor a remote GPU be van kapcsolva):
  1. Másold a scriptet a remote-ra
  2. Telepítsd: pip install torch sentencepiece
  3. Futtasd: python logicadapter_v6.py

Kimenet: logicadapter_v6_checkpoint.pt (csak az adapter súlyai)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import random, time, os, json
from collections import Counter

# ====== BEÁLLÍTÁSOK ======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ADAPTER_HIDDEN = 128      # 64 → 128 (több kapacitás)
EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-3
LOG_FILE = 'logicadapter_v6_log.txt'

# Tokenizer elérési útja
TOKENIZER_PATH = r'C:\Users\neura\NeuraNode\bitnet\data\bitnet_pretrain\tokenizer\tokenizer.model'
OUT_DIR = r'C:\Users\neura'

print(f"=== LogicAdapter v6 ===")
print(f"Device: {DEVICE}")
print(f"Adapter hidden: {ADAPTER_HIDDEN}")
print(f"Epochs: {EPOCHS}")

# ====== TOKENIZER ======
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)

# Digit token ID-k ellenőrzése
DIGIT_NAMES = [str(d) for d in range(1, 10)]
DIGIT_IDS = torch.tensor([sp.PieceToId(d) for d in DIGIT_NAMES], device=DEVICE)
print(f"Digit token IDs: {dict(zip(DIGIT_NAMES, DIGIT_IDS.tolist()))}")

# ====== 500+ PÉLDA GENERÁLÁSA ======
PROMPT_TEMPLATES = [
    # Alap (5 féle)
    "{a} - {b} =",
    "{a} minusz {b} az",
    "{a} almabol {b}-et megeszek, marad",
    "{a} almabol {b}-t megeszek, marad",
    "Van {a} almam, megeszek {b}-et,",
    # Változatok (10+ féle)
    "{a} tole {b}-vel kevesebb, az",
    "{a} es {b} kulonbozete",
    "{a} kavibol {b}-et elajandekozok, marad",
    "mennyi {a} - {b}?",
    "eredmeny: {a} - {b} =",
    "Ha {a} kavim van es {b}-t elkoltek,",
    "{a} almabol elvittem {b}-t, hat",
    "{a} - {b} = ?",
    "Szamitsd ki: {a} - {b}",
    "{a} minus {b} egyenlo",
    "Mi az eredmeny: {a} - {b}?",
    "{a} darabbol {b} eltunik, marad",
]

def build_large_dataset(prompts_per_digit=50):
    """50 példa / számjegy → 450 total + 50 extra = 500"""
    all_examples = []
    
    for a in range(2, 51):  # 2-től 50-ig
        for b in range(1, min(a, 10)):  # 1-től 9-ig (de < a)
            result = a - b
            if result < 1 or result > 9:
                continue
            
            # Minden eredményhez több prompt variáció
            for template in PROMPT_TEMPLATES:
                prompt = template.format(a=a, b=b)
                all_examples.append((prompt, str(result)))
    
    # Balance: pontosan prompts_per_digit / számjegy
    balanced = []
    counts = Counter(a for _, a in all_examples)
    
    for digit in DIGIT_NAMES:
        dex = [(p, a) for p, a in all_examples if a == digit]
        random.shuffle(dex)
        balanced.extend(dex[:prompts_per_digit])
    
    print(f"\nAdatkészlet: {len(balanced)} példa")
    print(f"Eloszlás: {dict(sorted(Counter(a for _, a in balanced).items()))}")
    
    return balanced

data = build_large_dataset(prompts_per_digit=50)
N = len(data)

# Tokenizálás
input_ids_list = []
target_digit_idx = []
for prompt, answer in data:
    input_ids_list.append(sp.EncodeAsIds(prompt))
    target_digit_idx.append(DIGIT_NAMES.index(answer))

# ====== MODELL ARCHITEKTÚRA (NEURA 300M) ======
VOCAB = 32000
DIM = 1024
LAYERS = 24
HEADS = 16
KV_HEADS = 4
FFN_HIDDEN = 3072

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class GQA(nn.Module):
    def __init__(self, dim, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, dim // nh
        self.wq = nn.Linear(dim, nh * self.hd, False)
        self.wk = nn.Linear(dim, nkv * self.hd, False)
        self.wv = nn.Linear(dim, nkv * self.hd, False)
        self.wo = nn.Linear(nh * self.hd, dim, False)
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
    def __init__(self, dim, h):
        super().__init__()
        self.w1 = nn.Linear(dim, h, False)
        self.w2 = nn.Linear(h, dim, False)
        self.w3 = nn.Linear(dim, h, False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
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

class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList([Block(DIM, HEADS, KV_HEADS, FFN_HIDDEN) for _ in range(LAYERS)])
        self.ln_f = RMSNorm(DIM)
        self.out = nn.Linear(DIM, VOCAB, False)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks:
            x = b(x)
        return self.out(self.ln_f(x))

# ====== LOGICADAPTER ======
class LogicAdapter(nn.Module):
    """1024 → 128 → 1024, zero-init"""
    def __init__(self, dim=1024, hidden=128):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden, bias=False)
        self.decoder = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.decoder.weight)  # CRITICAL: zero-init
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
    def forward(self, x):
        return self.decoder(F.relu(self.encoder(x)))

class NEURAWithAdapter(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.tok = base_model.tok
        self.blocks = base_model.blocks
        self.ln_f = base_model.ln_f
        self.out = base_model.out
        self.logic = LogicAdapter(dim=1024, hidden=ADAPTER_HIDDEN)
    def forward(self, x):
        x = self.tok(x)
        for b in self.blocks:
            x = b(x)
        x = x + self.logic(x)
        return self.out(self.ln_f(x))

# ====== MODELL BETÖLTÉS ======
print("\nModell betöltése...")
device = DEVICE

# Checkpoint keresés
checkpoint_paths = [
    r'C:\Users\neura\lm300m_v3_step390000.pt',
    r'C:\Users\neura\Desktop\forge_chat\models\lm300m_v3_step390000.pt',
]
ckpt_path = None
for p in checkpoint_paths:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path:
    print(f"Checkpoint: {ckpt_path}")
    base_model = LM().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    base_model.load_state_dict(ckpt.get('model_state', ckpt))
    print(f"Paraméterek: {sum(p.numel() for p in base_model.parameters()):,}")
else:
    print("❌ Nincs checkpoint! Hozz létre egy üres modellt teszteléshez.")
    base_model = LM().to(device)

model = NEURAWithAdapter(base_model).to(device)
model.eval()

# ====== TRAINING ======
model.train()
for p in model.parameters():
    p.requires_grad = False
for p in model.logic.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(model.logic.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\nAdapter paraméterek: {sum(p.numel() for p in model.logic.parameters()):,}")
print(f"Adat: {N} példa, {EPOCHS} epoch")
print(f"Várható idő GPU-n: ~{N * EPOCHS / 1000 * 0.4:.0f} másodperc")

def log(msg):
    with open(os.path.join(OUT_DIR, LOG_FILE), 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    try:
        print(msg)
    except:
        pass

start_time = time.time()
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    indices = list(range(N))
    random.shuffle(indices)
    
    for i in range(0, N, BATCH_SIZE):
        batch = indices[i:i + BATCH_SIZE]
        optimizer.zero_grad()
        
        batch_loss = 0.0
        for idx in batch:
            x = torch.tensor([input_ids_list[idx]], dtype=torch.long, device=device)
            logits = model(x)
            last_logits = logits[0, -1]
            
            # Digit ID a target tokenhez (FULL vocab, nem selective!)
            target_token_id = sp.PieceToId(f"▁{data[idx][1]}")
            
            # COMBINED LOSS: selective + full vocab
            digit_logits = last_logits[DIGIT_IDS]
            target_digit = torch.tensor([target_digit_idx[idx]], device=device)
            loss_selective = F.cross_entropy(digit_logits.unsqueeze(0), target_digit)
            
            loss_full = F.cross_entropy(
                last_logits.unsqueeze(0), 
                torch.tensor([target_token_id], device=device)
            )
            
            loss = loss_selective + 0.5 * loss_full
            batch_loss = batch_loss + loss
            
            if torch.argmax(digit_logits).item() == target_digit_idx[idx]:
                correct += 1
            total += 1
        
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.logic.parameters(), 1.0)
        optimizer.step()
        total_loss += batch_loss.item()
    
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        avg_loss = total_loss / max(1, (N + BATCH_SIZE - 1) // BATCH_SIZE)
        acc = correct / total * 100
        elapsed = time.time() - start_time
        
        if acc > best_acc:
            best_acc = acc
            # Mentés
            torch.save(model.logic.state_dict(), 
                       os.path.join(OUT_DIR, 'logicadapter_v6_best.pt'))
        
        log(f"Epoch {epoch:3d}: loss={avg_loss:.4f} acc={acc:.1f}% best={best_acc:.1f}% time={elapsed:.0f}s")

log(f"\n=== KÉSZ! ===")
log(f"Best accuracy: {best_acc:.1f}%")
log(f"Idő: {time.time() - start_time:.0f}s")

# ====== TESZT ======
log("\n=== TESZT ===")
test_prompts = [
    ("5 - 2 =", "3"),
    ("8 - 5 =", "3"),
    ("2 - 1 =", "1"),
    ("5 - 3 =", "2"),
    ("10 - 1 =", "9"),
    ("6 - 1 =", "5"),
    ("7 - 4 =", "3"),
    ("9 - 2 =", "7"),
    ("4 - 2 =", "2"),
    ("7 - 1 =", "6"),
    ("3 - 1 =", "2"),
    ("10 - 3 =", "7"),
    # ÚJ, nem látott formátumok
    ("5 - 2 = ?", "3"),
    ("mennyi 8 - 5?", "3"),
    ("Szamitsd ki: 6 - 1", "5"),
    ("10 minusz 1 az", "9"),
    ("3 almabol 1-et megeszek, marad", "2"),
]

model.eval()
correct_tests = 0
with torch.no_grad():
    for prompt, expected in test_prompts:
        x = torch.tensor([sp.EncodeAsIds(prompt)], dtype=torch.long, device=device)
        logits = model(x)
        digit_logits = logits[0, -1, DIGIT_IDS]
        pred_digit_idx = torch.argmax(digit_logits).item()
        pred_digit = DIGIT_NAMES[pred_digit_idx]
        
        # Full vocab predikció is
        full_pred_id = logits[0, -1].argmax().item()
        full_pred = sp.IdToPiece(full_pred_id)
        
        mark = "✅" if pred_digit == expected else "❌"
        if pred_digit == expected:
            correct_tests += 1
        log(f"{mark} '{prompt}' → digit={pred_digit} (expected {expected}) full='{full_pred}'")

log(f"\nTest accuracy: {correct_tests}/{len(test_prompts)} = {correct_tests/len(test_prompts)*100:.1f}%")
log(f"Best adapter mentve: logicadapter_v6_best.pt")
