#!/usr/bin/env python3
"""
LogicAdapter a Pulse 350M-re (lokális gép)
==========================================
Ugyanaz a koncepció, mint a NEURA 300M-en.
A Pulse 350M egy magyar nyelvi modell (FusedQKV MHA + SwiGLU FFN).
A LogicAdapter a residual stream végén dolgozik → architektúra-független.

Futtatás:
  cd C:/Users/iga/Desktop/forge_chat
  python ../neuron_modification_book/scripts/logicadapter_pulse.py

Ido: ~7-15 perc GPU-n (RTX 30xx)
Eredmeny: varhatoan 70%+ digit accuracy
"""

import sys, os, time, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

# ====== FORGE CHAT BACKEND ======
sys.path.insert(0, 'C:/Users/iga/Desktop/forge_chat')
from models.forge_model import create_backend, ForgeModel

# ====== KONFIG ======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ADAPTER_HIDDEN = 128      # 64 helyett 128
EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-3
PROMPTS_PER_DIGIT = 50    # 12 helyett 50 → 450+ példa
LOG_FILE = 'C:/Users/iga/Desktop/neuron_modification_book/scripts/pulse_adapter_log.txt'

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    try:
        print(msg)
    except:
        pass

log("=== LogicAdapter Pulse 350M ===")
log(f"Device: {DEVICE}")
log(f"Adapter hidden: {ADAPTER_HIDDEN}")

# ====== 1. MODELL BETÖLTÉS ======
log("\nModell betöltése...")
backend = create_backend()
backend.load_tokenizer()
backend.load_model()
model = backend.model.to(DEVICE)
sp = backend.tokenizer

log(f"Modell: {sum(p.numel() for p in model.parameters()):,} paraméter")
log(f"Tokenizer: {sp.GetPieceSize()} szókészlet")

# ====== 2. LOGICADAPTER ======
class LogicAdapter(nn.Module):
    """1024 → 128 → 1024, zero-init"""
    def __init__(self, dim=1024, hidden=128):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden, bias=False)
        self.decoder = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.decoder.weight)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
    def forward(self, x):
        return self.decoder(F.relu(self.encoder(x)))

class PulseWithAdapter(nn.Module):
    """Pulse 350M + LogicAdapter wrapper"""
    def __init__(self, base_model):
        super().__init__()
        self.embed = base_model.embed
        self.layers = base_model.layers
        self.norm = base_model.norm
        self.lm_head = base_model.lm_head
        self.logic = LogicAdapter(dim=1024, hidden=ADAPTER_HIDDEN)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = x + self.logic(x)  # ← Adapter itt!
        x = self.norm(x)
        return self.lm_head(x)

model_with_adapter = PulseWithAdapter(model).to(DEVICE)
model_with_adapter.eval()

# ====== 3. ADAT GENERÁLÁS ======
DIGIT_NAMES = [str(d) for d in range(1, 10)]
DIGIT_IDS = torch.tensor([sp.PieceToId(d) for d in DIGIT_NAMES], device=DEVICE)
log(f"Digit token IDs: {dict(zip(DIGIT_NAMES, DIGIT_IDS.tolist()))}")

PROMPT_TEMPLATES = [
    "{a} - {b} =", "{a} minusz {b} az",
    "{a} almabol {b}-et megeszek, marad",
    "{a} almabol {b}-t megeszek, marad",
    "Van {a} almam, megeszek {b}-et,",
    "{a} tole {b}-vel kevesebb, az",
    "{a} es {b} kulonbozete",
    "{a} kavibol {b}-et elajandekozok, marad",
    "mennyi {a} - {b}?", "eredmeny: {a} - {b} =",
    "Ha {a} kavim van es {b}-t elkoltek,",
    "{a} almabol elvittem {b}-t, hat",
    "{a} - {b} = ?", "Szamitsd ki: {a} - {b}",
    "{a} minus {b} egyenlo",
    "Mi az eredmeny: {a} - {b}?",
    "{a} darabbol {b} eltunik, marad",
]

def build_dataset(prompts_per_digit=50):
    all_examples = []
    for a in range(2, 51):
        for b in range(1, min(a, 10)):
            result = a - b
            if result < 1 or result > 9:
                continue
            for template in PROMPT_TEMPLATES:
                all_examples.append((template.format(a=a, b=b), str(result)))
    
    balanced = []
    for digit in DIGIT_NAMES:
        dex = [(p, a) for p, a in all_examples if a == digit]
        random.shuffle(dex)
        balanced.extend(dex[:prompts_per_digit])
    
    log(f"\nAdat: {len(balanced)} példa")
    log(f"Eloszlás: {dict(sorted(Counter(a for _,a in balanced).items()))}")
    return balanced

data = build_dataset(PROMPTS_PER_DIGIT)
N = len(data)

input_ids_list = []
target_digit_idx = []
for prompt, answer in data:
    input_ids_list.append(sp.EncodeAsIds(prompt))
    target_digit_idx.append(DIGIT_NAMES.index(answer))

# ====== 4. TRAINING ======
model_with_adapter.train()
for p in model_with_adapter.parameters():
    p.requires_grad = False
for p in model_with_adapter.logic.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(model_with_adapter.logic.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

adapter_params = sum(p.numel() for p in model_with_adapter.logic.parameters())
log(f"Adapter paraméterek: {adapter_params:,}")
log(f"Epochok: {EPOCHS}, Batch: {BATCH_SIZE}")

start_time = time.time()
best_acc = 0

for epoch in range(EPOCHS):
    model_with_adapter.train()
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
            x = torch.tensor([input_ids_list[idx]], dtype=torch.long, device=DEVICE)
            logits = model_with_adapter(x)
            last_logits = logits[0, -1]
            
            target_id = sp.PieceToId(DIGIT_NAMES[target_digit_idx[idx]])
            
            # Combined loss
            digit_logits = last_logits[DIGIT_IDS]
            target_digit = torch.tensor([target_digit_idx[idx]], device=DEVICE)
            loss_selective = F.cross_entropy(digit_logits.unsqueeze(0), target_digit)
            loss_full = F.cross_entropy(last_logits.unsqueeze(0), 
                                         torch.tensor([target_id], device=DEVICE))
            loss = loss_selective + 0.5 * loss_full
            batch_loss = batch_loss + loss
            
            if torch.argmax(digit_logits).item() == target_digit_idx[idx]:
                correct += 1
            total += 1
        
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_with_adapter.logic.parameters(), 1.0)
        optimizer.step()
        total_loss += batch_loss.item()
    
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        avg_loss = total_loss / max(1, (N + BATCH_SIZE - 1) // BATCH_SIZE)
        acc = correct / total * 100
        elapsed = time.time() - start_time
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model_with_adapter.logic.state_dict(),
                       r'C:\Users\iga\Desktop\neuron_modification_book\scripts\pulse_adapter_best.pt')
        
        log(f"Epoch {epoch:3d}: loss={avg_loss:.4f} acc={acc:.1f}% best={best_acc:.1f}% time={elapsed:.0f}s")

log(f"\n=== KÉSZ! ===")
log(f"Best accuracy: {best_acc:.1f}%")
log(f"Idő: {time.time() - start_time:.0f}s")

# ====== 5. TESZT ======
log("\n=== TESZT ===")
test_prompts = [
    ("5 - 2 =", "3"), ("8 - 5 =", "3"), ("2 - 1 =", "1"),
    ("5 - 3 =", "2"), ("10 - 1 =", "9"), ("6 - 1 =", "5"),
    ("7 - 4 =", "3"), ("9 - 2 =", "7"), ("4 - 2 =", "2"),
    ("7 - 1 =", "6"), ("3 - 1 =", "2"), ("10 - 3 =", "7"),
    ("5 - 2 = ?", "3"), ("mennyi 8 - 5?", "3"),
    ("Szamitsd ki: 6 - 1", "5"), ("10 minusz 1 az", "9"),
    ("3 almabol 1-et megeszek, marad", "2"),
    ("Mennyi 7 - 3?", "4"), ("eredmeny: 9 - 5 =", "4"),
    ("8 almabol 3-at megeszek, marad", "5"),
]

model_with_adapter.eval()
correct_tests = 0
with torch.no_grad():
    for prompt, expected in test_prompts:
        x = torch.tensor([sp.EncodeAsIds(prompt)], dtype=torch.long, device=DEVICE)
        logits = model_with_adapter(x)
        digit_logits = logits[0, -1, DIGIT_IDS]
        pred_digit = DIGIT_NAMES[torch.argmax(digit_logits).item()]
        
        mark = "✅" if pred_digit == expected else "❌"
        if pred_digit == expected:
            correct_tests += 1
        log(f"{mark} '{prompt}' → {pred_digit} (expected {expected})")

log(f"\nTest accuracy: {correct_tests}/{len(test_prompts)} = {correct_tests/len(test_prompts)*100:.1f}%")
log(f"Best adapter: pulse_adapter_best.pt")
log(f"\nKövetkező lépés: másold át a NEURA 300M checkpoint-hoz, és futtasd ugyanezt!")
