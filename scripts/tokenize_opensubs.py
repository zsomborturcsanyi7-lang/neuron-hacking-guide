#!/usr/bin/env python3
"""
OpenSubtitles HU tokenizálás NEURA tokenizerrel

1. Kicsomagolja a .gz fájlt
2. Tokenizálja a szöveget a NEURA SentencePiece tokenizerrel
3. Mentés shard-okba (512 token/szekvencia, 50K seq/shard)

HASZNÁLAT:
  python tokenize_opensubs.py --input opensubtitles_hu.txt.gz --output ./subs_tokenized
  
CPU-N fut! ~10-15 perc az 1.4B tokenhez.
"""

import gzip
import torch
import sentencepiece as spm
import os, argparse, time
from collections import Counter

SEQ_LEN = 512
SHARD_SIZE = 50000  # sequences per shard


def tokenize_opensubs(input_path, output_dir, tokenizer_path):
    """
    OpenSubtitles HU tokenizálása NEURA tokenizerrel.
    
    Args:
        input_path: opensubtitles_hu.txt.gz vagy .txt
        output_dir: hova mentsük a shard-okat
        tokenizer_path: a NEURA SentencePiece .model fájl
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tokenizer betöltése
    print(f"Tokenizer betöltése: {tokenizer_path}")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    vocab_size = sp.GetPieceSize()
    print(f"Szókészlet mérete: {vocab_size}")
    
    # Fájl megnyitása
    if input_path.endswith('.gz'):
        print(f"Tömörített fájl megnyitása: {input_path}")
        f = gzip.open(input_path, 'rt', encoding='utf-8', errors='replace')
    else:
        print(f"Szöveges fájl megnyitása: {input_path}")
        f = open(input_path, 'r', encoding='utf-8', errors='replace')
    
    # Tokenizálás stream-ben
    buffer = []
    shard = []
    shard_idx = 0
    total_tokens = 0
    total_seqs = 0
    skipped = 0
    start_time = time.time()
    
    print(f"\nTokenizálás {SEQ_LEN} token/szekvencia...")
    
    for line_idx, line in enumerate(f):
        text = line.strip()
        if len(text) < 10:  # Túl rövid sorok kihagyása
            skipped += 1
            continue
        
        # Tokenizálás
        ids = sp.EncodeAsIds(text)
        
        # Túl hosszú sorok eldobása (valószínűleg zaj)
        if len(ids) > 1000:
            skipped += 1
            continue
        
        buffer.extend(ids)
        
        # Buffer feltöltése 512 tokenes szekvenciákba
        while len(buffer) >= SEQ_LEN:
            shard.append(buffer[:SEQ_LEN])
            buffer = buffer[SEQ_LEN:]
            total_seqs += 1
            total_tokens += SEQ_LEN
            
            # Shard mentése
            if len(shard) >= SHARD_SIZE:
                out_path = os.path.join(output_dir, f'subs_shard_{shard_idx}.pt')
                torch.save(torch.tensor(shard, dtype=torch.int32), out_path)
                elapsed = time.time() - start_time
                tok_s = total_tokens / max(elapsed, 1)
                print(f"  Shard {shard_idx}: {len(shard)} seq → {out_path} "
                      f"({total_tokens:,} token, {tok_s:,.0f} tok/s)")
                shard_idx += 1
                shard = []
        
        if line_idx % 100000 == 0 and line_idx > 0:
            elapsed = time.time() - start_time
            tok_s = total_tokens / max(elapsed, 1)
            print(f"  Progress: {line_idx:,} sor, {total_seqs:,} seq, "
                  f"{total_tokens:,} token, {tok_s:,.0f} tok/s")
    
    f.close()
    
    # Utolsó shard mentése (padding)
    if shard:
        # Padding az utolsó szekvenciához
        while len(shard) > 0 and len(shard[-1]) < SEQ_LEN:
            shard[-1] = shard[-1] + [0] * (SEQ_LEN - len(shard[-1]))
        
        out_path = os.path.join(output_dir, f'subs_shard_{shard_idx}.pt')
        torch.save(torch.tensor(shard, dtype=torch.int32), out_path)
        print(f"  Shard {shard_idx} (utolsó): {len(shard)} seq → {out_path}")
        shard_idx += 1
    
    # Összegzés
    elapsed = time.time() - start_time
    print(f"\n=== KÉSZ! ===")
    print(f"Idő: {elapsed:.0f}s ({elapsed/60:.1f} perc)")
    print(f"Összes token: {total_tokens:,}")
    print(f"Összes szekvencia: {total_seqs:,}")
    print(f"Shard-ok: {shard_idx}")
    print(f"Kihagyott sorok: {skipped}")
    print(f"Átlagos sebesség: {total_tokens/elapsed:,.0f} tok/s")
    
    return shard_idx


def verify_shards(output_dir, num_shards):
    """Shard-ok ellenőrzése"""
    print(f"\n=== Shard-ok Ellenőrzése ===")
    total = 0
    for i in range(num_shards):
        path = os.path.join(output_dir, f'subs_shard_{i}.pt')
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu', weights_only=True)
            total += len(data)
            print(f"  subs_shard_{i}.pt: {data.shape}")
        else:
            print(f"  ❌ Hiányzik: subs_shard_{i}.pt")
    
    print(f"Összes szekvencia: {total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenSubtitles HU tokenizálás')
    parser.add_argument('--input', default=r'C:\Users\iga\Desktop\opensubtitles_hu.txt.gz',
                       help='Bemeneti .txt.gz fájl')
    parser.add_argument('--output', default=r'C:\Users\iga\Desktop\neuron_modification_book\data\subs_tokenized',
                       help='Kimeneti mappa')
    parser.add_argument('--tokenizer', 
                       default=r'C:\Users\iga\Desktop\MicroLanguageSwarm\data\bitnet_pretrain\tokenizer\tokenizer.model',
                       help='NEURA tokenizer .model fájl')
    parser.add_argument('--verify', action='store_true',
                       help='Csak ellenőrizze a shard-okat')
    args = parser.parse_args()
    
    if args.verify:
        # Shard-ok számának detektálása
        shards = [f for f in os.listdir(args.output) if f.endswith('.pt')]
        verify_shards(args.output, len(shards))
    else:
        tokenize_opensubs(args.input, args.output, args.tokenizer)
