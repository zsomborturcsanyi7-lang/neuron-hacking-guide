# Modifying Neurons in Small Language Models

**Complete Guide from Beginner to Advanced**  
**Model:** NEURA 300M (355M parameters, 24 layers, 1024 dim)  
**Based on experiments:** June–July 2026  
**Author:** Zsombi & Hermes Agent (Nous Research)  
**License:** CC BY 4.0

---

## Description

This project is a **step-by-step guidebook** that demonstrates how to **directly modify neurons** in small language models (such as NEURA 300M). The book guides you through 25 chapters, from fundamentals to the latest research methods, including activation patching, ROME, MEMIT, FiNE, sparse autoencoders, and advanced activation steering.

---

## File Structure

```
neuron_modification_book/
│
├── README.md                              # This document
├── STATUS.md                              # Project status
│
├── combined_book.md                       # The complete book (chapters 1-18)
├── neuron_modification_book.md            # Previous version of the book
│
├── neura_research_paper_01_training_dynamics.md     # 1. Research paper: Training dynamics
├── neura_research_paper_02_neuron_specialization.md # 2. Research paper: Neuron specialization
├── neura_research_paper_03_attention_hierarchy.md   # 3. Research paper: Attention hierarchy
├── neura_research_paper_04_why_cant_reason.md       # 4. Research paper: Why it can't reason
├── neura_research_paper_05_practical_guide.md       # 5. Research paper: Practical guide
│
├── ch19_activation_patching.md            # Chapter 19: Activation Patching
├── ch20_ROME.md                           # Chapter 20: Rank-One Model Editing
├── ch21_MEMIT.md                          # Chapter 21: Mass-Editing Memory
├── ch22_FiNE.md                           # Chapter 22: Fine-grained Neuron Editing
├── ch23_SAE.md                            # Chapter 23: Sparse Autoencoders
├── ch24_steering.md                       # Chapter 24: Activation Steering
├── ch25_update.md                         # Chapter 25: Decision tree + summary
│
├── pulse_adapter_results.md               # Pulse adapter results
│
└── scripts/                               # Experimental scripts
    ├── RUNME.bat                          # Launcher script
    ├── logicadapter_pulse.py              # LogicAdapter Pulse experiment
    ├── logicadapter_v6.py                 # LogicAdapter v6
    ├── activation_patching.py             # Activation patching experiment
    ├── rome_edit.py                       # ROME edit experiment
    └── tokenize_opensubs.py               # OpenSubtitles tokenization
```

---

## Book Contents

### Part 1: Fundamentals
| Chapter | Topic |
|---------|------|
| 1 | Introduction — What is neuron modification? |
| 2 | The Structure of a Transformer Neuron |
| 3 | Environment Setup |
| 4 | Your First Neuron Modification |

### Part 2: Methods
| Chapter | Topic |
|---------|------|
| 5-18 | Intermediate chapters (combined_book.md) |
| 19 | **Activation Patching & Causal Tracing** |
| 20 | **ROME** — Rank-One Model Editing |
| 21 | **MEMIT** — Mass-Editing Memory |
| 22 | **FiNE** — Neuron-level Knowledge Editing (ICLR 2025) |
| 23 | **Sparse Autoencoders** — Disentangling polysemantic neurons |
| 24 | **Advanced Activation Steering** |
| 25 | **Updated Decision Tree** — Comparison of 6+ methods |

---

## Usage

### Running Scripts

```bash
cd scripts

# Activation patching
python activation_patching.py

# ROME edit
python rome_edit.py

# LogicAdapter experiments
python logicadapter_pulse.py
python logicadapter_v6.py
```

### Reading the Book

The book is in Markdown format and can be opened with any Markdown reader:

```bash
# VS Code
code combined_book.md

# In browser
start combined_book.md
```

---

## Why Modify Neurons?

| Traditional Method | Neuron Modification |
|---------------------|------------------|
| Full fine-tuning (hours, heavy GPU) | Rewriting one neuron (seconds, CPU) |
| All parameters change | Only 1-2 neurons change |
| Risk of catastrophic forgetting | Targeted, other knowledge intact |
| Large dataset required (~10K+ examples) | 1-100 examples sufficient |

---

## Developer

Zsombi & Hermes Agent (Nous Research)
