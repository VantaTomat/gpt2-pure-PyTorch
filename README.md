# GPT-2 without Transformers (Pure PyTorch)

This repository contains a **minimal GPT-2 inference implementation written directly in PyTorch**, without using HuggingFace Transformers, safetensors, or high-level model wrappers.

The goal of this project is **not production usage**, but exploration, learning, and solving a very specific constraint:
> *Running GPT-2 style models when standard tooling is unavailable or undesirable.*

---

## Why this exists

Most GPT-2 examples rely heavily on `transformers`, which hides a lot of the underlying mechanics.
This script intentionally avoids that and instead:

- Loads raw PyTorch checkpoints (`.pt` / `.bin`)
- Implements a **byte-level BPE tokenizer** (GPT-2 style)
- Implements attention, MLP, layer norm, and causal masking manually
- Handles **messy / inconsistent checkpoint key names** with robust fallback logic

Some of the solutions here are **not ideal**, and that’s intentional.
They solve a *specific* problem under *specific constraints*.

---

## What this is
- Educational
- Experimental
- Debug-friendly
- Useful when you want to understand or bypass higher-level tooling

## What this is NOT
- Production-ready
- Optimized
- Feature-complete
- Intended to replace Transformers

---

## Requirements
- Python 3.x
- PyTorch
- `regex`

---

## Folder structure expected

```text
model_folder/
├─ config.json
├─ pytorch_model.pt   (or pytorch_model.bin)
├─ vocab.json
└─ merges.txt
```

---

## Usage

```text
python run_no_transformers.py path/to/model_folder
```
----

You’ll get an interactive prompt in the terminal.

Type quit to exit.

---

## Notes

CPU-only by default

Designed for GPT-2–style checkpoints

Key matching is intentionally flexible to support differently-named state_dicts

---

## Disclaimer

This project exists to solve a specific technical constraint, not to demonstrate best practices.
Some design choices are deliberately pragmatic rather than clean.

If you’re looking for production inference: use Transformers.
If you want to see how things actually work under the hood: this repo may be useful.
