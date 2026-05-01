# GPT-2 without Transformers (Pure PyTorch)

This repository contains a **minimal GPT-2 inference implementation written directly in PyTorch**, without using HuggingFace Transformers, safetensors, or high-level model wrappers.

---

## Why i made this script

Most GPT-2 examples rely heavily on the `transformers` library, which hides a lot of the underlying mechanics. This project started when I wanted to run a GPT-2 model locally but couldn't install Transformers on my outdated system. Rather than give up, I built a pure PyTorch inference script from scratch.

The script intentionally avoids standard tooling and instead:
- Loads raw PyTorch checkpoints (`.pt` / `.bin`)
- Implements a **byte-level BPE tokenizer** (GPT-2 style)
- Implements attention, MLP, layer norm, and causal masking manually
- Handles **messy / inconsistent checkpoint key names** with robust fallback logic

It was a deeply educational experience, and I'm sharing it here in case others find it useful or interesting.

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

This is not intended to replace Transformers.
If you’re looking for production inference: use Transformers.
If you want to see how things actually work under the hood: this repo may be useful.
