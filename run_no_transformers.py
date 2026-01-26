# run_no_transformers.py
# Pure PyTorch GPT-2 loader + byte-level BPE tokenizer (no transformers, no safetensors)
# Usage:
# python run_no_transformers.py "C:\Models\GPT2_pt"

import sys
import os
import json
import math
import regex as re
import torch
import torch.nn.functional as F
from collections import defaultdict

# ---------------------------
# Tokenizer (byte-level BPE, GPT-2 style)
# ---------------------------
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

_b2u = bytes_to_unicode()
_u2b = {v:k for k,v in _b2u.items()}

# regex pattern identical to GPT-2 tokenizer
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^ \s\p{L}\p{N}]+""", re.UNICODE)

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class BPETokenizer:
    def __init__(self, vocab_path, merges_path):
        # load vocab.json (token->id)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)  # token->id
        # invert
        self.decoder = {v:k for k,v in self.encoder.items()}
        # load merges
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        # skip first line if it contains header
        for i, line in enumerate(lines):
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    merges.append((parts[0], parts[1]))
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.cache = {}

    def bpe(self, token):
        # token is a string of unicode chars representing bytes mapping
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        bpe_ranks = self.bpe_ranks
        while True:
            min_pair = None
            min_rank = None
            for pair in pairs:
                rank = bpe_ranks.get(pair)
                if rank is not None:
                    if min_rank is None or rank < min_rank:
                        min_rank = rank
                        min_pair = pair
            if min_pair is None:
                break
            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(PAT, text):
            # convert token to bytes then to unicode mapping
            token_bytes = token.encode("utf-8")
            token_trans = "".join(_b2u[b] for b in token_bytes)
            bpe_tok = self.bpe(token_trans)
            for piece in bpe_tok.split(" "):
                if piece in self.encoder:
                    bpe_tokens.append(self.encoder[piece])
                else:
                    # fallback: try to look up piece as-is
                    bpe_tokens.append(self.encoder.get(piece, self.encoder.get("<|endoftext|>", 50256)))
        return bpe_tokens

    def decode(self, token_ids):
        tokens = [self.decoder[t] for t in token_ids]
        text = "".join(tokens)
        # convert back bytes
        byte_vals = [ord(ch) for ch in text]
        b = bytes([_u2b.get(ch, ord(ch)) if ch in _u2b else ord(ch) for ch in text])
        try:
            return b.decode("utf-8", errors="replace")
        except:
            # fallback
            s = ""
            for ch in tokens:
                for c in ch:
                    bval = _u2b.get(c, ord(c))
                    s += bytes([bval]).decode("utf-8", errors="replace")
            return s

# ---------------------------
# Simple direct GPT-2 implementation reading weights from state_dict
# ---------------------------
class DirectGPT2:
    def __init__(self, state_dict, config, device="cpu"):
        self.device = device
        # keep tensors on device
        self.sd = {k: v.to(device) for k, v in state_dict.items()}

        # config fields
        self.n_embd = config["n_embd"]
        self.n_head = config["n_head"]
        self.n_layer = config["n_layer"]
        self.n_ctx = config.get("n_ctx", config.get("n_positions", 1024))
        self.head_dim = self.n_embd // self.n_head
        self.eps = config.get("layer_norm_epsilon", 1e-5)

        # helper to robustly find a key by suffix (or exact)
        def find_key_by_suffix(suffixes):
            """suffixes: list of suffix strings (try in order). Return tensor or None."""
            if isinstance(suffixes, str):
                suffixes = [suffixes]
            for suf in suffixes:
                # prefer exact match first
                if suf in self.sd:
                    return self.sd[suf]
                # try ending-with match
                for k in self.sd:
                    if k.endswith(suf):
                        return self.sd[k]
            return None

        # token embedding
        self.wte = find_key_by_suffix(["transformer.wte.weight", "wte.weight", "model.wte.weight", "embeddings.word_embeddings.weight", "tok_embeddings.weight"])
        if self.wte is None:
            raise RuntimeError("token embedding 'wte' not found in state_dict (searched common names). Keys sample: " + ", ".join(list(self.sd.keys())[:20]))

        # positional embeddings
        self.wpe = find_key_by_suffix(["transformer.wpe.weight", "wpe.weight", "model.wpe.weight", "pos_emb.weight", "position_embeddings.weight"])
        if self.wpe is None:
            # allow missing wpe (some checkpoints use rotary / different scheme)
            print("Warning: positional embeddings not found — using zeros (model might use rotary embeddings).")
            self.wpe = torch.zeros((self.n_ctx, self.n_embd), dtype=self.wte.dtype, device=device)

        # final ln
        self.ln_f_w = find_key_by_suffix(["transformer.ln_f.weight", "ln_f.weight", "model.ln_f.weight", "ln_f.weight"])
        self.ln_f_b = find_key_by_suffix(["transformer.ln_f.bias", "ln_f.bias", "model.ln_f.bias", "ln_f.bias"])
        if self.ln_f_w is None or self.ln_f_b is None:
            raise RuntimeError("final layernorm weights 'ln_f' not found")

        # lm head (optional)
        self.lm_head = find_key_by_suffix(["lm_head.weight", "model.lm_head.weight", "lm_head.weight", "output_embedding.weight"])

        # per-layer weights collect (robust lookup)
        self.layers = []
        missing_keys = []
        for i in range(self.n_layer):
            # candidate suffixes for each tensor (we search by suffix)
            suffix_base = f".h.{i}."
            # common patterns to try: transformer.h.i.*, model.h.i.*, h.i.* , blocks.i.*
            candidates = {
                "c_attn_w": [f"transformer.h.{i}.attn.c_attn.weight", f"model.h.{i}.attn.c_attn.weight", f"h.{i}.attn.c_attn.weight", f"blocks.{i}.attn.c_attn.weight"],
                "c_attn_b": [f"transformer.h.{i}.attn.c_attn.bias", f"model.h.{i}.attn.c_attn.bias", f"h.{i}.attn.c_attn.bias"],
                "c_proj_w": [f"transformer.h.{i}.attn.c_proj.weight", f"model.h.{i}.attn.c_proj.weight", f"h.{i}.attn.c_proj.weight"],
                "c_proj_b": [f"transformer.h.{i}.attn.c_proj.bias", f"model.h.{i}.attn.c_proj.bias", f"h.{i}.attn.c_proj.bias"],
                "ln1_w":   [f"transformer.h.{i}.ln_1.weight", f"model.h.{i}.ln_1.weight", f"h.{i}.ln_1.weight", f"blocks.{i}.ln1.weight"],
                "ln1_b":   [f"transformer.h.{i}.ln_1.bias", f"model.h.{i}.ln_1.bias", f"h.{i}.ln_1.bias"],
                "ln2_w":   [f"transformer.h.{i}.ln_2.weight", f"model.h.{i}.ln_2.weight", f"h.{i}.ln_2.weight", f"blocks.{i}.ln2.weight"],
                "ln2_b":   [f"transformer.h.{i}.ln_2.bias", f"model.h.{i}.ln_2.bias", f"h.{i}.ln_2.bias"],
                "fc_w":    [f"transformer.h.{i}.mlp.c_fc.weight", f"model.h.{i}.mlp.c_fc.weight", f"h.{i}.mlp.c_fc.weight", f"blocks.{i}.ffn.fc_in.weight"],
                "fc_b":    [f"transformer.h.{i}.mlp.c_fc.bias", f"model.h.{i}.mlp.c_fc.bias", f"h.{i}.mlp.c_fc.bias", f"blocks.{i}.ffn.fc_in.bias"],
                "proj_w":  [f"transformer.h.{i}.mlp.c_proj.weight", f"model.h.{i}.mlp.c_proj.weight", f"h.{i}.mlp.c_proj.weight", f"blocks.{i}.ffn.fc_out.weight"],
                "proj_b":  [f"transformer.h.{i}.mlp.c_proj.bias", f"model.h.{i}.mlp.c_proj.bias", f"h.{i}.mlp.c_proj.bias", f"blocks.{i}.ffn.fc_out.bias"],
            }

            L = {}
            for name, cand_list in candidates.items():
                tensor = find_key_by_suffix(cand_list)
                L[name] = tensor
                if tensor is None:
                    missing_keys.append(f"layer{i}:{name}")

            self.layers.append(L)

        # after building, show any missing keys for debugging
        if missing_keys:
            print("Warning: some keys missing in state_dict (searched many common names). Missing entries:", missing_keys)
            # If too many missing items then it's likely a completely incompatible checkpoint
            # but we don't raise immediately — we print and proceed; later access will fail with clearer error.


    def ln(self, x, w, b):
        # x: (batch, seq, embd)
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean)**2).mean(-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * w.unsqueeze(0).unsqueeze(0) + b.unsqueeze(0).unsqueeze(0)

    def gelu_new(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, input_ids):
        # input_ids: LongTensor (batch, seq)
        bsz, seqlen = input_ids.shape
        device = self.wte.device
        # embeddings
        wte = self.wte
        wpe = self.wpe
        emb = F.embedding(input_ids, wte)  # (b,s,e)
        positions = torch.arange(seqlen, device=device).unsqueeze(0)
        pos_emb = F.embedding(positions, wpe)  # (1,s,e)
        x = emb + pos_emb

        for i in range(self.n_layer):
            L = self.layers[i]
            # ln1
            x_ln1 = self.ln(x, L["ln1_w"], L["ln1_b"])
            # c_attn -> qkv
            qkv = F.linear(x_ln1, L["c_attn_w"].t(), L["c_attn_b"])
            # split qkv: assume shape (..., 3*n_embd)
            q, k, v = qkv.split(self.n_embd, dim=2)
            # reshape for heads
            b, s, _ = q.shape
            q = q.view(b, s, self.n_head, self.head_dim).permute(0,2,1,3)  # (b,nh,s,hd)
            k = k.view(b, s, self.n_head, self.head_dim).permute(0,2,1,3)
            v = v.view(b, s, self.n_head, self.head_dim).permute(0,2,1,3)
            # attention scores
            att = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)  # (b,nh,s,s)
            # causal mask
            mask = torch.tril(torch.ones((s,s), device=device)).unsqueeze(0).unsqueeze(0)  # (1,1,s,s)
            att = att.masked_fill(mask == 0, float("-inf"))
            attn = F.softmax(att, dim=-1)
            out = torch.matmul(attn, v)  # (b,nh,s,hd)
            out = out.permute(0,2,1,3).contiguous().view(b, s, self.n_embd)  # (b,s,emb)
            out = F.linear(out, L["c_proj_w"].t(), L["c_proj_b"])
            x = x + out

            # MLP
            x_ln2 = self.ln(x, L["ln2_w"], L["ln2_b"])
            fc = F.linear(x_ln2, L["fc_w"].t(), L["fc_b"])
            fc_act = self.gelu_new(fc)
            proj = F.linear(fc_act, L["proj_w"].t(), L["proj_b"])
            x = x + proj

        # final ln
        x = self.ln(x, self.ln_f_w, self.ln_f_b)  # (b,s,e)

        # lm head
        if self.lm_head is not None:
            lm_w = self.lm_head
        else:
            # tie to token embedding
            lm_w = self.wte
        logits = torch.matmul(x, lm_w.t())  # (b,s,vocab)
        return logits

# ---------------------------
# Generation helpers
# ---------------------------
def sample_logits(logits, temperature=1.0, top_k=50):
    logits = logits / max(1e-8, temperature)
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
        min_topk = topk_vals[..., -1, None]
        logits = torch.where(logits < min_topk, torch.full_like(logits, -1e10), logits)
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate(model, tokenizer, prompt, max_new_tokens=64, temperature=0.7, top_k=50):
    """
    model   : DirectGPT2 instance
    tokenizer: BPETokenizer instance
    prompt  : string prompt (will be tokenized by tokenizer.encode)
    returns : decoded text of full sequence (prompt + generated)
    """
    # encode prompt
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        input_ids = torch.tensor([[50256]], dtype=torch.long, device=model.device)  # blank -> eos token fallback
    else:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)

    # autoregressive generation
    for _ in range(max_new_tokens):
        logits = model.forward(input_ids)           # (1, seq, vocab)
        next_logits = logits[0, -1, :]             # (vocab,)
        nxt = sample_logits(next_logits, temperature=temperature, top_k=top_k)
        input_ids = torch.cat([input_ids, torch.tensor([[nxt]], dtype=torch.long, device=model.device)], dim=1)
        if nxt == 50256:  # stop at eos token if model emits it
            break

    out_ids = input_ids[0].tolist()
    # decode (tokenizer.decode expects token ids list)
    out_text = tokenizer.decode(out_ids)
    return out_text


# ---------------------------
# Main runner
# ---------------------------
def load_config_json(folder):
    with open(os.path.join(folder, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # ensure required keys exist
    keys = ["n_embd", "n_layer", "n_head"]
    for k in keys:
        if k not in cfg:
            raise RuntimeError(f"config.json missing {k}")
    return cfg

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_no_transformers.py C:/path/to/TalkT2_pt")
        return
    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print("Folder not found:", folder)
        return
    
    pt_path = os.path.join(folder, "pytorch_model.pt")
    if not os.path.exists(pt_path):
        # fallback to .bin
        bin_path = os.path.join(folder, "pytorch_model.bin")
        if os.path.exists(bin_path):
            pt_path = bin_path
        else:
            print("pytorch_model.pt (or pytorch_model.bin) not found in folder")
            return
    
    # tokenizer files
    vocab_path = os.path.join(folder, "vocab.json")
    merges_path = os.path.join(folder, "merges.txt")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print("vocab.json and merges.txt required in the folder for tokenizer.")
        return

    print("Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_path, merges_path)

    print("Loading config...")
    cfg = load_config_json(folder)
    device = "cpu"
    print("Loading state_dict (this may take a while)...")
    sd = torch.load(pt_path, map_location="cpu")
    # if state dict has prefix 'model.' remove it
    new_sd = {}
    for k,v in sd.items():
        new_k = k
        if k.startswith("model."):
            new_k = k[len("model."):]
        new_sd[new_k] = v
    sd = new_sd

    print("Building model wrapper...")
    model = DirectGPT2(sd, cfg, device=device)

    # interactive prompt
    print("Ready. Type input and press enter. 'quit' to exit.")
    
    history = ""   # keeps track of the dialogue so far

    while True:
        text = input("You: ")
        if text.strip().lower() == "quit":
            break

        # Add user message to history
        history += f"You: {text}\nTalkT2:"

        # Feed the entire conversation so far
        prompt = history[-2000:]  # limit length to avoid GPU/CPU memory issues

        output = generate(model, tokenizer, prompt, max_new_tokens=128, temperature=0.8, top_k=50)
        reply = output.split("TalkT2:")[-1].strip()

        print(f"TalkT2: {reply}")

        # Append model reply to history
        history += f" {reply}\n"

if __name__ == "__main__":
    main()
