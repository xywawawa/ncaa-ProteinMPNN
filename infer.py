#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for a ProteinMPNN-style model on a single protein graph (.pt).

Input formats for --graph:
  A) "raw" dict with chains (seq_chain_X, res3_chain_X, coords_chain_X) plus masked_list/visible_list
  B) "featurized" dict with tensors: X, S, mask, chain_M, residue_idx, chain_encoding_all

Outputs:
  - FASTA
  - Pretty JSON (multi-line, indented) containing ONLY the B chain:
      {
        "greedy_b": {
          "aa3_tokens": [...],
          "aa1_string": "....",
          "top1_probs": [Lb],
          "full_probs": [Lb][V],
          "ncaa_count": int,
          "ncaa_list": [...],
          "ncaa_stats": {"count": int, "types": {AA3: freq, ...}}
        },
        "samples_b": [
          {
            "aa3_tokens": [...],
            "aa1_string": "....",
            "ids_b": [Lb],
            "chosen_probs": [Lb],    # probability of the chosen token at each B position under the sampling distribution
            "top1_probs": [Lb],      # same per-position top-1 probabilities as in greedy_b (for easy comparison)
            "ncaa_count": int,
            "ncaa_list": [...],
            "ncaa_stats": {...}
          }, ...
        ],
        "design_mask_b": [Lb bools],
        "meta": {...}
      }
"""
import argparse
import json
import os
import os.path as osp
import sys
from collections import Counter

import torch
import torch.nn.functional as F

from model import ProteinMPNN, build_tokenizer, featurize
import numpy as np


AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
    'THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}
CANONICAL20 = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
}

def tokens3_to_one(tokens3):
    out = []
    for t in tokens3:
        out.append(AA3_TO_1.get(t.upper(), 'X'))  # non-canonical -> 'X'
    return "".join(out)

def count_ncaa_and_list(aa3_list):
    """Return (count, list) for non-canonical AAs in a 3-letter token list."""
    nc_list = [aa for aa in aa3_list if aa.upper() not in CANONICAL20]
    return len(nc_list), nc_list

def ncaa_stats(aa3_list):
    """Return {"count": int, "types": {aa3: freq, ...}} for ncAAs in aa3_list."""
    nc = [aa for aa in aa3_list if aa.upper() not in CANONICAL20]
    return {"count": len(nc), "types": dict(Counter(nc))}


def _apply_topk_p_filtering(pv, topk=0, p=0.0):
    """
    Apply top-k and nucleus (top-p) filtering to a 1D probability vector pv (on CPU).
    Returns a re-normalized probability vector.
    """
    import torch
    V = pv.shape[0]
    probs = pv.clone()

    # Top-k
    if isinstance(topk, int) and 0 < topk < V:
        vals, idxs = torch.sort(probs, descending=True)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[idxs[:topk]] = True
        probs = torch.where(mask, probs, torch.tensor(0.0, dtype=probs.dtype))

    # Nucleus (top-p)
    if isinstance(p, float) and 0.0 < p < 1.0:
        vals2, idxs2 = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(vals2, dim=0)
        keep_n = (cumsum <= p).sum().item()
        keep_n = max(1, int(keep_n))
        mask2 = torch.zeros_like(probs, dtype=torch.bool)
        mask2[idxs2[:keep_n]] = True
        probs = torch.where(mask2, probs, torch.tensor(0.0, dtype=probs.dtype))

    s = probs.sum()
    if s.item() <= 0:
        probs = pv
        s = probs.sum()
    probs = probs / s
    return probs


def get_nc_indices(tokenizer):
    """Return a LongTensor of indices (in tokenizer.vocab) that are non-canonical AAs."""
    import torch
    nc = [i for i, t in enumerate(getattr(tokenizer, "vocab", [])) if str(t).upper() not in CANONICAL20]
    return torch.tensor(nc, dtype=torch.long) if len(nc) > 0 else torch.empty(0, dtype=torch.long)


def sample_sequences(
    tokenizer, probs, S_ids, design_mask,
    num_samples=1, topk=0, p=0.0, canonical_only=False,
    ncaa_ratio_target=0.0, ncaa_bias=1.0, ncaa_alpha_max=0.5, ncaa_min_prob=1e-3
):
    """
    Sample full-length sequences while only modifying designed positions (design_mask==True).
    Supports soft guidance toward ncAAs:
      - ncaa_ratio_target: target fraction of ncAAs among designed positions (0~1), 0 disables guidance
      - ncaa_bias: strength of the bias toward ncAAs
      - ncaa_alpha_max: per-position maximum mixing weight toward the ncAA sub-distribution
      - ncaa_min_prob: if the max ncAA prob at a position is below this threshold, skip the bias there
    Returns a list of dicts:
      {
        "aa3_tokens": [...],
        "aa1_string": "...",
        "ids": [...],                # full-length ids (kept internal; NOT written to JSON)
        "chosen_probs_full": [...],  # probability of the chosen token at each position under the sampling distribution
      }
    """
    import torch

    V = probs.shape[1]
    whitelist_idx = None
    if canonical_only:
        whitelist_idx = [i for i, t in enumerate(getattr(tokenizer, "vocab", [])) if str(t).upper() in CANONICAL20]
        whitelist_idx = torch.tensor(whitelist_idx, dtype=torch.long) if len(whitelist_idx) > 0 else None

    nc_idx = get_nc_indices(tokenizer)

    # Designed positions
    design_positions = torch.nonzero(design_mask, as_tuple=False).flatten().tolist()
    Ld = len(design_positions)

    results = []
    for _ in range(max(1, int(num_samples))):
        ids = S_ids.clone()
        chosen_probs_full = [None] * probs.shape[0]
        nc_so_far = 0
        for pos in range(probs.shape[0]):
            if not bool(design_mask[pos]):
                continue
            pv = probs[pos].clone()  # [V]

            # Canonical-only whitelist (optional)
            if whitelist_idx is not None and whitelist_idx.numel() > 0:
                pv2 = torch.zeros_like(pv)
                pv2[whitelist_idx] = pv[whitelist_idx]
                if pv2.sum().item() > 0:
                    pv = pv2 / pv2.sum()

            # Soft guidance toward ncAAs
            if ncaa_ratio_target > 0.0 and nc_idx.numel() > 0 and Ld > 0:
                remaining = max(1, Ld - nc_so_far)
                target_total = int(round(ncaa_ratio_target * Ld))
                desire = max(0, target_total - nc_so_far)
                alpha_raw = float(desire) / float(remaining)
                alpha = min(ncaa_alpha_max, alpha_raw * float(ncaa_bias))

                max_nc_prob = pv[nc_idx].max().item() if nc_idx.numel() > 0 else 0.0
                if max_nc_prob < float(ncaa_min_prob):
                    alpha = 0.0

                if alpha > 0.0:
                    pv_nc = torch.zeros_like(pv)
                    pv_nc[nc_idx] = pv[nc_idx]
                    s_nc = pv_nc.sum().item()
                    if s_nc > 0:
                        pv_nc = pv_nc / s_nc
                        pv = (1 - alpha) * pv + alpha * pv_nc
                        pv = pv / pv.sum()

            # Top-k / Top-p after biasing
            pv = _apply_topk_p_filtering(pv, topk=topk, p=p)

            choice = torch.multinomial(pv, 1).item()
            ids[pos] = int(choice)
            chosen_probs_full[pos] = float(pv[choice].item())

            tok3 = str(getattr(tokenizer, "vocab", [])[choice]).upper() if hasattr(tokenizer, "vocab") else None
            if tok3 is not None and tok3 not in CANONICAL20:
                nc_so_far += 1

        aa3 = tokenizer.decode(ids.tolist())
        aa1 = tokens3_to_one(aa3)
        results.append({
            "aa3_tokens": aa3,
            "aa1_string": aa1,
            "ids": ids.tolist(),
            "chosen_probs_full": chosen_probs_full,
        })
    return results


# ----------------------------
# Multi-chain helpers
# ----------------------------

def _pick_coords_from_chain_obj(obj):
    """
    Extract backbone coordinates from a chain object.
    Return dict: {'N': [L,3], 'CA': [L,3], 'C': [L,3], 'O': [L,3]}.
    """
    import numpy as np

    keys = ('N','CA','C','O')
    if all(k in obj for k in keys):
        return {k: np.asarray(obj[k], dtype=float) for k in keys}
    if 'coords' in obj and isinstance(obj['coords'], dict) and all(k in obj['coords'] for k in keys):
        return {k: np.asarray(obj['coords'][k], dtype=float) for k in keys}
    raise KeyError("Cannot find backbone coords (N/CA/C/O) in chain object.")


def _chain_files_to_raw_two_chain(vis_path, des_path, vis_tag='A', des_tag='B'):
    """
    Build a 'raw' dict compatible with featurize() by concatenating two chain .pt files.
    By convention: first is the visible chain, second is the design chain.
    """
    import torch
    one2three = {
        'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY',
        'H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER',
        'T':'THR','W':'TRP','Y':'TYR','V':'VAL','X':'UNK'
    }

    def _extract(chain_obj, tag):
        seq1 = str(chain_obj.get('seq', ''))
        if 'res3_seq' in chain_obj and chain_obj['res3_seq'] is not None:
            res3 = [str(t).upper() for t in chain_obj['res3_seq']]
        else:
            res3 = [one2three.get(a.upper(), 'UNK') for a in seq1]
            print("WARNING: res3_seq not found; converting 1-letter to 3-letter with UNK fallback.")
        bb = _pick_coords_from_chain_obj(chain_obj)
        coords_dict = {
            f'N_chain_{tag}': bb['N'],
            f'CA_chain_{tag}': bb['CA'],
            f'C_chain_{tag}': bb['C'],
            f'O_chain_{tag}': bb['O'],
        }
        return seq1, res3, coords_dict

    vis_obj = torch.load(vis_path, map_location='cpu')
    des_obj = torch.load(des_path, map_location='cpu')

    v_seq, v_res3, v_coords = _extract(vis_obj, vis_tag)
    d_seq, d_res3, d_coords = _extract(des_obj, des_tag)

    raw = {
        'name': f"{os.path.basename(vis_path)}+{os.path.basename(des_path)}",
        'masked_list': [des_tag],
        'visible_list': [vis_tag],
        'num_of_chains': 2,
        f'seq_chain_{vis_tag}': v_seq,
        f'res3_chain_{vis_tag}': v_res3,
        f'coords_chain_{vis_tag}': v_coords,
        f'seq_chain_{des_tag}': d_seq,
        f'res3_chain_{des_tag}': d_res3,
        f'coords_chain_{des_tag}': d_coords,
        'seq': v_seq + d_seq,
    }
    return raw


# ----------------------------
# I/O helpers
# ----------------------------

def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    vocab = ckpt.get("tokenizer_vocab", None)
    state = ckpt["model_state_dict"]
    return args, vocab, state


def ensure_batch(graph_obj, device, tokenizer):
    """
    Accept two input types:
      - raw dict with masked_list/visible_list + chain fields => run featurize([raw])
      - featurized dict with required tensors => batchify to B=1
    Returns:
      X,S,mask,chain_M,residue_idx,chain_encoding_all (all on device) and meta (dict)
    """
    if isinstance(graph_obj, (list, tuple)):
        assert len(graph_obj) == 1, "Pass a single protein at a time."
        graph_obj = graph_obj[0]

    needed = {"X", "S", "mask", "chain_M", "residue_idx", "chain_encoding_all"}
    if isinstance(graph_obj, dict) and needed.issubset(graph_obj.keys()):
        X = graph_obj["X"].to(device)
        S = graph_obj["S"].to(device)
        mask = graph_obj["mask"].to(device)
        chain_M = graph_obj["chain_M"].to(device)
        residue_idx = graph_obj["residue_idx"].to(device)
        chain_encoding_all = graph_obj["chain_encoding_all"].to(device)
        meta = {k: v for k, v in graph_obj.items() if k not in needed}

        # Ensure batch dimension
        if X.dim() == 3:
            X = X.unsqueeze(0)
        if S.dim() == 1:
            S = S.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if chain_M.dim() == 1:
            chain_M = chain_M.unsqueeze(0)
        if residue_idx.dim() == 1:
            residue_idx = residue_idx.unsqueeze(0)
        if chain_encoding_all.dim() == 1:
            chain_encoding_all = chain_encoding_all.unsqueeze(0)

        return X, S.long(), mask.float(), chain_M.float(), residue_idx.long(), chain_encoding_all.long(), meta

    # Otherwise assume "raw" and run featurize
    batch = [graph_obj]
    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
        batch, device=device, tokenizer=tokenizer
    )
    meta = {
        "masked_list": graph_obj.get("masked_list", []),
        "visible_list": graph_obj.get("visible_list", [])
    }
    return X, S.long(), mask.float(), chain_M.float(), residue_idx.long(), chain_encoding_all.long(), meta


def decode_ids_to_aa3(ids, tokenizer):
    toks = tokenizer.decode([int(i) for i in ids])
    return toks


def write_fasta(out_path, title, aa1_string, design_mask=None):
    """Write a single FASTA record. Designed positions are lower-cased (optional)."""
    if design_mask is not None and len(design_mask) == len(aa1_string):
        s = "".join(c.lower() if m else c for c, m in zip(aa1_string, design_mask))
    else:
        s = aa1_string
    with open(out_path, "w") as f:
        f.write(f">{title}\n")
        for i in range(0, len(s), 80):
            f.write(s[i:i+80] + "\n")


# ----------------------------
# Core inference
# ----------------------------

def build_model_from_ckpt(args, vocab, device):
    """Rebuild model & tokenizer from saved args (with fallbacks)."""
    hidden_dim = int(args.get("hidden_dim", 128))
    num_encoder_layers = int(args.get("num_encoder_layers", 3))
    num_decoder_layers = int(args.get("num_decoder_layers", 3))
    num_neighbors = int(args.get("num_neighbors", 48))
    dropout = float(args.get("dropout", 0.1))
    backbone_noise = float(args.get("backbone_noise", 0.2))
    use_chem_emb = bool(args.get("use_chem_emb", False))
    chem_freeze = bool(args.get("chem_freeze", True))
    chem_proj_bias = bool(args.get("chem_proj_bias", False))
    bilinear_rank = int(args.get("bilinear_rank", 64))
    num_letters = int(args.get("num_letters", 20))


    tokenizer = build_tokenizer()
    if vocab is not None:
        tokenizer = type(tokenizer)(vocab=vocab)  # e.g., Res3Tokenizer(vocab)
    V = tokenizer.size

    def build_W_from_dict(tokenizer, z_map, emb_dim=512):
        """
        tokenizer.vocab: ['ALA','ARG',...,'CR2',...,'UNK']
        z_map: {'ALA': np/torch(D,), 'ARG': ..., 'CR2': ..., ...} or None
        Returns: torch.FloatTensor [V, emb_dim], row order matches tokenizer.vocab
        """
        V = tokenizer.size
        W = np.zeros((V, emb_dim), dtype=np.float32)
        if z_map is None:
            return torch.from_numpy(W)

        for i, res3 in enumerate(getattr(tokenizer, "vocab", [None]*V)):
            if res3 == "UNK":
                W[i] = 0.0
            else:
                vec = z_map.get(str(res3).upper())
                if vec is None:
                    W[i] = 0.0
                else:
                    v = torch.as_tensor(vec, dtype=torch.float32).cpu().numpy()
                    assert v.shape[-1] == W.shape[-1], f"chem emb dim mismatch for {res3}: {v.shape[-1]} vs {W.shape[-1]}"
                    W[i] = v
        return torch.from_numpy(W)

    def load_zmap_from_all(
        cls_path=args.chem_embedding_cls_path,
        meta_path=args.chem_embedding_meta_path
    ):
        """
        Load merged CLS embeddings (20AA + CSV ncAA), keep only allowed residues (20AA + specified ncAA).
        Returns: z_map: {res3: np.ndarray[D]}, emb_dim: int
        """
        import torch, json, numpy as np
        allowed = {
            # canonical 20
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL",
            # ncAA 
            "4BF", "MLE", "TPO", "HYP", "SCY", "CGU", "ABA", "MLY", "CXM", "PCA",
            "CSX", "HIC", "SAR", "MLZ", "SCH", "ALY", "CSD", "CSS", "CSO", "ORN",
            "FME", "CME", "KCX", "M3L", "AIB", "XCP", "MVA", "YCM", "DHA", "NLE", "OCS",
            "DAL","DAR","DAS","DCY","DGL","DGN",
            "DHI","DIL","DLE","DLY","DPN","DPR",
            "DSN","DTH","DTR","DTY","DVA"
        }
        cls = torch.load(cls_path, map_location="cpu")
        if isinstance(cls, torch.Tensor):
            cls = cls.detach().cpu().numpy()
        with open(meta_path, "r") as f:
            meta = json.load(f)
        keys = meta["order"]
        aa1_to_aa3 = meta.get("aa1_to_aa3", {})

        z_map = {}
        for i, key in enumerate(keys):
            if isinstance(key, str) and len(key) == 1 and key.isalpha():
                aa3 = aa1_to_aa3.get(key)
                if aa3 and aa3.upper() in allowed:
                    z_map[aa3.upper()] = cls[i].astype(np.float32)
            else:
                if str(key).upper() in allowed:
                    z_map[str(key).upper()] = cls[i].astype(np.float32)
        emb_dim = cls.shape[1]
        print(f"[Chem] loaded {len(z_map)} embeddings (dim={emb_dim}) from allowed list")
        return z_map, emb_dim

    if args.use_chem_emb:
        z_map, D = load_zmap_from_all(args.chem_embedding_cls_path, args.chem_embedding_meta_path)
        W = build_W_from_dict(tokenizer, z_map=z_map, emb_dim=D)
    else:
        D = 512
        W = torch.zeros((tokenizer.size, D), dtype=torch.float32)

    # ---------------- model ----------------
    assert W.shape[0] == V, f"chem table rows {W.shape[0]} != vocab {V}"

    model = ProteinMPNN(
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        vocab=V,
        k_neighbors=num_neighbors,
        dropout=dropout,
        augment_eps=backbone_noise,
        use_chem_emb=use_chem_emb,
        chem_emb_weight=W if use_chem_emb else None,
        chem_freeze=chem_freeze,
        chem_proj_bias=chem_proj_bias,
        bilinear_rank=bilinear_rank,
        use_norm=True,
        num_letters=num_letters,
    ).to(device)
    return model, tokenizer

def run_inference(
    model, tokenizer, graph, *,
    temperature=1.0,
    num_samples=1,
    topk=0,
    nucleus_p=0.0,
    canonical_only=False,
    outputs_are_logits=True,
    ncaa_ratio_target=0.0,
    ncaa_bias=1.0,
    ncaa_alpha_max=0.5,
    ncaa_min_prob=1e-3,
):
    """
    auto regression sampling
    return:
      - samples_b: list of {
            aa3_tokens, aa1_string,
            ids_b,              
            score,              
            ncaa_count, ncaa_list, ncaa_stats
        }
      - design_mask_b: List[bool]  (length = Lb)
      - meta
    """
    unk_idx = None
    if hasattr(tokenizer, "vocab"):
        for i, t in enumerate(tokenizer.vocab):
            if str(t).upper() == "UNK":
                unk_idx = i
                break

    model.eval()
    device = next(model.parameters()).device
    X, S, mask, chain_M, residue_idx, chain_encoding_all, meta = ensure_batch(
        graph, device, tokenizer
    )

    # design chain B 
    design_mask = ((mask > 0.5) & (chain_M > 0.5)).bool()  # [B,L]
    dmask = design_mask[0].detach().cpu()                  # [L] bool
    idx_b = torch.nonzero(dmask, as_tuple=False).flatten().tolist()
    Lb = len(idx_b)

    def _decode_b(ids_full_1d):
        aa3_full = decode_ids_to_aa3(ids_full_1d.tolist(), tokenizer)  # full 3-letter
        aa3_b = [aa3_full[i] for i in idx_b]
        aa1_b = tokens3_to_one(aa3_b)
        return aa3_b, aa1_b

    samples_b = []

    with torch.no_grad():
        for k in range(int(num_samples)):
            sample_dict = model.sample_ar(
                X,
                S,             
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
                temperature=float(temperature),
                banned_idx=unk_idx,
            )
            S_sample = sample_dict["S"][0].detach().cpu()            # [L]
            chosen_logp = sample_dict["chosen_logp"][0].detach().cpu()  # [L]

            aa3_b, aa1_b = _decode_b(S_sample)

            if Lb > 0:
                score_val = float(-(chosen_logp[dmask].mean().item()))
            else:
                score_val = 0.0

            # ncAA 
            c_nc, list_nc = count_ncaa_and_list(aa3_b)
            stats_nc = ncaa_stats(aa3_b)

            ids_b = [int(S_sample[i].item()) for i in idx_b]

            samples_b.append({
                "aa3_tokens": aa3_b,
                "aa1_string": aa1_b,
                "ids_b": ids_b,
                "score": score_val,
                "ncaa_count": c_nc,
                "ncaa_list": list_nc,
                "ncaa_stats": stats_nc,
            })

    # debug 
    enc = chain_encoding_all[0].detach().cpu().numpy().astype(int)
    dmask_np = design_mask[0].detach().cpu().numpy().astype(bool)
    print("masked_list:", meta.get("masked_list"))
    print("visible_list:", meta.get("visible_list"))
    print("unique chain ids:", set(enc.tolist()))
    print("design_mask sum (designed positions):", int(dmask_np.sum()))
    from collections import Counter as _C
    per_chain_counts = _C(enc[dmask_np])
    print("design positions per chain:", per_chain_counts)

    out = {
        "samples_b": samples_b,
        "design_mask_b": [True] * Lb,
        "meta": meta,
    }
    return out

def write_token_sequences_txt(out, out_txt):
    """
    Write token sequences (3-letter AAs) for samples in a human-readable txt.
    Format:
      SAMPLE_1
      ALA ARG GLY ...
      SAMPLE_2
      ...
    """
    with open(out_txt, "w") as f:
        for i, s in enumerate(out["samples_b"]):
            f.write(f"SAMPLE_{i+1}\n")
            f.write(" ".join(s["aa3_tokens"]) + "\n")


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", default='best.pt' help="Path to trained checkpoint")
    ap.add_argument("--graph", required=False, help="Protein graph (.pt): raw dict or featurized tensors")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (<=1 is sharper)")
    ap.add_argument("--num_samples", type=int, default=1, help="Number of sequences to generate")
    ap.add_argument("--topk", type=int, default=0, help="Top-k sampling (0=disabled)")
    ap.add_argument("--p", type=float, default=0.0, help="Nucleus sampling threshold (0=disabled)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0=auto)")
    ap.add_argument("--canonical_only", action="store_true", help="Restrict sampling to canonical 20 AAs")

    ap.add_argument("--ncaa_ratio_target", type=float, default=0.2, help="Target fraction of ncAA among designed positions (0~1).")
    ap.add_argument("--ncaa_bias", type=float, default=2.0, help="Bias strength to tilt probability mass toward ncAA.")
    ap.add_argument("--ncaa_alpha_max", type=float, default=0.6, help="Max per-position mixing weight toward ncAA.")
    ap.add_argument("--ncaa_min_prob", type=float, default=1e-3, help="Skip bias at a position if max ncAA prob < threshold.")

    ap.add_argument("--out_fasta", default="design.fasta")
    ap.add_argument("--out_jsonl", default="design.jsonl",
                    help="Pretty JSON (not line-delimited) will be written here.")
    ap.add_argument("--out_txt", default="design_tokens.txt",
                help="Plain text file with token sequences (3-letter AAs) for greedy and samples.")


    ap.add_argument("--mode", choices=["single","multi"], default="single",
                    help="single: --graph is a single .pt (raw/featurized); multi: use --visible and --design for two chains")
    ap.add_argument("--visible", type=str, default=None, help="[multi] visible chain .pt path (A)")
    ap.add_argument("--design", type=str, default=None, help="[multi] design chain .pt path (B)")
    ap.add_argument("--chem_embedding_cls_path", default='unimol_cls_all_chiral.pt')
    ap.add_argument("--chem_embedding_meta_path", default='unimol_cls_all_chiral_meta.json')
    
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.seed:
        torch.manual_seed(args.seed)

    # Load checkpoint and rebuild model+tokenizer
    ckpt_args, ckpt_vocab, state = load_checkpoint(args.ckpt)
    model, tokenizer = build_model_from_ckpt(ckpt_args, ckpt_vocab, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys:", len(missing))
    if unexpected:
        print("[WARN] Unexpected keys:", len(unexpected))
    model.to(device)

    # Load graph
    if args.mode == "single":
        graph_obj = torch.load(args.graph, map_location="cpu")
    else:
        assert args.visible and args.design, "[multi] please provide --visible and --design"
        assert osp.exists(args.visible) and osp.exists(args.design), "[multi] --visible/--design file does not exist"
        graph_obj = _chain_files_to_raw_two_chain(args.visible, args.design, vis_tag='A', des_tag='B')

    # Determine whether model outputs logits or log-probs
    outputs_are_logits = bool(ckpt_args.get("outputs_are_logits", True))

    # Run inference (B chain only)
    out = run_inference(
        model, tokenizer, graph_obj,
        temperature=args.temperature, topk=args.topk, num_samples=args.num_samples,
        nucleus_p=args.p, canonical_only=args.canonical_only, outputs_are_logits=outputs_are_logits,
        ncaa_ratio_target=args.ncaa_ratio_target, ncaa_bias=args.ncaa_bias,
        ncaa_alpha_max=args.ncaa_alpha_max, ncaa_min_prob=args.ncaa_min_prob,
    )

    # FASTA: B chain only
    base = os.path.basename(args.graph) if args.mode == "single" else os.path.basename(args.design)
    first = out['samples_b'][0]
    write_fasta(
        args.out_fasta,
        title=f"{base}|B|sample=1",
        aa1_string=first['aa1_string'],
        design_mask=[True] * len(first['aa1_string'])  
    )
    with open(args.out_fasta, 'a') as _ap:
        for i, s in enumerate(out['samples_b']):
            score = s.get("score", None)
            if score is not None:
                _ap.write(f">{base}|B|sample={i+1}|score={score:.4f}\n")
            else:
                _ap.write(f">{base}|B|sample={i+1}\n")
            seq = s['aa1_string']
            for j in range(0, len(seq), 80):
                _ap.write(seq[j:j+80] + "\n")

    # Pretty JSON (multi-line, indented) for readability
    with open(args.out_jsonl, 'w') as fj:
        json.dump(out, fj, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {args.out_fasta} and pretty JSON at {args.out_jsonl} (B chain only)")

    write_token_sequences_txt(out, args.out_txt)
    print(f"[OK] Wrote token sequence txt at {args.out_txt}")


if __name__ == "__main__":
    main()
