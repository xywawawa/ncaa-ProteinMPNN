# ncAA ProteinMPNN

This repository provides an inference pipeline for **ProteinMPNN extended to non-canonical amino acids (ncAAs)**.

## environment:
We recommend using Conda to manage dependencies.

```bash
conda env create -f environment.yml
conda activate ncproteinmpnn
```

## 0. Convert `.pdb` to `.mmcif`

ProteinMPNN requires structures in **mmCIF** format.

Use the official PDBJ converter:

https://mmcif.pdbj.org/converter

---

## 1. Convert `.mmcif` to `.pt`

Parse mmCIF files into  `.pt` format.

```bash
python parse_cif_noX.py <input_dir> <output_dir>
```

**Arguments**
- `input_dir`: directory containing `.mmcif` files
- `output_dir`: directory to store generated `.pt` files

---

## 2. Convert `.pt` to raw ProteinMPNN input

Merge visible and design chains into a single raw graph input.

```bash
python to_raw.py \
  --mode multi \
  --visible CAND1_BACKBONE_A.pt \
  --design  CAND1_BACKBONE_B.pt \
  --vis_tag B \
  --des_tag A \
  --out CAND1_BACKBONE_AB_raw.pt
```

### Notes
- Default behavior is **designing Chain B**
- `--visible`: chain treated as context (fixed backbone)
- `--design`: chain to be redesigned
- `vis_tag` / `des_tag` must match chain IDs in the structure
- Output is a single `*_raw.pt` file used for inference

---

## 3. Inference (Sequence Design)

Run ProteinMPNN inference with optional ncAA control.

```bash
python infer.py \
  --graph CAND1_BACKBONE_AB_raw.pt \
  --ncaa_ratio_target 0.0 \
  --num_samples 1000 \
  --topk 5 \
  --temperature 1.0 \
  --out_txt design.txt \
  --out_jsonl design.jsonl
```

### Key arguments
- `--graph`: raw `.pt` file from step 2
- `--ncaa_ratio_target`:
  - `0.0`: do **not** force ncAA generation (standard ProteinMPNN behavior)
  - `>0`: target fraction of ncAAs in generated sequences
- `--num_samples`: number of sequences to sample
- `--topk`: top-k sampling
- `--temperature`: sampling temperature
- `--out_txt`: plain-text output sequences
- `--out_jsonl`: structured JSONL output for downstream analysis

---

## TODO

- Update training code

