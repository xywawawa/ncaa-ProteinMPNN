#!/usr/bin/env python3
import os, sys, torch, argparse
from parse_cif_noX import aa2idx  
import numpy as np

one2three = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY',
    'H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER',
    'T':'THR','W':'TRP','Y':'TYR','V':'VAL','X':'UNK'
}
def _extract_backbone_coords(chain_obj, res3_seq):
    xyz = chain_obj['xyz']   # [L, MAX_ATOMS, 3] (torch.Tensor or np.ndarray)
    mask = chain_obj['mask'] # [L, MAX_ATOMS] (torch.Tensor or np.ndarray)

    if isinstance(xyz, torch.Tensor): xyz = xyz.cpu().numpy()
    if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
    L = xyz.shape[0]

    def take_atom(name):
        out = np.full((L, 3), np.nan, dtype=np.float32)
        for i in range(L):
            res3 = str(res3_seq[i]).upper()
            key = (res3, name)
            j = aa2idx.get(key, None)
            if j is None:
                continue
            if j < mask.shape[1] and mask[i, j]:
                out[i] = xyz[i, j]
        return out

    N  = take_atom('N')
    CA = take_atom('CA')
    C  = take_atom('C')
    O  = take_atom('O')
    return {'N': N, 'CA': CA, 'C': C, 'O': O}

def make_chain_dict(obj, letter):
    seq1 = obj['seq']
    res3 = obj.get('res3_seq', [one2three.get(a.upper(), 'UNK') for a in str(seq1)])
    coords = _extract_backbone_coords(obj, res3)
    return {
        f'seq_chain_{letter}': str(seq1),
        f'res3_chain_{letter}': list(res3),
        f'coords_chain_{letter}': {
            f'N_chain_{letter}': coords['N'].tolist(),
            f'CA_chain_{letter}': coords['CA'].tolist(),
            f'C_chain_{letter}': coords['C'].tolist(),
            f'O_chain_{letter}': coords['O'].tolist(),
        },
    }

def build_single(in_pt, out_pt):
    obj = torch.load(in_pt, map_location='cpu')
    letter = os.path.splitext(os.path.basename(in_pt))[0].split('_')[-1]
    raw = {
        'name': os.path.splitext(os.path.basename(in_pt))[0],
        'masked_list': [letter],
        'visible_list': [],
        'num_of_chains': 1,
        'seq': str(obj['seq']),
    }
    raw.update(make_chain_dict(obj, letter))
    
    # >>> ADD print
    nres = len(raw[f'res3_chain_{letter}'])
    print(f"[INFO] Chain {letter}: {nres} residues")

    torch.save(raw, out_pt)
    print(f"[OK] Wrote single-chain raw → {out_pt}")


def build_multi(visible_pt, design_pt, out_pt, vis_tag, des_tag):
    vis_obj = torch.load(visible_pt, map_location='cpu')
    des_obj = torch.load(design_pt, map_location='cpu')

    raw = {
        'name': f"{os.path.splitext(os.path.basename(design_pt))[0]}_{vis_tag}{des_tag}",
        'masked_list': [des_tag],
        'visible_list': [vis_tag],
        'num_of_chains': 2,
        'seq': f"{vis_obj['seq']}|{des_obj['seq']}",
    }
    raw.update(make_chain_dict(vis_obj, vis_tag))
    raw.update(make_chain_dict(des_obj, des_tag))

    # >>> ADD print
    nres_vis = len(raw[f'res3_chain_{vis_tag}'])
    nres_des = len(raw[f'res3_chain_{des_tag}'])
    print(f"[INFO] Chain {vis_tag}: {nres_vis} residues | Chain {des_tag}: {nres_des} residues")

    torch.save(raw, out_pt)
    print(f"[OK] Wrote multi-chain raw (visible={vis_tag}, design={des_tag}) → {out_pt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert .pt chain(s) to raw format")
    ap.add_argument("--mode", choices=["single", "multi"], default="single")
    ap.add_argument("--in_pt", help="Input .pt (for single mode)")
    ap.add_argument("--visible", help="Visible chain .pt (for multi mode)")
    ap.add_argument("--design", help="Design chain .pt (for multi mode)")
    ap.add_argument("--vis_tag", default="A", help="Chain letter for visible")
    ap.add_argument("--des_tag", default="B", help="Chain letter for design")
    ap.add_argument("--out", required=True, help="Output raw .pt path")
    args = ap.parse_args()

    if args.mode == "single":
        assert args.in_pt, "--in_pt required for single mode"
        build_single(args.in_pt, args.out)
    else:
        assert args.visible and args.design, "--visible and --design required for multi mode"
        build_multi(args.visible, args.design, args.out, args.vis_tag, args.des_tag)
