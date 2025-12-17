#!/usr/bin/env python3
"""
Batch process mmCIF files to PyTorch format
Usage: python script.py <input_dir> <output_dir> [target_residue]
Example: python parse_cif_noX.py /scratch/yx4224/ProteinMPNN/downloads ./output YCM
"""

import pdbx
from pdbx.reader.PdbxReader import PdbxReader
import gzip
import numpy as np
import torch
import os
import sys
import glob
import re
from itertools import combinations
import tempfile
import subprocess
import shutil
from pathlib import Path

import io, gzip




def _smart_open_text(path):
    with open(path, 'rb') as fh:
        magic = fh.read(2)
    if magic == b'\x1f\x8b': 
        return io.TextIOWrapper(gzip.open(path, 'rb'), encoding='utf-8', errors='strict')
    return open(path, 'r', encoding='utf-8', errors='strict')

# ---------- Configuration ----------
def find_tmalign():
    """Find TMalign executable in common locations"""
    possible_paths = [
        '/home/aivan/prog/TMalign',
        shutil.which('TMalign'),
        os.path.expanduser('~/bin/TMalign'),
        '/usr/local/bin/TMalign',
    ]
    
    for path in possible_paths:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None

TMALIGN_PATH = find_tmalign()

def get_temp_dir():
    """Get best available temporary directory"""
    if os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK):
        return '/dev/shm'
    return None

TEMP_DIR = get_temp_dir()

# ---------- Residue and Atom Definitions ----------
RES_NAMES_STD = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
]

ATOM_ORDER_STD = [
    ("N","CA","C","O","CB"),
    ("N","CA","C","O","CB","CG","CD","NE","CZ","NH1","NH2"),
    ("N","CA","C","O","CB","CG","OD1","ND2"),
    ("N","CA","C","O","CB","CG","OD1","OD2"),
    ("N","CA","C","O","CB","SG"),
    ("N","CA","C","O","CB","CG","CD","OE1","NE2"),
    ("N","CA","C","O","CB","CG","CD","OE1","OE2"),
    ("N","CA","C","O"),
    ("N","CA","C","O","CB","CG","ND1","CD2","CE1","NE2"),
    ("N","CA","C","O","CB","CG1","CG2","CD1"),
    ("N","CA","C","O","CB","CG","CD1","CD2"),
    ("N","CA","C","O","CB","CG","CD","CE","NZ"),
    ("N","CA","C","O","CB","CG","SD","CE"),
    ("N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ"),
    ("N","CA","C","O","CB","CG","CD"),
    ("N","CA","C","O","CB","OG"),
    ("N","CA","C","O","CB","OG1","CG2"),
    ("N","CA","C","O","CB","CG","CD1","CD2","CE2","CE3","NE1","CZ2","CZ3","CH2"),
    ("N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","OH"),
    ("N","CA","C","O","CB","CG1","CG2")
]

RES_ATOM_ORDER_NCAA = {
    '3FG': ("N","CA","C","O","CB","CG1","CD1","OD1","CG2","CD2","OD2","CZ","OXT"),
    '4BF': ("N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","BR","OXT"),
    'ABA': ("N","CA","C","O","CB","CG","OXT"),
    'AIB': ("N","CA","C","O","CB1","CB2","OXT"),
    'ALY': ("N","CA","C","O","CB","CG","CD","CE","NZ","CH","OH","CH3","OXT"),
    'CGU': ("N","CA","C","O","CB","CG","CD1","CD2","OE11","OE12","OE21","OE22"),
    'CME': ("N","CA","C","O","CB","SG","SD","CE","CZ","OH","OXT"),
    'CSD': ("N","CA","C","O","CB","SG","OXT","OD1","OD2"),
    'CSO': ("N","CA","C","O","CB","SG","OXT","OD"),
    'CSS': ("N","CA","C","O","CB","SG","SD","OXT"),
    'CSX': ("N","CA","C","O","CB","SG","OXT","OD"),
    'CXM': ("N","CA","C","O","CB","CG","SD","CE","CN","ON1","ON2","OXT"),
    'DHA': ("N","CA","C","O","CB","OXT"),
    'FME': ("N","CA","C","O","CN","O1","CB","CG","SD","CE","OXT"),
    'HIC': ("N","CA","C","O","CB","CG","ND1","CD2","CE1","NE2","CZ","OXT"),
    'HYP': ("N","CA","C","O","CB","CG","CD","OD1","OXT"),
    'IAS': ("N","CA","C","O","CB","CG","OD1","OXT","OD2"),
    'KCX': ("N","CA","C","O","CB","CG","CD","CE","NZ","CX","OXT","OQ1","OQ2"),
    'M3L': ("N","CA","C","O","CB","CG","CD","CE","NZ","OXT","CM1","CM2","CM3"),
    'MLE': ("N","CA","C","O","CN","CB","CG","CD1","CD2","OXT"),
    'MLY': ("N","CA","C","O","CB","CG","CD","CE","NZ","CH1","CH2","OXT"),
    'MLZ': ("N","CA","C","O","CB","CG","CD","CE","NZ","CM","OXT"),
    'MVA': ("N","CA","C","O","CN","CB","CG1","CG2","OXT"),
    'NLE': ("N","CA","C","O","OXT","CB","CG","CD","CE"),
    'OCS': ("N","CA","C","O","CB","SG","OXT","OD1","OD2","OD3"),
    'ORN': ("N","CA","C","O","CB","CG","CD","NE","OXT"),
    'PCA': ("N","CA","C","O","CB","CG","CD","OE","OXT"),
    'SAR': ("N","CA","C","O","CN","OXT"),
    'SCH': ("N","CA","C","O","CB","SG","SD","CE","OXT"),
    'SCY': ("N","CA","C","O","CB","SG","CD","OCD","CE","OXT"),
    'TPO': ("N","CA","C","O","CB","CG2","OG1","P","O1P","O2P","O3P","OXT"),
    'XCP': ("N","CA","C","O","CB","CG","CD","CE","OXT"),
    'YCM': ("N","CA","C","O","CB","SG","CD","CE","OZ1","NZ2","OXT"),
    'DAL': ("N","CA","C","O"),  # D-Alanine
    'DAR': ("N","CA","C","O"),  # D-Arginine
    'DAS': ("N","CA","C","O"),  # D-Asparagine
    'DAN': ("N","CA","C","O"),  # D-Aspartate
    'DCY': ("N","CA","C","O"),  # D-Cysteine
    'DGN': ("N","CA","C","O"),  # D-Glutamine
    'DGL': ("N","CA","C","O"),  # D-Glutamate
    'DHI': ("N","CA","C","O"),  # D-Histidine
    'DIL': ("N","CA","C","O"),  # D-Isoleucine
    'DLE': ("N","CA","C","O"),  # D-Leucine
    'DLY': ("N","CA","C","O"),  # D-Lysine
    'DME': ("N","CA","C","O"),  # D-Methionine
    'DPN': ("N","CA","C","O"),  # D-Phenylalanine
    'DPR': ("N","CA","C","O"),  # D-Proline
    'DSN': ("N","CA","C","O"),  # D-Serine
    'DTH': ("N","CA","C","O"),  # D-Threonine
    'DTR': ("N","CA","C","O"),  # D-Tryptophan
    'DTY': ("N","CA","C","O"),  # D-Tyrosine
    'DVA': ("N","CA","C","O") # D-Valine
}

# Merge standard and NCAA residues
RES_ATOM_ORDER = {r: a for r, a in zip(RES_NAMES_STD, ATOM_ORDER_STD)}
RES_ATOM_ORDER.update(RES_ATOM_ORDER_NCAA)

aa2idx = {(res3, atom): j
          for res3, atoms in RES_ATOM_ORDER.items()
          for j, atom in enumerate(atoms)}

idx2ra = {(res3, j): (res3, atom)
          for res3, atoms in RES_ATOM_ORDER.items()
          for j, atom in enumerate(atoms)}

MAX_ATOMS = max(len(atoms) for atoms in RES_ATOM_ORDER.values())


def writepdb(f, xyz, res3_seq, bfac=None):
    """Write structure to PDB format"""
    f.seek(0)
    f.truncate()
    ctr = 1
    L = len(res3_seq)

    if bfac is None:
        bfac = np.zeros((L, MAX_ATOMS), dtype=float)

    idx = []
    for i in range(L):
        res3 = res3_seq[i]
        for j, xyz_ij in enumerate(xyz[i]):
            key = (res3, j)
            if key not in idx2ra:
                continue
            if np.isnan(xyz_ij).sum() > 0:
                continue
            r, a = idx2ra[key]
            f.write("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n" % (
                "ATOM", ctr, a, r, "A", i+1,
                xyz_ij[0], xyz_ij[1], xyz_ij[2],
                1.0, bfac[i, j]
            ))
            if a == 'CA':
                idx.append(i)
            ctr += 1
    f.flush()
    return np.array(idx)


def TMalign(chainA, chainB):
    """Run TMalign on two protein chains"""
    
    if TMALIGN_PATH is None:
        return None, None
    
    # Create temp files
    fA = tempfile.NamedTemporaryFile(mode='w+t', dir=TEMP_DIR, delete=False, suffix='.pdb')
    fB = tempfile.NamedTemporaryFile(mode='w+t', dir=TEMP_DIR, delete=False, suffix='.pdb')
    mtx = tempfile.NamedTemporaryFile(mode='w+t', dir=TEMP_DIR, delete=False, suffix='.txt')
    
    try:
        idxA = writepdb(fA, chainA['xyz'], chainA['res3_seq'], bfac=chainA['bfac'])
        idxB = writepdb(fB, chainB['xyz'], chainB['res3_seq'], bfac=chainB['bfac'])
        
        fA.close()
        fB.close()
        mtx.close()
        
        # Run TMalign
        cmd = f"{TMALIGN_PATH} {fA.name} {fB.name} -m {mtx.name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 or result.stderr:
            return None, None
        
        lines = result.stdout.split('\n')
        
        # Parse transformation matrix
        with open(mtx.name, 'r') as f:
            mtx_lines = f.readlines()
            if len(mtx_lines) < 5:
                return None, None
            tu = np.fromstring(''.join(l[2:] for l in mtx_lines[2:5]), 
                               dtype=float, sep=' ').reshape((3, 4))
        
        t = tu[:, 0]
        u = tu[:, 1:]
        
        # Parse metrics
        rmsd = float(lines[16].split()[4][:-1])
        seqid = float(lines[16].split()[-1])
        tm1 = float(lines[17].split()[1])
        tm2 = float(lines[18].split()[1])
        
        # Parse alignment
        seq1 = lines[-5]
        seq2 = lines[-3]
        
        ss1 = np.array(list(seq1.strip())) != '-'
        ss2 = np.array(list(seq2.strip())) != '-'
        mask = np.logical_and(ss1, ss2)
        
        alnAB = np.stack((idxA[(np.cumsum(ss1)-1)[mask]],
                          idxB[(np.cumsum(ss2)-1)[mask]]))
        alnBA = np.stack((alnAB[1], alnAB[0]))
        
        resAB = {'rmsd': rmsd, 'seqid': seqid, 'tm': tm1, 'aln': alnAB, 't': t, 'u': u}
        resBA = {'rmsd': rmsd, 'seqid': seqid, 'tm': tm2, 'aln': alnBA, 't': -u.T@t, 'u': u.T}
        
        return resAB, resBA
        
    finally:
        # Clean up temp files
        for fname in [fA.name, fB.name, mtx.name]:
            try:
                os.unlink(fname)
            except:
                pass


def get_tm_pairs(chains):
    """Compute TM-align for all chain pairs"""
    
    if TMALIGN_PATH is None:
        print("# WARNING: TMalign not found, skipping pairwise alignments", file=sys.stderr)
        tm_pairs = {}
        for A in chains.keys():
            L = chains[A]['xyz'].shape[0]
            aln = np.arange(L)[chains[A]['mask'][:, 1]]
            aln = np.stack((aln, aln))
            tm_pairs.update({(A, A): {'rmsd': 0.0, 'seqid': 1.0, 'tm': 1.0, 'aln': aln}})
        return tm_pairs
    
    tm_pairs = {}
    for A, B in combinations(chains.keys(), r=2):
        resAB, resBA = TMalign(chains[A], chains[B])
        if resAB is not None:
            tm_pairs.update({(A, B): resAB})
            tm_pairs.update({(B, A): resBA})
    
    # Add self-alignments
    for A in chains.keys():
        L = chains[A]['xyz'].shape[0]
        aln = np.arange(L)[chains[A]['mask'][:, 1]]
        aln = np.stack((aln, aln))
        tm_pairs.update({(A, A): {'rmsd': 0.0, 'seqid': 1.0, 'tm': 1.0, 'aln': aln}})
    
    return tm_pairs


def parseOperationExpression(expression):
    """Parse assembly operation expression"""
    expression = expression.strip('() ')
    operations = []
    for e in expression.split(','):
        e = e.strip()
        pos = e.find('-')
        if pos > 0:
            start = int(e[0:pos])
            stop = int(e[pos+1:])
            operations.extend([str(i) for i in range(start, stop+1)])
        else:
            operations.append(e)
    return operations


def parseAssemblies(data, chids):
    """Parse biological assembly information"""
    
    xforms = {
        'asmb_chains': None,
        'asmb_details': None,
        'asmb_method': None,
        'asmb_ids': None
    }

    assembly_data = data.getObj("pdbx_struct_assembly")
    assembly_gen = data.getObj("pdbx_struct_assembly_gen")
    oper_list = data.getObj("pdbx_struct_oper_list")

    if (assembly_data is None) or (assembly_gen is None) or (oper_list is None):
        return xforms

    # Build transformation dictionary
    opers = {}
    for k in range(oper_list.getRowCount()):
        key = oper_list.getValue("id", k)
        val = np.eye(4)
        for i in range(3):
            val[i, 3] = float(oper_list.getValue("vector[%d]" % (i+1), k))
            for j in range(3):
                val[i, j] = float(oper_list.getValue("matrix[%d][%d]" % (i+1, j+1), k))
        opers.update({key: val})
    
    chains, details, method, ids = [], [], [], []

    for index in range(assembly_gen.getRowCount()):
        assemblyId = assembly_gen.getValue("assembly_id", index)
        ids.append(assemblyId)

        oper_expression = assembly_gen.getValue("oper_expression", index)
        oper_list_parsed = [parseOperationExpression(expression) 
                            for expression in re.split('\(|\)', oper_expression) if expression]
        
        chains.append(assembly_gen.getValue("asym_id_list", index))

        index_asmb = min(index, assembly_data.getRowCount()-1)
        details.append(assembly_data.getValue("details", index_asmb))
        method.append(assembly_data.getValue("method_details", index_asmb))
        
        if len(oper_list_parsed) == 1:
            xform = np.stack([opers[o] for o in oper_list_parsed[0]])
        elif len(oper_list_parsed) == 2:
            xform = np.stack([opers[o1]@opers[o2] 
                              for o1 in oper_list_parsed[0] 
                              for o2 in oper_list_parsed[1]])
        else:
            continue
        
        xforms.update({'asmb_xform%d' % index: xform})
    
    xforms['asmb_chains'] = chains
    xforms['asmb_details'] = details
    xforms['asmb_method'] = method
    xforms['asmb_ids'] = ids

    return xforms


def parse_mmcif(filename):
    """
    Parse mmCIF file (supports gzipped files)
    Returns: (chains dict, metadata dict)
    """
    chains = {}

    # Handle gzipped files
    with _smart_open_text(filename) as cif:
        reader = PdbxReader(cif)
        data_blocks = []
        reader.read(data_blocks)


    if not data_blocks:
        return {}, {}
    
    data = data_blocks[0]

    # Parse entity and sequence information
    entity_poly = data.getObj('entity_poly')
    if entity_poly is None:
        return {}, {}

    pdbx_poly_seq_scheme = data.getObj('pdbx_poly_seq_scheme')
    if pdbx_poly_seq_scheme is None:
        return {}, {}

    # Build chain mappings
    pdb2asym = {
        (row[pdbx_poly_seq_scheme.getIndex('pdb_strand_id')],
         row[pdbx_poly_seq_scheme.getIndex('asym_id')])
        for row in pdbx_poly_seq_scheme.getRowList()
    }
    pdb2asym = dict(pdb2asym)

    _VALID_TYPES = {'polypeptide(L)', 'polypeptide(D)'}
    chs2num = {}
    type_idx = entity_poly.getIndex('type')
    strand_idx = entity_poly.getIndex('pdbx_strand_id')
    ent_id_idx = entity_poly.getIndex('entity_id')

    for row in entity_poly.getRowList():
        # Only keep polymer types we care about
        if row[type_idx] not in _VALID_TYPES:
            continue

        strand_field = row[strand_idx] or ""
        # Some entries may contain multiple comma-separated strand IDs
        for ch in strand_field.split(','):
            ch = ch.strip()

            # Skip unknown / invalid / placeholder chain IDs like '?' or '.'
            if ch in ("", "?", "."):
                print(f"# Warning: skipping invalid strand id {ch!r} in entity_poly")
                continue

            # Skip chains that do not have a mapping in pdb2asym
            if ch not in pdb2asym:
                print(f"# Warning: strand id {ch!r} not found in pdb2asym mapping. Skipping.")
                continue

            chid = pdb2asym[ch]
            ent_id = row[ent_id_idx]

            # If the same chid appears multiple times, last one wins; this is usually fine
            chs2num[chid] = ent_id


    num2seq = {
        row[entity_poly.getIndex('entity_id')]:
            row[entity_poly.getIndex('pdbx_seq_one_letter_code_can')].replace('\n', '')
        for row in entity_poly.getRowList()
        if row[entity_poly.getIndex('type')] in _VALID_TYPES
    }

    # Parse modified residues
    pdbx_struct_mod_residue = data.getObj('pdbx_struct_mod_residue')
    if pdbx_struct_mod_residue is None:
        modres = {}
    else:
        modres = {
            (r[pdbx_struct_mod_residue.getIndex('label_comp_id')]).upper():
            (r[pdbx_struct_mod_residue.getIndex('parent_comp_id')]).upper()
            for r in pdbx_struct_mod_residue.getRowList()
        }

    # Initialize chains
    for chid, ent_id in chs2num.items():
        seq1 = num2seq.get(ent_id, "")
        L = len(seq1)
        chains[chid] = {
            'seq': seq1,
            'res3_seq': ['UNK'] * L,
            'xyz': np.full((L, MAX_ATOMS, 3), np.nan, dtype=np.float32),
            'mask': np.zeros((L, MAX_ATOMS), dtype=bool),
            'bfac': np.full((L, MAX_ATOMS), np.nan, dtype=np.float32),
            'occ': np.zeros((L, MAX_ATOMS), dtype=np.float32),
        }

    if not chains:
        return {}, {}

    # Parse atom coordinates
    atom_site = data.getObj('atom_site')
    if atom_site is None:
        return chains, {}

    i = {
        'atm': atom_site.getIndex('label_atom_id'),
        'atype': atom_site.getIndex('type_symbol'),
        'res': atom_site.getIndex('label_comp_id'),
        'chid': atom_site.getIndex('label_asym_id'),
        'num': atom_site.getIndex('label_seq_id'),
        'alt': atom_site.getIndex('label_alt_id'),
        'x': atom_site.getIndex('Cartn_x'),
        'y': atom_site.getIndex('Cartn_y'),
        'z': atom_site.getIndex('Cartn_z'),
        'occ': atom_site.getIndex('occupancy'),
        'bfac': atom_site.getIndex('B_iso_or_equiv'),
        'model': atom_site.getIndex('pdbx_PDB_model_num'),
    }

    for row in atom_site.getRowList():
        # Skip hydrogen atoms
        if row[i['atype']] == 'H':
            continue

        chid = str(row[i['chid']])
        if chid not in chains:
            continue

        try:
            num = int(row[i['num']])
        except:
            continue

        # Only process model 1
        try:
            model = int(row[i['model']])
            if model > 1:
                continue
        except:
            pass

        res3_raw = str(row[i['res']]).upper()

        # Determine residue type to use
        if res3_raw in RES_ATOM_ORDER:
            res3_use = res3_raw
        elif res3_raw in modres and modres[res3_raw] in RES_ATOM_ORDER:
            res3_use = modres[res3_raw]
        else:
            res3_use = res3_raw

        if res3_use not in RES_ATOM_ORDER:
            continue

        chain = chains[chid]
        L = len(chain['res3_seq'])
        if 1 <= num <= L:
            chain['res3_seq'][num - 1] = res3_use
        else:
            continue

        atm = str(row[i['atm']]).upper()
        key = (res3_use, atm)
        if key not in aa2idx:
            continue

        atom_idx = aa2idx[key]

        try:
            x = float(row[i['x']])
            y = float(row[i['y']])
            z = float(row[i['z']])
            occ = float(row[i['occ']])
            bfac = float(row[i['bfac']])
        except:
            continue

        idx = (num - 1, atom_idx)
        if occ > chain['occ'][idx]:
            chain['xyz'][idx] = [x, y, z]
            chain['mask'][idx] = True
            chain['occ'][idx] = occ
            chain['bfac'][idx] = bfac

    # Build metadata
    res = None
    if data.getObj('refine') is not None:
        try:
            res = float(data.getObj('refine').getValue('ls_d_res_high', 0))
        except:
            res = None
    if (data.getObj('em_3d_reconstruction') is not None) and (res is None):
        try:
            res = float(data.getObj('em_3d_reconstruction').getValue('resolution', 0))
        except:
            res = None

    chids = list(chains.keys())
    seq_meta = []
    for ch in chids:
        mask_mainchain = chains[ch]['mask'][:, :3].sum(1) == 3
        ref_seq = chains[ch]['seq']
        atom_seq = ''.join([a if m else '-' for a, m in zip(ref_seq, mask_mainchain)])
        seq_meta.append([ref_seq, atom_seq])

    metadata = {
        'method': data.getObj('exptl').getValue('method', 0).replace(' ', '_')
                  if data.getObj('exptl') is not None else None,
        'date': data.getObj('pdbx_database_status').getValue('recvd_initial_deposition_date', 0)
                if data.getObj('pdbx_database_status') is not None else None,
        'resolution': res,
        'chains': chids,
        'seq': seq_meta,
        'id': data.getObj('entry').getValue('id', 0)
              if data.getObj('entry') is not None else None,
    }

    # Parse assemblies
    asmbs = parseAssemblies(data, chains)
    metadata.update(asmbs)

    return chains, metadata

def process_single_file(input_file, output_dir, target_res=None):
    """Process a single CIF file and save outputs under output_dir[/<AA>]/..."""
    try:
        # Parse the CIF file
        chains, metadata = parse_mmcif(input_file)
        if not chains:
            print(f"# Skip {input_file}: no valid chains found", file=sys.stderr)
            return False

        #ID = metadata.get('id', os.path.splitext(os.path.basename(input_file))[0])
        from pathlib import Path

        stem = Path(input_file).name
        if stem.endswith('.cif.gz'):
            stem = stem[:-7]   
        elif stem.endswith('.cif'):
            stem = stem[:-4]   

        ID = stem


        # -------------------------
        # Filter by target residue
        # -------------------------
        # keep all the chains even they don't contain ncaa
        file_target_res = target_res.upper() if target_res else None
        

        # -------------------------
        # Compute TM-align scores
        # -------------------------
        tm_pairs = get_tm_pairs(chains)
        chids = list(chains.keys())
        tm = []
        for a in chids:
            tm_a = []
            for b in chids:
                tm_ab = tm_pairs.get((a, b))
                if tm_ab is None:
                    tm_a.append([0.0, 0.0, 999.9])
                else:
                    tm_a.append([tm_ab[k] for k in ['tm', 'seqid', 'rmsd']])
            tm.append(tm_a)

        metadata['chains'] = chids
        metadata['tm'] = tm

        # -------------------------
        # Decide save directory
        # -------------------------
        if file_target_res and isinstance(file_target_res, str):
            sub_output_dir = os.path.join(output_dir, file_target_res.upper())
        else:
            parent_dir = os.path.basename(os.path.dirname(input_file))
            if len(parent_dir) == 3 and parent_dir.isupper():
                sub_output_dir = os.path.join(output_dir, parent_dir)
            else:
                sub_output_dir = output_dir
        os.makedirs(sub_output_dir, exist_ok=True)

        # ------------------------
        # remove same sequences 
        seen_seqs = set()
        unique_chains = {}
        for k, v in chains.items():
            seq = v.get("seq", "")
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                unique_chains[k] = v
            else:
                print(f"# Skip duplicate chain {k} (identical sequence)", file=sys.stderr)
        chains = unique_chains
        # -------------------------
        # Save per-chain tensors
        # -------------------------
        
        for k, v in chains.items():
            nres = (v['mask'][:, :3].sum(1) == 3).sum()
            print(
                f">{ID}_{k} {metadata.get('date', 'NA')} {metadata.get('method', 'NA')} "
                f"{metadata.get('resolution', 'NA')} {len(v['seq'])} {nres}\n{v['seq']}"
            )

            to_save = {}
            for kc, vc in v.items():
                if kc == 'seq':
                    to_save[kc] = str(vc)
                elif kc == 'res3_seq':
                    to_save[kc] = list(vc)
                elif isinstance(vc, np.ndarray):
                    to_save[kc] = torch.from_numpy(vc)
                else:
                    try:
                        to_save[kc] = torch.as_tensor(vc)
                    except Exception:
                        to_save[kc] = vc

            out_path = os.path.join(sub_output_dir, f"{ID}_{k}.pt")
            torch.save(to_save, out_path)

        # -------------------------
        # Save metadata
        # -------------------------
        meta_pt = {}
        for k, v in metadata.items():
            if isinstance(v, (list, tuple, np.ndarray)) and (k == "tm" or str(k).startswith("asmb_xform")):
                meta_pt[k] = torch.as_tensor(v)
            else:
                meta_pt[k] = v

        meta_path = os.path.join(sub_output_dir, f"{ID}.pt")
        torch.save(meta_pt, meta_path)

        return True

    except Exception as e:
        print(f"# Error processing {input_file}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_dir> <output_dir> [target_residue]")
        print("Example: python script.py /scratch/yx4224/ProteinMPNN/downloads ./output YCM")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    target_res = sys.argv[3] if len(sys.argv) >= 4 else None
    
    # If target_res not specified, try to infer from subdirectory structure
    if target_res is None:
        print("# No target residue specified, will process all chains", file=sys.stderr)
    
    print(f"# Input directory: {input_dir}", file=sys.stderr)
    print(f"# Output directory: {output_dir}", file=sys.stderr)
    print(f"# Target residue: {target_res if target_res else 'ALL'}", file=sys.stderr)
    print(f"# MAX_ATOMS: {MAX_ATOMS}", file=sys.stderr)
    print(f"# TMalign path: {TMALIGN_PATH if TMALIGN_PATH else 'NOT FOUND'}", file=sys.stderr)
    print(f"# Temp directory: {TEMP_DIR if TEMP_DIR else 'system default'}", file=sys.stderr)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CIF files (including gzipped)
    cif_patterns = [
        os.path.join(input_dir, '**', '*.cif'),
        os.path.join(input_dir, '**', '*.cif.gz'),
    ]
    
    cif_files = []
    for pattern in cif_patterns:
        cif_files.extend(glob.glob(pattern, recursive=True))
    
    cif_files = sorted(set(cif_files))
    
    if not cif_files:
        print(f"# ERROR: No CIF files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"# Found {len(cif_files)} CIF files to process", file=sys.stderr)
    
    # Process each file
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, cif_file in enumerate(cif_files, 1):
        print(f"\n# [{i}/{len(cif_files)}] Processing: {cif_file}", file=sys.stderr)
        
        # Determine target residue for this file
        # If target_res is specified globally, use it
        # Otherwise, try to infer from parent directory name
        if target_res:
            file_target_res = target_res
        else:
            parent_dir = os.path.basename(os.path.dirname(cif_file))
            if len(parent_dir) == 3 and parent_dir.isupper():
                file_target_res = parent_dir
                print(f"# Inferred target residue: {file_target_res}", file=sys.stderr)
            else:
                file_target_res = None
        
        result = process_single_file(cif_file, output_dir, file_target_res)
        
        if result:
            success_count += 1
        elif result is False:
            skip_count += 1
        else:
            fail_count += 1
    
    print(f"\n# ===== Summary =====", file=sys.stderr)
    print(f"# Total files: {len(cif_files)}", file=sys.stderr)
    print(f"# Successfully processed: {success_count}", file=sys.stderr)
    print(f"# Skipped (no target residue): {skip_count}", file=sys.stderr)
    print(f"# Failed: {fail_count}", file=sys.stderr)
    print(f"# Output saved to: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()