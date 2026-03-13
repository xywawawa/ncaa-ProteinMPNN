from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools

from parse_cif_noX import RES_ATOM_ORDER

class Res3Tokenizer:
    def __init__(self, vocab):
        self.vocab = [v.upper() for v in vocab]
        self.stoi = {t:i for i,t in enumerate(self.vocab)}
        assert len(self.vocab) == len(self.stoi), "vocab has redundant token"
        self.unk_index = len(self.vocab) - 1
    @property
    def size(self): return len(self.vocab)
    def encode(self, tokens):
        out = []
        for t in tokens:
            t = (t or "UNK").upper()
            out.append(self.stoi.get(t, self.unk_index))
        return out
    def decode(self, ids):
        return [self.vocab[i] if 0 <= i < self.size else "UNK" for i in ids]

def build_tokenizer(allowed_residues= {
        # canonical 20
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL",
        # ncAA
        "4BF", "MLE", "TPO", "HYP", "SCY", "CGU", "ABA", "MLY", "CXM", "PCA",
        "CSX", "HIC", "SAR", "MLZ", "SCH", "ALY", "CSD", "CSS", "CSO", "ORN",
        "FME", "CME", "KCX", "M3L", "AIB", "XCP", "MVA", "YCM", "DHA", "NLE", "OCS",
        #daa
        "DAL","DAR","DAS","DCY","DGL","DGN",
        "DHI","DIL","DLE","DLY","DPN","DPR",
        "DSN","DTH","DTR","DTY","DVA"

    }
):
    full_vocab = sorted(list(RES_ATOM_ORDER.keys()))
    if allowed_residues is not None:
        full_vocab = [v for v in full_vocab if v in allowed_residues]
    res3_vocab = full_vocab + ['UNK']
    return Res3Tokenizer(res3_vocab)


def featurize(batch, device, tokenizer=None):
    """
    - batch[i] includes:
        - masked_list, visible_list, num_of_chains
        - each chain:
            - seq_chain_{letter} : one letter(not used)
            - res3_chain_{letter}: list[str] three letters
            - coords_chain_{letter}: dict of N/CA/C/O coordinates [L,3]
    - return:
        X [B,L,4,3], S [B,L] (res3 index),
    """
    assert tokenizer is not None, "pls input Res3Tokenizer to featurize(tokenizer=...)"
    B = len(batch)

    # max length
    lengths = np.array([sum(len(b[f'seq_chain_{ch}']) for ch in (b['masked_list']+b['visible_list']))
                        for b in batch], dtype=np.int32)
    L_max = int(lengths.max())

    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    #mask_self = np.ones([B, L_max, L_max], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S = np.zeros([B, L_max], dtype=np.int32)

    # only when batch doesn't provide res3_chain_* 
    one2three = {
        'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY',
        'H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER',
        'T':'THR','W':'TRP','Y':'TYR','V':'VAL','X':'UNK'
    }

    for i, b in enumerate(batch):
        masked_chains = list(b['masked_list'])
        visible_chains = list(b['visible_list'])

        vis_map, mask_map = {}, {}
        for letter in masked_chains + visible_chains:
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                vis_map[letter] = chain_seq
            else:
                mask_map[letter] = chain_seq
        for km, vm in mask_map.items():
            for kv, vv in vis_map.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)

        all_chains = masked_chains + visible_chains
        #random.shuffle(all_chains)

        x_chain_list, chain_mask_list, chain_seq3_list, chain_encoding_list = [], [], [], []
        c = 1; l0 = 0; l1 = 0

        for letter in all_chains:
            seq1 = b[f'seq_chain_{letter}']
            if f'res3_chain_{letter}' in b:
                seq3_list = [str(t).upper() for t in b[f'res3_chain_{letter}']]
            else:
                seq3_list = [one2three.get(a.upper(),'UNK') for a in seq1]

            Lc = len(seq3_list)
            coords = b[f'coords_chain_{letter}']

            # [L,4,3] N/CA/C/O
            x_chain = np.stack([coords[f'{atom}_chain_{letter}'] for atom in ['N','CA','C','O']], 1)
            x_chain_list.append(x_chain)

            # mask: masked=1, visible=0
            m = np.ones(Lc, dtype=np.float32) if letter in masked_chains else np.zeros(Lc, dtype=np.float32)
            chain_mask_list.append(m)

            chain_seq3_list.extend(seq3_list)
            chain_encoding_list.append(c*np.ones(Lc, dtype=np.int32))

            l1 += Lc
            #mask_self[i, l0:l1, l0:l1] = 0.0
            residue_idx[i, l0:l1] = 100*(c-1) + np.arange(l0, l1, dtype=np.int32)
            l0 += Lc; c += 1

        # concate / pad
        x = np.concatenate(x_chain_list, 0)
        m = np.concatenate(chain_mask_list, 0)
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        l = len(chain_seq3_list)

        # seq -> index (res3)
        indices = np.asarray(tokenizer.encode(chain_seq3_list), dtype=np.int32)
        S[i, :l] = indices

        X[i, :l, :, :] = x
        if l < L_max:
            X[i, l:, :, :] = np.nan

        chain_M[i, :l] = m
        chain_encoding_all[i, :l] = chain_encoding

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.0

    # to tensor
    residue_idx = torch.from_numpy(residue_idx).long().to(device)
    S = torch.from_numpy(S).long().to(device) #S consists of token ids. type: long
    X = torch.from_numpy(X).float().to(device)
    mask = torch.from_numpy(mask).float().to(device)
    #mask_self = torch.from_numpy(mask_self).float().to(device)
    mask_self = None
    chain_M = torch.from_numpy(chain_M).float().to(device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).long().to(device)
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all

def featurize_one_nopad(raw, tokenizer, device):
    import numpy as np, torch
    one2three = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY',
                 'H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER',
                 'T':'THR','W':'TRP','Y':'TYR','V':'VAL','X':'UNK'}

    masked = list(raw['masked_list']); visible = list(raw['visible_list'])
    all_chains = masked + visible

    X_list, S_list, mask_list, chainM_list, chainEnc_list, resid_list = [], [], [], [], [], []
    c = 1
    for ch in all_chains:
        seq1 = raw[f'seq_chain_{ch}']
        res3 = raw.get(f'res3_chain_{ch}', [one2three.get(a.upper(),'UNK') for a in seq1])
        Lc = len(res3)
        coords = raw[f'coords_chain_{ch}']
        x = np.stack([coords[f'{a}_chain_{ch}'] for a in ['N','CA','C','O']], 1)  # [Lc,4,3]

        X_list.append(torch.tensor(x, dtype=torch.float))
        S_list.append(torch.tensor(tokenizer.encode(res3), dtype=torch.long))
        mask_list.append(torch.ones(Lc, dtype=torch.float))                     
        chainM_list.append(torch.ones(Lc) if ch in masked else torch.zeros(Lc)) 
        chainEnc_list.append(torch.full((Lc,), c, dtype=torch.long))            
        resid_list.append(torch.arange(Lc, dtype=torch.long) + 100*(c-1))       
        c += 1

    X   = torch.cat(X_list,   0).to(device)        # [L,4,3]
    S   = torch.cat(S_list,   0).to(device)        # [L]
    mask= torch.cat(mask_list,0).to(device)        # [L]
    chain_M = torch.cat(chainM_list,0).to(device)  # [L]
    chain_encoding_all = torch.cat(chainEnc_list,0).to(device)  # [L]
    residue_idx = torch.cat(resid_list,0).to(device)            # [L]

    return dict(X=X, S=S, mask=mask, chain_M=chain_M,
                chain_encoding_all=chain_encoding_all, residue_idx=residue_idx)

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    V = log_probs.size(-1)                 # vocab size
    S_onehot = torch.nn.functional.one_hot(S, V).float()
    S_onehot = S_onehot + weight / float(V)
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0
    return loss, loss_av



# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E



class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    def __init__(self, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False,
        use_chem_emb=False,
        chem_emb_weight=None,   # [vocab, Din] embeddings for each aa
        chem_freeze=True,
        chem_proj_bias=False,
        bilinear_rank: int = 64, 
        use_norm: bool = True,
        num_letters = 20):
        super(ProteinMPNN, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.use_chem_emb = use_chem_emb
        self.bilinear_rank = bilinear_rank
        self.use_norm = use_norm

        # Featurization layers
        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        # --- sequence embedding branch (choose one) ---
        if self.use_chem_emb:
            assert chem_emb_weight is not None, "use_chem_emb=True but didn't provide chem_emb_weight"
            V, Din = chem_emb_weight.shape
            assert V == vocab, f"chem_emb rows {V} != vocab {vocab}"
            self.chem_table = nn.Embedding.from_pretrained(chem_emb_weight, freeze=chem_freeze)
            self.chem_proj  = nn.Linear(Din, hidden_dim, bias=chem_proj_bias)

            self.U = nn.Linear(hidden_dim, bilinear_rank, bias=False)  # H -> r
            self.V = nn.Linear(hidden_dim, bilinear_rank, bias=False)  # H -> r

            # Optional: class bias + learnable temperature (logit scale)
            self.class_bias = nn.Parameter(torch.zeros(vocab))
            self.logit_scale = nn.Parameter(torch.tensor(1.0))  # start from 1.0, learnable temperature
        else:
            self.W_s = nn.Embedding(vocab, hidden_dim)
            self.W_out = nn.Linear(hidden_dim, vocab, bias=True)  

        # Encoder / Decoder
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
                                             for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
                                             for _ in range(num_decoder_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---- embeddings ----
    def embed_tokens(self, S: torch.LongTensor) -> torch.Tensor:
        if self.use_chem_emb:
            H = self.chem_proj(self.chem_table(S))  # [B, L, H]
            # set special token embedding as 0 to avoid leakage (assume UNK at vocab-1)
            special_idx = self.vocab - 1
            H = torch.where((S == special_idx).unsqueeze(-1), torch.zeros_like(H), H)
            return H
        else:
            return self.W_s(S)  # [B, L, H]

    # ---- class prototypes for similarity head ----
    def get_class_prototypes(self) -> torch.Tensor:
        # [vocab, H]
        return self.chem_proj(self.chem_table.weight)

    # ---- unified logits ----
    def compute_logits(self, h_V: torch.Tensor) -> torch.Tensor:
        # logits = (U*h_V)*(V*P)^T + b
        # h_V: [B, L, H]
        if self.use_chem_emb:
            P = self.get_class_prototypes()         # [V, H]
            class_bias = self.class_bias

            # Project both sides to r-dim
            hU = self.U(h_V)                        # [B, L, r]
            PU = self.V(P)                          # [V, r]  ← 不再写死 21

            if self.use_norm:
                hU = F.normalize(hU, dim=-1)
                PU = F.normalize(PU, dim=-1)

            logits = F.linear(hU, PU, class_bias)   # [B, L, V]
            logits = self.logit_scale * logits      # temperature
            return logits
        else:
            return self.W_out(h_V)

    # def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order=False, decoding_order=None):
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        B, L = E.shape[0], E.shape[1]
        h_V = torch.zeros((B, L, self.hidden_dim), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            # h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend, use_reentrant=False)
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.embed_tokens(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions
        decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=device))))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.compute_logits(h_V) 
        # NOT ALLOWED TO PREDICT UNK
        special_idx = logits.size(-1) - 1
        logits[..., special_idx] = -1e4

        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def sample_ar(
        self,
        X,
        S_true,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        temperature: float = 1.0,
        banned_idx: int | list[int] | torch.Tensor | None = None,
    ):
        """
        Autoregressive sampling

        input:
          X:               [B,L,4,3]
          S_true:          [B,L]  
          mask:            [B,L]  
          chain_M:         [B,L]  design mask(B chain is 1)
          residue_idx:     [B,L]
          chain_encoding_all: [B,L]
          temperature: float

        output:
          dict{
            "S":           [B,L]  
            "chosen_logp": [B,L]  
            "decoding_order": [B,L]  
          }
        """
        device = X.device
        B, L = S_true.shape

        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((B, L, self.hidden_dim), device=device)
        h_E = self.W_e(E)

        # encoder: unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        chain_mask = chain_M * mask  # [B,L], only be 1 at design position
        randn = torch.randn_like(chain_mask)
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * torch.abs(randn), dim=1
        )  # [B,L]

        mask_size = E_idx.shape[1]  # = L
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()  # [B,L,L]

        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )  # [B,L,L]

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)  # [B,L,K,1]
        mask_1D = mask.view(B, L, 1, 1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_S = torch.zeros_like(h_V, device=device)          # [B,L,H]
        S = torch.zeros((B, L), dtype=torch.long, device=device)  # final generated sequence
        chosen_logp = torch.zeros((B, L), dtype=torch.float32, device=device)

        banned_tensor = None
        if banned_idx is not None:
            if isinstance(banned_idx, torch.Tensor):
                banned_tensor = banned_idx.to(device=device, dtype=torch.long).view(-1)
            elif isinstance(banned_idx, (list, tuple, set)):
                banned_tensor = torch.tensor(list(banned_idx), dtype=torch.long, device=device)
            else:
                banned_tensor = torch.tensor([int(banned_idx)], dtype=torch.long, device=device)
            banned_tensor = banned_tensor[(banned_tensor >= 0) & (banned_tensor < self.vocab)]
            if banned_tensor.numel() == 0:
                banned_tensor = None

        # encoder features for decoder
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        # decoder
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))
        ]

        # simplify here, no  omit_AA / bias / pssm, maybe to be done in the future
        for t_ in range(L):
            t = decoding_order[:, t_]  # [B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])  # [B,1]
            mask_gathered = torch.gather(mask, 1, t[:, None])              # [B,1]

           
            if (mask_gathered == 0).all():
                S_t = torch.gather(S_true, 1, t[:, None])  # [B,1]
            else:
                # --- decoder ---
                E_idx_t = torch.gather(
                    E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
                )  # [B,1,K]
                h_E_t = torch.gather(
                    h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1])
                )  # [B,1,K,H]

                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)  # [B,1,K,H+H]
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw,
                    1,
                    t[:, None, None, None].repeat(
                        1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]
                    ),
                )  # [B,1,K,H+H]

                mask_t = torch.gather(mask, 1, t[:, None])  # [B,1]

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(
                        h_V_stack[l],
                        1,
                        t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]),
                    )  # [B,1,H]
                    h_ESV_t = torch.gather(
                        mask_bw,
                        1,
                        t[:, None, None, None].repeat(
                            1, 1, mask_bw.shape[-2], mask_bw.shape[-1]
                        ),
                    ) * h_ESV_decoder_t + h_EXV_encoder_t

                    h_V_stack[l + 1].scatter_(
                        1,
                        t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t),
                    )

                # sample
                h_V_t = torch.gather(
                    h_V_stack[-1],
                    1,
                    t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]),
                )[:, 0, :]  # [B,H]
                logits = self.compute_logits(h_V_t) / float(max(1e-6, temperature))  # [B,V]

                if banned_tensor is not None:
                    logits[:, banned_tensor] = -1e9

                probs = F.softmax(logits, dim=-1)  # [B,V]

                if banned_tensor is not None:
                    row_sum = probs.sum(dim=-1, keepdim=True)
                    zero_rows = (row_sum <= 1e-12).squeeze(-1)
                    if zero_rows.any():
                        fallback = torch.ones((int(zero_rows.sum().item()), probs.shape[1]), device=device, dtype=probs.dtype)
                        fallback[:, banned_tensor] = 0.0
                        fallback = fallback / (fallback.sum(dim=-1, keepdim=True) + 1e-12)
                        probs[zero_rows] = fallback
                        row_sum = probs.sum(dim=-1, keepdim=True)
                    probs = probs / (row_sum + 1e-12)
                S_t = torch.multinomial(probs, 1)  # [B,1]

                # log p（only for design location）
                logp_full = torch.log(probs + 1e-8)                # [B,V]
                logp_selected = logp_full.gather(1, S_t)[:, 0]     # [B]
                logp_selected = logp_selected * chain_mask_gathered[:, 0]  # [B]

                chosen_logp.scatter_(
                    1,
                    t[:, None],                      # [B,1]
                    logp_selected[:, None],          # [B,1]
                )

            # final token（design location use S_t，others use S_true）
            S_true_gathered = torch.gather(S_true, 1, t[:, None])  # [B,1]
            S_t_final = (
                S_t * chain_mask_gathered + S_true_gathered * (1.0 - chain_mask_gathered)
            ).long()  # [B,1]

            # update h_S / S
            h_embed = self.embed_tokens(S_t_final)  # [B,1,H]
            h_S.scatter_(
                1,
                t[:, None, None].repeat(1, 1, h_embed.shape[-1]),
                h_embed,
            )
            S.scatter_(1, t[:, None], S_t_final)

        return {
            "S": S,                               # [B,L]
            "chosen_logp": chosen_logp,           # [B,L]
            "decoding_order": decoding_order,     # [B,L]
        }





class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )
