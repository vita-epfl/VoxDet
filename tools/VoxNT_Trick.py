
import torch
import torch.nn.functional as F
import numpy as np

def run_length_positive(t, dim):
    shape = t.shape
    L = shape[dim]
    out = torch.empty_like(t, dtype=torch.int32)
    
    idx_last = [slice(None)] * len(shape)
    idx_last[dim] = -1
    out[tuple(idx_last)] = torch.tensor(1, dtype=torch.int32, device=t.device)
    
    for i in range(L - 2, -1, -1):
        idx = [slice(None)] * len(shape)
        idx[dim] = i
        idx_next = [slice(None)] * len(shape)
        idx_next[dim] = i + 1
        
        current = t[tuple(idx)]
        nxt = t[tuple(idx_next)]
        
        cond = (current == nxt)
        out_next = out[tuple(idx_next)]
        
        val = torch.where(cond, out_next + 1, torch.tensor(1, dtype=torch.int32, device=t.device))
        out[tuple(idx)] = val
        
    return out

def run_length_along_dim(t, dim, direction):
    if direction == 'positive':
        return run_length_positive(t, dim)
    else:
        t_flip = torch.flip(t, dims=(dim,))
        out_flip = run_length_positive(t_flip, dim)
        return torch.flip(out_flip, dims=(dim,))

def compute_all_direction_distances(gt_occ):
    B, X, Y, Z = gt_occ.shape

    dist_x_pos = run_length_along_dim(gt_occ, 1, 'positive')
    dist_x_neg = run_length_along_dim(gt_occ, 1, 'negative')
    dist_y_pos = run_length_along_dim(gt_occ, 2, 'positive')
    dist_y_neg = run_length_along_dim(gt_occ, 2, 'negative')
    dist_z_pos = run_length_along_dim(gt_occ, 3, 'positive')
    dist_z_neg = run_length_along_dim(gt_occ, 3, 'negative')
    
    distances = torch.stack([dist_x_pos, dist_x_neg, dist_y_pos, dist_y_neg, dist_z_pos, dist_z_neg], dim=1)
    return distances
