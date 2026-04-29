"""Graph Attention Autoencoder for mutation signature attribution.

Architecture:
  Encoder: 3-layer GAT (4-head) + Softplus → Z (n_bins, k), non-negative
  Decoder: V̂ = Z · H, H COSMIC-initialized, learnable, L1-normalized rows
  
Loss:
  L = L_recon(KL) + λ_tad·L_TAD(contrastive) + λ_cos·L_cosmic(cosine) + λ_l2·L_reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """3-layer GAT encoder with non-negative output."""
    
    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv3 = GATConv(hidden * heads, out_channels, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.softplus(x)  # non-negativity
        return x


class MLPEncoder(nn.Module):
    """MLP encoder (no graph) — ablation baseline."""
    
    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index=None):
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        x = F.softplus(x)
        return x


class GraphSignatureAE(nn.Module):
    """Graph-based Explainable Autoencoder for mutation signatures.
    
    V ≈ Z · H where Z = Encoder(V, A), H is COSMIC-initialized.
    """
    
    def __init__(self, H_cosmic: torch.Tensor, hidden: int = 64,
                 heads: int = 4, dropout: float = 0.1, use_graph: bool = True):
        """
        Args:
            H_cosmic: COSMIC signature matrix (k, 96), L1-normalized rows.
            hidden: Hidden dimension in encoder.
            heads: Number of attention heads.
            dropout: Dropout rate.
            use_graph: If True, use GAT encoder; if False, use MLP.
        """
        super().__init__()
        k = H_cosmic.shape[0]
        in_ch = H_cosmic.shape[1]  # 96
        
        if use_graph:
            self.encoder = GATEncoder(in_ch, hidden, k, heads=heads, dropout=dropout)
        else:
            self.encoder = MLPEncoder(in_ch, hidden, k, dropout=dropout)
        
        # Decoder: FROZEN H (COSMIC fixed — ensures fair comparison with SigProfiler GT)
        self.register_buffer('H', H_cosmic.clone())
        self.k = k
    
    def get_H_normalized(self):
        """L1-normalize H rows (for decoding). H is frozen COSMIC."""
        return self.H / (self.H.sum(dim=1, keepdim=True) + 1e-10)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (n_bins, 96) — mutation counts.
            edge_index: (2, E) edge indices.
        
        Returns:
            V_hat: Reconstructed (n_bins, 96).
            Z: Latent exposure (n_bins, k).
        """
        Z = self.encoder(x, edge_index)
        H = self.get_H_normalized()
        V_hat = Z @ H  # (n_bins, 96)
        return V_hat, Z
    
    def loss(self, V, V_hat, Z, tad_assignment=None,
             lambda_tad=0.1, lambda_cos=0.01, lambda_l2=1e-4,
             tad_margin=1.0, n_tad_pairs=1000):
        """Compute total loss.
        
        Args:
            V: Ground truth (n_bins, 96).
            V_hat: Reconstructed (n_bins, 96).
            Z: Latent (n_bins, k).
            tad_assignment: TAD label per bin, -1 = unassigned.
            lambda_tad: TAD contrastive loss weight.
            lambda_cos: COSMIC cosine regularization weight.
            lambda_l2: L2 regularization weight.
        """
        # --- L_recon: Generalized KL divergence ---
        V_hat_safe = V_hat + 1e-10
        # KL(V || V_hat) = V * log(V / V_hat) - V + V_hat
        # For zero entries in V: 0 * log(0) = 0, so only -V + V_hat = V_hat
        mask = V > 0
        kl = torch.zeros_like(V)
        kl[mask] = V[mask] * torch.log(V[mask] / V_hat_safe[mask]) - V[mask] + V_hat_safe[mask]
        kl[~mask] = V_hat_safe[~mask]
        L_recon = kl.sum() / V.shape[0]
        
        # --- L_TAD: Contrastive loss ---
        L_tad = torch.tensor(0.0, device=V.device)
        if tad_assignment is not None and lambda_tad > 0:
            valid = tad_assignment >= 0
            if valid.sum() > 10:
                valid_idx = torch.where(valid)[0]
                valid_tads = tad_assignment[valid]
                
                # Sample pairs
                n_pairs = min(n_tad_pairs, len(valid_idx) * 2)
                idx_a = valid_idx[torch.randint(len(valid_idx), (n_pairs,))]
                idx_b = valid_idx[torch.randint(len(valid_idx), (n_pairs,))]
                
                same_tad = (tad_assignment[idx_a] == tad_assignment[idx_b])
                
                dists = ((Z[idx_a] - Z[idx_b]) ** 2).sum(dim=1)
                
                # Same TAD: minimize distance; Different TAD: push apart
                pos_loss = dists[same_tad].mean() if same_tad.any() else torch.tensor(0.0, device=V.device)
                neg_dists = dists[~same_tad]
                neg_loss = F.relu(tad_margin - neg_dists).mean() if (~same_tad).any() else torch.tensor(0.0, device=V.device)
                
                L_tad = pos_loss + neg_loss
        
        # --- L_cosmic: Not needed (H is frozen) ---
        L_cos = torch.tensor(0.0, device=V.device)
        
        # --- L_l2: Regularization on Z ---
        L_l2 = (Z ** 2).mean()
        
        total = L_recon + lambda_tad * L_tad + lambda_cos * L_cos + lambda_l2 * L_l2
        
        return total, {
            "recon": L_recon.item(),
            "tad": L_tad.item(),
            "cosmic": L_cos.item(),
            "l2": L_l2.item(),
            "total": total.item(),
        }
