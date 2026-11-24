import torch
import torch.nn as nn
import torch.nn.functional as F

def _compute_stats(dist_matrix, mask_pos, mask_neg):
    """
    Helper to compute standard Euclidean statistics.
    Euclidean distances for L2-normalized embeddings range from 0.0 to 2.0.
    """
    pos_dists = dist_matrix[mask_pos]
    neg_dists = dist_matrix[mask_neg]

    # Means
    dist_pos_mean = pos_dists.mean().item() if pos_dists.numel() > 0 else 0.0
    dist_neg_mean = neg_dists.mean().item() if neg_dists.numel() > 0 else 0.0

    # Hards (Max positive distance, Min negative distance)
    dist_pos_hard = pos_dists.max().item() if pos_dists.numel() > 0 else 0.0
    dist_neg_hard = neg_dists.min().item() if neg_dists.numel() > 0 else 0.0

    return {
        "dist_pos_mean": dist_pos_mean,
        "dist_neg_mean": dist_neg_mean,
        "dist_pos_hard": dist_pos_hard,
        "dist_neg_hard": dist_neg_hard,
    }

def batch_hard_triplet_loss(embeddings, labels, margin=0.2, clip_margin=False):
    """
    Euclidean batch-hard triplet loss.
    """
    device = embeddings.device
    B = embeddings.size(0)
    huge_val = 1e9

    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    # --- STANDARDIZED MASKS ---
    labels = labels.view(-1)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1))
    mask_pos.fill_diagonal_(False)
    mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1))

    # --- Hardest Mining ---
    dist_pos = dist_matrix * mask_pos.float()
    hardest_pos, _ = dist_pos.max(dim=1)

    dist_neg = dist_matrix * mask_neg.float() + (~mask_neg).float() * huge_val
    hardest_neg, _ = dist_neg.min(dim=1)

    loss_vec = F.relu(hardest_pos - hardest_neg + margin)

    valid_mask = hardest_neg < huge_val * 0.5
    if valid_mask.sum() == 0:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss = loss_vec[valid_mask].mean()
        if torch.isnan(loss):
            print("[WARN] NaN in triplet loss, zeroing it out.")
            loss = torch.tensor(0.0, device=device, requires_grad=True)

    stats = _compute_stats(dist_matrix, mask_pos, mask_neg)
    active_count = (loss_vec[valid_mask] > 0).sum().item()
    stats['active_anchors_ratio'] = active_count / B

    return loss, stats, margin

def semi_hard_triplet_loss(embeddings, labels, margin=0.2):
    """
    Fully Vectorized Pure Euclidean semi-hard triplet loss.
    """
    device = embeddings.device
    B = embeddings.size(0)

    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    # --- STANDARDIZED MASKS ---
    labels = labels.view(-1)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1))
    mask_pos.fill_diagonal_(False)
    mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1))

    # ---------------------------------------------------------
    # Vectorized Random Positive Sampling
    # ---------------------------------------------------------
    has_pos = mask_pos.any(dim=1)
    prob_p = mask_pos.float() / (mask_pos.float().sum(dim=1, keepdim=True) + 1e-8)
    
    # Dummy probability trick to prevent torch.multinomial crash on empty rows
    prob_p_safe = prob_p.clone()
    prob_p_safe[~has_pos, 0] = 1.0 
    
    sampled_pos_idx = torch.multinomial(prob_p_safe, 1).squeeze(1)
    d_p = dist_matrix[torch.arange(B), sampled_pos_idx]

    # ---------------------------------------------------------
    # Vectorized Semi-Hard Negative Masking
    # ---------------------------------------------------------
    # Expand d_p to broadcast against the [B, B] distance matrix
    d_p_expanded = d_p.unsqueeze(1)
    
    # Pure Semi-Hard Window: d_p < d_n < d_p + margin
    sh_mask = (dist_matrix > d_p_expanded) & (dist_matrix < d_p_expanded + margin) & mask_neg
    
    # Which anchors actually have at least one semi-hard negative?
    has_sh = sh_mask.any(dim=1)

    # ---------------------------------------------------------
    # Vectorized Semi-Hard Sampling
    # ---------------------------------------------------------
    prob_sh = sh_mask.float()
    prob_sh_safe = prob_sh / (prob_sh.sum(dim=1, keepdim=True) + 1e-8)
    
    # Dummy probability trick for anchors with NO semi-hard negatives
    prob_sh_safe[~has_sh, 0] = 1.0 
    
    sampled_sh_idx = torch.multinomial(prob_sh_safe, 1).squeeze(1)
    d_n = dist_matrix[torch.arange(B), sampled_sh_idx]

    # ---------------------------------------------------------
    # Loss Calculation
    # ---------------------------------------------------------
    loss_vec = F.relu(d_p - d_n + margin)
     
    # Only count anchors that HAVE a positive, HAVE a semi-hard negative, AND produced a gradient > 0
    active_mask = (loss_vec > 0) & has_sh & has_pos
    active_count = active_mask.sum().item()

    if active_count > 0:
        loss = loss_vec[active_mask].mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    stats = _compute_stats(dist_matrix, mask_pos, mask_neg)
    stats['active_anchors_ratio'] = active_count / B
    
    return loss, stats, margin

def distance_weighted_triplet_loss(embeddings, labels, margin):
    """
    Triplet Loss using Distance Weighted Sampling (DWS) for negatives
    and Uniform Random Sampling for positives.
    """
    device = embeddings.device
    N = embeddings.size(0)
    dim = embeddings.size(1)
    
    l2_dist = torch.cdist(embeddings, embeddings, p=2)
    
    # --- STANDARDIZED MASKS ---
    labels = labels.view(-1)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1))
    mask_pos.fill_diagonal_(False)
    mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1))
    
    # ---------------------------------------------------------
    # Vectorized Random Positive Sampling
    # ---------------------------------------------------------
    has_pos = mask_pos.any(dim=1)
    prob_p = mask_pos.float() / (mask_pos.float().sum(dim=1, keepdim=True) + 1e-8)
    
    # Dummy probability trick to prevent torch.multinomial crash on empty rows
    prob_p_safe = prob_p.clone()
    prob_p_safe[~has_pos, 0] = 1.0 
    
    sampled_pos_idx = torch.multinomial(prob_p_safe, 1).squeeze(1)
    sampled_pos_dist = l2_dist[torch.arange(N), sampled_pos_idx]
    
    # ------------------------------------------------------------------
    # DWS for Negatives
    # ------------------------------------------------------------------
    has_neg = mask_neg.any(dim=1)
    
    d_n_clamped = torch.clamp(l2_dist, min=0.5, max=1.99)
    log_q_d_inv = ((2.0 - dim) * torch.log(d_n_clamped) - 
                   ((dim - 3.0) / 2.0) * torch.log(1.0 - 0.25 * (d_n_clamped.pow(2))))
    
    log_q_d_inv[~mask_neg] = -float('inf')
    
    max_log_q, _ = log_q_d_inv.max(dim=1, keepdim=True)
    q_d_inv = torch.exp(log_q_d_inv - max_log_q)
    q_d_inv[~mask_neg] = 0.0 
    
    prob_n = q_d_inv / (q_d_inv.sum(dim=1, keepdim=True) + 1e-8)
    
    # Mirroring the dummy trick for negatives just to be absolutely bulletproof
    prob_n_safe = prob_n.clone()
    prob_n_safe[~has_neg, 0] = 1.0
    
    sampled_neg_idx = torch.multinomial(prob_n_safe, 1).squeeze(1)
    sampled_neg_dist = l2_dist[torch.arange(N), sampled_neg_idx]
    
    # ------------------------------------------------------------------
    # Total Triplet Loss Calculation
    # ------------------------------------------------------------------
    # The standard Triplet math: max(0, d_p - d_n + margin)
    loss_vec = F.relu(sampled_pos_dist - sampled_neg_dist + margin)
    
    # Ensure anchors with no valid positives or negatives don't generate false gradients
    valid_anchors = has_pos & has_neg
    loss_vec = loss_vec * valid_anchors.float()
    
    active_mask = loss_vec > 0
    active_count = active_mask.sum().item()
    
    loss = loss_vec[active_mask].mean() if active_count > 0 else torch.tensor(0.0, device=device, requires_grad=True)
    
    # Assuming _compute_stats is available in your scope
    stats = _compute_stats(l2_dist, mask_pos, mask_neg)
    stats['active_anchors_ratio'] = active_count / N
    
    return loss, stats, margin

def distance_weighted_margin_loss(embeddings, labels, beta, margin):
    """
    Margin Loss using Distance Weighted Sampling (DWS) for negatives
    and Uniform Random Sampling for positives.
    """
    device = embeddings.device
    N = embeddings.size(0)
    dim = embeddings.size(1)
    
    l2_dist = torch.cdist(embeddings, embeddings, p=2)
    
    # --- STANDARDIZED MASKS ---
    labels = labels.view(-1)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1))
    mask_pos.fill_diagonal_(False)
    mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1))
    
    anchor_beta = beta[labels]

    # ---------------------------------------------------------
    # Vectorized Random Positive Sampling
    # ---------------------------------------------------------
    has_pos = mask_pos.any(dim=1)
    prob_p = mask_pos.float() / (mask_pos.float().sum(dim=1, keepdim=True) + 1e-8)
    
    # Dummy probability trick to prevent torch.multinomial crash on empty rows
    prob_p_safe = prob_p.clone()
    prob_p_safe[~has_pos, 0] = 1.0 
    
    sampled_pos_idx = torch.multinomial(prob_p_safe, 1).squeeze(1)
    sampled_pos_dist = l2_dist[torch.arange(N), sampled_pos_idx]
    
    pos_loss = F.relu(sampled_pos_dist - anchor_beta + margin)
    
    # ------------------------------------------------------------------
    # DWS for Negatives
    # ------------------------------------------------------------------
    d_n_clamped = torch.clamp(l2_dist, min=0.5, max=1.99)
    log_q_d_inv = ((2.0 - dim) * torch.log(d_n_clamped) - 
                   ((dim - 3.0) / 2.0) * torch.log(1.0 - 0.25 * (d_n_clamped.pow(2))))
    
    log_q_d_inv[~mask_neg] = -float('inf')
    
    max_log_q, _ = log_q_d_inv.max(dim=1, keepdim=True)
    q_d_inv = torch.exp(log_q_d_inv - max_log_q)
    q_d_inv[~mask_neg] = 0.0 
    
    prob_n = q_d_inv / (q_d_inv.sum(dim=1, keepdim=True) + 1e-8)
    sampled_neg_idx = torch.multinomial(prob_n, 1).squeeze(1)
    sampled_neg_dist = l2_dist[torch.arange(N), sampled_neg_idx]
    
    neg_loss = F.relu(anchor_beta - sampled_neg_dist + margin)
    
    # ------------------------------------------------------------------
    # Total Loss Calculation
    # ------------------------------------------------------------------
    loss_vec = pos_loss + neg_loss
    active_mask = loss_vec > 0
    active_count = active_mask.sum().item()
    
    loss = loss_vec[active_mask].mean() if active_count > 0 else torch.tensor(0.0, device=device, requires_grad=True)
    
    stats = _compute_stats(l2_dist, mask_pos, mask_neg)
    stats['active_anchors_ratio'] = active_count / N
    
    return loss, stats, margin

class MarginLossBaselines(nn.Module):
    """
    Lightweight nn.Module wrapper required to hold the learnable beta parameters
    when running baseline DWS Margin Loss.
    """
    def __init__(self, margin=0.2, init_beta=1.2, num_classes=100):
        super().__init__()
        self.margin = margin
        self.beta = nn.Parameter(torch.ones(num_classes) * init_beta)

    def forward(self, embeddings, labels):
        return distance_weighted_margin_loss(embeddings, labels, self.beta, self.margin)