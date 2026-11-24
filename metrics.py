import torch
from sklearn.metrics import normalized_mutual_info_score

# ----------------------------------------------------------------------
# Custom GPU K-Means Implementation
# ----------------------------------------------------------------------
def kmeans_pytorch(X, num_clusters, max_iter=20, tol=1e-4, device='cuda', seed=33):
    """
    Simple K-means implementation on GPU.
    Args:
        X: [N, D] tensor
        num_clusters: int
    Returns:
        cluster_labels: [N] tensor (int)
    """
    N, D = X.shape

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Randomly initialize centroids from the data
    # (Using random selection is faster than k-means++ for this purpose)
    indices = torch.randperm(N, device=device, generator=g)[:num_clusters]
    centroids = X[indices].clone()

    for i in range(max_iter):
        # Compute distances: ||X - C||^2
        # Using pure tensor broadcasting might be memory heavy, so we use efficient cdist or expansion
        # dists: [N, K]
        dists = torch.cdist(X, centroids)

        # Assign to nearest centroid
        labels = torch.argmin(dists, dim=1)

        # Update centroids
        # We scatter_add to compute new means
        # Prepare denominator (counts per cluster)
        counts = torch.bincount(labels, minlength=num_clusters).float().view(-1, 1)
        
        # Prepare numerator (sum of vectors per cluster)
        # We can use index_add_ or scatter_add_
        # Create a zero tensor for new centroids
        new_centroids = torch.zeros_like(centroids)

        # Add X values into the corresponding cluster bin
        # Note: index_add is faster but requires transposing X usually. 
        # A simple loop over unique labels is often clearer but slower.
        # Here is a vectorized mask approach (fast for K < 5000)
        
        # Optimization: One-hot encoding method uses too much RAM (N * K).
        # We use a loop over clusters if K is large, or scatter_add if efficient.
        # For simplicity and memory safety in ReID (K can be large):
        
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.any():
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                # If a cluster is empty, re-init it to a random point
                random_idx = torch.randint(0, N, (1,), device=device, generator=g).item()
                new_centroids[k] = X[random_idx]

        # Check convergence
        shift = torch.norm(centroids - new_centroids)
        centroids = new_centroids
        if shift < tol:
            break
            
    return labels

# ----------------------------------------------------------------------
# Metric Wrappers
# ----------------------------------------------------------------------

@torch.no_grad()
def compute_embeddings(model, loader, device):
    model.eval()
    all_emb = []
    all_labels = []
    
    # Extract features
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        emb = model(imgs)
        all_emb.append(emb)
        all_labels.append(labels)

    if not all_emb:
        return None, None

    # Concatenate on GPU
    emb = torch.cat(all_emb, dim=0)         # [N, D]
    labels = torch.cat(all_labels, dim=0)   # [N]
    
    return emb, labels

@torch.no_grad()
def compute_retrieval_metrics(emb, labels, chunk_size=5000, ranks=[1, 5, 10]):
    """
    Computes R@1, R@5, R@10, and mAP on GPU using Euclidean Distance.
    Uses chunking to avoid OOM for large N.
    """
    device = emb.device
    N = emb.shape[0]
    
    cmc = torch.zeros(len(ranks), device=device)
    aps = 0.0
    valid_queries = 0
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        query_labels = labels[i:end]
        
        # [Chunk, N] Euclidean distance matrix
        dists = torch.cdist(emb[i:end], emb, p=2)
        
        # Mask self-matches
        row_indices = torch.arange(end - i, device=device)
        global_indices = row_indices + i
        # Use positive infinity so self-matches have the highest distance
        dists[row_indices, global_indices] = float('inf')
        
        # Sort ascending to get ranking (smallest distance = rank 1)
        sorted_indices = dists.argsort(dim=1, descending=False)
        
        # Binary Match Matrix: [Chunk, N]. 1 if match, 0 if not.
        matches = (labels[sorted_indices] == query_labels.unsqueeze(1)).float()
        
        # Number of positive matches for each query
        num_pos = matches.sum(dim=1)
        
        # Valid queries are those with at least one true match in the gallery
        valid_mask = num_pos > 0
        if not valid_mask.any():
            continue
            
        matches_valid = matches[valid_mask]
        num_pos_valid = num_pos[valid_mask]
        
        # 1. Compute CMC (Recall @ K)
        for j, k in enumerate(ranks):
            # A query is successful at rank K if there is at least one match in the top K
            cmc[j] += matches_valid[:, :k].sum(dim=1).clamp(max=1.0).sum()
            
        # 2. Compute Average Precision (AP)
        cum_matches = torch.cumsum(matches_valid, dim=1) # Cumulative positives up to each rank
        rank_positions = torch.arange(1, N + 1, device=device, dtype=torch.float32).unsqueeze(0)
        
        # Precision at each rank position
        precisions = cum_matches / rank_positions
        
        # AP is the average of precisions ONLY at the ranks where a positive match occurred
        ap = (precisions * matches_valid).sum(dim=1) / num_pos_valid
        
        aps += ap.sum().item()
        valid_queries += valid_mask.sum().item()
        
    if valid_queries == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    cmc = cmc / valid_queries
    mAP = aps / valid_queries
    
    # Return r1, r5, r10, map
    return cmc[0].item(), cmc[1].item(), cmc[2].item(), mAP

@torch.no_grad()
def nmi_gpu(emb, labels, seed=33):
    """
    GPU-accelerated NMI calculation.
    """

    # Get number of clusters (identities)
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return 0.0

    # Run K-Means on GPU which is the heavy lifting
    cluster_labels_tensor = kmeans_pytorch(emb, num_clusters=n_clusters, max_iter=20, device=emb.device, seed=seed)
    
    # Compute Score
    # sklearn's NMI function expects CPU numpy arrays, but it's very fast
    # once the clustering is done. The transfer of just the labels (integers) is instant.
    labels_true = labels.cpu().numpy()
    labels_pred = cluster_labels_tensor.cpu().numpy()
    
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return float(nmi)

@torch.no_grad()
def evaluate_metrics(model, val_loader, device, slow_cycle=False, seed=33):

    torch.cuda.empty_cache() 
    
    # 1. Extraction (GPU)
    emb, labels = compute_embeddings(model, val_loader, device)
    if emb is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 2. Recall & mAP (GPU)
    r1, r5, r10, mAP = compute_retrieval_metrics(emb, labels)
    
    # 3. NMI (GPU-Accelerated)
    nmi = nmi_gpu(emb, labels, seed=seed)

    # 4. Global Means
    mu_p, mu_n, hard_p, hard_n = compute_validation_global_geometric_means(emb, labels)
    
    torch.cuda.empty_cache()
    
    return r1, r5, r10, mAP, nmi, mu_p, mu_n, hard_p, hard_n

# ----------------------------------------------------------------------
# Validation geometry
# ----------------------------------------------------------------------

@torch.no_grad()
def compute_validation_global_geometric_means(emb, labels, chunk_size=2048):
    """
    Calculates mu_p, mu_n, hard_p, and hard_n for the ENTIRE validation set 
    using Euclidean Distance.
    """
    N = emb.shape[0]
    
    # Accumulators for Means
    sum_p, sum_n = 0.0, 0.0
    count_p, count_n = 0, 0
    
    # Accumulators for Hards (Averaged per anchor)
    sum_hard_p, sum_hard_n = 0.0, 0.0
    valid_hard_p_count, valid_hard_n_count = 0, 0
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        query_chunk = emb[i:end]
        query_labels = labels[i:end]
        
        # Compute Euclidean distances for this chunk against ALL embeddings: [Chunk, N]
        dists_chunk = torch.cdist(query_chunk, emb, p=2)
        
        # --- Create Masks ---
        pos_mask = (query_labels.unsqueeze(1) == labels.unsqueeze(0))
        for row_idx in range(end - i):
            pos_mask[row_idx, i + row_idx] = False # Mask self
            
        neg_mask = ~pos_mask
        for row_idx in range(end - i):
            neg_mask[row_idx, i + row_idx] = False # Remove self from negs too

        # --- 1. Global Means ---
        sum_p += dists_chunk[pos_mask].sum().item()
        count_p += pos_mask.sum().item()
        
        sum_n += dists_chunk[neg_mask].sum().item()
        count_n += neg_mask.sum().item()
        
        # --- 2. Global Hards (Per Anchor) ---
        
        # A. Hardest Positive (Maximum Euclidean Distance)
        # Clone and fill non-positives with -inf so they are ignored by max()
        pos_dists_masked = dists_chunk.clone()
        pos_dists_masked[~pos_mask] = -float('inf')
        hard_p_vals, _ = pos_dists_masked.max(dim=1) # Shape: [Chunk_size]
        
        # Only count anchors that actually had at least 1 positive
        valid_p_mask = hard_p_vals != -float('inf')
        sum_hard_p += hard_p_vals[valid_p_mask].sum().item()
        valid_hard_p_count += valid_p_mask.sum().item()

        # B. Hardest Negative (Minimum Euclidean Distance)
        # Clone and fill non-negatives with +inf so they are ignored by min()
        neg_dists_masked = dists_chunk.clone()
        neg_dists_masked[~neg_mask] = float('inf')
        hard_n_vals, _ = neg_dists_masked.min(dim=1) # Shape: [Chunk_size]
        
        # Only count anchors that actually had at least 1 negative
        valid_n_mask = hard_n_vals != float('inf')
        sum_hard_n += hard_n_vals[valid_n_mask].sum().item()
        valid_hard_n_count += valid_n_mask.sum().item()
        
    # Final Averages
    mu_p = sum_p / max(1, count_p)
    mu_n = sum_n / max(1, count_n)
    hard_p = sum_hard_p / max(1, valid_hard_p_count)
    hard_n = sum_hard_n / max(1, valid_hard_n_count)
    
    return mu_p, mu_n, hard_p, hard_n