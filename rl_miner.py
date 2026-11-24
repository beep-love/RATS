import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
from collections import deque
from losses import _compute_stats, batch_hard_triplet_loss, semi_hard_triplet_loss, distance_weighted_triplet_loss, distance_weighted_margin_loss

class RLConfig:
    """Wrapper to access config safely"""
    def __init__(self, raw_config):
        self.cfg = raw_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.rl = self.cfg['rl']

class MetricTracker:
    """Tracks historical metrics to compute Rolling Averages 
    Tracks 4-5 Metrics: [mu_p, mu_n, hard_p, hard_n, (optional) margin]
    (Removed ig_ratio to streamline 2-region architecture)
    """
    def __init__(self, config, max_window=128, is_adaptive_margin=False):
        self.history = deque(maxlen=max_window)
        self.is_adaptive_margin = is_adaptive_margin
        # State size is either 5 [mu_p, mu_n, hard_p, hard_n, margin] or 4 [mu_p, mu_n, hard_p, hard_n]
        self.dim = 5 if is_adaptive_margin else 4
        self.state_windows = config.rl.get('state_windows', [2, 4, 8])

    def update(self, mu_p, mu_n, hard_p, hard_n, margin=None):
        # Store as a vector [Dim Metrics] for easy averaging later
        if self.is_adaptive_margin and margin is not None:
            self.history.append(torch.tensor([mu_p, mu_n, hard_p, hard_n, margin], dtype=torch.float32))
        else:
            self.history.append(torch.tensor([mu_p, mu_n, hard_p, hard_n], dtype=torch.float32))

    def get_running_averages(self, windows=None):
        if windows is None:
            windows = self.state_windows
            
        if len(self.history) == 0:
            return torch.zeros(len(windows) * self.dim, dtype=torch.float32)

        if len(windows) == 0:
            return torch.zeros(0, dtype=torch.float32)

        data = torch.stack(list(self.history)) # [History Length, Dim]
        means = []

        for w in windows:
            if len(data) >= w:
                means.append(data[-w:].mean(dim=0))
            else:
                means.append(data.mean(dim=0))
        return torch.cat(means) # Flatten: [n windows ~ 4/5 metrics] vector of rolling averages

class ActorCritic(nn.Module):
    """
    The Streamlined Policy Network.  (PPO Compatible)
    Input state includes Rolling Averages + Current Weights + Instant Metrics.
    Outputs 1 Discrete Action for Regions + 1 Discrete Action for Margin + Value
    """
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.is_adaptive_margin = self.cfg.rl.get('adaptive_margin', False)
        self.is_adaptive_sampling = self.cfg.rl.get('adaptive_sampling', True)      # Default to True for 2-region architecture
        num_windows = len(self.cfg.rl.get('state_windows', [2, 4, 8]))

        rolling_dim = (5 if self.is_adaptive_margin else 4) * num_windows 

        if self.is_adaptive_sampling:
            instant_dim = 8 if self.is_adaptive_margin else 7
        else:
            instant_dim = 6 if self.is_adaptive_margin else 5

        self.input_dim = rolling_dim + instant_dim  
                
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),              
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # --- Actor Heads ---
        if self.is_adaptive_sampling:
            # 1D Region Slider: 3 choices -> [Shift Semi, Keep, Shift Hard]
            self.actor_regions = nn.Linear(128, 3) 
        
        if self.is_adaptive_margin:
            # 1D Margin Slider: 3 choices -> [Decrease, Keep, Increase]
            self.actor_margin = nn.Linear(128, 3)
        
        # --- Critic Head ---
        self.critic = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        if self.is_adaptive_sampling:
            # Override Actor Heads (Gain = 0.01 for Maximum Entropy)
            # This mathematically forces the initial logits to be near [0, 0, 0]
            # resulting in a perfect [33%, 33%, 33%] Categorical probability.
            nn.init.orthogonal_(self.actor_regions.weight, gain=0.01)
        if self.is_adaptive_margin:
            nn.init.orthogonal_(self.actor_margin.weight, gain=0.01)

        # Override Critic Head (Gain = 1.0 for Value Prediction)
        # Prevents the initial value predictions from exploding
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, state):
        if state.dim() == 1: state = state.unsqueeze(0)
        features = self.feature_net(state)
        
        if self.is_adaptive_sampling:
            # --- Region Distribution (Categorical) ---
            r_logits = self.actor_regions(features)
            dist_regions = Categorical(logits=r_logits)
        
        if self.is_adaptive_margin:
            # --- Margin Distribution (Categorical) ---
            m_logits = self.actor_margin(features)
            dist_m = Categorical(logits=m_logits)
        
        value = self.critic(features)

        if self.is_adaptive_margin and self.is_adaptive_sampling:
            return dist_regions, dist_m, value 
        elif self.is_adaptive_sampling:
            return dist_regions, value
        elif self.is_adaptive_margin:
            return dist_m, value
        else:
            return value


class RLMiner(nn.Module):
    """
    Dynamic Percentile Bucketing & Structural Loss Calculation.
    Maintains the state of the current region weights and margin, and applies them to the loss calculation.
    (Optimized 2-Region Architecture: Semi and Hard only).
    Acts as a universal curriculum wrapper for BOTH Triplet and Margin loss.
    """
    def __init__(self, config, init_margin=0.2, num_classes=100, init_beta=1.2):
        super().__init__()
        self.cfg = config
        self.current_margin = init_margin

        # Read loss type from config: "triplet" or "margin"
        self.loss_type = self.cfg.cfg['loss_type']
        self.is_adaptive_sampling = self.cfg.rl.get('adaptive_sampling', True)

        # --- MARGIN LOSS REQUIREMENT ---
        if self.loss_type == 'margin':
            self.beta = nn.Parameter(torch.ones(num_classes) * init_beta)
            print(f"--> RLMiner initialized with Margin Loss (beta initialized to 1.2 for {num_classes} classes)")
        
        if self.is_adaptive_sampling:
            # Region Logits [Semi, Hard]. Initialize evenly.
            self.region_logits = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=False)
            self.region_weights = F.softmax(self.region_logits, dim=0)
        else:
            print("--> RLMiner: Adaptive Sampling DISABLED. Using baseline mining.")

    def update_weights(self, action_r, updates_tensor):
        """Pure 1D Seesaw Mechanics.
        Action is a single choice: [0: Shift Semi, 1: Keep, 2: Shift Hard]"""
        # Get the raw additive step values based on the agent's chosen actions
        delta_W = updates_tensor[action_r].squeeze()

        # Apply the seesaw directly: Add to Hard, Subtract from Semi
        self.region_logits[1] += delta_W
        self.region_logits[0] -= delta_W
        
        # --- LOGIT CLAMPING ---
        # Bounding between [-1.2, 1.2] gives the agent wide control (~8% floor, ~91% ceiling)
        # while preventing mathematical "dead zones" in the Softmax gradient.
        clamp_val = self.cfg.rl.get('logit_clamp', 1.2)
        self.region_logits.clamp_(min=-clamp_val, max=clamp_val)
        self.region_weights = F.softmax(self.region_logits, dim=0)
    
    def forward(self, embeddings, labels, clip_margin=False):
        '''
        Computes the RL-augmented loss using geometric boundaries.
        Logic:-
        1. Fast Bypass: Route to baselines if RL sampling is disabled.
        2. Create positive and negative masks based on labels.
        3. Compute hardness (d_n - d_p) for ALL anchor-positive-negative triplets.
        4. Apply the agent's region weights to the raw loss (Semi, Hard).
           *Easy triplets are completely ignored to save compute.*
        5. Normalize the weighted loss by the number of active anchors.

        Two curriculum-application strategies are supported:
          - 'weighting'  -> vectorized  (see _forward_weighting_vectorized)
          - 'sampling'   -> per-anchor Python loop (NOT yet vectorized)
        '''
        device = embeddings.device
        B = embeddings.size(0)

        # ---------------------------------------------------------
        # FAST BYPASS: Route to optimized baselines for sampling
        # ---------------------------------------------------------
        if not self.is_adaptive_sampling:
            strategy = self.cfg.cfg.get('sampling_strategy', 'batch_hard')

            if self.loss_type == 'triplet':
                if strategy == 'batch_hard': return batch_hard_triplet_loss(embeddings, labels, margin=self.current_margin)
                elif strategy == 'semi_hard': return semi_hard_triplet_loss(embeddings, labels, margin=self.current_margin)
                elif strategy == 'distance_weighted': return distance_weighted_triplet_loss(embeddings, labels, beta=self.beta, margin=self.current_margin)
            elif self.loss_type == 'margin':
                if strategy == 'distance_weighted': return distance_weighted_margin_loss(embeddings, labels, beta=self.beta, margin=self.current_margin)

            raise ValueError(f"Strategy {strategy} or Loss Type {self.loss_type} not supported.")
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # ADAPTIVE SAMPLING: "Batch All Weighted" Triplet Loss
        # ---------------------------------------------------------

        # --- Compute Euclidean Matrix ---
        # embeddings: [B, emb_dim]  ->  l2_dist: [B, B]
        # l2_dist[a, b] = || f(x_a) - f(x_b) ||_2
        l2_dist = torch.cdist(embeddings, embeddings, p=2)

        labels = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))     # [B, B]
        pos_mask.fill_diagonal_(False)                              # an anchor is not its own positive
        neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))     # [B, B]

        stats = _compute_stats(l2_dist, pos_mask, neg_mask)

        # Read the strategy from config (defaulting to sampling if missing)
        adaptive_sampling_strategy = self.cfg.rl.get('adaptive_sampling_strategy', 'sampling')

        # ---------------------------------------------------------
        # DISPATCH: the 'weighting' curriculum is fully deterministic,
        # so it is computed in a single batched pass. 'sampling' draws
        # one triplet per anchor from a per-anchor categorical and is
        # still handled by the loop below.
        # ---------------------------------------------------------
        if adaptive_sampling_strategy == 'weighting':
            return self._forward_weighting_vectorized(
                embeddings, labels, l2_dist, pos_mask, neg_mask, stats
            )

        # =========================================================
        # PER-ANCHOR LOOP  --  'sampling' strategy only
        #
        # TODO (future work): this branch has NOT been vectorized.
        # Each anchor draws its region from a categorical and then a
        # single triplet uniformly from within that region, so a batched
        # rewrite would change the order in which random numbers are
        # consumed and therefore would not reproduce existing runs.
        # The 'weighting' path above is the one used in the reported
        # experiments; vectorizing 'sampling' is left as an extension.
        # =========================================================
        total_active_anchors = 0.0
        weighted_loss_sum = 0.0

        for i in range(len(embeddings)):
            p_idxs = torch.where(pos_mask[i])[0]
            n_idxs = torch.where(neg_mask[i])[0]
            if len(p_idxs) == 0 or len(n_idxs) == 0: continue

            # --- BUCKETING  ---
            d_p = l2_dist[i, p_idxs].unsqueeze(1)     # [num_pos, 1]
            d_n = l2_dist[i, n_idxs].unsqueeze(0)     # [1, num_neg]

            # TRUE HARDNESS: (Lower = Harder)
            euclidean_hardness = d_n - d_p            # [num_pos, num_neg]

            # --- GEOMETRIC LITERATURE BOUNDARIES ---
            # Easy triplets (hardness >= current_margin) are structurally ignored.
            mask_hard = euclidean_hardness < 0.0
            mask_semi = (euclidean_hardness >= 0.0) & (euclidean_hardness < self.current_margin)

            # --- THE GRADIENT MATH SWITCH ---
            if self.loss_type == 'triplet':
                # Pure Gradients
                raw_loss = F.relu(-euclidean_hardness + self.current_margin)     # Same as d_p - d_n + margin

            elif self.loss_type == 'margin':
                # Sampling matters Margin Loss
                anchor_beta = self.beta[labels[i]]
                pos_loss = F.relu(d_p - anchor_beta + self.current_margin)
                neg_loss = F.relu(anchor_beta - d_n + self.current_margin)
                raw_loss = pos_loss + neg_loss
            # --------------------------------

            # Apply agent weights to regions
            weights = torch.zeros_like(euclidean_hardness)

            count_hard = mask_hard.sum().item()
            count_semi = mask_semi.sum().item()

            # Roll the die based on agent's current region weights
            sampled_choice = torch.multinomial(self.region_weights, 1).item()

            # Fallback Mechanism (Smooth failover if chosen region is empty)
            if sampled_choice == 1 and count_hard == 0 and count_semi > 0:
                sampled_choice = 0
            elif sampled_choice == 0 and count_semi == 0 and count_hard > 0:
                sampled_choice = 1

            # PURE SAMPLING: Pick exactly ONE triplet
            if sampled_choice == 1 and count_hard > 0:
                valid_idx = torch.where(mask_hard)                      # tuple of (row_indices, col_indices) for 2D tensors
                rand_idx = torch.randint(0, count_hard, (1,)).item()

                r, c = valid_idx[0][rand_idx], valid_idx[1][rand_idx]
                weights[r, c] = 1.0

            elif sampled_choice == 0 and count_semi > 0:
                valid_idx = torch.where(mask_semi)
                rand_idx = torch.randint(0, count_semi, (1,)).item()

                r, c = valid_idx[0][rand_idx], valid_idx[1][rand_idx]
                weights[r, c] = 1.0

            anchor_weighted_loss = (raw_loss * weights).sum()
            if anchor_weighted_loss > 0:
                weighted_loss_sum += anchor_weighted_loss
                total_active_anchors += 1.0

        if total_active_anchors > 0:
            loss = weighted_loss_sum / total_active_anchors
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        stats['active_anchors_ratio'] = total_active_anchors / len(embeddings)
        return loss, stats, self.current_margin

    # -----------------------------------------------------------------
    # Vectorized 'weighting' curriculum
    # -----------------------------------------------------------------
    def _forward_weighting_vectorized(self, embeddings, labels, l2_dist,
                                      pos_mask, neg_mask, stats):
        """
        Batched equivalent of the per-anchor 'weighting' loop.

        The loop builds, for each anchor i, a [num_pos_i, num_neg_i] hardness
        matrix. Here all B of those are computed at once as a single
        [B, P, N] tensor indexed by (anchor, positive, negative), with an
        explicit validity mask standing in for the variable per-anchor shapes.

        Verified numerically equivalent to the loop (loss, gradient w.r.t.
        embeddings, and active_anchors_ratio) -- see test_miner_equivalence.py.

        Shapes (B = batch size):
            l2_dist   [B, B]        pairwise Euclidean distances
            pos_mask  [B, B]        pos_mask[i, p] = p is a positive of anchor i
            neg_mask  [B, B]        neg_mask[i, n] = n is a negative of anchor i

        Memory note: the [B, B, B] intermediates are ~8.4 MB each at B=128
        (float32). Only ~2% of their entries are valid triplets, but the dense
        form removes B sequential kernel launches, which dominated the cost.
        """
        device = embeddings.device
        B = embeddings.size(0)

        # -------------------------------------------------------------
        # 1. JOINT HARDNESS TENSOR
        #    Both operands are row i of l2_dist, laid out on different axes,
        #    so subtracting them gives the outer difference within each row:
        #
        #        hardness[i, p, n] = l2_dist[i, n] - l2_dist[i, p]
        #                          = d_an - d_ap        (paper Eq. 5)
        # -------------------------------------------------------------
        d_p = l2_dist.unsqueeze(2)                 # [B, P, 1]  varies along positives
        d_n = l2_dist.unsqueeze(1)                 # [B, 1, N]  varies along negatives
        hardness = d_n - d_p                       # [B, P, N]

        # -------------------------------------------------------------
        # 2. VALIDITY MASK
        #    Entry (i, p, n) is a real triplet only if p is a positive of i
        #    AND n is a negative of i. Unsqueezing the two masks on different
        #    axes makes the AND a conjunction over the (p, n) grid.
        #
        #    Anchors with no positives or no negatives get an all-False plane,
        #    contribute zero loss, and are dropped by the active test in step 5
        #    -- reproducing the loop's `continue`.
        # -------------------------------------------------------------
        valid = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)       # [B, P, N]

        # --- GEOMETRIC LITERATURE BOUNDARIES (paper Sec. III-C) ---
        # Hard: ordering violated. Semi-hard: ordered but margin unmet.
        # Easy (hardness >= margin) falls in neither and is structurally ignored.
        mask_hard = (hardness < 0.0) & valid                        # [B, P, N]
        mask_semi = (hardness >= 0.0) & (hardness < self.current_margin) & valid

        # -------------------------------------------------------------
        # 3. RAW LOSS  (the gradient math switch, batched)
        # -------------------------------------------------------------
        if self.loss_type == 'triplet':
            raw_loss = F.relu(-hardness + self.current_margin)      # [B, P, N]

        elif self.loss_type == 'margin':
            # beta is per-class; gather per anchor and broadcast over (p, n).
            anchor_beta = self.beta[labels].view(B, 1, 1)           # [B, 1, 1]
            pos_loss = F.relu(d_p - anchor_beta + self.current_margin)   # [B, P, 1]
            neg_loss = F.relu(anchor_beta - d_n + self.current_margin)   # [B, 1, N]
            raw_loss = pos_loss + neg_loss                          # -> [B, P, N]
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        # -------------------------------------------------------------
        # 4. REGION WEIGHTS  (paper Eq. 12)
        #    Reducing over (p, n) collapses each anchor's plane to its bucket
        #    population; .view(B,1,1) broadcasts that anchor's share back out.
        #
        #        rho_hard = P_hard / max(1, N_hard)
        #
        #    Multiplying by float-cast masks (rather than boolean assignment)
        #    keeps the anchor axis intact so the per-anchor scalars broadcast.
        #    Invalid entries are False in both masks -> weight 0.
        # -------------------------------------------------------------
        count_hard = mask_hard.sum(dim=(1, 2))                      # [B]
        count_semi = mask_semi.sum(dim=(1, 2))                      # [B]

        count_h_safe = count_hard.clamp(min=1).to(raw_loss.dtype)   # avoid div by zero
        count_s_safe = count_semi.clamp(min=1).to(raw_loss.dtype)

        # Index 0 = Semi, Index 1 = Hard
        w_hard = (self.region_weights[1] / count_h_safe).view(B, 1, 1)
        w_semi = (self.region_weights[0] / count_s_safe).view(B, 1, 1)

        weights = mask_hard.to(raw_loss.dtype) * w_hard \
                + mask_semi.to(raw_loss.dtype) * w_semi             # [B, P, N]

        # -------------------------------------------------------------
        # 5. REDUCE PER ANCHOR, THEN AVERAGE OVER ACTIVE ANCHORS
        # -------------------------------------------------------------
        anchor_loss = (raw_loss * weights).sum(dim=(1, 2))          # [B]

        active = anchor_loss > 0
        total_active_anchors = active.sum()

        if total_active_anchors > 0:
            loss = anchor_loss[active].sum() / total_active_anchors.to(anchor_loss.dtype)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        stats['active_anchors_ratio'] = total_active_anchors.item() / B
        return loss, stats, self.current_margin


class RLAgent:
    """
    Manages State, Action, Fast Geometric Reward --> Updates the ActorCritic Policy
    """
    def __init__(self, miner, policy, config):
        self.miner = miner
        self.policy = policy
        self.cfg = config
        # self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.cfg.rl.get('learning_rate', 1e-4), weight_decay=self.cfg.rl.get('weight_decay', 0.0001))
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.cfg.rl.get('learning_rate', 1e-4))        
        
        self.is_adaptive_margin = self.cfg.rl.get('adaptive_margin', False)
        self.is_adaptive_sampling = self.cfg.rl.get('adaptive_sampling', True)

        self.tracker = MetricTracker(config=self.cfg, is_adaptive_margin=self.is_adaptive_margin)

        self.reward_ema = None
        self.ema_alpha = self.cfg.rl.get('ema_alpha', 0.0)   # Smoothing factor for Reward EMA
        
        # Clean, symmetric additive steps for Logit Space: [-step, 0.0, step]
        region_step = self.cfg.rl.get('region_step', 0.15)
        self.region_action_updates = torch.tensor([-region_step, 0.0, region_step], device=self.cfg.device)
        
        margin_step = self.cfg.rl.get('margin_step', 0.01)
        self.margin_action_updates = torch.tensor([-margin_step, 0.0, margin_step], device=self.cfg.device)
                
        # PPO Buffers
        self.saved_states, self.saved_values = [], []
        self.saved_actions_r, self.saved_actions_m = [], []
        self.saved_log_probs, self.rewards = [], []

    def get_state(self, epoch_progress, batch_stats):
        """Constructs the state vector including the agent's own current weights."""
        
        # -- Update History & Get Rolling Averages ---
        kwargs = {
            'mu_p': batch_stats['dist_pos_mean'], 
            'mu_n': batch_stats['dist_neg_mean'],
            'hard_p': batch_stats['dist_pos_hard'],
            'hard_n': batch_stats['dist_neg_hard']
        }

        if self.is_adaptive_margin:
            kwargs['margin'] = self.miner.current_margin

        self.tracker.update(**kwargs)
        
        rolling_context = self.tracker.get_running_averages().to(self.cfg.device)

        instant_list = []
        if self.is_adaptive_sampling:
            w = self.miner.region_weights.detach()
            instant_list.extend([w[0].item(), w[1].item()])
        
        instant_list.extend([
            batch_stats['dist_pos_mean'], batch_stats['dist_neg_mean'], 
            batch_stats['dist_pos_hard'], batch_stats['dist_neg_hard']
        ])

        if self.is_adaptive_margin:
            instant_list.append(self.miner.current_margin)
        
        instant_list.append(epoch_progress)
        
        instant_metrics = torch.tensor(instant_list, device=self.cfg.device)

        return torch.cat([rolling_context, instant_metrics])

    def calculate_immediate_reward(self, current_batch_stats):
        """
        Pure Geometric Reward (2-Region Architecture)
        """
        # --- Frontier Proximity ---

        current_hard_neg = current_batch_stats['dist_neg_hard']
        current_avg_pos = current_batch_stats['dist_pos_mean']

        # --- Triplet Utilization Bonus ---
        current_anchor_ratio = current_batch_stats['active_anchors_ratio']
        lambda_ratio = self.cfg.rl.get('lambda_ratio', 0.5)

        # Linear Margin Compliance (Dense Reward)
        absolute_margin = current_hard_neg - current_avg_pos

        if absolute_margin < 0.0:
            # Geometry is completely violated, hard negative is closer than the average positive
            margin_score = 0.0  
        elif absolute_margin >= self.miner.current_margin:
            # Margin is satisfied, hard negative is farther than the average positive by at least the margin
            margin_score = 1.0   
        else:
            # In the "gray zone" where the margin is not yet satisfied but the order is correct. 
            # This encourages the agent to push the frontier further rather than being content with just barely satisfying the geometry.
            margin_score = absolute_margin / self.miner.current_margin  
        
        total = lambda_ratio * current_anchor_ratio + (1 - lambda_ratio) * margin_score
        
        # --- EMA Smoothing ---
        if self.reward_ema is None:
            self.reward_ema = total
        else:
            self.reward_ema = self.ema_alpha * self.reward_ema + (1.0 - self.ema_alpha) * total
            
        self.rewards.append(self.reward_ema)
        
        return self.reward_ema, margin_score, current_anchor_ratio

    def step(self, state):

        if self.is_adaptive_margin and self.is_adaptive_sampling:
            dist_r, dist_m, value = self.policy(state)
            action_r, action_m = dist_r.sample(), dist_m.sample()
            log_p = dist_r.log_prob(action_r) + dist_m.log_prob(action_m)
            self.saved_actions_r.append(action_r)
            self.saved_actions_m.append(action_m)
        elif self.is_adaptive_sampling:
            dist_r, value = self.policy(state)
            action_r = dist_r.sample()
            log_p = dist_r.log_prob(action_r)
            self.saved_actions_r.append(action_r)
        elif self.is_adaptive_margin:
            dist_m, value = self.policy(state)
            action_m = dist_m.sample()
            log_p = dist_m.log_prob(action_m)
            self.saved_actions_m.append(action_m)
        
        self.saved_states.append(state)
        self.saved_values.append(value)
        self.saved_log_probs.append(log_p)          
        
        with torch.no_grad():
            if self.is_adaptive_sampling:
                self.miner.update_weights(action_r, self.region_action_updates)
            
            if self.is_adaptive_margin:
                # Map the discrete action [0, 1, 2] to [-step, 0.0, +step]
                self.miner.current_margin += self.margin_action_updates[action_m].item()

                # Clamp for safety (assuming margin should stay between min_margin and max_margin)
                self.miner.current_margin = max(self.cfg.rl['min_margin'], min(self.cfg.rl['max_margin'], self.miner.current_margin))
        
        info = {}
        if self.is_adaptive_sampling:
            info['action_r'] = int(action_r.item())
        if self.is_adaptive_margin:
            info['action_m'] = int(action_m.item())
            info['margin'] = self.miner.current_margin
        return info

    def update_policy(self, entropy_beta=0.05):
        """Standard PPO/A2C/REINFORCE Update Loop"""
        if not self.saved_log_probs:
            return {'policy_loss': 0.0, 'value_loss': 0.0,
                    'entropy': 0.0, 'entropy_region': 0.0, 'entropy_margin': 0.0}

        states = torch.stack(self.saved_states).detach()
        old_log_probs = torch.stack(self.saved_log_probs).detach().view(-1)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.cfg.device)
        rollout_values = torch.stack(self.saved_values).detach().view(-1)

        if self.is_adaptive_margin:
            actions_m = torch.stack(self.saved_actions_m).detach().view(-1)
        if self.is_adaptive_sampling:
            actions_r = torch.stack(self.saved_actions_r).detach().view(-1)

        # Calculate FIXED Advantage BEFORE the loop 
        # This prevents the Advantage from changing as the weights update
        fixed_advantages = rewards - rollout_values
        if len(fixed_advantages) > 1:
            std = fixed_advantages.std()
            if std > 1e-6:
                fixed_advantages = (fixed_advantages - fixed_advantages.mean()) / (std + 1e-8)
            else:
                fixed_advantages = fixed_advantages - fixed_advantages.mean()
        
        method = self.cfg.rl['method']

        # --- Multi-Epoch Update Logic (Standard for PPO) ---
        epochs = self.cfg.rl.get('ppo_epochs', 4) if method == "PPO" else 1
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_list = []
        ent_r_list, ent_m_list = [], []

        for _ in range(epochs):
            if self.is_adaptive_margin and self.is_adaptive_sampling:
                new_dist_r, new_dist_m, fresh_values = self.policy(states)
                new_log_probs = new_dist_r.log_prob(actions_r) + new_dist_m.log_prob(actions_m)
                ent_r = new_dist_r.entropy().mean()
                ent_m = new_dist_m.entropy().mean()
                entropy = ent_r + ent_m
            elif self.is_adaptive_sampling:
                new_dist_r, fresh_values = self.policy(states)
                new_log_probs = new_dist_r.log_prob(actions_r)
                ent_r = new_dist_r.entropy().mean()
                ent_m = torch.tensor(float('nan'))
                entropy = ent_r
            elif self.is_adaptive_margin:
                new_dist_m, fresh_values = self.policy(states)
                new_log_probs = new_dist_m.log_prob(actions_m)
                ent_r = torch.tensor(float('nan'))
                ent_m = new_dist_m.entropy().mean()
                entropy = ent_m

            fresh_values = fresh_values.view(-1)

            if not hasattr(self, '_shape_checked'):
                # states=[T, input_dim]  new_log_probs=[T]  fixed_advantages=[T]  rewards=[T]  fresh_values=[T]  actions_r=[T] where T = rl_steps_per_epoch
                self._shape_checked = True
                print(f"[SHAPE CHECK] states={states.shape} "
                      f"new_log_probs={new_log_probs.shape} "
                      f"fixed_advantages={fixed_advantages.shape} "
                      f"rewards={rewards.shape} fresh_values={fresh_values.shape}", flush=True)
                if self.is_adaptive_sampling:
                    print(f"[SHAPE CHECK] actions_r={actions_r.shape}", flush=True)
                if self.is_adaptive_margin:
                    print(f"[SHAPE CHECK] actions_m={actions_m.shape}", flush=True)
            
            entropy_list.append(entropy.item())
            ent_r_list.append(ent_r.item())
            ent_m_list.append(ent_m.item())

            if method == "PPO":
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * fixed_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.rl['ppo_clip'], 1.0 + self.cfg.rl['ppo_clip']) * fixed_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(fresh_values, rewards)
            elif method == "A2C":
                policy_loss = -(new_log_probs * fixed_advantages).mean()
                value_loss = F.mse_loss(fresh_values, rewards)

            elif method == "REINFORCE":
                policy_loss = -(new_log_probs * rewards).mean()
                value_loss = torch.tensor(0.0, device=self.cfg.device)

            # Combined Optimization Objective
            loss = policy_loss + 0.5 * value_loss - entropy_beta * entropy
            
            self.optimizer.zero_grad()
            loss.backward()

            # Improvement: Clip Gradients to norm 0.5 for stability
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()

        # Buffer Cleanup
        self.saved_states, self.saved_actions_r, self.saved_actions_m = [], [], []
        self.saved_log_probs, self.saved_values, self.rewards = [], [], []
        
        return {
            'policy_loss': policy_loss_total / epochs,
            'value_loss': value_loss_total / epochs,
            'entropy': float(np.mean(entropy_list)),
            'entropy_region': float(np.mean(ent_r_list)) if self.is_adaptive_sampling else 0.0,
            'entropy_margin': float(np.mean(ent_m_list)) if self.is_adaptive_margin else 0.0,
        }