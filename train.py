######################################################################################
### Author: Biplav Sharma Regmi <biplav01regmi@gmail.com>
### Date: JAN 2026 - NOV 2026
### Version: 1.0
### Description: Main training script for the vehicle re-identification task.
######################################################################################

import os
import yaml
import time
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from datetime import datetime

# --- Dataset Imports ---
from cub_200_dataset import build_cub_dataset
from sop_dataset import build_sop_dataset
from car196_dataset import build_cars_dataset
from veri_wild_dataset import build_veri_wild_datasets

# --- Sampler and Model Imports ---
from sampler import PKSampler
from model import ResNetEmbedding
from metrics import evaluate_metrics

# --- Loss & RL Imports ---
from losses import batch_hard_triplet_loss, semi_hard_triplet_loss, distance_weighted_triplet_loss, MarginLossBaselines
from rl_miner import RLConfig, RLMiner, ActorCritic, RLAgent

def set_global_seed(seed=33, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed} | cudnn.deterministic={deterministic} | cudnn.benchmark=False")

def get_unified_scheduler(optimizer, cfg):
    """
    Unified Scheduler: Handles both Warmup and Step Decay based on config flags.
    """
    def lr_lambda(epoch):
        factor = 1.0
        
        # 1. Warmup Phase (Linear increase to 1.0)
        if cfg.get('warmup', False) and epoch < cfg.get('warmup_epochs', 5):
            factor = float(epoch + 1) / cfg['warmup_epochs']
        else:
            # 2. Decay Phase (Applies after warmup, or from epoch 0 if no warmup)
            if cfg.get('lr_decay', False):
                decay_step = cfg.get('lr_decay_step', 20)
                gamma = cfg.get('lr_decay_gamma', 0.1)
                # Calculate how many decay steps have occurred
                decay_power = epoch // decay_step
                factor = factor * (gamma ** decay_power)
                
        return factor
    return LambdaLR(optimizer, lr_lambda)

def load_dataset(cfg):
    """Factory function for datasets based on Config"""
    name = cfg['dataset']
    root = cfg['data_root']
    
    print(f"Loading {name} from {root}...")
    
    if name == "cub_200":
        return build_cub_dataset(root, image_size=cfg['image_size'], crop_size=cfg['crop_size'])
    elif name == "sop":
        return build_sop_dataset(root, image_size=cfg['image_size'], crop_size=cfg['crop_size'])
    elif name == 'car196':
        return build_cars_dataset(root, image_size=cfg['image_size'], crop_size=cfg['crop_size'])
    elif name == "veri_wild":
        ram_root = os.environ.get("VERI_WILD_ROOT")
        if ram_root and os.path.exists(ram_root):
            print(f" FAST MODE: Using Dataset from RAM: {ram_root}")
            vw_root = ram_root
        else:
            print(" SLOW MODE: Using Dataset from HDD")
            vw_root = root
        xml_file = cfg.get('xml_file', "train_test_split/train_list_start0.txt")
        return build_veri_wild_datasets(vw_root, list_file=xml_file, train_val_split=cfg.get('train_val_split', 0.8), seed=cfg.get('seed', 33), image_size=cfg['image_size'], crop_size=cfg['crop_size'])
    else:
        raise ValueError(f"Unknown dataset: {name}")

# To automate the training with command-line overrides, we can apply them to the config dictionary.
def apply_overrides(cfg, overrides):
    """Apply CLI overrides of the form 'key=value' or 'rl.method=PPO' onto the config."""
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f"Malformed override '{item}', expected key=value")
        key, raw_val = item.split('=', 1)
        val = yaml.safe_load(raw_val)   # yields int/float/bool/list/str correctly

        parts = key.split('.')
        node = cfg
        for p in parts[:-1]:
            if p not in node:
                raise KeyError(f"Override path '{key}' not found in config (missing '{p}')")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Override key '{key}' not found in config")
        node[parts[-1]] = val
        print(f"[OVERRIDE] {key} = {val!r}")
    return cfg

def train(config_path="configs/cub200.yaml", overrides=None):
    # Load Configuration
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)

    # Apply any command-line overrides
    raw_cfg = apply_overrides(raw_cfg, overrides)

    set_global_seed(seed=raw_cfg.get('seed', 33), deterministic=raw_cfg.get('cuda_deterministic', True))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rl_enabled = raw_cfg['rl'].get('enabled', False)
    static_margin = raw_cfg.get('init_margin', 0.2) 

    # Setup TensorBoard & HParams
    if rl_enabled:
        rl_tag = (f"{raw_cfg['rl']['method']}"
                  f"_samp{raw_cfg['rl']['adaptive_sampling']}"
                  f"_marg{raw_cfg['rl']['adaptive_margin']}"
                  f"_dw{raw_cfg['rl']['region_step']}"
                  f"_win{'-'.join(str(w) for w in raw_cfg['rl']['state_windows'])}"
                  f"_lam{raw_cfg['rl']['lambda_ratio']}"
                  f"_ctrl{raw_cfg['rl'].get('hardness_control', 'scalar')}")
        mode_str = rl_tag
    else:
        mode_str = raw_cfg['sampling_strategy']

    run_name = (f"{raw_cfg['experiment_name']}_{raw_cfg['dataset']}_{mode_str}"
                # f"_{raw_cfg['loss_type']}_{raw_cfg['model_type']}"
                f"_seed{raw_cfg.get('seed', 33)}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(run_name)
    log_dir = os.path.join(raw_cfg['output_dir'], run_name)
    writer = SummaryWriter(log_dir=log_dir)

    # Save the config used for this run to the log directory for reproducibility
    with open(os.path.join(log_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(raw_cfg, f, sort_keys=False)

    os.makedirs(raw_cfg['checkpoint_dir'], exist_ok=True)
    
    # Flatten config for HParams - Commented out parameters are fixed throughout the experiment and not logged as hyperparameters
    hparams = {
        # --- High Level ---
        'batch_size': raw_cfg['batch_size_p'] * raw_cfg['batch_size_k'],
        'emb_dim': raw_cfg['emb_dim'],                                      
        'model_type': raw_cfg['model_type'],
        'pretrained': raw_cfg['pretrained'],
        'imagenet_pretrained': raw_cfg.get('imagenet_pretrained', True),
        'ssl_pretrained_path': raw_cfg.get('ssl_pretrained_path', 'None'),  
        'loss_type': raw_cfg['loss_type'],
        'initial_margin': raw_cfg['init_margin'], 

        'seed': raw_cfg.get('seed', 33),
        'cuda_deterministic': raw_cfg.get('cuda_deterministic', True),
        'dataset' : raw_cfg['dataset'],
        'epochs': raw_cfg['epochs'],
        'lr': raw_cfg['lr'],
        'sampling_strategy': raw_cfg['sampling_strategy'] if not rl_enabled else 'rl',
        
        'rl_enabled': rl_enabled,
        'adaptive_sampling': raw_cfg['rl']['adaptive_sampling'] if rl_enabled else False,
        'adaptive_margin': raw_cfg['rl']['adaptive_margin'] if rl_enabled else False,
    
        # --- LR Optimization ---
        'warmup': raw_cfg.get('warmup', False),
        'warmup_epochs': raw_cfg.get('warmup_epochs', 5),
        'lr_decay': raw_cfg.get('lr_decay', False),
        'lr_decay_step': raw_cfg.get('lr_decay_step', 25),
        'lr_decay_gamma': raw_cfg.get('lr_decay_gamma', 0.3),

        # --- Margin Loss Specific ---
        'beta_lr': raw_cfg.get('beta_lr', 0.0005),
        'init_beta': raw_cfg.get('init_beta', 1.2),
             
        # --- RL Optimization (Stability vs Plasticity) ---
        'method': raw_cfg['rl']['method'] if rl_enabled else 'baseline',
        'rl_steps_per_epoch': raw_cfg['rl']['rl_steps_per_epoch'],              
        'rl_lr': raw_cfg['rl']['learning_rate'],  
        'loss_strategy': raw_cfg['rl']['adaptive_sampling_strategy'],           # How do we calculate loss? (weigthed or sampled)
        
        # --- RL Exploration ---
        'entropy_beta_decay': raw_cfg['rl']['entropy_beta_decay'],          # Should we decay exploration over time?
        'entropy_beta': raw_cfg['rl'].get('entropy_beta', 0.05),            # Exploration rate
        
        # --- PPO Hyperparameters ---
        'ppo_epochs': raw_cfg['rl'].get('ppo_epochs', 4),                   # Training intensity
        'ppo_clip': raw_cfg['rl'].get('ppo_clip', 0.2),                     # Clipping for PPO stability
                                                    
        # --- Margin Scheduling ---
        'margin_step': raw_cfg['rl']['margin_step'],                        # How fast can margin grow?
        'margin_min': raw_cfg['rl']['min_margin'],                          # Minimum margin
        'margin_max': raw_cfg['rl']['max_margin'],                          # Maximum margin
        
        #--- Region Weight Scheduling ---
        'region_step': raw_cfg['rl']['region_step'],                        # How fast can region weights change?
        'logit_clamp': raw_cfg['rl']['logit_clamp'],                        # Clamp for action logits to prevent extreme probabilities
        
        # --- Reward Sensitivity ---
        "lambda_ratio": raw_cfg['rl']['lambda_ratio'],                      # Reward sensitivity to triplet utilization
        "ema_alpha": raw_cfg['rl']['ema_alpha'],                            # EMA alpha for reward smoothing
        
        # --- State Observation ---
        'state_windows': str(raw_cfg['rl']['state_windows']),
        'hardness_control': raw_cfg['rl'].get('hardness_control', 'scalar'),
        'viz_enabled': raw_cfg['rl'].get('viz_enabled', False) if rl_enabled else False,
        }

    train_ds, val_ds = load_dataset(raw_cfg)

    # Calculate dynamic number of classes
    unique_train_classes = len(set(train_ds.get_labels()))
    print(f"Detected {unique_train_classes} unique training classes.")

    sampler = PKSampler(train_ds.get_labels(), P=raw_cfg['batch_size_p'], K=raw_cfg['batch_size_k'], seed=raw_cfg.get('seed', 33))

    train_loader = DataLoader(train_ds, batch_size=raw_cfg['batch_size_p']*raw_cfg['batch_size_k'], 
                              sampler=sampler, drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    
    # Model & Optimizer - Initialization

    # If a custom SSL weight is provided, disable ImageNet pretraining
    is_pretrained = raw_cfg.get('pretrained', False)
    is_imagenet = raw_cfg.get('imagenet_pretrained', False)
    raw_ssl_path = raw_cfg.get('ssl_pretrained_path', None)

    use_imagenet = False
    custom_ssl_weight = None

    if is_pretrained:
        if is_imagenet:
            use_imagenet = True
        else:
            # Truth Table condition 3: SSL Weights
            if raw_ssl_path and str(raw_ssl_path).lower() != "none":
                custom_ssl_weight = raw_ssl_path
            else:
                raise ValueError("Config requests SSL weights (pretrained=True, imagenet_pretrained=False) but 'ssl_pretrained_path' is missing or invalid.")

    model = ResNetEmbedding(
        emb_dim=raw_cfg.get('emb_dim', 128), 
        pretrained=use_imagenet, 
        base_model=raw_cfg.get('model_type', 'resnet50'),
        freeze_bn=False if raw_cfg["dataset"] == "veri_wild" else True     # Freeze BN for standard DML stability; keep active for VeRi-Wild
    ).to(device)

    # --- Custom SSL Weights ---
    if custom_ssl_weight:
        if os.path.exists(custom_ssl_weight):
            print(f" Injecting custom SSL backbone weights from: {custom_ssl_weight}")
            ssl_state_dict = torch.load(custom_ssl_weight, map_location=device, weights_only=True)
            
            # Robust Dictionary Remapping: Handle full SimCLR checkpoints
            clean_state_dict = {}
            for k, v in ssl_state_dict.items():
                if k.startswith('encoder.'):
                    # Strip the 'encoder.' prefix to match standard ResNet keys
                    clean_state_dict[k.replace('encoder.', '', 1)] = v
                elif k.startswith('projector.'):
                    # Silently ignore the SimCLR non-linear projection head
                    continue
                else:
                    # Fallback for cleanly saved backbones (no prefix)
                    clean_state_dict[k] = v
            
            # CRITICAL: We load into model.backbone, NOT the whole model.
            # strict=False allows ignoring any minor mismatches (like num_batches_tracked)
            model.backbone.load_state_dict(clean_state_dict, strict=False)
        else:
            raise FileNotFoundError(f" Custom SSL weight path not found: {custom_ssl_weight}")
    
    # Initialize Loss Function / Miner First - for margin loss
    loss_module = None
    
    # RL Architecture Setup (Conditional)
    if rl_enabled:
        print(f" Starting RL Training Mode ({raw_cfg['rl']['method']}) on {raw_cfg['dataset']}")
        rl_cfg = RLConfig(raw_cfg)
        miner = RLMiner(rl_cfg, init_margin=raw_cfg.get('init_margin', 0.2), num_classes=unique_train_classes, init_beta=raw_cfg.get('init_beta', 1.2)).to(device)
        policy = ActorCritic(rl_cfg).to(device)
        agent = RLAgent(miner, policy, rl_cfg)
        loss_module = miner
        
        # ---------------------------------------------------------
        # STATE SPEC DUMP (reproducibility: reviewer R1-C1)
        # ---------------------------------------------------------
        
        n_windows = len(raw_cfg['rl']['state_windows'])
        per_window = 5 if raw_cfg['rl']['adaptive_margin'] else 4
        n_policy_params = sum(p.numel() for p in policy.parameters())

        fields = [f"MA_w{w}[{m}]" for w in raw_cfg['rl']['state_windows']
                  for m in (['mu_p','mu_n','hard_p','hard_n','margin']
                            if raw_cfg['rl']['adaptive_margin']
                            else ['mu_p','mu_n','hard_p','hard_n'])]
        if raw_cfg['rl']['adaptive_sampling']:
            fields += ['P_semi', 'P_hard']
        fields += ['mu_p', 'mu_n', 'hard_p', 'hard_n']
        if raw_cfg['rl']['adaptive_margin']:
            fields += ['margin']
        fields += ['tau']

        spec = (f"n (num MA horizons) = {n_windows} | windows = {raw_cfg['rl']['state_windows']}\n"
                f"rolling dim = {n_windows} x {per_window} = {n_windows * per_window}\n"
                f"instantaneous dim = {len(fields) - n_windows * per_window}\n"
                f"total input_dim = {policy.input_dim}\n"
                f"policy params = {n_policy_params}\n"
                f"fields = {fields}")
        print(spec)
        writer.add_text("RL/StateSpec", spec.replace("\n", "  \n"), 0)
        assert policy.input_dim == len(fields), \
            f"State spec mismatch: {policy.input_dim} vs {len(fields)}"

        # ---------------------------------------------------------
        # DYNAMIC EPOCH-SCALING LOGIC
        # ---------------------------------------------------------
        loader_len = len(train_loader)
        rl_steps_per_epoch = raw_cfg['rl'].get('rl_steps_per_epoch', 5)
        
        # How many ResNet batches between each RL Action?
        update_freq = max(1, loader_len // rl_steps_per_epoch)
        
        # How many RL Actions before a Policy Update?
        policy_update_interval = rl_steps_per_epoch
        
        total_rl_steps = raw_cfg['epochs'] * rl_steps_per_epoch
        total_policy_updates = raw_cfg['epochs']
        
        print(f"Dataset Size: {loader_len} Batches/Epoch")
        print(f"RL Action Frequency: Every {update_freq} Batches")
        print(f"Policy Update: Exactly 1 per Epoch ({policy_update_interval} actions in buffer)")
        print(f"Total RL Updates: {total_rl_steps}")

        steps_since_update = 0
        rl_step_count = 0
        window_stats = defaultdict(float)
    else:
        print(f" Starting Baseline Training Mode on {raw_cfg['dataset']}")
        if raw_cfg.get("loss_type") == "margin":
            loss_module = MarginLossBaselines(
                margin=raw_cfg.get('init_margin', 0.2),
                init_beta=raw_cfg.get('init_beta', 1.2),
                num_classes=unique_train_classes,
            ).to(device)            

    # Build Parameter Groups for Optimizer
    # We use standard LR for ResNet, but a specific LR for Beta if it exists
    param_groups = [{"params": model.parameters()}]
    
    if loss_module is not None:
        # Grab any parameters in the loss module that require gradients (e.g., self.beta)
        loss_params = [p for p in loss_module.parameters() if p.requires_grad]
        if len(loss_params) > 0:
            # PADS uses 0.0005 for beta_lr by default
            beta_lr = raw_cfg.get('beta_lr', 0.0005) 
            param_groups.append({"params": loss_params, "lr": beta_lr})
            print(f"--> Injected Learnable Loss Parameters into Optimizer with LR: {beta_lr}")

    optimizer = torch.optim.Adam(param_groups, lr=raw_cfg['lr'], weight_decay=raw_cfg['weight_decay'])
    scheduler = get_unified_scheduler(optimizer, raw_cfg)

    ### ---------------------------------------------------------------------------------------------------------
    step_global = 0
    best_r1 = 0.0
    best_r5 = 0.0
    best_r10 = 0.0
    best_mAP = 0.0
    best_nmi = 0.0

    # ---------------------------------------------------------
    # MAIN TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(raw_cfg['epochs']):
        model.train()
        epoch_loss = 0
        n_batches = 0
        epoch_stats = defaultdict(float)
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        rl_time_epoch = 0.0
        loss_time_epoch = 0.0
        bwd_time_epoch = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            step_global += 1
            
            embeddings = model(images)
            
            torch.cuda.synchronize()
            _t_loss = time.time()
            # --- BRANCH: Loss Calculation ---
            if rl_enabled:
                loss, batch_stats, effective_margin = miner(embeddings, labels)
            else:
                if raw_cfg.get("loss_type") == "margin":
                    # Margin Loss Baselines (Handles Batch Hard, Semi Hard, and Distance Weighted internally)
                    loss, batch_stats, effective_margin = loss_module(embeddings, labels)
                elif raw_cfg.get("loss_type") == "triplet":
                    if raw_cfg["sampling_strategy"] == "batch_hard":
                        loss, batch_stats, effective_margin = batch_hard_triplet_loss(embeddings, labels, margin=static_margin)
                    elif raw_cfg["sampling_strategy"] == "semi_hard":
                        loss, batch_stats, effective_margin = semi_hard_triplet_loss(embeddings, labels, margin=static_margin)
                    elif raw_cfg["sampling_strategy"] == "distance_weighted":
                        loss, batch_stats, effective_margin = distance_weighted_triplet_loss(embeddings, labels, margin=static_margin)
                    else:
                        raise ValueError(f"Unknown sampling strategy: {raw_cfg['sampling_strategy']}")
                else:
                    raise ValueError(f"Unknown loss type: {raw_cfg.get('loss_type')}")

            torch.cuda.synchronize()
            loss_time_epoch += time.time() - _t_loss

            # --- Standard Backprop ---
            optimizer.zero_grad()
            torch.cuda.synchronize()
            _t_bwd = time.time()
            loss.backward()
            torch.cuda.synchronize()
            bwd_time_epoch += time.time() - _t_bwd
            optimizer.step()
            
            # Accumulate Stats
            epoch_loss += loss.item()
            for k, v in batch_stats.items():
                epoch_stats[k] += v
            n_batches += 1

            # =========================================================
            # BRANCH: RL CURRICULUM LOGIC
            # =========================================================
            if rl_enabled:
                for key in ['dist_pos_mean', 'dist_neg_mean', 'dist_pos_hard', 'dist_neg_hard', 'active_anchors_ratio']:
                    if key in batch_stats:
                        window_stats[key] += batch_stats[key]

                # RL Update Cycle
                if step_global % update_freq == 0:
                    rl_step_count += 1
                    rl_progress = rl_step_count / total_rl_steps

                    avg_window_stats = {k: v / update_freq for k, v in window_stats.items()}

                    _t_rl = time.time()

                    # A. Immediate Feedback
                    reward, margin_score, util_score = agent.calculate_immediate_reward(avg_window_stats)
                    
                    # B. State Observation
                    state = agent.get_state(rl_progress, avg_window_stats)
                    
                    # C. Action Selection (Updates miner weights internally)
                    act_info = agent.step(state)

                    # Time Logging
                    rl_time_epoch += time.time() - _t_rl
                    
                    steps_since_update += 1
                    
                    # RL Policy Optimization
                    if steps_since_update >= policy_update_interval:
                        
                        # --- Calculate Entropy Beta for this Epoch ---
                        # Decay: Starts at 0.05, decays by 5% every epoch, floors at 0.001
                        if raw_cfg['rl']['entropy_beta_decay']:
                            current_entropy_beta = max(raw_cfg['rl']['entropy_beta'] * (0.99 ** epoch), 0.001)
                        else:
                            current_entropy_beta = raw_cfg['rl']['entropy_beta']
                        
                        _t_rl = time.time()

                        rl_stats = agent.update_policy(entropy_beta=current_entropy_beta)
                        
                        rl_time_epoch += time.time() - _t_rl
                        steps_since_update = 0                              # Reset counter

                        pu_step = step_global // (update_freq * policy_update_interval)

                        writer.add_scalar("RL/PolicyLoss", rl_stats['policy_loss'], pu_step)
                        writer.add_scalar("RL/ValueLoss", rl_stats['value_loss'], pu_step)
                        writer.add_scalar("RL/PolicyEntropy", rl_stats['entropy'], pu_step)
                        writer.add_scalar("RL/EntropyBeta", current_entropy_beta, pu_step)

                        if raw_cfg['rl']['adaptive_sampling']:
                            writer.add_scalar("RL/Entropy_Region", rl_stats['entropy_region'], pu_step)
                        if raw_cfg['rl']['adaptive_margin']:
                            writer.add_scalar("RL/Entropy_Margin", rl_stats['entropy_margin'], pu_step)

                    # Log RL Metrics
                    writer.add_scalar("RL/Margin", miner.current_margin, step_global//update_freq)  
                    writer.add_scalar("RL/Reward_EMA", reward, step_global//update_freq)
                    writer.add_scalar("RL/Reward_Margin", margin_score, step_global//update_freq)
                    writer.add_scalar("RL/Reward_Util", util_score, step_global//update_freq)
                    if raw_cfg['rl']['adaptive_sampling']:
                        writer.add_scalar("RL/Action_Weight_Semi", miner.region_weights[0].item(), step_global//update_freq)
                        writer.add_scalar("RL/Action_Weight_Hard", miner.region_weights[1].item(), step_global//update_freq)
                    if raw_cfg['rl']['adaptive_margin'] and 'action_m' in act_info:
                        writer.add_scalar("RL/Margin_Action", act_info['action_m'], step_global//update_freq)
                    if raw_cfg['rl']['adaptive_sampling'] and 'action_r' in act_info:
                        writer.add_scalar("RL/Region_Action", act_info['action_r'], step_global//update_freq)
                    
                    window_stats.clear() 
            # =========================================================

        # --- Epoch End Adjustments ---
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        if n_batches > 0:
            epoch_loss /= n_batches
            for k in epoch_stats:
                epoch_stats[k] /= n_batches
        
        t1 = time.time()
        train_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        train_reserved_mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

        # Validation
        r1, r5, r10, mAP, nmi, mu_p_val, mu_n_val, hard_p_val, hard_n_val = evaluate_metrics(model, val_loader, device, seed=raw_cfg.get('seed', 33))
        t2 = time.time()

        total_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        train_time = t1 - t0
        writer.add_scalar("perf/epoch_train_time_s", train_time, epoch + 1)
        writer.add_scalar("perf/epoch_val_time_s", t2 - t1, epoch + 1)
        writer.add_scalar("perf/loss_time_s", loss_time_epoch, epoch + 1)
        writer.add_scalar("perf/loss_time_pct", 100.0 * loss_time_epoch / max(1e-9, train_time), epoch + 1)
        writer.add_scalar("perf/backward_time_s", bwd_time_epoch, epoch + 1)
        writer.add_scalar("perf/backward_time_pct", 100.0 * bwd_time_epoch / max(1e-9, train_time), epoch + 1)

        writer.add_scalar("perf/peak_train_gpu_mem_gb", train_peak_mem_gb, epoch + 1)
        writer.add_scalar("perf/peak_gpu_mem_gb", total_peak_mem_gb, epoch + 1)
        writer.add_scalar("perf/peak_train_reserved_gpu_mem_gb", train_reserved_mem_gb, epoch + 1)

        if rl_enabled:
            writer.add_scalar("perf/rl_agent_time_s", rl_time_epoch, epoch + 1)
            writer.add_scalar("perf/rl_overhead_pct", 100.0 * rl_time_epoch / max(1e-9, t1 - t0), epoch + 1) 

        print(f"Epoch {epoch+1:03d} | Loss {epoch_loss:.4f} | Val R1 {r1:.4f} | Val R5 {r5:.4f} | Val R10 {r10:.4f} | Val mAP {mAP:.4f} | Val NMI {nmi:.4f} | LR {current_lr:.2e} | Train Time {t1-t0:.1f}s  | Val Time {t2-t1:.1f}s")
        
        # Universal TensorBoard Logs
        writer.add_scalar("train/loss", epoch_loss, epoch + 1)
        writer.add_scalar("train/lr", current_lr, epoch + 1)

        # Safely log whatever angle metrics the loss function outputted
        for key in ['dist_pos_mean', 'dist_neg_mean', 'dist_pos_hard', 'dist_neg_hard', 'active_anchors_ratio']:
            if key in epoch_stats:
                writer.add_scalar(f"train_metrics/{key}", epoch_stats[key], epoch + 1)

        writer.add_scalar("val/recall_at_1", r1, epoch + 1)
        writer.add_scalar("val/recall_at_5", r5, epoch + 1)
        writer.add_scalar("val/recall_at_10", r10, epoch + 1)
        writer.add_scalar("val/nmi", nmi, epoch + 1)
        writer.add_scalar("val/mAP", mAP, epoch + 1)
        writer.add_scalar("val/mean_pos_dist", mu_p_val, epoch + 1)
        writer.add_scalar("val/mean_neg_dist", mu_n_val, epoch + 1)
        writer.add_scalar("val/hard_pos_dist", hard_p_val, epoch + 1)
        writer.add_scalar("val/hard_neg_dist", hard_n_val, epoch + 1)
    
        # Checkpointing
        if r1 > best_r1:
            best_r1 = r1
            save_path = f"{raw_cfg['checkpoint_dir']}/best_model_{run_name}.pth"
            torch.save(model.state_dict(), save_path)
            print(f" --> 🏆 Saved Best Model at epoch {epoch+1:03d} (R@1 {r1:.4f})")

        if r5 > best_r5:
            best_r5 = r5
        if r10 > best_r10:
            best_r10 = r10
        if mAP > best_mAP:
            best_mAP = mAP
        if nmi > best_nmi:
            best_nmi = nmi

    final_path = f"{raw_cfg['checkpoint_dir']}/final_model_{run_name}.pth"
    torch.save(model.state_dict(), final_path)
    print(f" --> Saved final-epoch model: {final_path}")

    # Final Hyperparam Log
    writer.add_hparams(hparams, {'hparam/R1': best_r1, 'hparam/R5': best_r5, 'hparam/R10': best_r10, 'hparam/mAP': best_mAP, 'hparam/NMI': best_nmi })  
    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cub200.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set seed=1 --set rl.method=PPO")
    args = parser.parse_args()
    train(config_path=args.config, overrides=args.set)