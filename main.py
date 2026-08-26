import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import argparse
import os
import importlib
import subprocess
import sys
import utils.models as models, utils.datasets as datasets
import utils.optimizers as optim_utils
import utils.plotting as plotting
from attackers.fia_config import (
    add_fia_arguments,
    apply_fia_preset,
    needs_fia_public_data,
    print_fia_config,
    validate_fia_args,
)
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import torch
import random
import numpy as np

def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    hidden_help = argparse.SUPPRESS

    parser.add_argument('--attack',
                        help='name of attack approach;',
                        type=str,
                        choices=['sign', 'cluster', 'pmc', 'amc', 'noll', 'ressfl_fia'],
                        default='sign')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=datasets.datasets_choices,
                        default='mnist')
    parser.add_argument('--epochs',
                        help='number of epochs;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size;',
                        type=int,
                        default=128)
    parser.add_argument('--lr_passive',
                        help='learning rate for passive party;',
                        type=float,
                        default=0.01)
    parser.add_argument('--lr_active',
                        help='learning rate for active party;',
                        type=float,
                        default=0.01)
    parser.add_argument('--lr_attack',
                        help='learning rate for attacker;',
                        type=float,
                        default=0.01)
    parser.add_argument('--set_attack_epoch',
                        help='whether to set attack epoch;',
                        action='store_true',
                        default=False)
    parser.add_argument('--attack_epoch',
                        help='epoch at which attack happens;',
                        type=int,
                        default=1)
    parser.add_argument('--attack_id',
                        help='id of the attacker;',
                        type=int,
                        default=1)
    parser.add_argument('--num_passive',
                        help='number of passive parties;',
                        type=int,
                        default=4)
    parser.add_argument('--use_emb',
                        help='whether to use embedding, if not, will use gradients;',
                        action='store_true',
                        default=False)
    parser.add_argument('--attack_every_n_iter',
                        help='attack every n iterations;',
                        type=int,
                        default=100)
    parser.add_argument('--simple',
                        help='use simple model',
                        action='store_true',
                        default=False)
    parser.add_argument('--padding_mode',
                        help='using the extreme assumption that only one passive party has data and the rest are padded with random data in [0, 1);',
                        action='store_true',
                        default=False)
    parser.add_argument('--division_mode',
                        help='choose the data division mode;',
                        type=str,
                        choices=['vertical', 'random', 'imbalanced'],
                        default='vertical')
    parser.add_argument('--tsne',
                        help='whether to use tsne for attack;',
                        action='store_true',
                        default=False)
    parser.add_argument('--attack_model_epochs',
                        help='number of epochs for attack model (completion model);',
                        type=int,
                        default=5)
    parser.add_argument('--seed',
                        help='random seed used by a single experiment run.',
                        type=int,
                        default=42)
    parser.add_argument('--run_seeds',
                        help='explicit list of seeds used for repeated experiments; overrides --seed when provided.',
                        type=int,
                        nargs='+',
                        default=None)
    parser.add_argument('--lr_attack_model',
                        help='learning rate for attack model (completion model);',
                        type=float,
                        default=0.1)
    parser.add_argument('--sgd_momentum',
                        help='momentum used by standard SGD optimizers for main-task training.',
                        type=float,
                        default=0.9)
    parser.add_argument('--weight_decay_active',
                        help='weight decay for the active party main-task optimizer; use -1 for dataset-aware auto selection.',
                        type=float,
                        default=-1.0)
    parser.add_argument('--weight_decay_passive',
                        help='weight decay for passive party main-task optimizers; use -1 for dataset-aware auto selection.',
                        type=float,
                        default=-1.0)
    parser.add_argument('--fusion_dropout',
                        help='dropout applied on fused features before the active classifier head.',
                        type=float,
                        default=0.3)
    parser.add_argument('--cifar_random_erasing',
                        help='enable RandomErasing in CIFAR-10 training augmentation; 1 enables and 0 disables.',
                        type=int,
                        choices=[0, 1],
                        default=1)
    parser.add_argument('--cifar_random_erasing_p',
                        help='application probability for CIFAR-10 RandomErasing.',
                        type=float,
                        default=0.25)
    parser.add_argument('--cifar_release_dim',
                        help='low-dimensional release size for CIFAR passive-party shared representations; set <= 0 to keep the original 512-d release.',
                        type=int,
                        default=256)

    parser.add_argument('--main_lr_scheduler',
                        help='main-task learning-rate scheduler; auto uses cosine for CIFAR and none otherwise.',
                        type=str,
                        choices=['auto', 'none', 'cosine', 'multistep'],
                        default='auto')
    parser.add_argument('--main_lr_min_factor',
                        help='minimum learning-rate factor relative to the initial LR for cosine decay.',
                        type=float,
                        default=0.05)
    parser.add_argument('--main_lr_decay_gamma',
                        help='multiplicative decay factor for multistep scheduler.',
                        type=float,
                        default=0.1)
    parser.add_argument('--main_lr_milestones',
                        help='epoch milestones for multistep LR decay; ignored unless --main_lr_scheduler=multistep.',
                        type=int,
                        nargs='+',
                        default=[6, 8])
    parser.add_argument('--balanced',
                        help='use balanced Criteo dataset;',
                        action='store_true',
                        default=False)

    parser.add_argument('--defense_all',
                        help='use defense for all passive parties;',
                        action='store_true',
                        default=False)
    parser.add_argument('--adavfed',
                        help='enable the isolated paper-literal Ada-VFed path.',
                        action='store_true',
                        default=False)
    parser.add_argument('--epsilon',
                        help='epsilon for defense;',
                        type=float,
                        default=8.0)
    parser.add_argument('--round',
                        help='round for log;',
                        type=int,
                        default=0)
    parser.add_argument('--result_root',
                        help='root directory for experiment outputs;',
                        type=str,
                        default='.')
    parser.add_argument('--aux_public_dataset',
                        help='public auxiliary dataset used for task prior / canonical ID / public calibration; '
                             'auto resolves mnist->emnist_letters, fashionmnist->kmnist, '
                             'cifar10->cifar100, criteo->criteo_1tb.',
                        type=str,
                        choices=datasets.aux_public_dataset_choices,
                        default='auto')
    parser.add_argument('--dispersion',
                        help='calculate dispersion',
                        action='store_true',
                        default=False)

    parser.add_argument('--hg',
                        help='(HG) Use heterogeneous models for passive parties (exp);'
                             'supports image-model heterogeneity and Criteo with --num_passive=3.',
                        action='store_true',
                        default=False)
    parser.add_argument('--mechanism',
                    help='differential privacy mechanism (laplace or gaussian)',
                    type=str,
                    choices=['laplace', 'gaussian'],
                    default='gaussian')
    parser.add_argument('--rl',
                        action='store_true',
                        default=False)
    
    parser.add_argument('--plot_rl_curves',
                        help='whether to save RL reward/evaluation plots after training; 1 enables plotting and 0 disables it.',
                        type=int,
                        choices=[0, 1],
                        default=0)

    #completion
    parser.add_argument('--completion_aux_num_labels',
                        help='number of auxiliary labeled samples for completion attacks; use 0 to disable completion training.',
                        type=int,
                        default=60)
    parser.add_argument('--completion_aux_seed',
                        help='random seed used when sampling auxiliary data for completion attack.',
                        type=int,
                        default=0)

    # SAC
    parser.add_argument('--sac_lr_actor',
                        help='learning rate for RL Actor network',
                        type=float, default=3e-4)
    parser.add_argument('--sac_lr_critic',
                        help='learning rate for RL Critic networks (Q and R)',
                        type=float, default=3e-4)
    parser.add_argument('--sac_lr_alpha',
                        help='learning rate for RL Alpha (temperature)',
                        type=float, default=3e-4)
    parser.add_argument('--sac_gamma',
                        help='discount factor for RL',
                        type=float, default=0.99)
    parser.add_argument('--sac_batch_size',
                        help='batch size for RL replay buffer',
                        type=int, default=128)
    parser.add_argument('--sac_tau',
                        help='polyak (soft) update rate for target networks',
                        type=float, default=0.005)
    parser.add_argument('--sac_alpha',
                        help='Initial alpha (entropy coefficient)',
                        type=float, default=0.2)
    parser.add_argument('--sac_update_interval',
                        help='number of training batches between SAC updates; larger values usually reduce policy oscillation.',
                        type=int, default=4)
    parser.add_argument('--sac_updates_per_step',
                        help='number of SAC gradient steps performed at each update event.',
                        type=int, default=1)
    parser.add_argument('--sac_grad_clip',
                        help='gradient clipping norm for SAC actor/critic/alpha; set <= 0 to disable clipping.',
                        type=float, default=5.0)
    parser.add_argument('--sac_action_ema',
                        help='EMA coefficient for smoothing executed RL actions across batches; 0 disables smoothing.',
                        type=float, default=0.8)
    parser.add_argument('--sac_log_std_min',
                        help='minimum log standard deviation for SAC actor exploration.',
                        type=float, default=-5.0)
    parser.add_argument('--sac_log_std_max',
                        help='maximum log standard deviation for SAC actor exploration.',
                        type=float, default=0.5)
    parser.add_argument('--sac_window_size',
                        help='Size of the smoothing window (batches) for RL state.',
                        type=int,
                        default=50)
    
    # AMC
    parser.add_argument('--amc_momentum',
                        help='momentum parameter used by the active model completion malicious optimizer.',
                        type=float,
                        default=0.9)
    parser.add_argument('--amc_gamma',
                        help='resetting/scaling parameter used by the active model completion malicious optimizer.',
                        type=float,
                        default=1.0)
    parser.add_argument('--amc_disable_malicious_optimizer',
                        help='use normal SGD for AMC attack_id instead of the malicious local optimizer.',
                        action='store_true',
                        default=False)
    parser.add_argument('--amc_rmax',
                        help='maximum acceleration rate used by the active model completion malicious optimizer.',
                        type=float,
                        default=5.0)
    parser.add_argument('--amc_rmin',
                        help='minimum acceleration rate used by the active model completion malicious optimizer.',
                        type=float,
                        default=1.0)
    parser.add_argument('--amc_ssl_lambda',
                        help='weight for AMC semi-supervised unlabeled loss.',
                        type=float,
                        default=0.5)
    parser.add_argument('--amc_pseudo_threshold',
                        help='confidence threshold for AMC pseudo labels.',
                        type=float,
                        default=0.95)
    parser.add_argument('--amc_attack_seed',
                        help='base random seed for AMC completion-head training.',
                        type=int,
                        default=0)
    parser.add_argument('--amc_num_restarts',
                        help='number of independently trained AMC completion heads whose held-out accuracies are averaged.',
                        type=int,
                        default=5)
    parser.add_argument('--amc_stratified_aux',
                        help='use class-stratified auxiliary-label sampling for AMC (1 enables, 0 disables).',
                        type=int,
                        choices=[0, 1],
                        default=0)
    parser.add_argument('--amc_pseudo_warmup_epochs',
                        help='number of supervised-only AMC epochs before pseudo-label training starts.',
                        type=int,
                        default=0)
    
    parser.add_argument('--rl_forward_clip_target_min',
                        help='deprecated compatibility option; SAC no longer controls forward clipping.',
                        type=float, default=0.08)
    parser.add_argument('--rl_forward_clip_target_max',
                        help='deprecated compatibility option; SAC no longer controls forward clipping.',
                        type=float, default=0.30)
    parser.add_argument('--rl_forward_clip_cap_min',
                        help='deprecated compatibility option; SAC no longer controls forward clipping.',
                        type=float, default=1.00)
    parser.add_argument('--rl_forward_clip_cap_max',
                        help='fixed SAC-era forward clipping cap scale used after SAC forward control was removed.',
                        type=float, default=1.35)
    


    # DePriV Ablation Controls
    # Keep the method-specific switches together for quick ablation edits.
    parser.add_argument('--clip_threshold', type=float, default=0.0005, #ppdl 0.03
                        help='Clipping threshold (C) for adaptive constraints')#mnist0.0005 fashionmnist0.005
    parser.add_argument('--clip_threshold_forward', type=float, default=2.0, 
                    help='Clipping threshold (C) for FORWARD embeddings')
    
    #加权裁剪
    parser.add_argument('--forward_weighted_clipping',
                        help='enable task-aware weighted forward clipping/noise; 0 uses uniform clipping/noise for forward embeddings.',
                        type=int,
                        choices=[0, 1],
                        default=1)
    parser.add_argument('--forward_public_clip_calibration',
                        help='enable one-shot public-data calibration for fixed forward clipping in weighted space.',
                        type=int,
                        choices=[0, 1],
                        default=1)
    parser.add_argument('--forward_public_clip_target_hit',
                        help='target public clip-hit rate used to calibrate fixed forward clipping in weighted space.',
                        type=float,
                        default=0.25)
    parser.add_argument('--forward_public_clip_scale_min',
                        help='minimum multiplicative scale allowed for public calibrated forward clip.',
                        type=float,
                        default=0.90)
    parser.add_argument('--forward_public_clip_scale_max',
                        help='maximum multiplicative scale allowed for public calibrated forward clip.',
                        type=float,
                        default=1.35)
    parser.add_argument('--forward_freeze_public_clip',
                        help='freeze forward clip at the public-data calibrated initial value and disable online clip updates.',
                        type=int,
                        choices=[0, 1],
                        default=0)
    parser.add_argument('--forward_clip_log_interval',
                        help='print DP-safe forward clipping diagnostics every N training batches; 0 disables these logs.',
                        type=int,
                        default=100)
    parser.add_argument('--forward_sparse_keep_rate',
                        help='importance-aware forward keep rate applied after weighted forward DP release in RL mode; 1 disables forward sparsification.',
                        type=float,
                        default=1.0)
    parser.add_argument('--forward_sparse_random_ratio',
                        help='fraction of forward kept coordinates sampled randomly; the remaining kept coordinates are selected by task importance.',
                        type=float,
                        default=0.0)
    parser.add_argument('--backward_topk_mask',
                        help='RL backward release mode: 0 sends all coordinates, 1 uses task-prior Top-K, 2 uses random sparsification.',
                        type=int,
                        choices=[0, 1, 2],
                        default=1)
    parser.add_argument('--tc_smoother',
                        help='enable Temporal-Consensus Smoother for post-DP; 1 enables it and 0 disables it globally.',
                        type=int,
                        choices=[0, 1],
                        default=1)
    parser.add_argument('--structure_public_batches',
                        help='number of auxiliary held-out validation batches used to calibrate the task-aware prior; '
                             'set to 0 to disable auxiliary calibration.',
                        type=int,
                        default=20)

    parser.add_argument('--w_comm_weak', type=float, default=1.0, help='Cost weight for weak party communication')

    parser.add_argument('--ppdl', action='store_true', default=False, 
                        help='Enable PPDL (Privacy-Preserving Deep Learning) defense baseline.')
    parser.add_argument('--ppdl_tau', type=float, default=0.001, 
                        help='PPDL: Threshold value tau for gradient selection (default: 0.1).')
    parser.add_argument('--ppdl_theta_u', type=float, default=0.1, 
                        help='PPDL: Max fraction (theta_u) of parameters to update (default: 0.5).')
    parser.add_argument('--ppdl_noise_std', type=float, default=0.01, 
                        help='PPDL: Standard deviation for Gaussian noise (default: 0.05).')
    
    parser.add_argument('--acvfl',help='use TIFS 2024 adaptive clipping;',action='store_true')
    parser.add_argument('--acvfl_bidirectional',
                        help='ACVFL ablation switch: 1 enables bidirectional DP (forward + backward), 0 keeps the original backward-only ACVFL path.',
                        type=int,
                        choices=[0, 1],
                        default=1)
    parser.add_argument('--acvfl_gamma', type=float, default=0.1)
    parser.add_argument('--acvfl_lr_c', type=float, default=0.2,
                        help='ACVFL adaptive clipping update rate.')
    parser.add_argument('--acvfl_initial_c', type=float, default=-1.0,
                        help='Initial ACVFL clipping norm; -1 uses --clip_threshold.')
    parser.add_argument('--acvfl_min_c', type=float, default=1e-8,
                        help='Minimum ACVFL adaptive clipping norm.')
    parser.add_argument('--acvfl_max_c', type=float, default=-1.0,
                        help='Maximum ACVFL adaptive clipping norm; -1 uses --clip_threshold.')


    add_fia_arguments(parser, hidden_help)

    # --- 1. 收益塑形与对偶更新 (Reward Shaping & Dual Update) ---
    parser.add_argument('--beta_payment', 
                        help='Leader收益中支付成本(Payment Cost)的惩罚权重;', 
                        type=float, 
                        default=0.01)
    parser.add_argument('--w_utility', 
                        help='Leader收益中模型准确率(Utility)的放大权重 (用于对齐Reward的量级);', 
                        type=float, 
                        default=50.0)

    parser.add_argument('--alpha_lambda_eps', 
                        help='隐私预算拉格朗日乘子(lambda_eps)的更新步长(学习率);', 
                        type=float, 
                        default=0.01)
    parser.add_argument('--alpha_lambda_risk', 
                        help='攻击风险拉格朗日乘子(lambda_risk)的更新步长(学习率);', 
                        type=float, 
                        default=0.01)

    # DePriV released-message risk proxy
    parser.add_argument('--rl_risk_mode',
                        choices=['geometry', 'public_quality'],
                        default='geometry',
                        help='Risk state: released-gradient geometry or a public Cluster prior updated by prior DP-release quality.')
    parser.add_argument('--rl_public_risk_file',
                        type=str,
                        default='',
                        help='Raw or summary JSON produced by run_public_lia_risk_probe.py; required for --rl_risk_mode public_quality.')
    parser.add_argument('--rl_risk_window_samples',
                        help='Maximum number of previously released per-sample gradients retained for spectral risk estimation.',
                        type=int,
                        default=256)
    parser.add_argument('--rl_risk_min_samples',
                        help='Minimum number of released samples required before updating the spectral risk estimate.',
                        type=int,
                        default=64)
    parser.add_argument('--rl_risk_update_interval',
                        help='Number of released batches between released-message risk updates.',
                        type=int,
                        default=10)
    parser.add_argument('--rl_risk_ema_beta',
                        help='EMA coefficient for the released-gradient geometry risk score.',
                        type=float,
                        default=0.9)
    parser.add_argument('--rl_quality_ema_beta',
                        help='EMA coefficient for each party\'s historical DP-release quality.',
                        type=float,
                        default=0.9)
    parser.add_argument('--rl_quality_power',
                        help='Strength of relative DP-release quality in public-prior risk updates.',
                        type=float,
                        default=2.0)
    parser.add_argument('--rl_quality_ratio_min', type=float, default=0.8)
    parser.add_argument('--rl_quality_ratio_max', type=float, default=1.2)
    parser.add_argument('--rl_risk_max_features',
                        help='Maximum number of released-gradient coordinates used by each geometry risk update.',
                        type=int,
                        default=256)
    parser.add_argument('--rl_proxy_threshold',
                        help='Fixed released-geometry risk threshold on the normalized [0, 100] scale.',
                        type=float,
                        default=40.0)
    
    # DePriV RL budget and keep-rate controls
    parser.add_argument('--delta_c',  
                        type=float, 
                        default=1e-5)
    
    parser.add_argument('--eps_strong_min_ratio',
                        help='Minimum epsilon ratio for strongest parties relative to weak-party epsilon in RL mode.',
                        type=float,
                        default=0.85)
    parser.add_argument('--eps_strong_max_ratio',
                        help='Maximum epsilon ratio for strongest parties relative to weak-party epsilon in RL mode.',
                        type=float,
                        default=1.0)
    parser.add_argument('--q_weak_fixed',
                        help='Fixed backward post-noise coordinate keep rate for weak parties in RL mode. This is not the DP accounting subsampling rate.',
                        type=float,
                        default=0.1)
    parser.add_argument('--rl_q_min',
                        help='Minimum backward post-noise coordinate keep rate that RL may set. This is not the DP accounting subsampling rate.',
                        type=float, default=0.05)
    parser.add_argument('--rl_q_max',
                        help='Maximum backward post-noise coordinate keep rate that RL may set. This is not the DP accounting subsampling rate.',
                        type=float, default=0.15)
    parser.add_argument('--rl_risk_q_min',
                        help='Minimum backward keep rate used after a party exceeds the fixed proxy-risk threshold.',
                        type=float, default=0.15)
    parser.add_argument('--rl_risk_q_max',
                        help='Maximum backward keep rate used after a party exceeds the fixed proxy-risk threshold.',
                        type=float, default=0.50)
    parser.add_argument('--forward_budget_ratio',
                        help='Fraction of total privacy budget initially reserved for the forward path in RL mode.',
                        type=float,
                        default=0.90)
    parser.add_argument('--forward_fixed_eps',
                        help='Optional manual fixed forward-path step epsilon in RL mode. Use -1 to let the planner derive it from total_eps and forward_budget_ratio.',
                        type=float,
                        default=-1.0)
    parser.add_argument('--budget_sweep_mode',
                        help='Observation-only budget allocation mode for defense_all: joint keeps bidirectional DP, forward_only spends the full epsilon on the forward path, backward_only spends the full epsilon on the backward path.',
                        type=str,
                        choices=['joint', 'forward_only', 'backward_only'],
                        default='joint')
    parser.add_argument('--observation3_mode',
                        help='Observation-3-only sparsification mode for the standard defense_all path: forward sparsifies noised forward embeddings, backward sparsifies noised backward gradients, none disables the sweep.',
                        type=str,
                        choices=['none', 'forward', 'backward'],
                        default='none')
    parser.add_argument('--observation3_keep_rate',
                        help='Observation-3-only random coordinate keep rate applied after DP noise; used as the x-axis for the forward/backward sparsification sweep.',
                        type=float,
                        default=1.0)


    args = parser.parse_args()
    args.command = "python {}".format(subprocess.list2cmdline(sys.argv))
    completion_aux_seed_explicit = cli_arg_explicitly_provided('--completion_aux_seed')

    if args.set_attack_epoch and args.attack_epoch > args.epochs:
        raise ValueError('--attack_epoch should be smaller than or equals to --epochs')
    if not args.set_attack_epoch and args.attack_epoch != 1:
        raise ValueError('--attack_epoch should be 1 if not use `--set_attack_epoch`')
    if not (0.0 <= args.amc_ssl_lambda):
        raise ValueError('--amc_ssl_lambda should be non-negative.')
    if not (0.0 < args.amc_pseudo_threshold <= 1.0):
        raise ValueError('--amc_pseudo_threshold should be in the interval (0, 1].')
    if args.amc_num_restarts < 1:
        raise ValueError('--amc_num_restarts should be at least 1.')
    if args.amc_pseudo_warmup_epochs < 0:
        raise ValueError('--amc_pseudo_warmup_epochs should be non-negative.')
    if not (0.0 < args.q_weak_fixed <= 1.0):
        raise ValueError('--q_weak_fixed should be in the interval (0, 1].')
    if not (0.0 < args.rl_q_min <= args.rl_q_max <= 1.0):
        raise ValueError('--rl_q_min and --rl_q_max should satisfy 0 < min <= max <= 1.')
    if not (0.0 < args.rl_risk_q_min <= args.rl_risk_q_max <= 1.0):
        raise ValueError('--rl_risk_q_min and --rl_risk_q_max should satisfy 0 < min <= max <= 1.')
    if args.rl_risk_window_samples <= 0:
        raise ValueError('--rl_risk_window_samples should be positive.')
    if args.rl_risk_min_samples <= 0 or args.rl_risk_min_samples > args.rl_risk_window_samples:
        raise ValueError('--rl_risk_min_samples should be in (0, rl_risk_window_samples].')
    if args.rl_risk_update_interval <= 0:
        raise ValueError('--rl_risk_update_interval should be positive.')
    if not (0.0 <= args.rl_risk_ema_beta < 1.0):
        raise ValueError('--rl_risk_ema_beta should be in [0, 1).')
    if not (0.0 <= args.rl_quality_ema_beta < 1.0):
        raise ValueError('--rl_quality_ema_beta should be in [0, 1).')
    if args.rl_quality_power < 0.0:
        raise ValueError('--rl_quality_power should be non-negative.')
    if not (0.0 < args.rl_quality_ratio_min <= args.rl_quality_ratio_max):
        raise ValueError('--rl_quality_ratio_min/max should satisfy 0 < min <= max.')
    if args.rl_risk_max_features <= 0:
        raise ValueError('--rl_risk_max_features should be positive.')
    if not (0.0 <= args.rl_proxy_threshold <= 100.0):
        raise ValueError('--rl_proxy_threshold should be in [0, 100].')
    if args.backward_topk_mask not in [0, 1, 2]:
        raise ValueError('--backward_topk_mask should be 0 (dense), 1 (Top-K), or 2 (random).')
    if args.acvfl_bidirectional not in [0, 1]:
        raise ValueError('--acvfl_bidirectional should be either 0 or 1.')
    if not (0.0 < args.acvfl_gamma < 1.0):
        raise ValueError('--acvfl_gamma should be in the interval (0, 1).')
    if args.acvfl_lr_c < 0.0:
        raise ValueError('--acvfl_lr_c should be non-negative.')
    if args.acvfl_initial_c != -1.0 and args.acvfl_initial_c <= 0.0:
        raise ValueError('--acvfl_initial_c should be positive, or -1 to use --clip_threshold.')
    if args.acvfl_min_c <= 0.0:
        raise ValueError('--acvfl_min_c should be positive.')
    if args.acvfl_max_c != -1.0 and args.acvfl_max_c < args.acvfl_min_c:
        raise ValueError('--acvfl_max_c should be >= --acvfl_min_c, or -1 to use --clip_threshold.')
    if not (0.0 < args.forward_budget_ratio < 1.0):
        raise ValueError('--forward_budget_ratio should be in the interval (0, 1).')
    if args.forward_fixed_eps != -1.0 and args.forward_fixed_eps <= 0.0:
        raise ValueError('--forward_fixed_eps should be positive, or -1 to use the fixed fallback.')
    if args.forward_weighted_clipping not in [0, 1]:
        raise ValueError('--forward_weighted_clipping should be either 0 or 1.')
    if args.forward_public_clip_calibration not in [0, 1]:
        raise ValueError('--forward_public_clip_calibration should be either 0 or 1.')
    if not (0.0 < args.forward_public_clip_target_hit < 1.0):
        raise ValueError('--forward_public_clip_target_hit should be in the interval (0, 1).')
    if not (0.0 < args.forward_public_clip_scale_min <= args.forward_public_clip_scale_max):
        raise ValueError('--forward_public_clip_scale_min and --forward_public_clip_scale_max should satisfy 0 < min <= max.')
    if args.forward_freeze_public_clip not in [0, 1]:
        raise ValueError('--forward_freeze_public_clip should be either 0 or 1.')
    if args.forward_clip_log_interval < 0:
        raise ValueError('--forward_clip_log_interval should be greater than or equal to 0.')
    if not (0.0 < args.forward_sparse_keep_rate <= 1.0):
        raise ValueError('--forward_sparse_keep_rate should be in the interval (0, 1].')
    if not (0.0 <= args.forward_sparse_random_ratio <= 1.0):
        raise ValueError('--forward_sparse_random_ratio should be in the interval [0, 1].')
    if args.structure_public_batches < 0:
        raise ValueError('--structure_public_batches should be non-negative.')
    validate_fia_args(args)
    if args.completion_aux_num_labels < 0:
        raise ValueError('--completion_aux_num_labels should be non-negative.')
    if args.attack_id >= args.num_passive:
        raise ValueError('--attack_id should be smaller than --num_passive')
    if args.padding_mode and args.num_passive < 2:
        raise ValueError('--padding_mode should be used with --num_passive >= 2')
    if args.padding_mode and args.dataset == "criteo":
        raise ValueError("Dataset Criteo can not use padding_mode.")
    if not (0.0 < args.eps_strong_min_ratio <= args.eps_strong_max_ratio <= 1.0):
        raise ValueError('--eps_strong_min_ratio and --eps_strong_max_ratio should satisfy 0 < min <= max <= 1.')
    if args.num_passive != 1 and not args.padding_mode:
        if args.dataset in ['mnist', 'fashionmnist'] and args.num_passive not in [2, 4, 7]:#247
            raise ValueError("The number of passive parties for {} must be 1, 2, 4 or 7.".format(datasets.datasets_name[args.dataset]))
        elif args.dataset in ['cifar10', 'cifar100'] and args.num_passive not in [2, 4, 8]:
            raise ValueError("The number of passive parties for {} must be 1, 2, 4 or 8.".format(datasets.datasets_name[args.dataset]))
        elif args.dataset == "criteo" and args.num_passive != 3:
            raise ValueError("The number of passive parties for {} must be 1 or 3.".format(datasets.datasets_name[args.dataset]))
    if args.balanced and args.dataset != "criteo":
        raise ValueError("{} dataset should not use --balanced.".format(datasets.datasets_name[args.dataset]))
    if args.tsne and args.attack != 'cluster':
        raise ValueError("--tsne should be used with --attack='cluster'")
    if args.use_emb and args.attack != 'cluster':
        raise ValueError("--use_emb should be used with --attack='cluster'")
    if args.division_mode in ['random', 'imbalanced'] and args.dataset not in ['mnist', 'cifar10']:
        raise ValueError("Dataset {} can not use division_mode={}.".format(datasets.datasets_name[args.dataset], args.division_mode))
    if args.ppdl and args.defense_all:
        raise ValueError("Can not use both --ppdl and --defense_all.")
    if args.ppdl and getattr(args, 'acvfl', False):
        raise ValueError("Can not use both --ppdl and --acvfl.")
    if args.ppdl and args.rl:
        raise ValueError("Can not use both --ppdl and --rl.")
    if args.adavfed and args.defense_all:
        raise ValueError("Can not use both --adavfed and --defense_all.")
    if args.adavfed and args.ppdl:
        raise ValueError("Can not use both --adavfed and --ppdl.")
    if args.adavfed and args.rl:
        raise ValueError("Can not use both --adavfed and --rl.")
    if args.adavfed and getattr(args, 'acvfl', False):
        raise ValueError("Can not use both --adavfed and --acvfl.")
    if args.budget_sweep_mode != 'joint':
        if not args.defense_all:
            raise ValueError('--budget_sweep_mode requires --defense_all.')
        if args.ppdl:
            raise ValueError('--budget_sweep_mode is only supported for the standard --defense_all path, not --ppdl.')
        if args.rl:
            raise ValueError('--budget_sweep_mode is only supported for the standard --defense_all path, not --rl.')
        if getattr(args, 'acvfl', False):
            raise ValueError('--budget_sweep_mode is only supported for the standard --defense_all path, not --acvfl.')
    if not (0.0 < args.observation3_keep_rate <= 1.0):
        raise ValueError('--observation3_keep_rate should be in the interval (0, 1].')
    if args.observation3_mode != 'none':
        if not args.defense_all:
            raise ValueError('--observation3_mode requires --defense_all.')
        if args.budget_sweep_mode != 'joint':
            raise ValueError('--observation3_mode only supports the default joint budget accounting path.')
        if args.ppdl:
            raise ValueError('--observation3_mode is only supported for the standard --defense_all path, not --ppdl.')
        if args.rl:
            raise ValueError('--observation3_mode is only supported for the standard --defense_all path, not --rl.')
        if getattr(args, 'acvfl', False):
            raise ValueError('--observation3_mode is only supported for the standard --defense_all path, not --acvfl.')
    if args.weight_decay_active != -1.0 and args.weight_decay_active < 0.0:
        raise ValueError('--weight_decay_active should be non-negative, or -1 for auto.')
    if args.weight_decay_passive != -1.0 and args.weight_decay_passive < 0.0:
        raise ValueError('--weight_decay_passive should be non-negative, or -1 for auto.')
    if not (0.0 <= args.fusion_dropout < 1.0):
        raise ValueError('--fusion_dropout should be in the interval [0, 1).')
    if not (0.0 <= args.cifar_random_erasing_p <= 1.0):
        raise ValueError('--cifar_random_erasing_p should be in the interval [0, 1].')

    if not (0.0 <= args.main_lr_min_factor <= 1.0):
        raise ValueError('--main_lr_min_factor should be in the interval [0, 1].')
    if not (0.0 < args.main_lr_decay_gamma < 1.0):
        raise ValueError('--main_lr_decay_gamma should be in the interval (0, 1).')
    if any(m <= 0 for m in args.main_lr_milestones):
        raise ValueError('--main_lr_milestones should only contain positive epoch indices.')
    apply_fia_preset(args, cli_arg_explicitly_provided)
    run_seeds = list(args.run_seeds) if args.run_seeds else [args.seed + idx for idx in range(5)]
    test_acc_list = []
    attack_acc_runs = []
    amc_train_attack_acc_runs = []
    fia_metric_runs = []

    for run_idx, seed in enumerate(run_seeds):
        run_result = run_single_experiment(
            args,
            seed,
            run_idx,
            len(run_seeds),
            completion_aux_seed_explicit,
        )
        test_acc_list.append(run_result['test_acc'])
        attack_acc_runs.append(run_result['attack_accs'])
        amc_train_attack_acc_runs.append(run_result['amc_train_attack_accs'])
        fia_metric_runs.append(run_result['fia_metrics'])

    mean_acc, std_acc, summary_text = format_acc_summary(test_acc_list)
    per_run_text = ", ".join("{:.2f}%".format(acc) for acc in test_acc_list)
    print("\n" + "=" * 72)
    print("Multi-run seeds: {}".format(run_seeds))
    print("Per-run test_acc: {}".format(per_run_text))
    print("Final test_acc summary: {}".format(summary_text))
    if any(any(acc is not None for acc in run_attack_accs) for run_attack_accs in attack_acc_runs):
        for passive_id in range(args.num_passive):
            party_run_accs = [
                run_attack_accs[passive_id] if passive_id < len(run_attack_accs) else None
                for run_attack_accs in attack_acc_runs
            ]
            valid_party_accs = [acc for acc in party_run_accs if acc is not None]
            if len(valid_party_accs) == 0:
                continue
            _, _, party_summary_text = format_acc_summary(valid_party_accs)
            party_per_run_text = ", ".join(format_optional_acc(acc) for acc in party_run_accs)
            print("Per-run attack_acc of Passive {}: {}".format(passive_id, party_per_run_text))
            print("Final attack_acc summary of Passive {}: {}".format(passive_id, party_summary_text))
    if any(any(acc is not None for acc in run_accs) for run_accs in amc_train_attack_acc_runs):
        for passive_id in range(args.num_passive):
            party_run_accs = [
                run_accs[passive_id] if passive_id < len(run_accs) else None
                for run_accs in amc_train_attack_acc_runs
            ]
            valid_party_accs = [acc for acc in party_run_accs if acc is not None]
            if len(valid_party_accs) == 0:
                continue
            _, _, party_summary_text = format_acc_summary(valid_party_accs)
            party_per_run_text = ", ".join(format_optional_acc(acc) for acc in party_run_accs)
            print("Per-run AMC train-heldout_acc of Passive {}: {}".format(passive_id, party_per_run_text))
            print("Final AMC train-heldout_acc summary of Passive {}: {}".format(passive_id, party_summary_text))
    valid_fia_metrics = [metrics for metrics in fia_metric_runs if metrics is not None]
    if valid_fia_metrics:
        mse_values = [metrics['mse_mean'] for metrics in valid_fia_metrics]
        ssim_values = [metrics['ssim_mean'] for metrics in valid_fia_metrics]
        _, _, mse_summary = format_fia_summary(mse_values)
        _, _, ssim_summary = format_fia_summary(ssim_values)
        per_seed_text = ", ".join(
            "seed {}: MSE={:.6f}, SSIM={:.6f}".format(seed, metrics['mse_mean'], metrics['ssim_mean'])
            for seed, metrics in zip(run_seeds, fia_metric_runs)
            if metrics is not None
        )
        print("Per-run FIA metrics: {}".format(per_seed_text))
        print("Final FIA MSE summary across seeds: {}".format(mse_summary))
        print("Final FIA SSIM summary across seeds: {}".format(ssim_summary))
    print("=" * 72)


def cli_arg_explicitly_provided(flag_name):
    return any(
        arg == flag_name or arg.startswith(flag_name + "=")
        for arg in sys.argv[1:]
    )

def clone_args(args):
    return argparse.Namespace(**vars(args))

def format_acc_summary(acc_list):
    values = np.asarray(
        [float(acc) for acc in acc_list if acc is not None and np.isfinite(acc)],
        dtype=float,
    )
    mean_acc = float(values.mean()) if len(values) > 0 else 0.0
    std_acc = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean_acc, std_acc, "{:.2f}%±{:.2f}%".format(mean_acc, std_acc)


def format_fia_summary(values):
    values = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=float)
    mean_value = float(values.mean()) if len(values) > 0 else 0.0
    std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean_value, std_value, "{:.6f}+/-{:.6f}".format(mean_value, std_value)


def format_optional_acc(acc):
    if acc is None or not np.isfinite(acc):
        return "NA"
    return "{:.2f}%".format(float(acc))


def normalize_attack_accs(raw_attack_acc, num_passive, attack_id):
    if num_passive <= 0:
        return []

    if isinstance(raw_attack_acc, np.ndarray):
        raw_attack_acc = raw_attack_acc.tolist()

    if isinstance(raw_attack_acc, (list, tuple)):
        attack_accs = [None] * num_passive
        for passive_id in range(min(num_passive, len(raw_attack_acc))):
            value = raw_attack_acc[passive_id]
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                attack_accs[passive_id] = float(np.clip(value, 0.0, 100.0))
        return attack_accs

    try:
        scalar_acc = float(raw_attack_acc)
    except (TypeError, ValueError):
        return [None] * num_passive

    if not np.isfinite(scalar_acc):
        return [None] * num_passive

    scalar_acc = float(np.clip(scalar_acc, 0.0, 100.0))
    attack_accs = [None] * num_passive
    passive_id = min(max(int(attack_id), 0), num_passive - 1)
    attack_accs[passive_id] = scalar_acc
    return attack_accs


def extract_final_attack_accs(trainer):
    metrics = getattr(trainer, 'metrics', None)
    raw_attack_acc = None
    if metrics is not None and getattr(metrics, 'attack_acc', None):
        raw_attack_acc = metrics.attack_acc[-1]

    return normalize_attack_accs(
        raw_attack_acc=raw_attack_acc,
        num_passive=int(getattr(trainer.args, 'num_passive', 0)),
        attack_id=getattr(trainer.args, 'attack_id', 0),
    )


def extract_final_amc_train_attack_accs(trainer):
    if str(getattr(trainer.args, 'attack', '')).lower() != 'amc':
        return [None] * int(getattr(trainer.args, 'num_passive', 0))
    metrics = getattr(trainer, 'metrics', None)
    raw_attack_acc = None
    if metrics is not None and getattr(metrics, 'amc_train_attack_acc', None):
        raw_attack_acc = metrics.amc_train_attack_acc[-1]
    return normalize_attack_accs(
        raw_attack_acc=raw_attack_acc,
        num_passive=int(getattr(trainer.args, 'num_passive', 0)),
        attack_id=getattr(trainer.args, 'attack_id', 0),
    )


def extract_final_fia_metrics(trainer):
    if str(getattr(trainer.args, 'attack', '')).lower() != 'ressfl_fia':
        return None
    metrics = getattr(trainer, 'metrics', None)
    summaries = getattr(metrics, 'fia_sample_metrics', None) if metrics is not None else None
    return summaries[-1] if summaries else None

def run_single_experiment(base_args, seed, run_idx, total_runs, completion_aux_seed_explicit):
    args = clone_args(base_args)
    args.seed = int(seed)
    use_fixed_amc_aux_seed = (
        str(getattr(args, 'attack', '')).lower() == 'amc'
        and not completion_aux_seed_explicit
    )
    if not completion_aux_seed_explicit and not use_fixed_amc_aux_seed:
        args.completion_aux_seed = args.seed

    set_global_seed(args.seed)

    print("\n" + "=" * 72)
    print("Starting run {}/{} with seed {}".format(run_idx + 1, total_runs, args.seed))
    if use_fixed_amc_aux_seed:
        print(
            "Using fixed AMC completion_aux_seed={} across VFL seeds; "
            "restart offsets are applied inside AMC.".format(args.completion_aux_seed)
        )
    elif not completion_aux_seed_explicit:
        print("Using completion_aux_seed={} to match the experiment seed.".format(args.completion_aux_seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(
        "Run Config -> dataset={}, attack={}, attack_id={}, parties={}, division={}, batch_size={}, epochs={}, hg={}, rl={}, seed={}".format(
            args.dataset,
            args.attack,
            args.attack_id,
            args.num_passive,
            args.division_mode,
            args.batch_size,
            args.epochs,
            bool(args.hg),
            bool(args.rl),
            args.seed,
        )
    )
    print(
        "Optim Config -> lr_active={}, lr_passive={}, lr_attack={}, lr_attack_model={}".format(
            args.lr_active,
            args.lr_passive,
            args.lr_attack,
            args.lr_attack_model,
        )
    )
    resolved_wd_active, resolved_wd_passive = optim_utils.resolve_weight_decay(args)
    print(
        "Regularization -> wd_active={}, wd_passive={}, momentum={}, fusion_dropout={}, cifar_random_erasing={}, erasing_p={}".format(
            resolved_wd_active,
            resolved_wd_passive,
            optim_utils.resolve_sgd_momentum(args),
            args.fusion_dropout,
            bool(args.cifar_random_erasing),
            args.cifar_random_erasing_p,
        )
    )
    scheduler_name = optim_utils.resolve_main_lr_scheduler_name(args)
    print(
        "LR Schedule -> mode={}, min_factor={}, gamma={}, milestones={}".format(
            scheduler_name,
            args.main_lr_min_factor,
            args.main_lr_decay_gamma,
            args.main_lr_milestones,
        )
    )
    if args.adavfed:
        print(
            "AdaVFed Paper Config -> total_eps={} | paper defaults are isolated in adavfed/config.py".format(
                args.epsilon,
            )
        )
    print_fia_config(args)

    dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    output_root = os.path.abspath(args.result_root)

    data_dir = os.path.join(output_root, "data", args.attack, args.dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset_path = os.path.join(dir, 'dataset')
    if args.dataset == "criteo":
        data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True, balanced=args.balanced)
    else:
        train_transform = datasets.build_train_transform(args.dataset, args)
        data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True, download=True, transform=train_transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=False)
    if args.dataset == "criteo":
        data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False, balanced=args.balanced)
    else:
        data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False, transform=datasets.transforms_default[args.dataset])

    aux_public_data = None
    resolved_aux_public_dataset = "none"
    aux_public_label_compatible = False
    final_test_data = data_test
    needs_aux_public_data = needs_fia_public_data(args)
    if needs_aux_public_data:
        aux_public_data, resolved_aux_public_dataset, aux_public_label_compatible = (
            datasets.build_aux_public_dataset(
                root=dataset_path,
                target_dataset=args.dataset,
                aux_dataset_name=args.aux_public_dataset,
                download=True,
            )
        )
        if args.dataset == "criteo" and args.rl and aux_public_data is None:
            raise ValueError(
                "Criteo --rl requires the independent Criteo 1TB public auxiliary "
                "set. Use --aux_public_dataset auto or criteo_1tb."
            )
    else:
        aux_public_data = None
        resolved_aux_public_dataset = "none"
        aux_public_label_compatible = False

    args.resolved_aux_public_dataset = resolved_aux_public_dataset
    args.aux_public_label_compatible = bool(aux_public_label_compatible)

    dataloader_aux_public = None if aux_public_data is None else DataLoader(aux_public_data, batch_size=args.batch_size, shuffle=False)
    dataloader_test = DataLoader(final_test_data, batch_size=args.batch_size, shuffle=False)
    if needs_aux_public_data:
        if aux_public_data is None:
            print("Aux Public Dataset -> disabled")
        else:
            print(
                "Aux Public Dataset -> {} | samples={} | label_compatible={}".format(
                    datasets.aux_public_dataset_names.get(
                        resolved_aux_public_dataset,
                        resolved_aux_public_dataset,
                    ),
                    len(aux_public_data),
                    bool(aux_public_label_compatible),
                )
            )
    if args.simple:
        entire_model = models.entire_simple[args.dataset](num_passive=args.num_passive, padding_mode=args.padding_mode, division_mode=args.division_mode, args=args)
    else:
        entire_model = models.entire[args.dataset](num_passive=args.num_passive, padding_mode=args.padding_mode, division_mode=args.division_mode, args=args)

    entire_model = entire_model.to(device)
    attacker_path = 'attackers.%s' % args.attack
    attacker = getattr(importlib.import_module(attacker_path), 'Attacker')

    trainer = None
    test_acc = None
    try:
        trainer = attacker(
            args,
            entire_model,
            dataloader_train,
            (dataloader_aux_public, dataloader_test),
        )
        trainer.train()

        if args.rl:
            plotting.plot_rl_training_curves(args, trainer, output_root)
        test_acc = trainer.test()

    finally:
        if trainer is not None and hasattr(trainer, 'restore_stdout'):
            trainer.restore_stdout()

    if test_acc is None:
        raise RuntimeError("trainer.test() returned None. Please make sure the selected attacker returns super().test().")

    final_attack_accs = extract_final_attack_accs(trainer)
    final_amc_train_attack_accs = extract_final_amc_train_attack_accs(trainer)
    final_fia_metrics = extract_final_fia_metrics(trainer)
    final_attack_text = ", ".join(
        "P{}={}".format(passive_id, format_optional_acc(acc))
        for passive_id, acc in enumerate(final_attack_accs)
    )
    print("Finished run {}/{} with seed {} -> test_acc={:.2f}%".format(
        run_idx + 1,
        total_runs,
        args.seed,
        float(test_acc),
    ))
    if any(acc is not None for acc in final_attack_accs):
        print("Final attack_accs of this run: {}".format(final_attack_text))
    if any(acc is not None for acc in final_amc_train_attack_accs):
        amc_train_text = ", ".join(
            "P{}={}".format(passive_id, format_optional_acc(acc))
            for passive_id, acc in enumerate(final_amc_train_attack_accs)
        )
        print("Final AMC train-heldout_accs of this run: {}".format(amc_train_text))
    if final_fia_metrics is not None:
        print(
            "Final FIA metrics of this run: MSE={:.6f}+/-{:.6f} (sample std), "
            "SSIM={:.6f}+/-{:.6f} (sample std), n={}".format(
                final_fia_metrics['mse_mean'],
                final_fia_metrics['mse_std'],
                final_fia_metrics['ssim_mean'],
                final_fia_metrics['ssim_std'],
                final_fia_metrics['num_samples'],
            )
        )
    return {
        'test_acc': float(test_acc),
        'attack_accs': final_attack_accs,
        'amc_train_attack_accs': final_amc_train_attack_accs,
        'fia_metrics': final_fia_metrics,
    }

if __name__ == '__main__':
    main()


