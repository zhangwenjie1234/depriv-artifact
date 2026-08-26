import torch
from utils.metrics import Metrics
import time
import os
import sys
import numpy as np
import utils.datasets as datasets
import utils.metrics as metrics_utils
import utils.privacy as privacy_utils
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from collections import deque
import attackers.RL as rl_agent_module
from utils.privacy import PrivacyMonitor
import utils.dp as dp
import utils.ppdl as ppdl
import utils.rl_helpers as rl_helpers
import utils.optimizers as optim_utils
from datetime import datetime
from utils.task_prior import TaskPrior
from utils.smoother import TemporalConsensusSmoother
from utils.risk_proxy import (
    PublicPriorReleaseRiskTracker,
    ReleasedGradientRiskTracker,
    load_public_cluster_scores,
)

try:
    from utils import rl_runtime_logger
except ImportError:  # Optional temporary diagnostics; safe to delete the file.
    rl_runtime_logger = None

try:
    from adavfed import AdaVFedConfig, AdaVFedEngine
    from adavfed.attack_support import build_completion_cache_payload, completion_attack_mode
except ImportError:  # pragma: no cover - optional isolated paper-repro package
    AdaVFedConfig = None
    AdaVFedEngine = None
    build_completion_cache_payload = None
    completion_attack_mode = None


def build_main_task_loss(args):
    if bool(getattr(args, 'adavfed', False)):
        # Eq. (1) defines the paper task loss as the mean over samples.
        return torch.nn.CrossEntropyLoss(reduction='mean')
    return torch.nn.CrossEntropyLoss()


def normalize_reported_loss(total_loss, num_batches, num_examples, uses_sum_reduction):
    if uses_sum_reduction:
        return float(total_loss) / float(max(num_examples, 1))
    return float(total_loss) / float(max(num_batches, 1))


def main_task_uses_sum_reduction(args):
    return False


def normalize_step_loss(loss_value, batch_examples, uses_sum_reduction):
    return normalize_reported_loss(
        total_loss=loss_value,
        num_batches=1,
        num_examples=batch_examples,
        uses_sum_reduction=uses_sum_reduction,
    )


class BaseVFL(object):
    def __init__(self, args, entire_model, train_dataset, test_dataset):
        # setup arguments
        self.args = args
        self._original_stdout = sys.stdout
        self._stdout_logger = None
        self.output_root = os.path.abspath(getattr(self.args, 'result_root', '.'))
        if isinstance(test_dataset, tuple) and len(test_dataset) == 3:
            aux_public_dataset, _, final_test_dataset = test_dataset
        elif isinstance(test_dataset, tuple) and len(test_dataset) == 2:
            aux_public_dataset, final_test_dataset = test_dataset
        else:
            aux_public_dataset = None
            final_test_dataset = test_dataset

        self._setup_output_dirs_and_logging()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adavfed_engine = None
        # process dataset
        self._process_data(train_dataset, aux_public_dataset, final_test_dataset)
        self.model = entire_model.to(self.device)
        if getattr(self.args, 'adavfed', False):
            if self.args.defense_all or getattr(self.args, 'acvfl', False) or self.args.ppdl or self.args.rl:
                raise ValueError('AdaVFed isolated mode can not be mixed with defense_all/acvfl/ppdl/rl.')
            if AdaVFedConfig is None or AdaVFedEngine is None:
                raise ImportError('adavfed package is not available.')
            self.adavfed_engine = AdaVFedEngine(
                config=AdaVFedConfig.from_args(self.args),
                num_passive=self.args.num_passive,
                device=self.device,
            ).to(self.device)
            self.adavfed_engine.configure_training_steps(
                self.args.epochs * len(self.train_dataset)
            )
            self._prime_adavfed_engine_parameters()

        self.task_prior = TaskPrior(self.args, self)
        # AMC's passive adversary retains the local embedding before any
        # forward perturbation; other attacks continue to use released data.
        self._last_raw_forward_embeddings = [None] * self.args.num_passive
        self._amc_raw_cache_logged = False
        for i in range(self.args.num_passive):
            self.model.passive[i].register_forward_hook(self.task_prior.get_hook(i))

        self.loss = build_main_task_loss(self.args)

        self.optimizer_entire, self.optimizer_active, self.optimizer_passive = (
            optim_utils.build_main_task_optimizers(self.args, self.model)
        )
        self._register_adavfed_optimizer_params()
        self.main_lr_schedulers = optim_utils.build_main_task_schedulers(
            self.args,
            self.optimizer_entire,
            self.optimizer_active,
            self.optimizer_passive,
        )

        # setup metrics
        self.metrics = Metrics(args)
        self.iteration = None
        self.num_classes = datasets.datasets_classes[self.args.dataset]

        self.local_history_grads = {}
        self.train_dataset_len = len(train_dataset.dataset) if hasattr(train_dataset, 'dataset') else len(train_dataset)
        sample_rate = self.args.batch_size / self.train_dataset_len
        monitor_delta = (
            self.adavfed_engine.config.delta
            if self.adavfed_engine is not None and self.adavfed_engine.config.enabled
            else 1e-5
        )
        self.privacy_monitor = PrivacyMonitor(
            num_parties=self.args.num_passive, 
            sample_rate=sample_rate, 
            delta=monitor_delta,
            mechanism='gaussian',
            use_dim_amplification=False,
            verbose=not self.args.rl,
        )
        self.forward_privacy_monitor = PrivacyMonitor(
            num_parties=self.args.num_passive,
            sample_rate=sample_rate,
            delta=monitor_delta,
            mechanism='gaussian',
            use_dim_amplification=False,
            verbose=False,
        )
        self.backward_privacy_monitor = PrivacyMonitor(
            num_parties=self.args.num_passive,
            sample_rate=sample_rate,
            delta=monitor_delta,
            mechanism='gaussian',
            use_dim_amplification=False,
            verbose=False,
        )
        self.released_gradient_risk_tracker = None
        self.esi_tracker = deque(maxlen=self.args.sac_window_size)

        self.norm_tracker = {}
        self.forward_norm_tracker = {}
        self.tc_smoother = TemporalConsensusSmoother(num_parties=self.args.num_passive)
        self.smooth_sizes = [0.0] * self.args.num_passive
        self.smooth_waits = [0] * self.args.num_passive
        self.epoch = 0
        self.batch_idx = 0

        self.adaptive_clipping_state = {}
        self.current_forward_defense_plan = None
        self.q_weak_fixed = float(np.clip(getattr(self.args, 'q_weak_fixed', 0.5), 1e-6, 1.0))
        self.fixed_forward_eps = 0.0
        self.forward_total_eps_target = 0.0
        self.backward_total_eps_target = 0.0
        self.projected_composed_eps = 0.0
        self.sample_rate = sample_rate
        self.total_training_steps = self.args.epochs * len(self.train_dataset)
        self.target_avg_epsilon = self.args.epsilon
        # Cache the exact post-accounting value from the previous batch. The
        # first batch sees the same zero-history conversion as the former
        # force=True call; later batches reuse the preceding post-step value.
        self._cached_avg_privacy_spent = self.privacy_monitor.get_avg_spent(force=True)
        if self.adavfed_engine is not None and self.adavfed_engine.config.enabled:
            config = self.adavfed_engine.config
            if config.privacy_calibration == "rdp_subsampled_algorithm2":
                rdp_plan = dp.ensure_dp_noise_plan(
                    self.args,
                    steps_per_epoch=len(self.train_dataset),
                    q=self.args.batch_size / self.train_dataset_len,
                )
                self.adavfed_engine.configure_rdp_step_epsilons(
                    rdp_plan["forward_step_eps"],
                    rdp_plan["backward_step_eps"],
                )
            print(
                "[AdaVFed] mode={} | training={} | entire_step={} | total_eps={} | steps={} | C_fwd={} | C_bwd={} | delta={} | noise_delta={} | lambda1={} | lambda2={} | constraint={} | gates={} | dynamic_fwd={} | fwd_dp={} | bwd_dp={} | tau={} | temperature={} | knn_k={} | rho={} | step_eps=({:.6f},{:.6f})".format(
                    config.privacy_calibration,
                    config.training_mode,
                    "on" if config.use_entire_optimizer_update else "off",
                    config.total_epsilon,
                    config.total_training_steps,
                    config.clip_C,
                    config.backward_clip_C,
                    config.delta,
                    self.adavfed_engine._noise_delta(),
                    config.lambda_constraint,
                    config.lambda_gate,
                    "on" if config.enable_constraint else "off",
                    "on" if config.enable_gates else "off",
                    "on" if config.enable_dynamic_forward_noise else "off",
                    "on" if config.enable_forward_noise else "off",
                    "on" if config.enable_backward_noise else "off",
                    config.gate_tau,
                    config.temperature,
                    config.knn_k,
                    config.noise_injection_ratio,
                    config.forward_step_epsilon,
                    config.backward_step_epsilon,
                )
            )
        self.rl_agent = None 
        if self.args.rl:
            actual_steps_per_epoch = len(self.train_dataset)  
            q = self.args.batch_size / self.train_dataset_len 
            self.total_training_steps = self.args.epochs * actual_steps_per_epoch
            self.sample_rate = q
            privacy_utils.initialize_bidirectional_privacy_plan(self)
            state_dim = self.args.num_passive + 3
            # SAC controls only backward per-party epsilon ratios and keep rates.
            action_dim = 2 * self.args.num_passive
            self.rl_agent = rl_agent_module.RLPDPAgent(state_dim, action_dim, args)
            self.risk_mode = str(getattr(self.args, 'rl_risk_mode', 'geometry'))
            if self.risk_mode == 'public_quality':
                public_risk_file = str(getattr(self.args, 'rl_public_risk_file', '')).strip()
                if not public_risk_file:
                    raise ValueError(
                        '--rl_public_risk_file is required when '
                        '--rl_risk_mode public_quality is selected.'
                    )
                public_scores = load_public_cluster_scores(
                    public_risk_file,
                    dataset=self.args.dataset,
                    num_parties=self.args.num_passive,
                )
                self.released_gradient_risk_tracker = PublicPriorReleaseRiskTracker(
                    public_scores=public_scores,
                    ema_beta=getattr(self.args, 'rl_risk_ema_beta', 0.9),
                    quality_ema_beta=getattr(self.args, 'rl_quality_ema_beta', 0.9),
                    quality_power=getattr(self.args, 'rl_quality_power', 2.0),
                    quality_ratio_min=getattr(self.args, 'rl_quality_ratio_min', 0.8),
                    quality_ratio_max=getattr(self.args, 'rl_quality_ratio_max', 1.2),
                )
            else:
                self.released_gradient_risk_tracker = ReleasedGradientRiskTracker(
                    num_parties=self.args.num_passive,
                    num_classes=self.num_classes,
                    window_samples=getattr(self.args, 'rl_risk_window_samples', 256),
                    min_samples=getattr(self.args, 'rl_risk_min_samples', 64),
                    update_interval=getattr(self.args, 'rl_risk_update_interval', 10),
                    ema_beta=getattr(self.args, 'rl_risk_ema_beta', 0.9),
                    max_features=getattr(self.args, 'rl_risk_max_features', 256),
                    seed=getattr(self.args, 'seed', 0),
                )
            self.proxy_risk_threshold_pct = float(np.clip(
                getattr(self.args, 'rl_proxy_threshold', 40.0),
                0.0,
                100.0,
            ))
            self.rl_agent.R_target = self.proxy_risk_threshold_pct / 100.0
            self.risk_gate_trigger_counts = [0] * self.args.num_passive
            self.risk_gate_decision_counts = [0] * self.args.num_passive
            self.last_risk_gate_flags = [False] * self.args.num_passive
            self.last_risk_gate_values = [0.0] * self.args.num_passive
            print(
                "[RL Control] fixed proxy-risk threshold={:.4f}; "
                "risk gating is enabled from the start of training.".format(
                    self.proxy_risk_threshold_pct
                )
            )
            self.all_rewards_log = []
            self.static_identification_done = True
            
            self.state_window_size = self.args.sac_window_size
            self.utility_tracker = deque(maxlen=self.state_window_size)
            self.party_risk_trackers = [
                deque(maxlen=self.state_window_size)
                for _ in range(self.args.num_passive)
            ]
            self.last_dp_utility = None
            self.last_dp_party_qualities = [100.0] * self.args.num_passive
            
            self.last_state = None
            self.last_action = None
            self.last_action_plan = None
            self.prev_executed_action = None
            self.sac_update_interval = max(1, int(getattr(self.args, 'sac_update_interval', 4)))
            self.sac_updates_per_step = max(1, int(getattr(self.args, 'sac_updates_per_step', 1)))
            self.sac_action_ema = float(np.clip(getattr(self.args, 'sac_action_ema', 0.8), 0.0, 0.999))
            
    def restore_stdout(self):
        if getattr(self, '_stdout_logger', None) is not None:
            sys.stdout = self._original_stdout
            self._stdout_logger.close()
            self._stdout_logger = None

    def _setup_output_dirs_and_logging(self):
        if getattr(self.args, 'acvfl', False):
            defense_name = 'ACVFL'
        elif getattr(self.args, 'adavfed', False):
            defense_name = 'ADAVFED'
        elif self.args.defense_all:
            defense_name = 'DPALL'
        elif self.args.ppdl:
            defense_name = 'PPDL'
        else:
            defense_name = 'NONE'

        if self.args.rl:
            defense_name += '_RL'
        if getattr(self.args, 'hg', False):
            defense_name += '_HG'

        run_name = 'parties{}_epochs{}_div{}_{}_eps{}'.format(
            self.args.num_passive,
            self.args.epochs,
            self.args.division_mode,
            defense_name,
            self.args.epsilon,
        )
        observation3_mode = getattr(self.args, 'observation3_mode', 'none')
        if observation3_mode != 'none':
            run_name += '_obs3{}_keep{}'.format(
                observation3_mode,
                '{:.2f}'.format(float(getattr(self.args, 'observation3_keep_rate', 1.0))),
            )

        self.data_dir = os.path.join(
            self.output_root,
            'data',
            self.args.attack,
            self.args.dataset,
            'data_{}'.format(run_name),
        )
        self.log_dir = os.path.join(
            self.output_root,
            'log',
            self.args.attack,
            self.args.dataset,
            'log_{}'.format(run_name),
        )

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.args.attack in ['pmc', 'amc', 'ressfl_fia']:
            for file_name in os.listdir(self.data_dir):
                if file_name.endswith('.pt'):
                    os.remove(os.path.join(self.data_dir, file_name))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(self.log_dir, 'run_{}.txt'.format(timestamp))
        self._stdout_logger = Logger(log_file_path)
        sys.stdout = self._stdout_logger

        print("Time: {}".format(datetime.now()))
        print("Command: {}".format(getattr(self.args, 'command', 'python main.py')))
        print("Log Directory: {}".format(self.log_dir))
        print("")

    def _process_data(self, train_dataset, aux_public_dataset, test_dataset):
        sample_list = None
        if self.args.division_mode == 'random':
            if self.args.dataset not in ['mnist', 'cifar10']:
                raise ValueError("Random division only supports MNIST and CIFAR-10.")

            sample_list = []
            if self.args.num_passive == 1:
                sample_list.append(list(range(28)) if self.args.dataset == "mnist" else list(range(32)))
            elif self.args.num_passive == 2:
                if self.args.dataset == "mnist":
                    list_0 = [1, 4, 5, 7, 9, 11, 13, 14, 16, 18, 19, 23, 24, 26]
                    list_1 = [0, 2, 3, 6, 8, 10, 12, 15, 17, 20, 21, 22, 25, 27]
                else:
                    list_0 = [0, 3, 4, 5, 6, 7, 9, 14, 15, 16, 22, 23, 28, 29, 30, 31]
                    list_1 = [1, 2, 8, 10, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26, 27]
                sample_list.extend([list_0, list_1])
            elif self.args.num_passive == 4:
                if self.args.dataset == "mnist":
                    list_0 = [1, 2, 3, 17, 20, 23, 27]
                    list_1 = [6, 8, 11, 12, 13, 14, 24]
                    list_2 = [5, 7, 9, 10, 22, 25, 26]
                    list_3 = [0, 4, 15, 16, 18, 19, 21]
                else:
                    list_0 = [0, 4, 6, 14, 20, 24, 25, 28]
                    list_1 = [1, 3, 9, 15, 17, 19, 22, 27]
                    list_2 = [5, 7, 11, 16, 18, 21, 23, 31]
                    list_3 = [2, 8, 10, 12, 13, 26, 29, 30]
                sample_list.extend([list_0, list_1, list_2, list_3])
        elif self.args.division_mode == 'imbalanced':
            if self.args.dataset not in ['mnist', 'cifar10']:
                raise ValueError("Imbalance division only supports MNIST and CIFAR-10.")

            sample_list = []
            if self.args.num_passive == 1:
                sample_list.append(list(range(28)) if self.args.dataset == "mnist" else list(range(32)))
            elif self.args.num_passive == 2:
                if self.args.dataset == "mnist":
                    list_0 = [0, 2, 3, 4, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26, 27]
                    list_1 = [1, 5, 6, 8, 13, 17, 20, 24]
                else:
                    list_0 = [0, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 18, 21, 22, 24, 25, 26, 28, 30, 31]
                    list_1 = [1, 2, 5, 6, 11, 12, 17, 19, 20, 23, 27, 29]
                sample_list.extend([list_0, list_1])
            elif self.args.num_passive == 4:
                if self.args.dataset == "mnist":
                    list_0 = [1, 3, 4, 5, 7, 11, 14, 15, 19, 21, 23, 27]
                    list_1 = [2, 6, 9, 10, 12, 22]
                    list_2 = [0, 13, 17]
                    list_3 = [8, 16, 18, 20, 24, 25, 26]
                else:
                    list_0 = [0, 3, 6, 7, 12, 14, 15, 16, 23, 27, 29, 30, 31]
                    list_1 = [1, 2, 10, 13, 19, 22, 24]
                    list_2 = [8, 9, 11, 26]
                    list_3 = [4, 5, 17, 18, 20, 21, 25, 28]
                sample_list.extend([list_0, list_1, list_2, list_3])

        def process_loader(loader):
            processed_dataset = []
            processed_len = 0
            for batch_data in loader:
                data, labels = batch_data

                if self.args.padding_mode:
                    processed_data = []
                    for i in range(self.args.num_passive):
                        if i == self.args.attack_id:
                            processed_data.append(data)
                        else:
                            processed_data.append(torch.rand(data.shape))
                elif self.args.division_mode == 'vertical':
                    if self.args.dataset == "criteo":
                        processed_data = list(torch.chunk(data, self.args.num_passive, dim=1))
                    else:
                        processed_data = list(torch.chunk(data, self.args.num_passive, dim=3))
                elif self.args.division_mode in ['random', 'imbalanced']:
                    processed_data = []
                    for i in range(self.args.num_passive):
                        processed_data.append(data.index_select(3, torch.tensor(sample_list[i])))
                else:
                    raise ValueError("Unsupported division mode: {}".format(self.args.division_mode))

                processed_dataset.append([processed_data, labels])
                processed_len += len(data)

            return processed_dataset, processed_len

        self.train_dataset, self.train_dataset_len = process_loader(train_dataset)
        self.test_dataset, self.test_dataset_len = process_loader(test_dataset)
        if aux_public_dataset is not None:
            self.aux_val_dataset, self.aux_val_dataset_len = process_loader(aux_public_dataset)
        else:
            # Never silently repurpose the private final-test split as public
            # calibration data.  RL/FORA callers must provide a real auxiliary
            # release or explicitly operate without public calibration.
            self.aux_val_dataset = []
            self.aux_val_dataset_len = 0

        print("Finish processing dataset.")

    def _prime_adavfed_engine_parameters(self):
        if self.adavfed_engine is None or not self.adavfed_engine.config.enabled or len(self.train_dataset) == 0:
            return

        data, _ = self.train_dataset[0]
        data = [d.to(self.device) for d in data]
        with torch.no_grad():
            self.adavfed_engine.ensure_input_gates(data)

    def _register_adavfed_optimizer_params(self):
        if self.adavfed_engine is None or not self.adavfed_engine.config.enabled:
            return
        if not self.adavfed_engine.config.enable_gates:
            return

        if len(self.adavfed_engine.gate_means) != self.args.num_passive:
            raise ValueError("AdaVFed gate parameters were not initialized for all passive parties.")

        for party_id, gate_param in enumerate(self.adavfed_engine.gate_means):
            base_group = self.optimizer_passive[party_id].param_groups[0]
            hyperparams = {
                key: value
                for key, value in base_group.items()
                if key != 'params'
            }
            self.optimizer_passive[party_id].add_param_group({
                'params': [gate_param],
                **hyperparams,
            })
    
    def _clear_current_forward_defense(self):
        self.current_forward_defense_plan = None

    def _maybe_log_adavfed_diagnostics(self, batch_idx, ce_loss_value, total_loss_value, batch_examples):
        if self.adavfed_engine is None or not self.adavfed_engine.config.enabled:
            return
        interval = int(self.adavfed_engine.config.debug_interval)
        if interval <= 0:
            return
        is_first_paper_step = self.epoch == 0 and batch_idx == 0
        if not is_first_paper_step and (batch_idx + 1) % interval != 0:
            return

        snapshot = self.adavfed_engine.get_debug_snapshot()
        regularization = snapshot.get('regularization', {})
        gate_stats = snapshot.get('gate', [])
        forward_stats = snapshot.get('forward', [])
        backward_stats = snapshot.get('backward', [])
        def _stat_mean(stats_list, key, default=0.0):
            values = [float(stats.get(key, default)) for stats in stats_list]
            if len(values) == 0:
                return float(default)
            return float(sum(values) / len(values))

        def _stat_min(stats_list, key, default=0.0):
            values = [float(stats.get(key, default)) for stats in stats_list]
            if len(values) == 0:
                return float(default)
            return float(min(values))

        def _stat_max(stats_list, key, default=0.0):
            values = [float(stats.get(key, default)) for stats in stats_list]
            if len(values) == 0:
                return float(default)
            return float(max(values))

        prefix = f"[AdaVFed-Diag][E{self.epoch + 1} S{batch_idx + 1}]"
        print(
            "{} loss={:.3f}(ce={:.3f},c={:.3f},g={:.3f}) | gate={:.3f}/z{:.2f} "
            "| fwd=clip{:.0%},eps[{:.2f},{:.2f}],sig[{:.1f},{:.1f},{:.1f}] "
            "| bwd=clip{:.0%},sig{:.4g}".format(
                prefix,
                float(total_loss_value),
                float(ce_loss_value),
                float(regularization.get('constraint', 0.0)),
                float(regularization.get('gate', 0.0)),
                _stat_mean(gate_stats, 'mean', default=1.0),
                _stat_mean(gate_stats, 'zero_ratio', default=0.0),
                _stat_mean(forward_stats, 'clip_hit_rate', default=0.0),
                _stat_min(forward_stats, 'epsilon_min', default=0.0),
                _stat_max(forward_stats, 'epsilon_max', default=0.0),
                _stat_min(forward_stats, 'sigma_min', default=0.0),
                _stat_mean(forward_stats, 'sigma_mean', default=0.0),
                _stat_max(forward_stats, 'sigma_max', default=0.0),
                _stat_mean(backward_stats, 'clip_hit_rate', default=0.0),
                _stat_mean(backward_stats, 'sigma', default=0.0),
            )
        )

    def _maybe_update_fora_online(self, embeddings):
        if (
            self.args.attack == 'ressfl_fia'
            and getattr(self.args, 'fia_mode', 'decoder') == 'fora'
            and hasattr(self, 'fora_online_alignment_step')
        ):
            self.fora_online_alignment_step(embeddings[self.args.attack_id])
    
    def _train_adavfed(self):
        """Algorithm 3/4 training path with an explicit VFL communication boundary."""
        self.iteration = len(self.train_dataset)
        self.adavfed_engine.train()
        if self.adavfed_engine.config.privacy_calibration == "theorem_one_joint":
            print(
                "[AdaVFed] Theorem 1 joint calibration enabled. Forward and backward "
                "noise are jointly calibrated over all training steps."
            )
        elif self.adavfed_engine.config.privacy_calibration == "rdp_subsampled_algorithm2":
            if self.adavfed_engine.config.training_mode == "defense_all_aligned":
                print(
                    "[AdaVFed] defense_all-aligned control enabled. It reuses the shared "
                    "RDP planner, retained forward graph, and double optimizer update."
                )
            else:
                print(
                    "[AdaVFed] Subsampled-RDP Algorithm 2 calibration enabled. The step "
                    "budgets come from the shared defense_all RDP planner; the VFL boundary remains detached."
                )
        else:
            print(
                "[AdaVFed] Paper Algorithm 2 calibration enabled. This empirical path "
                "does not provide a strict jointly composed total-epsilon guarantee."
            )

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.model.train()
            self.model.active.train()
            for passive_model in self.model.passive:
                passive_model.train()
            self.adavfed_engine.train()

            train_loss = 0.0
            correct = 0
            total = 0
            epoch_attack_acc = None
            for batch_idx, batch_data in enumerate(self.train_dataset):
                self.batch_idx = batch_idx
                data, labels = batch_data
                data = [item.to(self.device) for item in data]
                labels = labels.to(self.device)

                self.optimizer_entire.zero_grad()
                self.optimizer_active.zero_grad()
                for optimizer in self.optimizer_passive:
                    optimizer.zero_grad()

                self.adavfed_engine.reset_batch_state()
                gated_inputs = self.adavfed_engine.apply_input_gates(data)
                raw_embeddings = [
                    self.model.passive[party_id](gated_inputs[party_id])
                    for party_id in range(self.args.num_passive)
                ]
                self.adavfed_engine.record_gated_embeddings(raw_embeddings)
                released_embeddings = self.adavfed_engine.privatize_forward_embeddings(raw_embeddings)

                active_inputs = [
                    self.adavfed_engine.prepare_active_embedding(release)
                    for release in released_embeddings
                ]
                logits, pred = self.model.forward_from_embeddings(active_inputs)
                ce_loss = self.loss(logits, labels)
                returned_gradients = torch.autograd.grad(
                    ce_loss,
                    active_inputs,
                    retain_graph=True,
                    create_graph=False,
                )
                ce_loss.backward(
                    retain_graph=(
                        self.adavfed_engine.config.training_mode == "defense_all_aligned"
                    )
                )

                noisy_gradients = self.adavfed_engine.privatize_backward_grads(returned_gradients)
                backward_targets = (
                    active_inputs
                    if self.adavfed_engine.config.training_mode == "defense_all_aligned"
                    else raw_embeddings
                )
                torch.autograd.backward(backward_targets, grad_tensors=noisy_gradients, retain_graph=True)
                regularization = self.adavfed_engine.build_regularization()
                regularization.backward()

                if self.adavfed_engine.config.use_entire_optimizer_update:
                    self.optimizer_entire.step()
                self.optimizer_active.step()
                for optimizer in self.optimizer_passive:
                    optimizer.step()

                total_loss = ce_loss.detach() + regularization.detach()
                self._maybe_log_adavfed_diagnostics(
                    batch_idx,
                    ce_loss.item(),
                    total_loss.item(),
                    labels.size(0),
                )

                # Keep the same epoch-level sign-attack metric as defense_all:
                # collect every batch, suppress its per-batch output, and print the
                # accumulated average once after the epoch evaluation.
                if self.args.attack == 'sign':
                    self.attack(noisy_gradients, labels, batch_idx)
                    if (batch_idx + 1) == self.iteration and self.metrics.attack_acc:
                        epoch_attack_acc = float(self.metrics.attack_acc[-1])
                elif (batch_idx + 1) == self.iteration:
                    if self.args.attack == 'cluster':
                        attack_data = released_embeddings if self.args.use_emb else noisy_gradients
                        self.attack(attack_data, labels, batch_idx)

                if self.args.attack in ['pmc', 'amc']:
                    cache_embeddings = (
                        raw_embeddings if self.args.attack == 'amc' else released_embeddings
                    )
                    cache_payload = build_completion_cache_payload(
                        cache_embeddings,
                        noisy_gradients,
                        labels,
                    )
                    if self.args.attack == 'amc' and not self._amc_raw_cache_logged:
                        print(
                            "[AMC Cache] using client-local pre-forward-noise embeddings; "
                            "returned gradients remain defended."
                        )
                        self._amc_raw_cache_logged = True
                    torch.save(cache_payload, os.path.join(self.data_dir, "data_{}.pt".format(batch_idx)))
                elif self.args.attack == 'ressfl_fia':
                    self._maybe_update_fora_online(released_embeddings)
                    raw_data = [party_data.detach().cpu() for party_data in data]
                    passive_emb = [embedding.detach().cpu() for embedding in released_embeddings]
                    passive_grad = [gradient.detach().cpu() for gradient in noisy_gradients]
                    cache_payload = [raw_data, passive_emb, passive_grad, labels.detach().cpu()]
                    torch.save(cache_payload, os.path.join(self.data_dir, "data_{}.pt".format(batch_idx)))

                train_loss += float(total_loss.item())
                total += labels.size(0)
                correct += pred.argmax(dim=1).eq(labels).sum().item()
                self.adavfed_engine.reset_batch_state()

            print("\nEpoch:{}/{}".format(epoch + 1, self.args.epochs))
            self._evaluate()
            if self.args.attack == 'amc' and epoch == self.args.epochs - 1:
                start_time = time.time_ns()
                self.attack()
                attack_nseconds = time.time_ns() - start_time
                print("Attack Runtime: {}:{}:{} (ns)".format(
                    int(attack_nseconds / (1000 * 1000)),
                    int(attack_nseconds / 1000) % 1000,
                    int(attack_nseconds) % 1000,
                ))
                self.metrics.attack_runtime.append(attack_nseconds)
            elif self.args.attack == 'pmc':
                mode = completion_attack_mode(
                    epoch=epoch,
                    set_attack_epoch=bool(self.args.set_attack_epoch),
                    attack_epoch=int(self.args.attack_epoch),
                )
                if mode is not None:
                    start_time = time.time_ns()
                    self.attack(init=(mode == 'init'))
                    attack_nseconds = time.time_ns() - start_time
                    second = int(attack_nseconds / (1000 * 1000))
                    msecond = int(attack_nseconds / 1000) % 1000
                    nsecond = int(attack_nseconds) % 1000
                    print("Attack Runtime: {}:{}:{} (ns)".format(second, msecond, nsecond))
                    self.metrics.attack_runtime.append(attack_nseconds)
            elif self.args.attack == 'ressfl_fia' and epoch == self.args.epochs - 1:
                start_time = time.time_ns()
                self.attack(init=True)
                attack_nseconds = time.time_ns() - start_time
                print("Attack Runtime: {}:{}:{} (ns)".format(
                    int(attack_nseconds / (1000 * 1000)),
                    int(attack_nseconds / 1000) % 1000,
                    int(attack_nseconds) % 1000,
                ))
                self.metrics.attack_runtime.append(attack_nseconds)
            if epoch_attack_acc is not None:
                print("Attack Accuracy: {:.2f}%".format(epoch_attack_acc))
            for scheduler in self.main_lr_schedulers[1:]:
                scheduler.step()

        if self.adavfed_engine.config.privacy_calibration == "theorem_one_joint":
            print(
                "[AdaVFed] Training completed under Theorem 1 joint calibration over "
                "all configured training steps."
            )
        elif self.adavfed_engine.config.privacy_calibration == "rdp_subsampled_algorithm2":
            if self.adavfed_engine.config.training_mode == "defense_all_aligned":
                print(
                    "[AdaVFed] Training completed under the defense_all-aligned control path; "
                    "this is not an isolated AdaVFed communication boundary."
                )
            else:
                print(
                    "[AdaVFed] Training completed under the shared subsampled-RDP accounting "
                    "ablation with an isolated AdaVFed communication boundary."
                )
        else:
            print(
                "[AdaVFed] Training completed under an empirical Algorithm 2 path; it is "
                "not a strictly calibrated total-epsilon implementation of Theorem 1."
            )

    def train(self):
        self.iteration = len(self.train_dataset)
        if self.adavfed_engine is not None and self.adavfed_engine.config.enabled:
            return self._train_adavfed()
        stop_training = False
        q = self.args.batch_size / self.train_dataset_len
        self.eval_reward_history = []
        dp_debug_logged = False

        use_acvfl_bidirectional = bool(getattr(self.args, 'acvfl', False) and getattr(self.args, 'acvfl_bidirectional', 1))
        budget_sweep_mode = getattr(self.args, 'budget_sweep_mode', 'joint')
        observation3_mode = getattr(self.args, 'observation3_mode', 'none')
        observation3_keep_rate = float(getattr(self.args, 'observation3_keep_rate', 1.0))
        observation3_logged = False
        if (self.args.defense_all or self.args.ppdl or use_acvfl_bidirectional) and not self.args.rl:
            self.privacy_monitor.verbose = False
            dp.ensure_dp_noise_plan(self.args, steps_per_epoch=self.iteration, q=q)

        if getattr(self.args, 'rl', False) and len(self.train_dataset) > 0:
            backward_mode = int(getattr(self.args, 'backward_topk_mask', 1))
            backward_release_mode = {
                0: "dense_all",
                1: "public_topk",
                2: "random_keep",
            }[backward_mode]
            max_public_batches = getattr(self.args, 'structure_public_batches', 20)
            if max_public_batches > 0 and len(self.task_prior.static_importance) == 0:
                self.task_prior.calibrate_from_aux_data(
                    self.aux_val_dataset,
                    self.model,
                    self.device,
                    max_batches=max_public_batches,
                )
            if rl_runtime_logger is not None:
                rl_runtime_logger.print_config(self)

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            if self.args.rl and rl_runtime_logger is not None:
                rl_runtime_logger.start_epoch(self)

            self.model.train()
            self.model.active.train()
            for i in range(self.args.num_passive):
                self.model.passive[i].train()
            train_loss = 0.0
            correct = 0
            total = 0
            dispersion_list = []


            total_updates_count = [0] * self.args.num_passive
            total_accumulated_gradients = [0] * self.args.num_passive

            for batch_idx, batch_data in enumerate(self.train_dataset):
                self.batch_idx = batch_idx
                if self.args.rl:
                    current_avg_spent = float(self._cached_avg_privacy_spent)
                else:
                    current_avg_spent = self.privacy_monitor.get_avg_spent(force=True)
                if current_avg_spent > self.args.epsilon + 1e-5: 
                    print(f"\n[Privacy Alert] Privacy budget ({self.args.epsilon}) exhausted at Epoch {epoch+1}, Step {batch_idx+1}!")
                    self.privacy_monitor.report() 
                    return
                data, labels = batch_data

                current_step_total_eps_list = None
                current_step_forward_eps_list = None
                current_step_backward_eps_list = None
                current_step_q_list = None
                grad_emb_noised_for_update = None
                current_backward_cap_eps = 0.0

                if self.args.rl:
                    current_backward_cap_eps = privacy_utils.get_current_backward_cap_eps(self, epoch, batch_idx)
                    control_risk = rl_helpers.get_control_party_risks(self)
                    self.last_control_party_risks = [float(value) for value in control_risk]

                    if self.last_dp_utility is not None:
                        self.utility_tracker.append(self.last_dp_utility)
                    for party_id in range(self.args.num_passive):
                        self.party_risk_trackers[party_id].append(control_risk[party_id])

                    done = (batch_idx == self.iteration - 1)

                    if len(self.utility_tracker) < self.state_window_size:
                        current_step_total_eps_list = [current_backward_cap_eps] * self.args.num_passive
                        current_step_q_list = [self.q_weak_fixed] * self.args.num_passive
                        if len(self.rl_agent.replay_buffer) > self.rl_agent.batch_size:
                            if batch_idx % self.sac_update_interval == 0:
                                for _ in range(self.sac_updates_per_step):
                                    self.rl_agent.learn()
                    else:
                        smooth_party_risks = [
                            float(np.mean(tracker)) if len(tracker) > 0 else 0.0
                            for tracker in self.party_risk_trackers
                        ]
                        self.last_smooth_party_risks = list(smooth_party_risks)
                        current_utility = (
                            float(np.mean(self.utility_tracker))
                            if len(self.utility_tracker) > 0 else 0.0
                        )
                        state_t = rl_helpers.construct_rl_state(
                            self,
                            smooth_party_risks,
                            current_utility,
                        )

                        if self.last_state is not None and self.last_action is not None:
                            executed_plan = self.last_action_plan
                            if executed_plan is None:
                                executed_plan = self.rl_agent.scale_action(
                                    self.last_action, base_step_eps=current_backward_cap_eps
                                )
                            reward_t = self.rl_agent.calculate_reward(self.last_state, state_t, executed_plan)
                            punishment_t = self.rl_agent.calculate_punishment(state_t)
                            final_reward = reward_t - punishment_t
                            self.all_rewards_log.append(final_reward)
                            self.rl_agent.replay_buffer.push(
                                self.last_state, self.last_action, state_t,
                                final_reward, done
                            )

                        action_t_tanh, _, _ = self.rl_agent.choose_action(state_t)
                        action_t_tanh_numpy = action_t_tanh.cpu().numpy().flatten()
                        if self.prev_executed_action is None or self.sac_action_ema <= 0.0:
                            executed_action = action_t_tanh_numpy
                        else:
                            executed_action = (
                                self.sac_action_ema * self.prev_executed_action
                                + (1.0 - self.sac_action_ema) * action_t_tanh_numpy
                            )

                        raw_action_plan = self.rl_agent.scale_action(
                            executed_action,
                            base_step_eps=current_backward_cap_eps
                        )
                        action_plan = rl_agent_module.apply_backward_risk_gate(
                            self,
                            raw_action_plan,
                            smooth_party_risks,
                            current_backward_cap_eps,
                        )
                        current_step_total_eps_list = [float(eps) for eps in action_plan.get('eps_list', [])]
                        current_step_q_list = [float(qv) for qv in action_plan.get('q_list', [])]

                        if len(self.rl_agent.replay_buffer) > self.rl_agent.batch_size:
                            if batch_idx % self.sac_update_interval == 0:
                                for _ in range(self.sac_updates_per_step):
                                    self.rl_agent.learn()
                        self.last_state = state_t
                        self.last_action = executed_action
                        self.last_action_plan = action_plan
                        self.prev_executed_action = executed_action
                        if done:
                            self.last_state = None
                            self.last_action = None
                            self.last_action_plan = None
                            self.prev_executed_action = None

                    current_step_forward_eps_list = privacy_utils.get_fixed_forward_eps_list(self)
                    current_step_backward_eps_list = [float(eps) for eps in current_step_total_eps_list]
                    rl_agent_module.set_current_forward_defense(self, current_step_forward_eps_list)
                    if rl_runtime_logger is not None:
                        rl_runtime_logger.record_step(
                            self,
                            current_step_backward_eps_list,
                            current_step_q_list,
                        )
                data = [d.to(self.device) for d in data]
                labels = labels.to(self.device)
                emb, logit, pred = self.model(data)
                passive_emb_for_backward = emb
                forward_used_eps = None
                forward_stats_list = None
                use_acvfl_bidirectional = bool(getattr(self.args, 'acvfl', False) and getattr(self.args, 'acvfl_bidirectional', 1))
                use_standard_defense_all = bool(
                    self.args.defense_all
                    and not self.args.rl
                    and not self.args.ppdl
                    and not getattr(self.args, 'acvfl', False)
                )
                apply_forward_dp = False
                if not self.args.rl:
                    if self.args.ppdl or use_acvfl_bidirectional:
                        apply_forward_dp = True
                    elif self.args.defense_all and budget_sweep_mode != 'backward_only':
                        apply_forward_dp = True
                if apply_forward_dp:
                    emb, forward_used_eps, forward_stats_list = dp.dp_noise(
                        emb,
                        self.args,
                        device=self.device,
                        steps_per_epoch=self.iteration,
                        q=q,
                        direction='forward',
                        return_stats=True,
                    )
                    if observation3_mode == 'forward':
                        emb, observation3_stats = dp.apply_random_sparsification(
                            emb,
                            observation3_keep_rate,
                            return_stats=True,
                        )
                        if not observation3_logged and observation3_stats:
                            keep_values = " ".join(
                                "P{}:{:.3f}".format(i, stats['actual_keep_rate'])
                                for i, stats in enumerate(observation3_stats)
                            )
                            print(
                                "[obs3] forward random sparsification | target keep={:.3f} | actual keep {}".format(
                                    observation3_keep_rate,
                                    keep_values,
                                )
                            )
                            observation3_logged = True
                if use_standard_defense_all:
                    emb = [tensor.detach().requires_grad_(True) for tensor in emb]
                if apply_forward_dp or use_standard_defense_all:
                    logit, pred = self.model.forward_from_embeddings(emb)
                batch_main_acc = metrics_utils.calculate_main_task_acc(pred, labels)
                ce_loss = self.loss(pred, labels)
                loss = ce_loss
              
                # zero grad for all optimizers
                self.optimizer_entire.zero_grad()
                self.optimizer_active.zero_grad()
                for i in range(self.args.num_passive):
                    self.optimizer_passive[i].zero_grad()
                loss.backward(retain_graph=True)
                grad = torch.autograd.grad(loss, emb, retain_graph=True, create_graph=True)
                if not (self.args.defense_all or getattr(self.args, 'acvfl', False) or self.args.rl):
                    if batch_idx % 50 == 0:  # Print every 50 batches to avoid flooding the console
                        with torch.no_grad():
                            norms = [f"P{i}: {torch.norm(g, p=2).item():.4f}" for i, g in enumerate(grad)]
                            print(f"[Step {batch_idx}] Clean Grad Norms: {', '.join(norms)}")
                if getattr(self.args, 'acvfl', False):
                    use_acvfl_bidirectional = bool(getattr(self.args, 'acvfl_bidirectional', 1))
                    if use_acvfl_bidirectional:
                        plan = dp.ensure_dp_noise_plan(self.args, steps_per_epoch=self.iteration, q=q)
                        step_epsilon = plan['backward_step_eps']
                    else:
                        step_epsilon = dp.compute_step_epsilon(self.args, self.iteration ,q)
                    
                    acvfl_initial_c = float(getattr(self.args, 'acvfl_initial_c', -1.0))
                    if acvfl_initial_c <= 0.0:
                        acvfl_initial_c = float(getattr(self.args, 'clip_threshold', 0.1))
                    acvfl_min_c = float(max(getattr(self.args, 'acvfl_min_c', 1e-8), 1e-12))
                    acvfl_max_c = float(getattr(self.args, 'acvfl_max_c', -1.0))
                    if acvfl_max_c <= 0.0:
                        acvfl_max_c = float(getattr(self.args, 'clip_threshold', acvfl_initial_c))
                    acvfl_max_c = max(acvfl_max_c, acvfl_min_c)
                    noisy_grad_list = []
                    for i in range(self.args.num_passive):

                        noisy_g = dp.dp_defense_adaptive(
                            grad[i], 
                            passive_id=i, 
                            epsilon=step_epsilon, 
                            device=self.device,
                            adaptive_state=self.adaptive_clipping_state,
                            target_quantile=getattr(self.args, 'acvfl_gamma', 0.1),
                            lr_c=getattr(self.args, 'acvfl_lr_c', 0.2),
                            initial_c=acvfl_initial_c,
                            min_c=acvfl_min_c,
                            max_c=acvfl_max_c,
                        )

                        if emb[i].requires_grad:
                            emb[i].backward(noisy_g, retain_graph=True)
                            
                        noisy_grad_list.append(noisy_g)

                    if use_acvfl_bidirectional and forward_used_eps is not None:
                        full_release = [1.0] * self.args.num_passive
                        self.forward_privacy_monitor.step(
                            [forward_used_eps] * self.args.num_passive,
                            q_sparse_list=full_release,
                        )
                        self.backward_privacy_monitor.step(
                            [step_epsilon] * self.args.num_passive,
                            q_sparse_list=full_release,
                        )
                        self.privacy_monitor.compose(
                            [
                                ([forward_used_eps] * self.args.num_passive, full_release),
                                ([step_epsilon] * self.args.num_passive, full_release),
                            ]
                        )
                        if not dp_debug_logged:
                            forward_logs = []
                            for i, stats in enumerate(forward_stats_list or []):
                                forward_logs.append(
                                    f"P{i}:C={stats['clip_threshold']:.3f}/sig={stats['sigma']:.3f}"
                                    f"/hit={stats['clip_hit_rate']*100:.0f}%"
                                    f"/coef={stats['mean_clip_coef']:.3f}"
                                    f"/mean={stats['mean_norm']:.3f}"
                                    f"/p95={stats['norm_p95']:.3f}"
                                )
                            if forward_logs:
                                print(f"[ACVFL] forward diag: {' '.join(forward_logs)}")
                            print(
                                f"[ACVFL] backward: step_eps={step_epsilon:.6f}, "
                                f"mode=bidirectional, clip_init={acvfl_initial_c:.6f}, "
                                f"clip_max={acvfl_max_c:.6f}, gamma={getattr(self.args, 'acvfl_gamma', 0.1):.3f}"
                            )
                            dp_debug_logged = True
                    else:
                        self.privacy_monitor.step([step_epsilon] * self.args.num_passive)
                    grad = tuple(noisy_grad_list)

                elif self.args.rl:
                    released_gradients = []
                    released_noise_stds = []
                    # 1. 闈欐€佽瘑鍒己寮卞娍鏂?(Epoch 0)
                    if not self.static_identification_done:
                        grad_emb_noised_for_update = [None] * self.args.num_passive
                        privacy_utils.step_rl_privacy_accounting(
                            self,
                            current_step_forward_eps_list,
                            current_step_backward_eps_list,
                            current_step_q_list,
                        )
                        for i in range(self.args.num_passive):
                            effective_q = rl_agent_module.get_backward_release_keep_rate(
                                self, current_step_q_list[i]
                            )
                            backward_mask = rl_agent_module.get_backward_release_mask(
                                self,
                                party_id=i,
                                grad_tensor=grad[i].detach(),
                                keep_rate=effective_q,
                            )
                            noisy_g, release_stats = dp.dp_perturb_sampling_adaptive(
                                grad[i], passive_id=i, epsilon=current_step_backward_eps_list[i],
                                sampling_rate=effective_q, device=self.device,
                                fixed_mask=backward_mask,
                                history_grads=self.local_history_grads,
                                norm_tracker=self.norm_tracker,
                                initial_clip=getattr(self.args, 'clip_threshold', 1.0),
                                return_stats=True,
                            )
                            grad_emb_noised_for_update[i] = noisy_g
                            released_gradients.append(noisy_g)
                            released_noise_stds.append(float((release_stats or {}).get('noise_std', 0.0)))
                            if self.released_gradient_risk_tracker is not None:
                                self.released_gradient_risk_tracker.update(i, noisy_g)
                        # 鍙嶅悜浼犳挱
                        for i in range(self.args.num_passive):
                            if emb[i].requires_grad:
                                emb[i].backward(grad_emb_noised_for_update[i], retain_graph=True)
                        
                        self.static_identification_done = True

                    else:
                        grad_emb_noised_for_update = [None] * self.args.num_passive
                        privacy_utils.step_rl_privacy_accounting(
                            self,
                            current_step_forward_eps_list,
                            current_step_backward_eps_list,
                            current_step_q_list,
                        )

                        for i in range(self.args.num_passive):
                            cur_eps = current_step_backward_eps_list[i]
                            cur_q = rl_agent_module.get_backward_release_keep_rate(
                                self, current_step_q_list[i]
                            )

                            grad_for_update = grad[i].detach().clone()
                            backward_mask = rl_agent_module.get_backward_release_mask(
                                self,
                                party_id=i,
                                grad_tensor=grad_for_update,
                                keep_rate=cur_q,
                            )

                            noisy_sparse_grad, release_stats = dp.dp_perturb_sampling_adaptive(
                                grad_for_update,
                                passive_id=i,
                                epsilon=cur_eps,
                                sampling_rate=cur_q,
                                device=self.device,
                                fixed_mask=backward_mask,
                                history_grads=self.local_history_grads,
                                norm_tracker=self.norm_tracker,
                                initial_clip=getattr(self.args, 'clip_threshold', 1.0),
                                return_stats=True,
                            )
                            released_gradients.append(noisy_sparse_grad)
                            released_noise_stds.append(float((release_stats or {}).get('noise_std', 0.0)))
                            if self.args.rl and getattr(self.args, 'tc_smoother', False):
                                should_apply, final_smoothed_grad, current_size = self.tc_smoother.step(
                                    party_id=i,
                                    new_noisy_grad=noisy_sparse_grad,
                                    current_eps=cur_eps
                                )
                            else:
                                should_apply = True
                                final_smoothed_grad = noisy_sparse_grad
                                current_size = 1

                            self.smooth_waits[i] = current_size

                            if should_apply:
                                if emb[i].requires_grad and final_smoothed_grad.shape == emb[i].shape:
                                    emb[i].backward(final_smoothed_grad, retain_graph=True)
                                    grad_emb_noised_for_update[i] = final_smoothed_grad
                                    if self.released_gradient_risk_tracker is not None:
                                        self.released_gradient_risk_tracker.update(i, final_smoothed_grad)
                                    self.smooth_sizes[i] = 0.9 * self.smooth_sizes[i] + 0.1 * current_size
                                    total_updates_count[i] += 1
                                    total_accumulated_gradients[i] += current_size
                        for i in range(self.args.num_passive):
                            if grad_emb_noised_for_update[i] is None:
                                grad_emb_noised_for_update[i] = torch.zeros_like(emb[i].detach())
                        grad = grad_emb_noised_for_update
                    self.last_dp_party_qualities = rl_helpers.released_gradient_party_qualities(
                        released_gradients,
                        released_noise_stds,
                    )
                    self.last_dp_utility = float(np.mean(self.last_dp_party_qualities)) \
                        if self.last_dp_party_qualities else 0.0
                    if hasattr(self.released_gradient_risk_tracker, 'update_qualities'):
                        # The quality of releases from step t updates the risk
                        # state consumed at step t+1; current raw gradients and
                        # labels never enter this update.
                        self.released_gradient_risk_tracker.update_qualities(
                            self.last_dp_party_qualities
                        )
                else:
                    if self.args.ppdl:
                        plan = dp.ensure_dp_noise_plan(self.args, steps_per_epoch=self.iteration, q=q)
                        step_epsilon = plan['backward_step_eps']
                        grad, used_eps = ppdl.ppdl_defense(grad, self.args, step_eps=step_epsilon, device=self.device) 
                        if hasattr(self, 'privacy_monitor'):
                            full_release = [1.0] * self.args.num_passive
                            if forward_used_eps is not None:
                                self.forward_privacy_monitor.step(
                                    [forward_used_eps] * self.args.num_passive,
                                    q_sparse_list=full_release,
                                )
                                self.backward_privacy_monitor.step(
                                    [used_eps] * self.args.num_passive,
                                    q_sparse_list=full_release,
                                )
                                self.privacy_monitor.compose(
                                    [
                                        ([forward_used_eps] * self.args.num_passive, full_release),
                                        ([used_eps] * self.args.num_passive, full_release),
                                    ]
                                )
                            else:
                                self.privacy_monitor.step([used_eps] * self.args.num_passive, q_sparse_list=full_release)
                        if not dp_debug_logged:
                            forward_logs = []
                            for i, stats in enumerate(forward_stats_list or []):
                                forward_logs.append(
                                    f"P{i}:C={stats['clip_threshold']:.3f}/sig={stats['sigma']:.3f}"
                                    f"/hit={stats['clip_hit_rate']*100:.0f}%"
                                    f"/coef={stats['mean_clip_coef']:.3f}"
                                    f"/mean={stats['mean_norm']:.3f}"
                                    f"/p95={stats['norm_p95']:.3f}"
                                )
                            if forward_logs:
                                print(f"[PPDL] forward diag: {' '.join(forward_logs)}")
                            print(
                                f"[PPDL] backward: step_eps={used_eps:.6f}, "
                                f"clip={getattr(self.args, 'clip_threshold', 1.0):.6f}, "
                                f"theta_u={getattr(self.args, 'ppdl_theta_u', 1.0):.3f}, "
                                f"tau={getattr(self.args, 'ppdl_tau', 0.001):.6f}"
                            )
                            if self.released_gradient_risk_tracker is not None:
                                self.released_gradient_risk_tracker.update(i, noisy_sparse_grad)
                            dp_debug_logged = True
                    elif self.args.defense_all:
                        if budget_sweep_mode == 'forward_only':
                            used_eps = 0.0
                            backward_stats_list = None
                        else:
                            grad, used_eps, backward_stats_list = dp.dp_noise(
                                grad,
                                self.args,
                                device=self.device,
                                steps_per_epoch=self.iteration,
                                q=q,
                                direction='backward',
                                return_stats=True,
                            )
                            if observation3_mode == 'backward':
                                grad, observation3_stats = dp.apply_random_sparsification(
                                    grad,
                                    observation3_keep_rate,
                                    return_stats=True,
                                )
                                if not observation3_logged and observation3_stats:
                                    keep_values = " ".join(
                                        "P{}:{:.3f}".format(i, stats['actual_keep_rate'])
                                        for i, stats in enumerate(observation3_stats)
                                    )
                                    print(
                                        "[obs3] backward random sparsification | target keep={:.3f} | actual keep {}".format(
                                            observation3_keep_rate,
                                            keep_values,
                                        )
                                    )
                                    observation3_logged = True
                        full_release = [1.0] * self.args.num_passive
                        release_specs = []
                        if forward_used_eps is not None and forward_used_eps > 0.0:
                            self.forward_privacy_monitor.step(
                                [forward_used_eps] * self.args.num_passive,
                                q_sparse_list=full_release,
                            )
                            release_specs.append(([forward_used_eps] * self.args.num_passive, full_release))
                        if used_eps > 0.0:
                            self.backward_privacy_monitor.step(
                                [used_eps] * self.args.num_passive,
                                q_sparse_list=full_release,
                            )
                            release_specs.append(([used_eps] * self.args.num_passive, full_release))
                        if len(release_specs) == 1:
                            eps_list, q_sparse_list = release_specs[0]
                            self.privacy_monitor.step(eps_list, q_sparse_list=q_sparse_list)
                        elif len(release_specs) > 1:
                            self.privacy_monitor.compose(release_specs)
                        if not dp_debug_logged:
                            forward_logs = []
                            for i, stats in enumerate(forward_stats_list or []):
                                forward_logs.append(
                                    f"P{i}:C={stats['clip_threshold']:.3f}/sig={stats['sigma']:.3f}"
                                    f"/hit={stats['clip_hit_rate']*100:.0f}%"
                                    f"/coef={stats['mean_clip_coef']:.3f}"
                                    f"/mean={stats['mean_norm']:.3f}"
                                    f"/p95={stats['norm_p95']:.3f}"
                                )
                            backward_logs = []
                            for i, stats in enumerate(backward_stats_list or []):
                                backward_logs.append(
                                    f"P{i}:C={stats['clip_threshold']:.6f}/sig={stats['sigma']:.6f}"
                                    f"/hit={stats['clip_hit_rate']*100:.0f}%"
                                    f"/coef={stats['mean_clip_coef']:.6f}"
                                    f"/mean={stats['mean_norm']:.6f}"
                                    f"/p95={stats['norm_p95']:.6f}"
                                )
                            if forward_logs:
                                print(f"[dp] forward diag: {' '.join(forward_logs)}")
                            if backward_logs:
                                print(f"[dp] backward diag: {' '.join(backward_logs)}")
                            dp_debug_logged = True
                    if not self.args.acvfl:
                        backward_targets = passive_emb_for_backward if use_standard_defense_all else emb
                        for i in range(self.args.num_passive):
                            if backward_targets[i].requires_grad:
                                backward_targets[i].backward(grad[i], retain_graph=True)

                # attack
                if self.args.attack in ['pmc', 'amc', 'ressfl_fia']:
                    # record embeddings, gradients, and labels
                    # process emb
                    passive_emb = []
                    emb_to_cache = emb
                    if self.args.attack == 'amc':
                        raw_forward_embeddings = getattr(
                            self, '_last_raw_forward_embeddings', None
                        )
                        if (
                            raw_forward_embeddings is not None
                            and len(raw_forward_embeddings) == self.args.num_passive
                            and all(item is not None for item in raw_forward_embeddings)
                        ):
                            emb_to_cache = raw_forward_embeddings
                        elif not self.args.rl:
                            # Non-RL forward DP is applied after this raw value is
                            # saved in passive_emb_for_backward.
                            emb_to_cache = passive_emb_for_backward
                        if not self._amc_raw_cache_logged:
                            print(
                                "[AMC Cache] using client-local pre-forward-noise embeddings; "
                                "returned gradients remain defended."
                            )
                            self._amc_raw_cache_logged = True
                    grad_to_cache = grad
                    self._maybe_update_fora_online(emb_to_cache)
                    for passive_id in range(self.args.num_passive):
                        passive_emb.append(emb_to_cache[passive_id].clone().detach().cpu())

                    # process grad
                    # grad = torch.autograd.grad(loss, emb, create_graph=True)
                    if self.args.dispersion:
                        if (batch_idx + 1) % self.args.attack_every_n_iter == 0 or (batch_idx + 1) == self.iteration:
                            dispersion_list.append(self.dispersion(emb_to_cache, grad_to_cache, labels))
                    passive_grad = []
                    for passive_id in range(self.args.num_passive):
                        passive_grad.append(grad_to_cache[passive_id].clone().detach().cpu())

                    if self.args.attack == 'ressfl_fia':
                        raw_data = [party_data.clone().detach().cpu() for party_data in data]
                        cache_payload = [raw_data, passive_emb, passive_grad, labels.cpu()]
                    else:
                        cache_payload = [passive_emb, passive_grad, labels.cpu()]

                    torch.save(cache_payload, os.path.join(self.data_dir, "data_{}.pt".format(batch_idx)))
                    del passive_emb, passive_grad, grad

                if (self.args.set_attack_epoch and epoch == self.args.attack_epoch) or not self.args.set_attack_epoch:
                    if (batch_idx + 1) % self.args.attack_every_n_iter == 0 or (batch_idx + 1) == self.iteration:
                        start_time = time.time_ns()
                        performed_attack = False

                        if self.args.attack == 'sign':
                            grad_logit = torch.autograd.grad(loss, logit, create_graph=True)
                            grad_for_attack = grad_logit
                            if self.args.rl:
                                grad_for_attack = grad_emb_noised_for_update
                            elif self.args.defense_all or getattr(self.args, 'acvfl', False):
                                grad_for_attack = grad

                            if self.args.dispersion:
                                self.dispersion(emb, grad, labels)
                            self.attack(grad_for_attack, labels, batch_idx)
                            performed_attack = True
                            del grad_logit, grad
                        elif self.args.attack == 'cluster':
                            # grad = torch.autograd.grad(loss, emb, create_graph=True)
                            
                            emb_for_attack = emb

                            if self.args.dispersion:
                                dispersion_list.append(self.dispersion(emb_for_attack, grad, labels))

                            if self.args.use_emb:
                                self.attack(emb_for_attack, labels, batch_idx)
                            else:                                
                                self.attack(grad, labels, batch_idx)
                            performed_attack = True
                            del grad

                        if performed_attack:
                            end_time = time.time_ns()
                            attack_nseconds = end_time - start_time
                            second = int(attack_nseconds / (1000 * 1000))
                            msecond = int(attack_nseconds / 1000) % 1000
                            nsecond = int(attack_nseconds) % 1000
                            if not self.args.rl:
                                print("Attack Runtime: {}:{}:{} (ns)".format(second, msecond, nsecond))
                            self.metrics.attack_runtime.append(attack_nseconds)

                
                self.optimizer_entire.step()
                self.optimizer_active.step()
                for i in range(self.args.num_passive):
                    self.optimizer_passive[i].step()
                if self.args.attack == 'amc':
                    import attackers.amc as amc_module
                    amc_module.log_amc_optimizer_stats(self, batch_idx)
                if self.args.rl:
                    self._clear_current_forward_defense()
                train_loss += loss.item()
                total += labels.size(0)
                pred_labels = pred.argmax(dim=1)
                correct += pred_labels.eq(labels).sum().item()

                # --- 鏍稿績淇敼 2锛氭彃鍏ラ殣绉佹鏌ヤ笌寮哄埗鎵撳嵃閫昏緫 ---
                current_eps = self.privacy_monitor.get_avg_spent(force=True)
                if self.args.rl:
                    # Single authoritative forced conversion for this batch,
                    # after its forward/backward releases have been composed.
                    self._cached_avg_privacy_spent = float(current_eps)
                if current_eps > self.args.epsilon + 1e-5:
                    print(f"\n[Privacy Alert] Privacy budget ({self.args.epsilon}) exhausted at Epoch {epoch+1}, Step {batch_idx+1}!")
                    
                    # 1. 鎵撳嵃涓讳换鍔″綋鍓嶇殑璁粌鍑嗙‘鐜?
                    tmp_loss = normalize_reported_loss(
                        total_loss=train_loss,
                        num_batches=batch_idx + 1,
                        num_examples=total,
                        uses_sum_reduction=main_task_uses_sum_reduction(self.args),
                    )
                    tmp_acc = 100. * correct / total
                    print(f"Train set (Partial): Average loss: {tmp_loss:.4f}, Accuracy: {correct}/{total} ({tmp_acc:.2f}%)")
                    
                    # 2. 寮哄埗瑙﹀彂鏀诲嚮璇勪及
                    forced_idx = self.iteration - 1 
                    print(f">>> [Final Attack Evaluation at step {batch_idx+1}]")
                    
                    if self.args.attack == 'cluster':
                        if self.args.use_emb:
                            self.attack(emb, labels, forced_idx)
                        else:
                            self.attack(grad, labels, forced_idx)
                    elif self.args.attack == 'sign':
                        self.attack(grad, labels, forced_idx)
                    
                    elif self.args.attack in ['pmc', 'amc', 'ressfl_fia']:
                        self.attack()

                    stop_training = True
                    break

                if (
                    self.args.rl
                    and self.args.plot_rl_curves
                    and ((batch_idx + 1) % 100 == 0)
                ):
                    was_training = self.model.training
                    
                    eval_reward = rl_helpers.evaluate_rl_performance(self)
                    if eval_reward is not None:
                        self.eval_reward_history.append(eval_reward)

                    if was_training:
                        self.model.train()
                        self.model.active.train()
                        for i in range(self.args.num_passive):
                            self.model.passive[i].train()
                if (batch_idx + 1) % self.args.attack_every_n_iter == 0 or (batch_idx + 1) == self.iteration:
                    step_loss_avg = normalize_step_loss(
                        loss_value=loss.item(),
                        batch_examples=labels.size(0),
                        uses_sum_reduction=main_task_uses_sum_reduction(self.args),
                    )
                    if not self.args.rl and main_task_uses_sum_reduction(self.args):
                        print(
                            'Epoch:{}/{}, Step:{} \tLoss(avg): {:.6f} [raw_sum={:.6f}]'.format(
                                epoch + 1,
                                self.args.epochs,
                                batch_idx + 1,
                                step_loss_avg,
                                loss.item(),
                            )
                        )
                    elif not self.args.rl:
                        print('Epoch:{}/{}, Step:{} \tLoss: {:.6f}'.format(epoch+1, self.args.epochs, batch_idx+1, loss.item()))

            if stop_training:
                break

            if self.args.attack == 'amc':
                if epoch == self.args.epochs - 1:
                    start_time = time.time_ns()
                    self.attack()
                    attack_nseconds = time.time_ns() - start_time
                    if not self.args.rl:
                        print("Attack Runtime: {}:{}:{} (ns)".format(
                            int(attack_nseconds / (1000 * 1000)),
                            int(attack_nseconds / 1000) % 1000,
                            int(attack_nseconds) % 1000,
                        ))
                    self.metrics.attack_runtime.append(attack_nseconds)
            elif self.args.attack in ['pmc', 'ressfl_fia']:
                skip_epoch_end_fia = (
                    self.args.attack == 'ressfl_fia'
                    and bool(getattr(self.args, 'fia_run_final_only', 1))
                )
                should_run_epoch_end_attack = (
                    not skip_epoch_end_fia or epoch == self.args.epochs - 1
                )
                if should_run_epoch_end_attack:
                    start_time = time.time_ns()

                    if not self.args.set_attack_epoch:
                        if epoch == 0:
                            self.attack(init=True)
                        else:
                            self.attack()
                    else:
                        if epoch + 1 < self.args.attack_epoch:
                            pass
                        elif epoch + 1 == self.args.attack_epoch:
                            self.attack(init=True)
                        else:
                            self.attack()

                    end_time = time.time_ns()
                    attack_nseconds = end_time - start_time
                    second = int(attack_nseconds / (1000 * 1000))
                    msecond = int(attack_nseconds / 1000) % 1000
                    nsecond = int(attack_nseconds) % 1000
                    if not self.args.rl:
                        print("Attack Runtime: {}:{}:{} (ns)".format(second, msecond, nsecond))
                    self.metrics.attack_runtime.append(attack_nseconds)

            # deal the dispersion list
            if self.args.attack in ['cluster', 'pmc', 'amc'] and self.args.dispersion:
                dispersion_list = np.array(dispersion_list)
                dispersion = dispersion_list.mean(axis=0)
                dispersion = dispersion.tolist()
                if not self.args.rl:
                    print("Dispersion: {}".format(dispersion))
                self.metrics.dispersion.append(dispersion)
                self.metrics.write()

            if self.args.rl:
                # Report the epoch main-task metric with the same evaluation
                # protocol as final test().  The former online accumulator was
                # computed while the per-batch forward DP plan was active,
                # whereas test() runs after that transient plan is cleared;
                # labeling those two different protocols as the same accuracy
                # made healthy models appear to jump by 20+ points at test time.
                self._evaluate()
                if rl_runtime_logger is not None:
                    rl_runtime_logger.finish_epoch(self, epoch)
            else:
                self._evaluate()
            if self.main_lr_schedulers:
                optim_utils.step_main_task_schedulers(self.main_lr_schedulers)
                if not self.args.rl:
                    print(
                    "[LR Schedule] Epoch {}/{} -> {}".format(
                        epoch + 1,
                        self.args.epochs,
                        optim_utils.summarize_main_task_lrs(
                            self.args,
                            self.optimizer_active,
                            self.optimizer_passive,
                        ),
                    )
                    )
            if getattr(self.args, 'acvfl', False):
                print("\n" + "="*20 + " ACVFL Global Status Check " + "="*20)
                for pid in range(self.args.num_passive):
                    c_val = self.adaptive_clipping_state.get(pid, 0.1)
                    print(f"  [Party {pid}] Final Adaptive Norm C: {c_val:.6f}")
                print("="*60 + "\n")
        if self.args.rl:
            print("\n" + "="*20 + " Final RL Privacy Verification " + "="*20)
        else:
            print("\n" + "="*20 + " Final Average Privacy Analysis " + "="*20)
        
        total_final_eps = 0.0
        final_party_eps = []
        show_split_privacy = bool(
            getattr(self.args, 'rl', False)
            or getattr(self.args, 'defense_all', False)
            or getattr(self.args, 'ppdl', False)
            or bool(getattr(self.args, 'acvfl', False) and getattr(self.args, 'acvfl_bidirectional', 1))
        )
        for i in range(self.args.num_passive):
            total_rdp_i = self.privacy_monitor.rdp_history[i]
            total_eps_at_alphas = total_rdp_i + (np.log(1.0 / self.privacy_monitor.delta) / (self.privacy_monitor.alphas - 1))
            final_eps_i = np.min(total_eps_at_alphas)
            total_final_eps += final_eps_i
            final_party_eps.append(float(final_eps_i))
            if show_split_privacy:
                forward_rdp_i = self.forward_privacy_monitor.rdp_history[i]
                backward_rdp_i = self.backward_privacy_monitor.rdp_history[i]
                forward_eps_at_alphas = forward_rdp_i + (np.log(1.0 / self.forward_privacy_monitor.delta) / (self.forward_privacy_monitor.alphas - 1))
                backward_eps_at_alphas = backward_rdp_i + (np.log(1.0 / self.backward_privacy_monitor.delta) / (self.backward_privacy_monitor.alphas - 1))
                final_forward_eps_i = np.min(forward_eps_at_alphas)
                final_backward_eps_i = np.min(backward_eps_at_alphas)
                print(
                    f"  Party {i} Composed eps: {final_eps_i:.4f} "
                    f"(Forward-only view: {final_forward_eps_i:.4f}, "
                    f"Backward-only view: {final_backward_eps_i:.4f})"
                )
            else:
                print(f"  Party {i} Cumulative eps: {final_eps_i:.4f}")
        
        final_system_avg = total_final_eps / self.args.num_passive
        print("-" * 60)
        if self.args.rl:
            target_eps = float(self.args.epsilon)
            worst_party_eps = max(final_party_eps) if final_party_eps else 0.0
            tolerance = 1e-5
            privacy_ok = worst_party_eps <= target_eps + tolerance
            print(
                "  Target eps: {:.6f} | Final average eps: {:.6f} | "
                "Worst-party eps: {:.6f} | Status: {}".format(
                    target_eps,
                    final_system_avg,
                    worst_party_eps,
                    "PASS" if privacy_ok else "FAIL",
                )
            )
            self._cached_avg_privacy_spent = float(final_system_avg)
            if not privacy_ok:
                raise RuntimeError(
                    "Final worst-party privacy loss {:.6f} exceeds target epsilon {:.6f}.".format(
                        worst_party_eps,
                        target_eps,
                    )
                )
        print("="*72 + "\n")

    def _forward_for_evaluation(self, data):
        if self.adavfed_engine is None or not self.adavfed_engine.config.enabled:
            return self.model(data)

        # Test-time gates are deterministic: z = clip(mu, 0, 1).
        gated_inputs = self.adavfed_engine.apply_input_gates(data)
        embeddings = [
            self.model.passive[party_id](gated_inputs[party_id])
            for party_id in range(self.args.num_passive)
        ]
        logit, pred = self.model.forward_from_embeddings(embeddings)
        return embeddings, logit, pred

    def test(self):
        if not self.args.rl:
            print("\n============== Test ==============")
        # Final test only reports main-task utility on the test set.
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()
        if self.adavfed_engine is not None and self.adavfed_engine.config.enabled:
            self.adavfed_engine.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_dataset):
                data, labels = batch_data
                
                data = [d.to(self.device) for d in data]
                labels = labels.to(self.device)

                emb, logit, pred = self._forward_for_evaluation(data)

                test_loss += self.loss(logit if self.adavfed_engine is not None else pred, labels).item()
                pred = pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss = normalize_reported_loss(
            total_loss=test_loss,
            num_batches=len(self.test_dataset),
            num_examples=self.test_dataset_len,
            uses_sum_reduction=main_task_uses_sum_reduction(self.args),
        )
        test_acc = 100. * correct / self.test_dataset_len
        if self.args.rl:
            print('Test Accuracy: {:.2f}%'.format(test_acc))
        else:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, self.test_dataset_len, test_acc))
        
        self.metrics.test_loss.append(test_loss)
        self.metrics.test_acc.append(test_acc)
        self.metrics.write()

        return test_acc
    def _construct_rl_state(self, smooth_party_risks, smooth_utility):
        return rl_helpers.construct_rl_state(
            self,
            smooth_party_risks,
            smooth_utility,
        )

    def evaluate_rl_performance(self):
        return rl_helpers.evaluate_rl_performance(self)

    def _record_online_epoch_metrics(self, train_loss, correct, total, num_batches):
        average_loss = normalize_reported_loss(
            total_loss=train_loss,
            num_batches=num_batches,
            num_examples=total,
            uses_sum_reduction=main_task_uses_sum_reduction(self.args),
        )
        train_acc = 100.0 * correct / max(total, 1)
        print('Epoch {}/{} Main-task Accuracy: {:.2f}%\n'.format(
            self.epoch + 1, self.args.epochs, train_acc))
        self.metrics.train_loss.append(average_loss)
        self.metrics.train_acc.append(train_acc)
        self.metrics.write()
        return train_acc

    def _evaluate(self):
        # evaluate entire model and show training loss and accuracy
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()
        if self.adavfed_engine is not None and self.adavfed_engine.config.enabled:
            self.adavfed_engine.eval()

        train_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_data in self.train_dataset:
                data, labels = batch_data 

                data = [d.to(self.device) for d in data]
                labels = labels.to(self.device)

                _, logit, pred = self._forward_for_evaluation(data)
                train_loss += self.loss(logit if self.adavfed_engine is not None else pred, labels).item()
                pred = pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        train_loss = normalize_reported_loss(
            total_loss=train_loss,
            num_batches=len(self.train_dataset),
            num_examples=self.train_dataset_len,
            uses_sum_reduction=main_task_uses_sum_reduction(self.args),
        )
        train_acc = 100. * correct / self.train_dataset_len
        if self.args.rl:
            print('Epoch {}/{} Main-task Accuracy: {:.2f}%\n'.format(
                self.epoch + 1, self.args.epochs, train_acc))
        else:
            print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                train_loss, correct, self.train_dataset_len, train_acc))
        
        self.metrics.train_loss.append(train_loss)
        self.metrics.train_acc.append(train_acc)
        self.metrics.write()

        return train_acc

    def attack(self, data, labels, batch_idx=0):
        pass

    def dispersion(self, emb, grad, labels):
        num_classes = datasets.datasets_classes[self.args.dataset]
        passive_emb = []
        for passive_id in range(len(emb)):
            passive_emb.append(emb[passive_id].reshape(emb[passive_id].shape[0], -1).detach().cpu().numpy())  # KMeans expected dim <= 2.
        passive_emb = np.array(passive_emb)

        dispersion_list = []
        for passive_id in range(self.args.num_passive):
            # algorithm{'lloyd', 'elkan', 'auto', 'full'}, default='lloyd'
            kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init='auto')
            kmeans.fit(passive_emb[passive_id])

            # calculate the closest point to the center
            dis = kmeans.transform(passive_emb[passive_id]).min(axis=1)  # n_samples * n_clusters
            dispersion = dis.sum()
            dispersion_list.append(dispersion)
            print("Passive {} dispersion: {}".format(passive_id, dispersion))
        return dispersion_list


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass

