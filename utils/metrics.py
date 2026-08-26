import os
import json


class Metrics(object):
    def __init__(self, args):
        self.args = args        
        self.test_acc = []
        self.test_loss = []
        self.train_acc = []
        self.train_loss = []
        self.attack_acc = []
        self.amc_train_attack_acc = []
        self.attack_mse = []
        self.attack_ssim = []
        self.fia_sample_metrics = []
        self.privacy_loss = []
        self.risk = []
        self.attack_runtime = []
        self.dispersion = []
        self.dir = os.path.join(os.path.abspath(getattr(args, 'result_root', '.')), 'metrics')

    def _collect_hyperparameters_in_order(self):
        hyperparams = {}
        for key, value in vars(self.args).items():
            if key == 'command':
                continue
            hyperparams[key] = value
        return hyperparams

    def write(self):
        '''write existing history records into a json file'''
        metrics = self._collect_hyperparameters_in_order()
        metrics['test_acc'] = self.test_acc
        metrics['test_loss'] = self.test_loss
        metrics['train_acc'] = self.train_acc
        metrics['train_loss'] = self.train_loss
        metrics['attack_acc'] = self.attack_acc
        metrics['amc_train_attack_acc'] = self.amc_train_attack_acc
        metrics['attack_mse'] = self.attack_mse
        metrics['attack_ssim'] = self.attack_ssim
        metrics['fia_sample_metrics'] = self.fia_sample_metrics
        metrics['privacy_loss'] = self.privacy_loss
        metrics['risk'] = self.risk
        metrics['attack_runtime'] = self.attack_runtime
        metrics['dispersion'] = self.dispersion

        defense_mode = "no_defense"
        if getattr(self.args, 'defense_all', False):
            defense_mode = "defense_all"
        budget_sweep_mode = getattr(self.args, 'budget_sweep_mode', 'joint')
        observation3_mode = getattr(self.args, 'observation3_mode', 'none')
        observation3_keep_rate = float(getattr(self.args, 'observation3_keep_rate', 1.0))
        forward_sparse_keep_rate = float(getattr(self.args, 'forward_sparse_keep_rate', 1.0))

        filename = "metrics_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.json".format(self.args.num_passive,
            self.args.batch_size,
            self.args.epochs,
            self.args.lr_passive, 
            self.args.lr_attack, 
            self.args.attack_epoch, 
            self.args.attack_id,
            self.args.use_emb,
            self.args.simple,
            self.args.division_mode,
            self.args.balanced,
            defense_mode,
            budget_sweep_mode,
            observation3_mode,
            "{:.4f}".format(observation3_keep_rate),
            "{:.4f}".format(forward_sparse_keep_rate),
            self.args.epsilon,
            self.args.round,
            getattr(self.args, 'seed', 'noseed'))
        if getattr(self.args, 'hg', False):
            filename = filename[:-5] + "_HG.json"
        if self.args.dataset == "criteo":
            filedir = "balanced" if self.args.balanced else "imbalanced"
            metrics_path = os.path.join(self.dir, self.args.attack, self.args.dataset, filedir, filename)
        else:
            metrics_path = os.path.join(self.dir, self.args.attack, self.args.dataset, filename)

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


def calculate_main_task_acc(pred, labels):
    pred_labels = pred.argmax(dim=1)
    correct = pred_labels.eq(labels).sum().item()
    if labels.shape[0] == 0:
        return 0.0
    return 100.0 * correct / labels.shape[0]


def make_empty_interval_stats(num_parties):
    return {
        i: {
            'eps': [],
            'q': [],
            'zero': [],
            'zero_ratio': [],
            'sig': [],
            'sigma': [],
            'noise_std': [],
            'clip': [],
            'norm': [],
            'clip_coef': [],
            'clipped_norm': [],
            'released_norm': [],
            'debiased_norm': [],
            'next_clip_target': [],
            'final_norm': [],
            'snr': [],
            'f_eps': [],
            'f_targeted': [],
            'f_enabled': [],
            'f_noise_std': [],
            'f_noise_std_min': [],
            'f_noise_std_max': [],
            'f_weighted_noise_std': [],
            'f_weight_min': [],
            'f_weight_max': [],
            'f_clip': [],
            'f_clip_reference': [],
            'f_pre_noise_hit': [],
            'f_released_hit': [],
            'f_debiased_hit': [],
            'f_control_hit': [],
            'f_clip_hit': [],
            'f_norm': [],
            'f_norm_p50': [],
            'f_norm_p90': [],
            'f_norm_p95': [],
            'f_norm_max': [],
            'f_weighted_norm': [],
            'f_weighted_norm_p50': [],
            'f_weighted_norm_p90': [],
            'f_weighted_norm_p95': [],
            'f_weighted_norm_max': [],
            'f_weighted_clipped_norm': [],
            'f_weighted_clipped_norm_p95': [],
            'f_weighted_clipped_norm_max': [],
            'f_raw_released_norm': [],
            'f_next_clip': [],
            'f_sigma_proxy': [],
            'f_adaptive_target': [],
            'f_max_clip': [],
            'f_sparse_keep': [],
            'f_sparse_zero': [],
            'f_sparse_norm_ratio': [],
        }
        for i in range(num_parties)
    }
