import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward", "done"),
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class RewardCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer1_q1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2_q1 = nn.Linear(256, 128)
        self.layer3_q1 = nn.Linear(128, 1)

        self.layer1_q2 = nn.Linear(state_dim + action_dim, 256)
        self.layer2_q2 = nn.Linear(256, 128)
        self.layer3_q2 = nn.Linear(128, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.layer1_q1(sa))
        q1 = F.relu(self.layer2_q1(q1))
        q1 = self.layer3_q1(q1)

        q2 = F.relu(self.layer1_q2(sa))
        q2 = F.relu(self.layer2_q2(q2))
        q2 = self.layer3_q2(q2)
        return q1, q2


def apply_backward_risk_gate(vfl, action_plan, party_risk_pct_list, base_step_eps):
    gated_plan = dict(action_plan)
    risk_threshold_pct = float(vfl.proxy_risk_threshold_pct)
    party_risks = list(party_risk_pct_list)
    if len(party_risks) < vfl.args.num_passive:
        party_risks.extend([0.0] * (vfl.args.num_passive - len(party_risks)))

    eps_list = [float(v) for v in gated_plan.get('eps_list', [])]
    q_list = [float(v) for v in gated_plan.get('q_list', [])]
    eps_ratio_list = [float(v) for v in gated_plan.get('eps_ratio_list', [])]
    q_min = float(getattr(vfl.args, 'rl_q_min', 0.05))
    q_max = float(getattr(vfl.args, 'rl_q_max', 0.15))
    risk_q_min = float(getattr(vfl.args, 'rl_risk_q_min', q_max))
    risk_q_max = float(getattr(vfl.args, 'rl_risk_q_max', 0.50))
    if not hasattr(vfl, 'risk_gate_trigger_counts'):
        vfl.risk_gate_trigger_counts = [0] * vfl.args.num_passive
        vfl.risk_gate_decision_counts = [0] * vfl.args.num_passive
    gate_flags = [False] * vfl.args.num_passive
    for party_id in range(vfl.args.num_passive):
        gate_on = float(party_risks[party_id]) > risk_threshold_pct
        gate_flags[party_id] = bool(gate_on)
        vfl.risk_gate_decision_counts[party_id] += 1
        if gate_on:
            vfl.risk_gate_trigger_counts[party_id] += 1
        if gate_on and party_id < len(q_list):
            action_fraction = (q_list[party_id] - q_min) / max(q_max - q_min, 1e-12)
            action_fraction = float(np.clip(action_fraction, 0.0, 1.0))
            q_list[party_id] = risk_q_min + action_fraction * (risk_q_max - risk_q_min)
        else:
            if party_id < len(eps_list):
                eps_list[party_id] = float(base_step_eps)
            if party_id < len(eps_ratio_list):
                eps_ratio_list[party_id] = 1.0

    vfl.last_risk_gate_values = [float(value) for value in party_risks[:vfl.args.num_passive]]
    vfl.last_risk_gate_flags = gate_flags

    gated_plan['eps_list'] = eps_list
    gated_plan['q_list'] = q_list
    gated_plan['eps_ratio_list'] = eps_ratio_list
    return gated_plan


def set_current_forward_defense(vfl, forward_eps_list):
    base_clip = float(getattr(vfl.args, 'clip_threshold_forward', 1.0))
    use_public_clip = bool(getattr(vfl.args, 'forward_public_clip_calibration', 1))
    vfl.current_forward_defense_plan = {}
    # Preserve the original SAC-era forward clipping defaults, but keep them
    # outside the action space now that SAC controls only the backward path.
    forward_clip_target = float(
        np.clip(
            getattr(vfl.args, 'forward_public_clip_target_hit', 0.25),
            1e-4,
            1.0 - 1e-4,
        )
    )
    forward_clip_cap = float(max(
        getattr(vfl.args, 'rl_forward_clip_cap_max', 1.35),
        1.0,
    ))
    initial_clips = [
        vfl.task_prior.get_forward_clip_value(party_id, base_clip) if use_public_clip else base_clip
        for party_id in range(len(forward_eps_list))
    ]
    for party_id, forward_eps in enumerate(forward_eps_list):
        forward_eps = float(max(forward_eps, 0.0))
        enabled = bool(forward_eps > 0.0)
        initial_clip = initial_clips[party_id]
        vfl.current_forward_defense_plan[party_id] = {
            'enabled': enabled,
            'targeted': enabled,
            'reason': 'active' if enabled else 'zero_eps',
            'epsilon': forward_eps,
            'initial_clip': initial_clip,
            'clip_target': forward_clip_target,
            'clip_cap_scale': forward_clip_cap,
        }


def get_backward_release_mask(vfl, party_id, grad_tensor, keep_rate):
    mode = int(getattr(vfl.args, 'backward_topk_mask', 1))
    if mode != 1:
        return None

    flat_feature_dim = int(grad_tensor.reshape(grad_tensor.shape[0], -1).shape[1])
    return vfl.task_prior.get_backward_topk_mask(
        party_id=party_id,
        feature_dim=flat_feature_dim,
        keep_rate=keep_rate,
        device=grad_tensor.device,
    )


def get_backward_release_keep_rate(vfl, keep_rate):
    mode = int(getattr(vfl.args, 'backward_topk_mask', 1))
    return 1.0 if mode == 0 else float(keep_rate)


class RLPDPAgent:
    def __init__(self, state_dim, action_dim, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_parties = int(getattr(args, "num_passive", 1))

        self.gamma = args.sac_gamma
        self.tau = args.sac_tau
        self.batch_size = getattr(args, "sac_batch_size", args.batch_size)
        self.alpha = args.sac_alpha
        self.target_entropy = -float(self.action_dim)
        self.grad_clip = float(getattr(args, "sac_grad_clip", 5.0))

        self.beta = getattr(args, "beta_payment", 0.1)
        self.w_utility = getattr(args, "w_utility", 100.0)

        self.lambda_eps = 0.0
        self.lambda_risk = 0.0
        self.alpha_lambda_eps = getattr(args, "alpha_lambda_eps", 0.01)
        self.alpha_lambda_risk = getattr(args, "alpha_lambda_risk", 0.01)
        self.E_target = 1.0
        self.R_target = float(np.clip(
            getattr(args, "rl_proxy_threshold", 40.0) / 100.0,
            0.0,
            1.0,
        ))

        self.actor = Actor(
            state_dim,
            action_dim,
            log_std_min=getattr(args, "sac_log_std_min", -5.0),
            log_std_max=getattr(args, "sac_log_std_max", 0.5),
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.sac_lr_actor)

        self.reward_critic = RewardCritic(state_dim, action_dim).to(self.device)
        self.reward_critic_optimizer = optim.Adam(self.reward_critic.parameters(), lr=args.sac_lr_critic)
        self.reward_critic_target = RewardCritic(state_dim, action_dim).to(self.device)
        self.reward_critic_target.load_state_dict(self.reward_critic.state_dict())

        initial_alpha = max(float(args.sac_alpha), 1e-8)
        self.log_alpha = torch.tensor(
            [math.log(initial_alpha)],
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.sac_lr_alpha)

        self.replay_buffer = ReplayBuffer(10000)

        self.q_default = float(np.clip(getattr(args, "q_weak_fixed", 0.6), 1e-6, 1.0))
        self.q_min = float(np.clip(getattr(args, "rl_q_min", 0.05), 1e-6, 1.0))
        self.q_max = float(np.clip(getattr(args, "rl_q_max", self.q_default), self.q_min, 1.0))
        self.eps_ratio_min = float(getattr(args, "eps_strong_min_ratio", 0.85))
        self.eps_ratio_max = float(getattr(args, "eps_strong_max_ratio", 1.0))
        self.w_comm = float(getattr(args, "w_comm_weak", 1.0))

    @staticmethod
    def _scale_to_interval(value, low, high):
        return low + 0.5 * (value + 1.0) * (high - low)

    def choose_action(self, state, evaluate=False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean, log_std = self.actor(state_tensor)

        if evaluate:
            action_tanh = torch.tanh(mean)
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            z = dist.rsample()
            action_tanh = torch.tanh(z)
        return action_tanh.detach(), mean.detach(), log_std.detach()

    def scale_action(self, action_tanh, base_step_eps=None):
        if isinstance(action_tanh, np.ndarray):
            action_tensor = torch.tensor(action_tanh, dtype=torch.float32).view(-1)
        else:
            action_tensor = action_tanh.view(-1)

        if base_step_eps is None:
            base_step_eps = 0.05

        eps_ratio_list = []
        q_list = []
        for party_id in range(self.num_parties):
            if party_id < action_tensor.numel():
                eps_ratio = self._scale_to_interval(
                    action_tensor[party_id].item(),
                    self.eps_ratio_min,
                    self.eps_ratio_max,
                )
            else:
                eps_ratio = 1.0
            eps_ratio_list.append(float(eps_ratio))

        q_offset = self.num_parties
        for party_id in range(self.num_parties):
            action_idx = q_offset + party_id
            if action_idx < action_tensor.numel():
                q_value = self._scale_to_interval(
                    action_tensor[action_idx].item(),
                    self.q_min,
                    self.q_max,
                )
            else:
                q_value = self.q_default
            q_list.append(float(q_value))

        eps_list = [float(base_step_eps) * ratio for ratio in eps_ratio_list]
        return {
            "eps_list": eps_list,
            "q_list": q_list,
            "eps_ratio_list": eps_ratio_list,
        }

    def calculate_reward(self, state_t, state_t_plus_1, action_plan):
        utility_t1 = state_t_plus_1[self.num_parties]

        r_utility = utility_t1 * self.w_utility
        eps_ratio_list = action_plan.get("eps_ratio_list", [])
        q_list = action_plan.get("q_list", [])
        protect_cost = float(np.mean([(1.0 - ratio) ** 2 for ratio in eps_ratio_list])) if eps_ratio_list else 0.0
        communication_cost = float(np.mean(q_list)) if q_list else self.q_default
        protect_penalty = self.beta * protect_cost
        communication_penalty = self.w_comm * communication_cost
        reward = r_utility - protect_penalty - communication_penalty
        self.last_reward_components = {
            "utility": float(r_utility),
            "protect_penalty": float(protect_penalty),
            "communication_penalty": float(communication_penalty),
            "base_reward": float(reward),
        }
        return reward

    def calculate_punishment(self, state_t_plus_1):
        party_risks = state_t_plus_1[:self.num_parties]
        current_budget_usage = state_t_plus_1[self.num_parties + 2]

        risk_terms = [max(0.0, float(risk) - self.R_target) ** 2 for risk in party_risks]
        risk_violation = float(np.mean(risk_terms)) if len(risk_terms) > 0 else 0.0
        eps_violation = max(0.0, current_budget_usage - self.E_target) ** 2

        total_punishment = (self.lambda_risk * risk_violation) + (self.lambda_eps * eps_violation)

        self.last_punishment_components = {
            "risk_violation": float(risk_violation),
            "epsilon_violation": float(eps_violation),
            "risk_multiplier": float(self.lambda_risk),
            "epsilon_multiplier": float(self.lambda_eps),
            "constraint_penalty": float(total_punishment),
        }

        self.lambda_risk = max(
            0.0,
            0.99 * self.lambda_risk + self.alpha_lambda_risk * risk_violation,
        )
        self.lambda_eps = max(
            0.0,
            0.99 * self.lambda_eps + self.alpha_lambda_eps * eps_violation,
        )

        return total_punishment

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            mean, log_std = self.actor(next_state_batch)
            std = log_std.exp()
            dist = Normal(mean, std)
            z = dist.rsample()
            next_action_tanh = torch.tanh(z)
            next_log_prob = dist.log_prob(z) - torch.log(1 - next_action_tanh.pow(2) + 1e-6)
            next_log_prob = next_log_prob.sum(1, keepdim=True)

        with torch.no_grad():
            target_q1, target_q2 = self.reward_critic_target(next_state_batch, next_action_tanh)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + (1.0 - done_batch) * self.gamma * (target_q - self.alpha * next_log_prob)

        current_q1, current_q2 = self.reward_critic(state_batch, action_batch)
        reward_critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.reward_critic_optimizer.zero_grad()
        reward_critic_loss.backward()
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.grad_clip)
        self.reward_critic_optimizer.step()

        mean, log_std = self.actor(state_batch)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action_tanh = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        actor_q1, actor_q2 = self.reward_critic(state_batch, action_tanh)
        actor_q = torch.min(actor_q1, actor_q2)

        actor_loss = (self.alpha * log_prob - actor_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        if self.grad_clip > 0.0 and self.log_alpha.grad is not None:
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.grad_clip)
        self.alpha_optimizer.step()

        self.update_target_nets()

    def update_target_nets(self):
        for param, target_param in zip(self.reward_critic.parameters(), self.reward_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def decay_lr(self, factor=0.1):
        optimizers = [
            self.actor_optimizer,
            self.reward_critic_optimizer,
            self.alpha_optimizer
        ]

        for opt in optimizers:
            for param_group in opt.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * factor
                param_group['lr'] = new_lr
