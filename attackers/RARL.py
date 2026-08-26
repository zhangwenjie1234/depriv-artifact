import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F

# 论文参数配置
HIDDEN_DIM = 32
LR_RL = 0.001
GAMMA = 0.90
MEMORY_CAPACITY = 5000
BATCH_SIZE = 32
TARGET_UPDATE_ITER = 50

EPS_CANDIDATES = [ 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
DELTA_CANDIDATES = [1e-4, 1e-5, 1e-6]
def get_noise(grad, mechanism, epsilon, delta, device, history_grad=None):

    # --- 1. 确定裁剪阈值 (Sensitivity C) ---
    clip_factor = 1.1 # 经验值，允许梯度轻微增长
    
    if history_grad is not None:
        # 使用历史梯度进行 Coordinate-wise 自适应裁剪
        threshold = clip_factor * torch.abs(history_grad)
    else:
        # 冷启动或无历史时，使用默认安全阈值
        default_val = 5.0
        threshold = torch.full_like(grad, default_val).to(device)
    
    # 防止阈值过小导致数值不稳定
    threshold = torch.clamp(threshold, min=1e-6)

    # --- 2. 执行裁剪 (Clipping) ---
    # 这一步是满足 DP 的关键：必须将梯度限制在 Sensitivity (threshold) 范围内
    grad_clipped = torch.min(torch.max(grad, -threshold), threshold)

    # --- 3. 计算噪声并添加 ---
    # 如果 Epsilon 极小或为负，通常意味着不加噪或全噪声(这里做个保护)
    if epsilon <= 1e-6:
        return grad_clipped

    if mechanism == 'laplace':
        # Laplace Scale = Sensitivity / Epsilon
        # Sensitivity 这里就是 threshold (坐标级敏感度)
        scale = threshold / epsilon
        m = torch.distributions.laplace.Laplace(0, scale)
        noise = m.sample().to(device)
    else:
        # Gaussian Scale = Sensitivity * sqrt(2 * ln(1.25/delta)) / Epsilon
        gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
        sigma = (threshold * gaussian_factor) / epsilon
        noise = torch.randn_like(grad, device=device) * sigma

    noisy_grad = grad_clipped + noise
    
    return noisy_grad

def calc_zeta(mechanism, epsilon, delta, sensitivity=1.0):
    """
    辅助计算隐私损失 Zeta (Eq. 3)，考虑动态 Delta
    """
    EPS_MIN, EPS_MAX = min(EPS_CANDIDATES), max(EPS_CANDIDATES)
    S_ref = 1.0

    if mechanism == 'gaussian':
        tau_curr = S_ref * np.sqrt(2 * np.log(1.25/delta)) / epsilon
        # 估算边界时假设 delta 取极端值
        DELTA_MIN, DELTA_MAX = min(DELTA_CANDIDATES), max(DELTA_CANDIDATES)
        tau_max = S_ref * np.sqrt(2 * np.log(1.25/DELTA_MIN)) / EPS_MIN
        tau_min = S_ref * np.sqrt(2 * np.log(1.25/DELTA_MAX)) / EPS_MAX
    else:
        tau_curr = S_ref / epsilon
        tau_max = S_ref / EPS_MIN
        tau_min = S_ref / EPS_MAX
    
    norm_tau = (tau_curr - tau_min) / (tau_max - tau_min + 1e-9)
    norm_tau = np.clip(norm_tau, 0.0, 1.0)
    
    return (1.0 - norm_tau) * epsilon

def init_probes(emb, num_passive, num_classes, device):
    """
    初始化探针网络 (RARL专用)
    返回: privacy_probes (ModuleList), probe_opt (list of optimizers)
    """
    privacy_probes = nn.ModuleList()
    probe_opt = []
    print(f"Initializing {num_passive} probes for RARL...")
    for i in range(num_passive):
        # 自动推断 Embedding 维度
        dim = emb[i].view(emb[i].size(0), -1).size(1)
        p = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, num_classes)
        ).to(device)
        privacy_probes.append(p)
        probe_opt.append(torch.optim.Adam(p.parameters(), lr=0.01))
    
    return privacy_probes, probe_opt
class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a1, a2, r, psi, s_next):
        self.buffer.append((s, a1, a2, r, psi, s_next))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a1, a2, r, psi, s_next = zip(*batch)
        return np.stack(s), np.array(a1), np.array(a2), np.array(r), np.array(psi), np.stack(s_next)
    def __len__(self):
        return len(self.buffer)

class RARLAgent:
    def __init__(self, state_dim, device):
        self.device = device
        self.state_dim = state_dim 
        
        # Level 1: Mechanism Selection (Gaussian vs Laplace)
        self.l1_action_dim = 2 
        
        # Level 2: Parameter Selection (Epsilon & Delta)
        self.eps_list = EPS_CANDIDATES
        self.delta_list = DELTA_CANDIDATES
        self.n_eps = len(self.eps_list)
        self.n_delta = len(self.delta_list)
        
        self.l2_input_dim = state_dim + 1 # State + Chosen Mechanism
        self.l2_action_dim = self.n_eps * self.n_delta # 笛卡尔积

        # Level 1 Networks (Q & R)
        self.q1 = Network(state_dim, self.l1_action_dim).to(device)
        self.r1 = Network(state_dim, self.l1_action_dim).to(device)
        self.target_q1 = Network(state_dim, self.l1_action_dim).to(device)
        self.target_r1 = Network(state_dim, self.l1_action_dim).to(device)
        
        # Level 2 Networks (Q & R)
        self.q2 = Network(self.l2_input_dim, self.l2_action_dim).to(device)
        self.r2 = Network(self.l2_input_dim, self.l2_action_dim).to(device)
        self.target_q2 = Network(self.l2_input_dim, self.l2_action_dim).to(device)
        self.target_r2 = Network(self.l2_input_dim, self.l2_action_dim).to(device)

        self.optimizers = [
            optim.Adam(self.q1.parameters(), lr=LR_RL), optim.Adam(self.r1.parameters(), lr=LR_RL),
            optim.Adam(self.q2.parameters(), lr=LR_RL), optim.Adam(self.r2.parameters(), lr=LR_RL)
        ]
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.learn_step_counter = 0
        self.update_targets(force=True)

    def update_targets(self, force=False):
        if force or self.learn_step_counter % TARGET_UPDATE_ITER == 0:
            self.target_q1.load_state_dict(self.q1.state_dict())
            self.target_r1.load_state_dict(self.r1.state_dict())
            self.target_q2.load_state_dict(self.q2.state_dict())
            self.target_r2.load_state_dict(self.r2.state_dict())

    # 严格按照论文 Eq. 15 实现 Boltzmann 策略分布
    def _boltzmann_choice(self, q_val, r_val):
        logits = q_val - r_val
        probs = F.softmax(logits, dim=1).cpu().data.numpy()[0]
        probs = np.nan_to_num(probs, nan=1.0/logits.shape[1])
        probs = probs / probs.sum()
        return np.random.choice(range(logits.shape[1]), p=probs)

    def choose_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Level 1 Decision
            a1 = self._boltzmann_choice(self.q1(state_t), self.r1(state_t))
            
            # Level 2 Decision
            l1_act_vec = torch.FloatTensor([[a1]]).to(self.device)
            state_l2 = torch.cat([state_t, l1_act_vec], dim=1)
            a2 = self._boltzmann_choice(self.q2(state_l2), self.r2(state_l2))
            
        return a1, a2

    def get_params(self, a1, a2):
        mech = 'gaussian' if a1 == 0 else 'laplace'
        eps_idx = a2 // self.n_delta
        delta_idx = a2 % self.n_delta        
        eps = self.eps_list[eps_idx]
        delta = self.delta_list[delta_idx]
        
        return mech, eps, delta

    def learn(self):
        if len(self.memory) < BATCH_SIZE: return
        self.learn_step_counter += 1
        self.update_targets()
        
        s, a1, a2, r, psi, s_next = self.memory.sample(BATCH_SIZE)
        s = torch.FloatTensor(s).to(self.device)
        a1 = torch.LongTensor(a1).unsqueeze(1).to(self.device)
        a2 = torch.LongTensor(a2).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        psi = torch.FloatTensor(psi).unsqueeze(1).to(self.device) # Risk value
        s_next = torch.FloatTensor(s_next).to(self.device)

        # 修正 Reward: r_hat = r - v5 * psi
        v5 = 2.0
        r_hat = r - v5 * psi

        # --- Update Level 1 ---
        with torch.no_grad():
            q1_next = self.target_q1(s_next).max(1)[0].unsqueeze(1)
            # R-network 更新：最小化未来风险
            r1_next = self.target_r1(s_next).min(1)[0].unsqueeze(1)
            q1_target = r_hat + GAMMA * q1_next
            r1_target = psi + GAMMA * r1_next

        loss_q1 = F.mse_loss(self.q1(s).gather(1, a1), q1_target)
        loss_r1 = F.mse_loss(self.r1(s).gather(1, a1), r1_target)
        
        self.optimizers[0].zero_grad(); loss_q1.backward(); self.optimizers[0].step()
        self.optimizers[1].zero_grad(); loss_r1.backward(); self.optimizers[1].step()

        # --- Update Level 2 ---
        with torch.no_grad():
            # 预测下一个状态的 Level 1 动作以辅助 Level 2 更新
            next_a1 = (self.target_q1(s_next) - self.target_r1(s_next)).argmax(1).unsqueeze(1).float()
            s_next_l2 = torch.cat([s_next, next_a1], dim=1)
            
            q2_next = self.target_q2(s_next_l2).max(1)[0].unsqueeze(1)
            r2_next = self.target_r2(s_next_l2).min(1)[0].unsqueeze(1)
            q2_target = r_hat + GAMMA * q2_next
            r2_target = psi + GAMMA * r2_next

        s_l2 = torch.cat([s, a1.float()], dim=1)
        loss_q2 = F.mse_loss(self.q2(s_l2).gather(1, a2), q2_target)
        loss_r2 = F.mse_loss(self.r2(s_l2).gather(1, a2), r2_target)

        self.optimizers[2].zero_grad(); loss_q2.backward(); self.optimizers[2].step()
        self.optimizers[3].zero_grad(); loss_r2.backward(); self.optimizers[3].step()