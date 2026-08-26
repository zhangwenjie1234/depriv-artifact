import torch
import numpy as np

class TemporalConsensusSmoother:

    # 恢复了原版参数: base_threshold=1.0, max_group_size=5
    def __init__(self, num_parties, filter_a=-0.81818181, filter_b=None, base_threshold=1.0, max_group_size=5):
        self.num_parties = num_parties
        self.max_group_size = max_group_size
        
        if filter_b is None:
            filter_b = [0.09090909, 0.09090909]

        self.filter_a = filter_a
        self.filter_b = list(filter_b)
        self.g_buffer = [[] for _ in range(num_parties)]
        self.m_buffer = [[] for _ in range(num_parties)]
        
        self.group_buffer = [[] for _ in range(num_parties)]
        self.base_thresholds = [base_threshold] * num_parties
        self.noisy_thresholds = [None] * num_parties

    def _apply_temporal_filter(self, party_id, current_grad):
        """应用一阶 IIR 低通滤波抑制高频噪声"""
        # [核心修复] 防止 Epoch 边界（如最后一个 batch 是 96，下一个是 128）导致的维度崩溃
        if len(self.g_buffer[party_id]) > 0 and self.g_buffer[party_id][0].shape != current_grad.shape:
            self.g_buffer[party_id] = []
            self.m_buffer[party_id] = []

        self.g_buffer[party_id].insert(0, current_grad)
        if len(self.g_buffer[party_id]) > len(self.filter_b):
            self.g_buffer[party_id].pop()
            
        m_t = torch.zeros_like(current_grad)
        for tau, b_val in enumerate(self.filter_b):
            if tau < len(self.g_buffer[party_id]):
                m_t += b_val * self.g_buffer[party_id][tau]
                
        if len(self.m_buffer[party_id]) > 0:
            m_t -= self.filter_a * self.m_buffer[party_id][0]
            
        self.m_buffer[party_id].insert(0, m_t)
        if len(self.m_buffer[party_id]) > 1:
            self.m_buffer[party_id].pop()
            
        return m_t

    def _calculate_deviation(self, buffer_list):
        if len(buffer_list) <= 1:
            return 0.0
        avg_grad = torch.mean(torch.stack(buffer_list), dim=0)
        deviations = [torch.norm(g - avg_grad).item() for g in buffer_list]
        return max(deviations)

    def step(self, party_id, new_noisy_grad, current_eps):
        filtered_grad = self._apply_temporal_filter(party_id, new_noisy_grad)
        
        if self.noisy_thresholds[party_id] is None:
            noise = np.random.laplace(0, 4.0 / current_eps)
            self.noisy_thresholds[party_id] = self.base_thresholds[party_id] + noise
            
        current_buffer = self.group_buffer[party_id]
        
        # Case 1: Buffer 为空
        if len(current_buffer) == 0:
            self.group_buffer[party_id].append(filtered_grad)
            return False, None, 1
            
        # 形状突变防护
        if current_buffer[0].shape != filtered_grad.shape:
            group_size = len(current_buffer)
            final_grad = torch.mean(torch.stack(current_buffer), dim=0)
            self.group_buffer[party_id] = [filtered_grad]
            noise = np.random.laplace(0, 4.0 / current_eps)
            self.noisy_thresholds[party_id] = self.base_thresholds[party_id] + noise
            return True, final_grad, group_size

        temp_buffer = current_buffer + [filtered_grad]
        
        # [恢复原版逻辑] 强制关闭：如果组太大了
        if len(temp_buffer) >= self.max_group_size:
            group_size = len(current_buffer)
            final_grad = torch.mean(torch.stack(current_buffer), dim=0)
            self.group_buffer[party_id] = [filtered_grad]
            noise = np.random.laplace(0, 4.0 / current_eps)
            self.noisy_thresholds[party_id] = self.base_thresholds[party_id] + noise
            return True, final_grad, group_size

        deviation = self._calculate_deviation(temp_buffer)
        noisy_deviation = deviation + np.random.laplace(0, 8.0 / current_eps)
        
        # 阈值判断
        if noisy_deviation > self.noisy_thresholds[party_id]:
            group_size = len(current_buffer)
            final_grad = torch.mean(torch.stack(current_buffer), dim=0)
            self.group_buffer[party_id] = [filtered_grad]
            noise = np.random.laplace(0, 4.0 / current_eps)
            self.noisy_thresholds[party_id] = self.base_thresholds[party_id] + noise
            return True, final_grad, group_size
        else:
            self.group_buffer[party_id].append(filtered_grad)
            return False, None, len(self.group_buffer[party_id])
