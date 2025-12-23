"""
CartPole Double DQN Agent (PyTorch Version)
-------------------------------------------
Double DQN (DDQN) 算法实现
目标计算：target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Double DQN 超参数
# -----------------------------
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 100000
INITIAL_EXPLORATION_STEPS = 2000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.997
TARGET_UPDATE_STEPS = 200  # 比DQN更频繁的更新


class QNet(nn.Module):
    """与DQN相同的网络结构"""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )
        # Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """与DQN相同的经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        s = np.asarray(s)
        s2 = np.asarray(s2)
        if s.ndim == 2 and s.shape[0] == 1:
            s = s.squeeze(0)
        if s2.ndim == 2 and s2.shape[0] == 1:
            s2 = s2.squeeze(0)
        self.buf.append((s, a, r, s2, 0.0 if done else 1.0))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, m = zip(*batch)
        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2, axis=0),
            np.array(m, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class DDQNConfig:
    """Double DQN 配置类"""
    gamma: float = GAMMA
    lr: float = LR
    batch_size: int = BATCH_SIZE
    memory_size: int = MEMORY_SIZE
    initial_exploration: int = INITIAL_EXPLORATION_STEPS
    eps_start: float = EPS_START
    eps_end: float = EPS_END
    eps_decay: float = EPS_DECAY
    target_update: int = TARGET_UPDATE_STEPS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DDQNSolver:
    """
    PyTorch Double DQN 智能体
    关键区别：使用 Double DQN 的目标值计算
    """

    def __init__(self, observation_space: int, action_space: int, cfg: DDQNConfig | None = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or DDQNConfig()
        self.device = torch.device(self.cfg.device)

        # 创建 online 和 target 网络
        self.online = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.target = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.update_target(hard=True)  # 初始同步

        # 优化器（添加权重衰减）
        self.optim = optim.Adam(self.online.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        
        # 经验回放
        self.memory = ReplayBuffer(self.cfg.memory_size)
        
        # 计数器
        self.steps = 0
        self.exploration_rate = self.cfg.eps_start

    # -----------------------------
    # 行动选择
    # -----------------------------
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        """ε-greedy 行动选择"""
        if not evaluation_mode and np.random.rand() < self.exploration_rate:
            return random.randrange(self.act_dim)
        
        with torch.no_grad():
            s_np = np.asarray(state_np, dtype=np.float32)
            if s_np.ndim == 1:
                s_np = s_np[None, :]
            s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
            q = self.online(s)  # [1, act_dim]
            a = int(torch.argmax(q, dim=1).item())
        return a

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)

    # -----------------------------
    # Double DQN 学习
    # -----------------------------
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """主学习函数（与DQN兼容）"""
        self.remember(state, action, reward, next_state, done)
        self.experience_replay()

    def experience_replay(self):
        """Double DQN 经验回放"""
        # 1) 检查是否有足够的数据
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
            self._decay_eps()
            return

        # 2) 采样
        s, a, r, s2, m = self.memory.sample(self.cfg.batch_size)
        
        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a_t  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r_t  = torch.as_tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        m_t  = torch.as_tensor(m,  dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3) Double DQN 目标值计算
        with torch.no_grad():
            # 使用 online 网络选择动作
            next_actions = self.online(s2_t).argmax(dim=1, keepdim=True)  # [B, 1]
            # 使用 target 网络评估 Q 值
            q_next = self.target(s2_t).gather(1, next_actions)  # [B, 1]
            target = r_t + m_t * self.cfg.gamma * q_next  # [B, 1]

        # 4) 计算当前 Q 值
        q_sa = self.online(s_t).gather(1, a_t)  # [B, 1]

        # 5) 损失和优化
        loss = nn.functional.mse_loss(q_sa, target)
        
        self.optim.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        
        self.optim.step()

        # 6) 衰减探索率
        self._decay_eps()

        # 7) 定期更新目标网络
        if self.steps % self.cfg.target_update == 0:
            self.update_target(hard=True)

    def update_target(self, hard: bool = True, tau: float = 0.001):
        """更新目标网络"""
        if hard:
            self.target.load_state_dict(self.online.state_dict())
        else:
            # 软更新（可选）
            with torch.no_grad():
                for p_t, p in zip(self.target.parameters(), self.online.parameters()):
                    p_t.data.mul_(1 - tau).add_(tau * p.data)

    # -----------------------------
    # 保存/加载
    # -----------------------------
    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        """加载模型"""
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])

    # -----------------------------
    # 辅助函数
    # -----------------------------
    def _decay_eps(self):
        """衰减探索率"""
        self.exploration_rate = max(self.cfg.eps_end, self.exploration_rate * self.cfg.eps_decay)
        self.steps += 1