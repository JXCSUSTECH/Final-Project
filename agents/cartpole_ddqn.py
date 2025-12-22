"""
CartPole Double DQN Agent
-------------------------
参考 agents/cartpole_dqn.py，使用 Double DQN(DDQN) 的目标计算：
  target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
并引入 target 网络的周期性同步。
"""

import random
from collections import deque
from typing import Deque, Tuple, List

import numpy as np
from tensorflow import keras

load_model = keras.models.load_model
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Adam = keras.optimizers.Adam

# -----------------------------
# Hyperparameters
# -----------------------------
GAMMA = 0.98                 # 折扣因子
LEARNING_RATE = 0.001        # 学习率
MEMORY_SIZE = 100000         # 经验回放容量
BATCH_SIZE = 32              # 批大小
EXPLORATION_MAX = 1.0        # 初始探索率
EXPLORATION_MIN = 0.01       # 最小探索率
EXPLORATION_DECAY = 0.997    # 探索率衰减
TARGET_UPDATE_EVERY = 100    # 多少次训练步后同步 target 网络


class DDQNSolver:
    """
    Double DQN for CartPole
    - online 网络进行行为选择
    - target 网络进行目标评估
    """

    def __init__(self, observation_space: int, action_space: int):
        # 探索率
        self.exploration_rate: float = EXPLORATION_MAX
        # 动作空间
        self.action_space: int = action_space
        # 经验池 (s, a, r, s', done)
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=MEMORY_SIZE)

        # online Q 网络
        self.model = self._build_model(observation_space, action_space)
        # target Q 网络
        self.target_model = self._build_model(observation_space, action_space)
        # 初始对齐
        self.target_model.set_weights(self.model.get_weights())

        # 训练步计数（用于周期性更新 target）
        self.train_steps: int = 0

    def _build_model(self, observation_space: int, action_space: int):
        model = Sequential()
        model.add(Dense(32, input_shape=(observation_space,), activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(action_space, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    # -----------------------------
    # Memory handling
    # -----------------------------
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    # -----------------------------
    # Action selection (epsilon-greedy)
    # -----------------------------
    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    # -----------------------------
    # Learning
    # -----------------------------
    def experience_replay(self) -> None:
        if len(self.memory) < BATCH_SIZE:
            return

        batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = random.sample(self.memory, BATCH_SIZE)

        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        # 当前 Q 估计
        q_values = self.model.predict(states, verbose=0)
        # Double DQN: online 用于选择，target 用于评估
        q_next_online = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            a_max = int(np.argmax(q_next_online[i]))  # online 选择
            target = rewards[i]
            if not dones[i]:
                target += GAMMA * q_next_target[i][a_max]  # target 评估
            q_values[i][actions[i]] = target

        # 单次批量更新
        self.model.fit(states, q_values, verbose=0, epochs=1, batch_size=BATCH_SIZE)

        # 递增训练步并周期性同步 target
        self.train_steps += 1
        if self.train_steps % TARGET_UPDATE_EVERY == 0:
            self.target_model.set_weights(self.model.get_weights())

        # 衰减探索率
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        mdl = load_model(path)
        self.model.set_weights(mdl.get_weights())
        self.target_model.set_weights(self.model.get_weights())

