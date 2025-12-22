"""
CartPole PPO Agent
------------------
一个针对 CartPole-v1 环境的简单 PPO( Proximal Policy Optimization )智能体实现，
使用 TensorFlow / Keras 搭建 actor-critic 网络结构。

设计目标：
  - 只依赖现有项目中的 TensorFlow / Keras、NumPy 等依赖；
  - 接口尽量简洁，便于在 train.py 中调用；
  - 注释清晰，方便课程作业阅读和修改。
"""

from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

Dense = keras.layers.Dense
Adam = keras.optimizers.Adam


# -----------------------------
# PPO 超参数（可自行调整）
# -----------------------------
GAMMA = 0.99           # 折扣因子 γ
LAMBDA = 0.95          # GAE λ
CLIP_RATIO = 0.2       # PPO clip 参数 ε
ACTOR_LR = 3e-4        # 策略网络学习率
CRITIC_LR = 1e-3       # 价值网络学习率
TRAIN_EPOCHS = 10      # 每轮对同一批数据迭代轮数
BATCH_SIZE = 64        # 每个小批量大小
ENTROPY_COEF = 0.0     # 熵正则系数（增大可鼓励探索）


class PPOAgent:
    """
    一个最小可用版本的离散动作 PPO 智能体：
      - actor：输出每个动作的未归一化 logit；
      - critic：输出状态价值 V(s)；
      - 通过简单的 GAE + PPO-clip 更新。
    """

    def __init__(self, observation_space: int, action_space: int):
        self.obs_dim = observation_space
        self.act_dim = action_space

        # --------- 构建网络 ---------
        obs_input = keras.Input(shape=(self.obs_dim,), name="obs")

        # 策略网络（actor）
        x = Dense(64, activation="tanh")(obs_input)
        x = Dense(64, activation="tanh")(x)
        logits = Dense(self.act_dim, activation=None, name="logits")(x)
        self.actor = keras.Model(obs_input, logits, name="actor")

        # 价值网络（critic）
        v = Dense(64, activation="tanh")(obs_input)
        v = Dense(64, activation="tanh")(v)
        value = Dense(1, activation=None, name="value")(v)
        self.critic = keras.Model(obs_input, value, name="critic")

        self.actor_optimizer = Adam(learning_rate=ACTOR_LR)
        self.critic_optimizer = Adam(learning_rate=CRITIC_LR)

        # --------- 经验缓冲区（按 episode 采样） ---------
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    # -----------------------------
    # 与环境交互
    # -----------------------------
    def get_action(self, state: np.ndarray):
        """
        给定 state（形状应为 (1, obs_dim)），
        返回：
          - action: int
          - log_prob: float
          - value: float
        """
        state_tf = tf.convert_to_tensor(state, dtype=tf.float32)

        logits = self.actor(state_tf)
        value = self.critic(state_tf)

        # 将 logits 转为概率分布并采样
        probs = tf.nn.softmax(logits)
        action_dist = tf.squeeze(probs, axis=0)
        # 采样一个动作
        action = tf.random.categorical(tf.math.log([action_dist]), 1)[0, 0]

        # 对应采样动作的 log π(a|s)
        log_prob = tf.math.log(action_dist[action] + 1e-8)

        action_int = int(action.numpy())
        log_prob_float = float(log_prob.numpy())
        value_float = float(tf.squeeze(value, axis=-1).numpy())

        return action_int, log_prob_float, value_float

    def store(self, state, action, reward, value, log_prob, done):
        """把一步交互的数据存入缓冲区。"""
        # 为了方便后续堆叠，这里存入扁平向量
        self.states.append(state.reshape(-1))  # (obs_dim,)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    # -----------------------------
    # PPO 训练逻辑
    # -----------------------------
    def _compute_gae(self, last_value: float = 0.0):
        """
        使用 GAE(λ) 计算优势函数和折扣回报。
        假设当前缓冲区中是一整个 episode 的数据。
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.bool_)

        gae = 0.0
        advantages = np.zeros_like(rewards, dtype=np.float32)

        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + GAMMA * values[t + 1] * non_terminal - values[t]
            gae = delta + GAMMA * LAMBDA * non_terminal * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def train(self):
        """
        使用当前缓冲区中的完整 episode 数据进行一次 PPO 更新。
        调用后会清空缓冲区。
        """
        if len(self.states) == 0:
            return

        # 计算 advantage 和 return
        advantages, returns = self._compute_gae(last_value=0.0)

        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)
        returns = returns.astype(np.float32)
        advantages = advantages.astype(np.float32)

        # 标准化优势，有助于训练稳定
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data_size = states.shape[0]
        indices = np.arange(data_size)

        for _ in range(TRAIN_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, data_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = indices[start:end]

                s_batch = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
                a_batch = tf.convert_to_tensor(actions[batch_idx], dtype=tf.int32)
                r_batch = tf.convert_to_tensor(returns[batch_idx], dtype=tf.float32)
                adv_batch = tf.convert_to_tensor(advantages[batch_idx], dtype=tf.float32)
                old_logp_batch = tf.convert_to_tensor(old_log_probs[batch_idx], dtype=tf.float32)

                with tf.GradientTape() as tape_pi, tf.GradientTape() as tape_v:
                    # --- 策略损失 ---
                    logits = self.actor(s_batch)
                    probs = tf.nn.softmax(logits)
                    log_probs_all = tf.math.log(probs + 1e-8)

                    # 选中实际动作对应的 log_prob
                    one_hot_actions = tf.one_hot(a_batch, depth=self.act_dim)
                    logp = tf.reduce_sum(one_hot_actions * log_probs_all, axis=1)

                    ratio = tf.exp(logp - old_logp_batch)  # π(a|s) / π_old(a|s)

                    unclipped = ratio * adv_batch
                    clipped = tf.clip_by_value(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * adv_batch
                    policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

                    # 熵正则（可选）
                    entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs_all, axis=1))
                    policy_loss -= ENTROPY_COEF * entropy

                    # --- 价值函数损失 ---
                    values = tf.squeeze(self.critic(s_batch), axis=1)
                    value_loss = tf.reduce_mean(tf.square(r_batch - values))

                # 反向传播并更新参数
                actor_grads = tape_pi.gradient(policy_loss, self.actor.trainable_variables)
                critic_grads = tape_v.gradient(value_loss, self.critic.trainable_variables)

                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # 清空缓冲区，等待下一轮采样
        self.clear_buffer()

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    # -----------------------------
    # 保存 / 加载
    # -----------------------------
    def save(self, path: str):
        """
        仅保存策略网络（actor），评估时只需要策略即可。
        """
        import os
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 保存模型
        self.actor.save(path)
        print(f"[PPOAgent] Actor model saved to: {path}")


