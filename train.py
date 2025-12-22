"""
CartPole Training & Evaluation with Save/Load (Gymnasium)
"""

import os
import numpy as np
import gymnasium as gym
from tensorflow import keras

load_model = keras.models.load_model

from agents.cartpole_dqn import DQNSolver
from agents.cartpole_ddqn import DDQNSolver
from agents.cartpole_ppo import PPOAgent
from scores.score_logger import ScoreLogger
import time

ENV_NAME = "CartPole-v1"
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "cartpole_dqn.keras")
DDQN_MODEL_PATH = os.path.join(ARTIFACT_DIR, "cartpole_ddqn.keras")
PPO_MODEL_PATH = os.path.join(ARTIFACT_DIR, "cartpole_ppo.keras")


def train_and_save(num_episodes, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = DQNSolver(obs_dim, act_dim)

    saved_500 = False  # 一旦出现500分即保存一次

    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, [1, obs_dim])
        step = 0

        while True:
            step += 1
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = reward if not done else -reward

            next_state = np.reshape(next_state, [1, obs_dim])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Run: {run}, Exploration: {agent.exploration_rate:.4f}, Score: {step}")
                score_logger.add_score(step, run)

                # 达到一次500就保存（只保存一次，避免重复写盘）
                if (not saved_500) and step >= 500:
                    agent.model.save(save_path)
                    saved_500 = True
                    print(f"[Train-DQN] Reached 500 on run {run}, saved model to: {save_path}")
                break

            agent.experience_replay()

    env.close()

    # 如果从未达到500，仍旧保存最终权重
    if not saved_500:
        agent.model.save(save_path)
        print(f"[Train-DQN] Final model saved to: {save_path}")


def train_and_save_ddqn(num_episodes, save_path=DDQN_MODEL_PATH):
    """
    使用 Double DQN(DDQN) 训练并保存模型。
    变更：只有当“连续5次达到500分”时才保存一次并立刻退出训练。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = DDQNSolver(obs_dim, act_dim)

    consecutive_perfect = 0
    saved_on_streak = False
    early_stopped = False

    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, [1, obs_dim])
        step = 0

        while True:
            step += 1
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = reward if not done else -reward

            next_state = np.reshape(next_state, [1, obs_dim])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"[DDQN] Run: {run}, Exploration: {agent.exploration_rate:.4f}, Score: {step}")
                score_logger.add_score(step, run)

                # 连续5次达到500就保存并退出训练
                if step >= 500:
                    consecutive_perfect += 1
                else:
                    consecutive_perfect = 0

                if (not saved_on_streak) and consecutive_perfect >= 5:
                    agent.save(save_path)
                    saved_on_streak = True
                    early_stopped = True
                    print(f"[Train-DDQN] Reached 500 for {consecutive_perfect} consecutive episodes (ending at run {run}). Saved to: {save_path}")
                break

            agent.experience_replay()

        if early_stopped:
            break

    env.close()

    # 不再进行“最终保存”，仅当满足连续5次500分时才保存并提前退出。
    if not saved_on_streak:
        print("[Train-DDQN] Training finished without 5 consecutive 500-step episodes. No model saved.")


def _add_suffix(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def _evaluate_actor_in_memory(model, episodes=5, seed_base=9000, render=False, fps=60, deterministic=True):
    """
    对给定的策略网络(Actor, 输出logits)做快速评估，返回平均步数。
    评估采用确定性动作(Argmax)以避免随机性影响选择“最佳”模型。
    """
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]

    scores = []
    dt = 1.0 / float(fps) if fps and render else 0.0

    for ep in range(episodes):
        state, _ = env.reset(seed=seed_base + ep)
        state = np.reshape(state, [1, obs_dim])
        done = False
        steps = 0
        while not done:
            logits = model.predict(state, verbose=0)[0]
            action = int(np.argmax(logits)) if deterministic else int(np.random.choice(len(logits), p=np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, [1, obs_dim])
            steps += 1
            if dt > 0:
                time.sleep(dt)
        scores.append(steps)
    env.close()
    return float(np.mean(scores))


def train_and_save_ppo(num_episodes, save_path, *,
                       eval_every=10, eval_episodes=5,
                       ema_alpha=0.1,
                       save_last=True,
                       early_stop_on_perfect=True,
                       perfect_score=500,
                       perfect_patience=8,
                       save_on_early_stop=True):
    """
    使用 PPOAgent 训练 CartPole，并将策略网络保存到指定路径。

    需求："只要达到一次500就保存" — 已在每个 episode 结束后检测 step>=500 并立即保存一次。

    其余：
    - 训练期间维护分数的指数滑动平均(EMA)，用于监控训练稳定性；
    - 每 eval_every 个 episode 做一次确定性评估(Argmax) eval_episodes 局，
      以评估指标(平均步数)挑选最优模型；
    - 将“最佳”权重保存到 save_path，同时将“最后一次”权重保存到 *_last.keras。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOAgent(obs_dim, act_dim)

    best_metric = -np.inf
    best_path = save_path  # 推断时默认加载最佳
    last_path = _add_suffix(save_path, "_last")
    ema_score = None
    consecutive_perfect = 0
    early_stopped = False

    saved_500 = False

    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, [1, obs_dim])
        step = 0
        episode_return = 0.0
        done = False

        while not done:
            step += 1
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state_reshaped = np.reshape(next_state, [1, obs_dim])

            agent.store(state, action, reward, value, log_prob, done)

            state = next_state_reshaped
            episode_return += reward

        # 一整个 episode 结束后，用收集到的轨迹做一次 PPO 更新
        agent.train()

        # 记录
        ema_score = step if ema_score is None else (ema_alpha * step + (1.0 - ema_alpha) * ema_score)
        print(f"Run: {run}, Score: {step}, EMA:{ema_score:.2f}, Return: {episode_return:.2f}")
        score_logger.add_score(step, run)

        # 达到一次500就保存（只保存一次）
        if (not saved_500) and step >= 500:
            agent.save(best_path)
            saved_500 = True
            print(f"[Train-PPO] Reached 500 on run {run}, saved policy to: {best_path}")

        # 早停判定：连续 perfect_patience 次达到 perfect_score
        if early_stop_on_perfect:
            if step >= perfect_score:
                consecutive_perfect += 1
            else:
                consecutive_perfect = 0
            if consecutive_perfect >= perfect_patience:
                print(f"[EarlyStop] 连续 {consecutive_perfect} 次达到 {perfect_score} 分，提前结束训练。")
                early_stopped = True
                if save_on_early_stop:
                    # 将当前权重作为最佳保存
                    agent.save(best_path)
                    if save_last:
                        agent.save(last_path)
                break

        # 周期性做确定性评估，选择最佳
        if run % eval_every == 0:
            avg_steps = _evaluate_actor_in_memory(agent.actor, episodes=eval_episodes, deterministic=True)
            print(f"[Val] episode {run}: avg_steps={avg_steps:.2f} (best={best_metric:.2f})")
            if avg_steps > best_metric:
                best_metric = avg_steps
                agent.save(best_path)
                print(f"[Checkpoint] New best ({best_metric:.2f}) saved to: {best_path}")

    env.close()

    # 保存最后一次/补存最佳
    try:
        if save_last:
            agent.save(last_path)
            print(f"[Train-PPO] Last policy saved to: {last_path}")
        # 如果未产生best，至少存一份到best_path
        if not os.path.exists(best_path):
            agent.save(best_path)
        print(f"[Train-PPO] Best policy kept at: {best_path}")
    except Exception as e:
        print(f"[Train-PPO] Error saving model: {e}")
        raise


def evaluate_from_disk(load_path, episodes=3, render=True, fps=60, wait_on_finish=True):
    model = load_model(load_path)

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    scores = []
    dt = 1.0 / float(fps) if fps and render else 0.0

    for ep in range(episodes):
        state, _ = env.reset(seed=2000 + ep)
        state = np.reshape(state, [1, obs_dim])
        done = False
        steps = 0

        while not done:
            # Greedy action (no exploration)
            q = model.predict(state, verbose=0)[0]
            action = int(np.argmax(q))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, [1, obs_dim])
            steps += 1

            if dt > 0:
                time.sleep(dt)   

        scores.append(steps)
        print(f"[Eval] Episode {ep+1}: steps={steps}")

    env.close()
    print(f"[Eval] Average over {episodes} episodes: {np.mean(scores):.2f}")
    return scores


def evaluate_from_disk_ppo(load_path, episodes=3, render=True, fps=60, deterministic=True):
    """
    使用磁盘上的 PPO 策略网络进行评估。
    deterministic=True 时用贪心(Argmax)选择动作，避免评估阶段的随机性。
    """
    model = load_model(load_path)

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    scores = []
    dt = 1.0 / float(fps) if fps and render else 0.0

    for ep in range(episodes):
        state, _ = env.reset(seed=3000 + ep)
        state = np.reshape(state, [1, obs_dim])
        done = False
        steps = 0

        while not done:
            # 根据策略网络输出的 logits 选择动作
            logits = model.predict(state, verbose=0)[0]
            if deterministic:
                action = int(np.argmax(logits))  # 评估阶段使用贪心，稳定表现
            else:
                # 数值稳定的 softmax 采样（保留可选）
                logits_shifted = logits - np.max(logits)
                probs = np.exp(logits_shifted)
                probs /= np.sum(probs)
                action = int(np.random.choice(len(probs), p=probs))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, [1, obs_dim])
            steps += 1

            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval-PPO] Episode {ep+1}: steps={steps}")

    env.close()
    print(f"[Eval-PPO] Average over {episodes} episodes: {np.mean(scores):.2f}")
    return scores


if __name__ == "__main__":
    # 如需继续使用 DQN 版本，请改回调用 train_and_save / evaluate_from_disk。
    # 这里默认演示 PPO 版本。
    # train_and_save_ppo(num_episodes=400, save_path=PPO_MODEL_PATH, eval_every=10, eval_episodes=5, ema_alpha=0.1, save_last=True)

    evaluate_from_disk_ppo(load_path=PPO_MODEL_PATH, episodes=10, render=True, deterministic=True)
    
    # train_and_save_ddqn(num_episodes=400, save_path=DDQN_MODEL_PATH)

    # evaluate_from_disk(load_path=DDQN_MODEL_PATH, episodes=10, render=True)
    # The number of times needs to be adjusted, and you need to explore and modify it yourself.
