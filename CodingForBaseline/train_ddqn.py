"""
CartPole Double DQN Training (PyTorch + Gymnasium)
---------------------------------------------------
- Trains a DDQN agent and logs scores via ScoreLogger (PNG + CSV)
- Saves model to ./models/cartpole_ddqn.torch
- Evaluates from a saved model (render optional)

输出格式与 train.py 保持一致
"""

from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import torch
from collections import deque

from agents.cartpole_ddqn import DDQNSolver, DDQNConfig
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cartpole_ddqn.torch")


def train(num_episodes: int = 200, terminal_penalty: bool = True) -> DDQNSolver:
    """
    Main training loop for Double DQN:
      - Creates the environment and agent
      - For each episode:
          * Reset env → get initial state
          * Loop: select action, step environment, call agent.step()
          * Log episode score with ScoreLogger
      - Saves the trained model to disk
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Create CartPole environment (no render during training for speed)
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    # Infer observation/action dimensions from the env spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # DDQN优化配置（与train.py中的格式保持一致）
    cfg = DDQNConfig(
        lr=2e-4,                    # 降低学习率
        batch_size=64,
        eps_start=1.0,
        eps_end=0.1,                # 保持一定的探索
        eps_decay=0.998,            # 更慢的衰减
        target_update=200,          # 目标网络更新频率
        initial_exploration=2000,
        memory_size=100000,
    )

    # Construct agent with DDQN config
    agent = DDQNSolver(obs_dim, act_dim, cfg=cfg)
    print(f"[Info] Using device: {agent.device}")
    print(f"[Info] Training Double DQN with {num_episodes} episodes")

    # Episode loop (与train.py格式相同)
    for run in range(1, num_episodes + 1):
        # Gymnasium reset returns (obs, info). Seed for repeatability.
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1

            # 1. ε-greedy action from the agent (training mode)
            #    state shape is [1, obs_dim]
            action = agent.act(state)

            # 2. Gymnasium step returns: obs', reward, terminated, truncated, info
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. Optional small terminal penalty (encourage agent to avoid failure)
            if terminal_penalty and done:
                reward = -1.0
            
            # 4. Reshape next_state for agent and next loop iteration
            next_state = np.reshape(next_state_raw, (1, obs_dim))

            # 5. Give (s, a, r, s', done) to the agent, which handles
            #    remembering and learning internally.
            agent.step(state, action, reward, next_state, done)

            # 6. Move to next state
            state = next_state

            # 7. Episode end: log and break (与train.py完全相同的输出格式)
            if done:
                print(f"Run: {run}, Epsilon: {agent.exploration_rate:.3f}, Score: {steps}")
                logger.add_score(steps, run)  # writes CSV + updates score PNG
                break

    env.close()
    # Persist the trained model
    agent.save(MODEL_PATH)
    print(f"[Train] Double DQN model saved to {MODEL_PATH}")
    return agent


def evaluate(model_path: str | None = None,
             algorithm: str = "ddqn",
             episodes: int = 5,
             render: bool = True,
             fps: int = 60):
    """
    Evaluate a trained DDQN agent in the environment using greedy policy (no ε).
    - Loads weights from disk
    - Optionally renders (pygame window)
    - Reports per-episode steps and average
    
    与train.py中的evaluate函数格式保持一致
    """
    # Resolve model path
    model_dir = MODEL_DIR
    if model_path is None:
        candidates = [f for f in os.listdir(model_dir) if f.endswith(".torch")]
        if not candidates:
            raise FileNotFoundError(f"No saved model found in '{model_dir}/'. Please train first.")
        model_path = os.path.join(model_dir, candidates[0])
        print(f"[Eval] Using detected model: {model_path}")
    else:
        print(f"[Eval] Using provided model: {model_path}")

    # Create env for evaluation; 'human' enables pygame-based rendering
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 根据算法选择智能体
    if algorithm.lower() == "ddqn":
        agent = DDQNSolver(obs_dim, act_dim, cfg=DDQNConfig())
    elif algorithm.lower() == "dqn":
        # 为了兼容，也支持DQN评估
        from agents.cartpole_dqn import DQNSolver, DQNConfig
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Load trained weights
    agent.load(model_path)
    print(f"[Eval] Loaded {algorithm.upper()} model from: {model_path}")

    scores = []
    # Sleep interval to approximate fps; set 0 for fastest evaluation
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            # Greedy action (no exploration) by calling act() in evaluation mode
            action = agent.act(state, evaluation_mode=True)

            # Step env forward
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            # Slow down rendering to be watchable
            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"[Eval] Average over {episodes} episodes: {avg:.2f}")
    return scores


def train_with_early_stop(num_episodes: int = 1000, 
                         target_score: int = 475, 
                         patience: int = 50):
    """
    DDQN专用训练函数，包含早停机制
    (可选功能，如果需要可以调用)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # DDQN配置
    cfg = DDQNConfig(
        lr=2e-4,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay=0.999,
        target_update=100,
        initial_exploration=2000,
        memory_size=100000,
    )
    
    agent = DDQNSolver(obs_dim, act_dim, cfg=cfg)
    print(f"[DDQN] Training on device: {agent.device}")
    
    # 早停机制
    best_avg_score = 0
    no_improve_count = 0
    scores_window = deque(maxlen=100)
    
    for run in range(1, num_episodes + 1):
        state, _ = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0
        
        while True:
            steps += 1
            action = agent.act(state)
            
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 与train.py相同的奖励设计
            if done:
                reward = -1.0
                
            next_state = np.reshape(next_state_raw, (1, obs_dim))
            agent.step(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                # 与train.py相同的输出格式
                print(f"Run: {run}, Epsilon: {agent.exploration_rate:.3f}, Score: {steps}")
                logger.add_score(steps, run)
                
                # 早停检查
                scores_window.append(steps)
                if len(scores_window) == 100:
                    current_avg = np.mean(scores_window)
                    if current_avg > best_avg_score:
                        best_avg_score = current_avg
                        no_improve_count = 0
                        # 保存最佳模型
                        agent.save(MODEL_PATH.replace(".torch", "_best.torch"))
                        print(f"  → New best average: {current_avg:.1f}")
                    else:
                        no_improve_count += 1
                        
                    # 达到目标分数
                    if current_avg >= target_score:
                        print(f"✓ Target reached! Average score: {current_avg:.1f}")
                        env.close()
                        agent.save(MODEL_PATH)
                        return agent
                        
                    # 早停
                    if no_improve_count >= patience:
                        print(f"Early stopping at episode {run}, best avg: {best_avg_score:.1f}")
                        env.close()
                        agent.save(MODEL_PATH)
                        return agent
                
                break
    
    env.close()
    agent.save(MODEL_PATH)
    print(f"[DDQN] Training completed, model saved to {MODEL_PATH}")
    return agent


if __name__ == "__main__":
    # 与train.py完全相同的调用格式
    print("="*60)
    print("Double DQN Training for CartPole-v1")
    print("="*60)
    
    # 训练DDQN智能体
    agent = train(num_episodes=800, terminal_penalty=True)
    
    # 评估DDQN智能体（100个episodes，不渲染）
    print("\n" + "="*60)
    print("Evaluating Double DQN Agent")
    print("="*60)
    scores = evaluate(
        model_path="models/cartpole_ddqn.torch", 
        algorithm="ddqn", 
        episodes=100, 
        render=False, 
        fps=60
    )
    
    # 输出统计信息
    avg_score = np.mean(scores)
    print(f"\nDouble DQN Final Results:")
    print(f"  Average score over 100 episodes: {avg_score:.2f}")
    print(f"  Maximum score: {max(scores)}")
    print(f"  Minimum score: {min(scores)}")
    
    if avg_score >= 475:
        print("  ✓ Double DQN achieved the target score of 475+!")
    else:
        print(f"  ✗ Double DQN did not reach target (475 needed, got {avg_score:.2f})")