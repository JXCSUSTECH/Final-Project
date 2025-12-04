"""
CartPole Training & Evaluation with Save/Load (Gymnasium)
"""

import os
import numpy as np
import gymnasium as gym
from keras.models import load_model
from agents.cartpole_dqn import DQNSolver
from scores.score_logger import ScoreLogger
import time

ENV_NAME = "CartPole-v1"
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "cartpole_dqn.keras")


def train_and_save(num_episodes, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = DQNSolver(obs_dim, act_dim)

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
                break

            agent.experience_replay()

    env.close()

    agent.model.save(save_path)
    print(f"[Train] Model saved to: {save_path}")


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


if __name__ == "__main__":
    train_and_save(num_episodes=30, save_path=MODEL_PATH)
    evaluate_from_disk(load_path=MODEL_PATH, episodes=3, render=True)
    # The number of times needs to be adjusted, and you need to explore and modify it yourself.
