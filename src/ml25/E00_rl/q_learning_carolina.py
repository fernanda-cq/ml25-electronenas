pip install gymnasium
pip install numpy

import gymnasium as gym
import numpy as np

class RandomAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, observation):
        return self.action_space.sample()

    def step(self, state, action, reward, next_state, done=False):
        pass


class QLearningAgent(RandomAgent):
    def __init__(self, env, alpha=0.5, gamma=0.99, epsilon=1.0):
        super().__init__(env, alpha, gamma, epsilon)

    def act(self, observation):
        # ε-greedy corto
        return (self.action_space.sample()
                if np.random.rand() < self.epsilon
                else int(np.argmax(self.Q[observation])))

    def step(self, s, a, r, s_next, done=False):
        # Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
        td_target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])


if __name__ == "__main__":
    env = gym.make("CliffWalking-v1", render_mode="human")

    n_episodes, episode_length = 800, 200
    agent = QLearningAgent(env, alpha=0.5, gamma=0.99, epsilon=1.0)

    
    eps_min, eps_decay = 0.01, 0.99

    for e in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0
        for _ in range(episode_length):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(obs, action, reward, next_obs, done)
            ep_return += reward
            obs = next_obs
            if done:
                break

        agent.epsilon = max(eps_min, agent.epsilon * eps_decay)

        if (e + 1) % 50 == 0:
            print(f"Ep {e+1}: Return={ep_return:4d} | ε={agent.epsilon:.2f}")

    env.close()
