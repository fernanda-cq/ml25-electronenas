pip install gymnasium
pip install numpy

import gymnasium as gym
import numpy as np
from collections import deque

class RandomAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n

      
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
       
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  
        else:
            return int(np.argmax(self.Q[observation]))  

    def step(self, state, action, reward, next_state, done=False):
        """
        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
        Si el episodio terminó, no se añade el término futuro.
        """
        current = self.Q[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        self.Q[state, action] = current + self.alpha * (target - current)


if __name__ == "__main__":

    env = gym.make("CliffWalking-v1", render_mode="human")

   
    n_episodes = 1000
    max_steps = 200
    agent = QLearningAgent(env, alpha=0.5, gamma=0.99, epsilon=1.0)


    eps_min = 0.05
    eps_decay = 0.995 

    alpha_min = 0.1
    alpha_decay = 0.999

    returns = deque(maxlen=100)

    for e in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0

        for t in range(max_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

         
            agent.step(obs, action, reward, next_obs, done)

            ep_return += reward
            obs = next_obs

            if done:
                break

        agent.epsilon = max(eps_min, agent.epsilon * eps_decay)
        agent.alpha   = max(alpha_min, agent.alpha * alpha_decay)

        returns.append(ep_return)

        if (e + 1) % 50 == 0:
            avg_last_100 = np.mean(returns) if len(returns) > 0 else ep_return
            print(f"Ep {e+1:4d} | Return: {ep_return:4d} | "
                  f"Avg100: {avg_last_100:6.2f} | "
                  f"ε={agent.epsilon:.3f} α={agent.alpha:.3f}")

    env.close()
