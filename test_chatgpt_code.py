import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
eps_clip = 0.1
K_epochs = 4
T_horizon = 128

# Frostbite 환경 초기화
env = gym.make('Frostbite-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 정책 신경망 (Policy Network) 클래스 정의
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        state_values = self.fc3(x)
        return action_probs, state_values

# PPO 에이전트 클래스 정의
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = Policy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        action_probs, _ = self.policy(state)
        m = Categorical(action_probs)
        action = m.sample()
        return action.item()

    def update(self, memory):
        states = torch.Tensor(memory.states)
        actions = torch.Tensor(memory.actions)
        rewards = torch.Tensor(memory.rewards)
        next_states = torch.Tensor(memory.next_states)
        masks = torch.Tensor(memory.masks)

        _, values = self.policy(states)
        _, next_values = self.policy(next_states)

        returns = torch.zeros_like(rewards)
        deltas = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        returns[-1] = rewards[-1] + gamma * next_values[-1] * masks[-1]
        for t in reversed(range(T_horizon - 1)):
            returns[t] = rewards[t] + gamma * returns[t + 1] * masks[t]
        deltas = returns - values

        for _ in range(K_epochs):
            _, new_values = self.policy(states)
            action_probs, _ = self.policy(states)
            m = Categorical(action_probs)
            action_log_probs = m.log_prob(actions)
            ratios = torch.exp(action_log_probs - torch.Tensor(memory.action_log_probs))

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.smooth_l1_loss(values, returns.detach())

            self.optimizer.zero_grad()
            loss = actor_loss + 0.5 * critic_loss
            loss.backward()
            self.optimizer.step()

    def train(self, num_epochs, max_steps_per_epoch):
        for epoch in range(num_epochs):
            memory = Memory()

            for _ in range(max_steps_per_epoch):
                state = env.reset()
                done = False

                for _ in range(T_horizon):
                    action = self.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    mask = 0 if done else 1

                    memory.states.append(state)
                    memory.actions.append(action)
                    memory.rewards.append(reward)
                    memory.next_states.append(next_state)
                    memory.masks.append(mask)

                    state = next_state

                    if done:
                        break

                _, last_value = self.policy(torch.Tensor(state))
                memory.compute_returns(last_value)

            self.update(memory)

            # 에포크 진행 상황 출력
            print(f"Epoch {epoch + 1}/{num_epochs} completed.")

# 메모리 클래스 정의
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.masks = []
        self.action_log_probs = []

    def compute_returns(self, last_value):
        returns = 0
        advantages = 0

        for t in reversed(range(len(self.rewards))):
            returns = self.rewards[t] + gamma * returns * self.masks[t]
            deltas = returns - self.values[t]
            advantages = deltas + gamma * advantages * self.masks[t]

            self.returns[t] = returns
            self.advantages[t] = advantages

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.masks[:]
        del self.action_log_probs[:]

# PPO 에이전트 생성 및 학습
agent = PPOAgent(state_dim, action_dim)
agent.train(num_epochs=100, max_steps_per_epoch=1000)

