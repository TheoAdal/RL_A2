#Agent.py

#QNetwork: neural network that approximates Q(s,a)
#DQNAgent: handles action selection and Q-value updates

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from ReplayBuffer import ReplayBuffer


# ============================================================
# Q-NETWORK
# ============================================================

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=1e-3,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 hidden_size=64,
                 batch_size=64,
                 update_every=1,
                 target_update_freq=500,
                 buffer_size=10000,
                 use_target_network=False,
                 use_replay_buffer=False):

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update_freq = target_update_freq
        self.use_target_network = use_target_network
        self.use_replay_buffer = use_replay_buffer
        self.step_count = 0

        self.q_network = QNetwork(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        if use_target_network:
            self.target_network = QNetwork(state_dim, action_dim, hidden_size)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

        if use_replay_buffer:
            self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        #Epsilon-greedy action selection.
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):

        #Store a single transition and decay epsilon
        #Called once per environment per step in the vectorized loop
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.use_replay_buffer:
            self.replay_buffer.push(state, action, reward, next_state, float(done))

    def update_network(self, step):
        #Perform one gradient update step and called once per environment step
       
        self.step_count += 1

        if not self.use_replay_buffer:
            return
        if len(self.replay_buffer) < self.batch_size:
            return
        if self.step_count % self.update_every != 0:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_target_network:
                next_q = self.target_network(next_states).max(1)[0]
            else:
                next_q = self.q_network(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_target_network and \
                self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def update(self, state, action, reward, next_state, done):

        #Single-transition update for the naive agent
        #Used when use_replay_buffer=FALSE

        self.step_count += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        states      = torch.FloatTensor(state).unsqueeze(0)
        actions     = torch.LongTensor([action])
        rewards     = torch.FloatTensor([reward])
        next_states = torch.FloatTensor(next_state).unsqueeze(0)
        dones       = torch.FloatTensor([float(done)])

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_target_network:
                next_q = self.target_network(next_states).max(1)[0]
            else:
                next_q = self.q_network(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_target_network and \
                self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


if __name__ == '__main__':
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        use_target_network=True,
        use_replay_buffer=True
    )
    state = [0.1, 0.2, 0.3, 0.4]
    action = agent.select_action(state)
    print(f"Agent test passed! Selected action: {action}")
