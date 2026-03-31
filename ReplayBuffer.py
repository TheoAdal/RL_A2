#ReplayBuffer.py



import numpy as np
import torch
import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=10000):
        # deque automatically removes oldest entries when full
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        ''' Add a transition to the buffer '''
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ''' Sample a random batch of transitions '''
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    # Quick test
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push([0.1, 0.2, 0.3, 0.4], 0, 1.0, [0.2, 0.3, 0.4, 0.5], False)
    states, actions, rewards, next_states, dones = buf.sample(5)
    print("ReplayBuffer test passed!")
    print("Sampled states shape:", states.shape)
