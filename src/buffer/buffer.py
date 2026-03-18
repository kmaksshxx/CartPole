from collections import deque
import random
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        s, a, r, s2, d = map(np.array, zip(*batch))

        return (
            torch.tensor(s, dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.long, device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.tensor(s2, dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device)
        )

    def __len__(self):
        return len(self.buffer)

