import torch
import collections
from collections import OrderedDict, namedtuple

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
    """
    Replay Buffer for storing past-experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int,device='cpu') -> None:
        self.device = device
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):  # -> Tuple:
        # indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        indices = torch.multinomial(torch.ones(len(self.buffer)), batch_size, replacement=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        return states, actions, rewards, dones, next_states