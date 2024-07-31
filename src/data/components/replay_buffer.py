import numpy as np
import collections
from collections import OrderedDict, namedtuple

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
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
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[idx] for idx in indices]

        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences], dtype=np.float32)
        dones = np.array([exp.done for exp in experiences], dtype=np.bool_)
        new_states = np.array([exp.new_state for exp in experiences])

        return states, actions, rewards, dones, new_states