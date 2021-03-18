from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, image_shape, state_shape, action_shape, device, seed, gamma):
        np.random.seed(int(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.image_shape = image_shape
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def explore(self, image, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        image = torch.tensor(image, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor_old.sample(
                image.unsqueeze(0), state.unsqueeze_(0))
        return action, log_pi.item()

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
