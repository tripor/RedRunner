import torch
from torch import nn
from torch.distributions import Categorical

from ..util import reparameterize, evaluate_lop_pi


class ActorPolicy(nn.Module):

    def __init__(self, image_shape, state_shape, action_shape):
        super().__init__()
        state_shape = state_shape[0]
        action_shape = action_shape[0]
        h, w, c = image_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32,
                      stride=4, kernel_size=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.net = nn.Sequential(
            nn.Linear(in_features=state_shape + 1536, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=action_shape),
            nn.Softmax(dim=-1)
        )

        self.log_stds = nn.Parameter(torch.zeros(1, action_shape))

    def forward(self, images, states):
        x1 = self.cnn(images.permute(0, 3, 1, 2))
        x = torch.cat((x1, states), dim=1)
        return self.net(x)

    def distribution(self, images, states):
        x1 = self.cnn(images.permute(0, 3, 1, 2))
        x = torch.cat((x1, states), dim=1)
        action_probs = self.net(x)
        dist = Categorical(action_probs)
        return dist

    def sample(self, images, states):
        x1 = self.cnn(images.permute(0, 3, 1, 2))
        x = torch.cat((x1, states), dim=1)
        action_probs = self.net(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_log_pi(self, images, states, actions):
        x1 = self.cnn(images.permute(0, 3, 1, 2))
        x = torch.cat((x1, states), dim=1)
        action_probs = self.net(x)
        dist = Categorical(action_probs)
        return dist.log_prob(actions)
