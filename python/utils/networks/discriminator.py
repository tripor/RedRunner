import torch
from torch import nn
import torch.nn.functional as F


class AIRLDiscrim(nn.Module):

    def __init__(self, image_shape, state_shape, gamma):
        super().__init__()

        c, h, w = image_shape

        self.g_cnn = nn.Sequential(
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
        self.g_net = nn.Sequential(
            nn.Linear(in_features=state_shape + 1536, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )
        self.h_cnn = nn.Sequential(
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
        self.h_net = nn.Sequential(
            nn.Linear(in_features=state_shape + 1536, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

        self.gamma = gamma

    def f(self, images, states, dones, next_images, next_states):
        g1 = self.g_cnn(images.permute(0, 3, 1, 2))
        g_state = torch.cat((g1, states), dim=1)
        rs = self.g(g_state)
        h1 = self.h_cnn(images.permute(0, 3, 1, 2))
        h_state = torch.cat((h1, states), dim=1)
        vs = self.h(h_state)
        h2 = self.h_cnn(next_images.permute(0, 3, 1, 2))
        h2_state = torch.cat((h2, next_states), dim=1)
        next_vs = self.h(h2_state)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, images, states, dones, log_pis, next_images, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(images, states, dones, next_images, next_states) - log_pis

    def calculate_reward(self, images, states, dones, log_pis, next_images, next_states):
        with torch.no_grad():
            logits = self.forward(images, states, dones,
                                  log_pis, next_images, next_states)
            return -F.logsigmoid(-logits)
