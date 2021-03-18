import torch
from torch import nn
import torch.nn.functional as F


class AIRLDiscrim(nn.Module):

    def __init__(self, image_shape, state_shape, gamma):
        super().__init__()
        state_shape = state_shape[0]

        h, w, c = image_shape

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
        rs = self.g_net(g_state)
        h1 = self.h_cnn(images.permute(0, 3, 1, 2))
        h_state = torch.cat((h1, states), dim=1)
        vs = self.h_net(h_state)
        h2 = self.h_cnn(next_images.permute(0, 3, 1, 2))
        h2_state = torch.cat((h2, next_states), dim=1)
        next_vs = self.h_net(h2_state)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, images, states, dones, log_pis, next_images, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(images, states, dones, next_images, next_states) - log_pis

    def calculate_reward(self, images, states, dones, log_pis, next_images, next_states):
        with torch.no_grad():
            logits = self.forward(images, states, dones,
                                  log_pis, next_images, next_states)
            return -F.logsigmoid(-logits)


class GAILDiscrim(nn.Module):

    def __init__(self, image_shape, state_shape, action_shape):
        super().__init__()
        state_shape = state_shape[0]

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
            nn.Linear(in_features=1+state_shape + 1536, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, images, states, actions):
        x1 = self.cnn(images.permute(0, 3, 1, 2))
        x2 = torch.cat((x1, states), dim=1)
        x3 = torch.cat((x2, actions), dim=1)
        out = self.net(x3)
        return out.squeeze(1)

    def calculate_reward(self, images, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(images, states, actions))
