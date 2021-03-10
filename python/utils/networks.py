import torch
from torch.nn.modules.activation import ReLU

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

from torch import nn


class PolicyNetwork(torch.nn.Module):
    def __init__(self, image_dim, state_dim, action_dim, discrete):
        super(PolicyNetwork, self).__init__()
        c, h, w = image_dim
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
            nn.Linear(in_features=state_dim + 1536, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=action_dim),
        )

        self.image_dim = image_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, image, state):
        x1 = self.cnn(image.permute(0, 3, 1, 2))
        x = torch.cat((x1, state), dim=1)
        if self.discrete:
            probs = torch.nn.functional.softmax(self.net(x))
            distb = torch.distributions.Categorical(probs)
        else:
            mean = self.net(x)

            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim) * (std ** 2)

            distb = torch.distributions.MultivariateNormal(mean, cov_mtx)

        return distb


class ValueNetwork(torch.nn.Module):
    def __init__(self, image_dim, state_dim):
        super(ValueNetwork, self).__init__()
        c, h, w = image_dim
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
            nn.Linear(in_features=state_dim + 1536, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, image, state):
        x1 = self.cnn(image.permute(0, 3, 1, 2))
        x = torch.cat((x1, state), dim=1)
        return self.net(x)


class Discriminator(torch.nn.Module):
    def __init__(self, image_dim, state_dim, action_dim, discrete):
        super(Discriminator, self).__init__()

        c, h, w = image_dim
        self.state_dim = state_dim
        self.image_dim = image_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if self.discrete:
            self.act_emb = nn.Embedding(
                action_dim, (1536+state_dim)
            )
            self.net_in_dim = 2 * (1536+state_dim)
        else:
            self.net_in_dim = (1536+state_dim) + action_dim

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
            nn.Linear(in_features=self.net_in_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, image, state, actions):
        return torch.sigmoid(self.get_logits(image, state, actions))

    def get_logits(self, image, state, actions):
        x1 = self.cnn(image.permute(0, 3, 1, 2))
        x = torch.cat((x1, state), dim=1)

        if self.discrete:
            actions = self.act_emb(actions.long())

        sa = torch.cat([x, actions], dim=-1)

        return self.net(sa)
