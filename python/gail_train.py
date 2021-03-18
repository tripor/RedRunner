from utils.environment import load_environment, load_environment_editor
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import torch
from gym.spaces import Box, Discrete

from utils.networks_gail import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, conjugate_gradient, rescale_and_linesearch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class GAIL:
    def __init__(
        self,
        image_dim,
        state_dim,
        action_dim,
        discrete,
        train_config=None,
        save_path='./model.ckpt'
    ):
        self.image_dim = image_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config
        self.save_path = save_path

        self.pi = PolicyNetwork(
            self.image_dim, self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.image_dim, self.state_dim)

        self.d = Discriminator(
            self.image_dim, self.state_dim, self.action_dim, self.discrete)

        self.opt_d = torch.optim.Adam(self.d.parameters())

        if torch.cuda.is_available():
            for net in self.get_networks():
                net.to(torch.device("cuda"))

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, image, obs):
        self.pi.eval()

        image = FloatTensor(image)
        obs_scalar = FloatTensor(obs)
        distb = self.pi(image.unsqueeze(0), obs_scalar.unsqueeze(0))

        action = distb.sample().detach().cpu().numpy()

        return action

    def save(self, path):
        torch.save({
            'policy_network_state_dict': self.pi.state_dict(),
            'value_network_state_dict': self.v.state_dict(),
            'discriminator_network_state_dict': self.d.state_dict(),
            'discriminator_network_optimizer_state_dict': self.opt_d.state_dict(),
        }, path)
        print("Model saved")

    def load(self, path):
        loaded = torch.load(path)
        self.pi.load_state_dict(loaded['policy_network_state_dict'])
        self.v.load_state_dict(loaded['value_network_state_dict'])
        self.d.load_state_dict(loaded['discriminator_network_state_dict'])
        self.opt_d.load_state_dict(
            loaded['discriminator_network_optimizer_state_dict'])
        if torch.cuda.is_available():
            for net in self.get_networks():
                net.to(torch.device("cuda"))
        print("Model loaded")

    def train(self, env: UnityEnvironment, expert_observations, render=False):
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        print("Tranforming demonstrations")
        exp_obs = [at for at in expert_observations[:, 1]]
        exp_images = [at for at in expert_observations[:, 0]]
        exp_acts = [at for at in expert_observations[:, 2]]

        print("Creating tensors")
        exp_obs = FloatTensor(exp_obs)
        exp_images = FloatTensor(exp_images)
        exp_acts = FloatTensor(exp_acts)

        # Main training
        for i in range(num_iters):
            print("Iteration {}".format(i))
            obs = []
            images = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < num_steps_per_iter:
                ep_obs = []
                ep_images = []
                ep_acts = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False
                env.reset()
                while not done and steps < num_steps_per_iter:
                    behaviour_names = list(env.behavior_specs.keys())
                    behaviour_specs = list(env.behavior_specs.values())
                    for name in behaviour_names:
                        str_split = name.split("?")
                        spec = env.behavior_specs[name]
                        decision_steps, terminal_steps = env.get_steps(
                            behavior_name=name)
                        if str_split[0] == "Player":
                            if len(decision_steps) == 1:
                                for agent_id in decision_steps.agent_id:
                                    observation = decision_steps[agent_id].obs
                                    image = observation[0]
                                    obs_scalar = observation[1][0:6]
                                    # What action to take
                                    act = self.act(image, obs_scalar)
                                    # Set the action to the agent
                                    env.set_action_for_agent(
                                        behavior_name=name, agent_id=agent_id, action=np.array([act]))
                                    # Save the observations (image and scalar values)
                                    ep_obs.append(obs_scalar)
                                    ep_images.append(image)
                                    # Save the actions taken
                                    ep_acts.append(act[0])
                                    ep_gms.append(gae_gamma ** t)
                                    ep_lmbs.append(gae_lambda ** t)
                                    t += 1
                            if len(terminal_steps) == 1:
                                done = True
                                for agent_id in terminal_steps.agent_id:
                                    observation = terminal_steps[agent_id].obs
                                    image = observation[0]
                                    obs_scalar = observation[1][0:6]
                    env.step()
                    if not done:
                        steps += 1

                if len(ep_obs) < 20:
                    continue
                for j in range(len(ep_obs)):
                    obs.append(ep_obs[j])
                    images.append(ep_images[j])
                    acts.append(ep_acts[j])
                ep_obs = FloatTensor(ep_obs)
                ep_images = FloatTensor(ep_images)
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_costs = (-1) * torch.log(self.d(ep_images, ep_obs, ep_acts))\
                    .squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[k:]) for k in range(t)]
                )
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_images, ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_images, ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                ep_deltas = ep_costs.unsqueeze(-1) + \
                    gae_gamma * next_vals - curr_vals

                ep_advs = torch.FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            # Training
            print("Training networks")
            obs = FloatTensor(obs)
            images = FloatTensor(images)
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets).cuda()
            advs = torch.cat(advs).cuda()
            gms = torch.cat(gms).cuda()

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            print("Training discriminator network")
            # Train the discriminator
            self.d.train()
            exp_scores = self.d.get_logits(exp_images, exp_obs, exp_acts)
            nov_scores = self.d.get_logits(images, obs, acts)

            self.opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores)
            )
            loss.backward()
            self.opt_d.step()

            print("Training value network")
            # Train the Value network
            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(images, obs).detach()

            def constraint():
                return ((old_v - self.v(images, obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(images, obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            print("Training policy network")
            # Train the Policy network
            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(images, obs)

            def L():
                distb = self.pi(images, obs)

                return (advs * torch.exp(distb.log_prob(acts) - old_distb.log_prob(acts).detach()).cuda()).mean()

            def kld():
                distb = self.pi(images, obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                        (old_cov / cov).sum(-1)
                        + (((old_mean - mean) ** 2) / cov).sum(-1)
                        - self.action_dim
                        + torch.log(cov).sum(-1)
                        - torch.log(old_cov).sum(-1)
                    ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            disc_causal_entropy = (
                (-1) * gms * self.pi(images, obs).log_prob(acts)).mean()
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)

            if i+1 % 10 == 0:
                self.save(self.save_path)

        return

    def test(self, env: UnityEnvironment, iteration_number):
        env.reset()
        for i in range(iteration_number):
            done = False
            while not done:
                behaviour_names = list(env.behavior_specs.keys())
                behaviour_specs = list(env.behavior_specs.values())
                for name in behaviour_names:
                    str_split = name.split("?")
                    spec = env.behavior_specs[name]
                    decision_steps, terminal_steps = env.get_steps(
                        behavior_name=name)
                    if str_split[0] == "Player":
                        if len(decision_steps) == 1:
                            for agent_id in decision_steps.agent_id:
                                observation = decision_steps[agent_id].obs
                                image = observation[0]
                                obs_scalar = observation[1][0:6]
                                # What action to take
                                act = self.act(image, obs_scalar)
                                # Set the action to the agent
                                env.set_action_for_agent(
                                    behavior_name=name, agent_id=agent_id, action=np.array([act]))
                        if len(terminal_steps) == 1:
                            done = True
                            for agent_id in terminal_steps.agent_id:
                                observation = terminal_steps[agent_id].obs
                                image = observation[0]
                                obs_scalar = observation[1][0:6]
                env.step()


env = load_environment_editor(graphics=True)

demonstrations = np.load('data.npy', allow_pickle=True)

image_dim = (1, 60, 80)
obs_dim = len(Box(np.array([0, 0, 0, 0, 0, 0]), np.array(
    [np.inf, np.inf, 3, np.inf, np.inf, np.inf])).high)
action_dim = Discrete(6).n
model = GAIL(image_dim, obs_dim, action_dim, True, {
    "num_iters": 20000,
    "num_steps_per_iter": 2000,
    "horizon": None,
    "lambda": 1e-3,
    "gae_gamma": 0.99,
    "gae_lambda": 0.99,
    "epsilon": 0.01,
    "max_kl": 0.01,
    "cg_damping": 0.1,
    "normalize_advantage": True
}, './model.ckpt')
model.load('./model.ckpt')
model.test(env, 200)
#model.train(env, demonstrations)
