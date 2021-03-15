from mlagents_envs.environment import UnityEnvironment
import torch
import numpy as np
from torch import nn
from torch.optim import Adam

from .base import Algorithm
from .buffer import RolloutBuffer
from .networks.network import ActorPolicy
from .networks.value import StateFunction


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, image_shape, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4,
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(image_shape, state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            image_shape=image_shape,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Actor.
        self.actor = ActorPolicy(
            image_shape=image_shape,
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            image_shape=image_shape,
            state_shape=state_shape
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env: UnityEnvironment, before_image, before_state, before_action, before_done, t, step):
        t += 1

        # What action to take
        done = False
        behaviour_names = list(env.behavior_specs.keys())
        for name in behaviour_names:
            str_split = name.split("?")
            decision_steps, terminal_steps = env.get_steps(
                behavior_name=name)
            if str_split[0] == "Player":
                if len(decision_steps) == 1:
                    for agent_id in decision_steps.agent_id:
                        observation = decision_steps[agent_id].obs
                        image = observation[0]
                        state = observation[1][0:6]
                        action, log_pi = self.explore(image, state)
                        # Set the action to the agent
                        env.set_action_for_agent(
                            behavior_name=name, agent_id=agent_id, action=np.array([action]))
                if len(terminal_steps) == 1:
                    done = True
                    for agent_id in terminal_steps.agent_id:
                        observation = terminal_steps[agent_id].obs
                        image = observation[0]
                        state = observation[1][0:6]
                        action, log_pi = self.explore(image, state)
        env.step()
        if step != 1 and not before_done:
            self.buffer.append(before_image, before_state,
                               before_action, 0, done, log_pi, image, state)

        if done:
            t = 0

        return image, state, action, done, t

    def update(self, writer):
        self.learning_steps += 1
        images, states, actions, rewards, dones, log_pis, next_images, next_states = self.buffer.get()
        self.update_ppo(images, states, actions, rewards, dones,
                        log_pis, next_images, next_states, writer)

    def update_ppo(self, images, states, actions, rewards, dones, log_pis, next_images, next_states, writer):
        with torch.no_grad():
            values = self.critic(images, states)
            next_values = self.critic(next_images, next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(images, states, targets, writer)
            self.update_actor(images, states, actions, log_pis, gaes, writer)

    def update_critic(self, images, states, targets, writer):
        loss_critic = (self.critic(images, states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, images, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(images, states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        pass