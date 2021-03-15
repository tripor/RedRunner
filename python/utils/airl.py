import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPO
from .networks.discriminator import AIRLDiscrim


class AIRL(PPO):

    def __init__(self, buffer_exp, image_shape, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma, rollout_length,
                         mix_buffer, lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm)

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            image_shape=image_shape,
            state_shape=state_shape,
            gamma=gamma
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            images, states, _, _, dones, log_pis, next_images, next_states = self.buffer.sample(
                self.batch_size)
            # Samples from expert's demonstrations.
            images_exp, states_exp, actions_exp, _, dones_exp, next_images_exp, next_states_exp = self.buffer_exp.sample(
                self.batch_size)
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(
                    images_exp, states_exp, actions_exp)
            # Update discriminator.
            self.update_disc(
                images, states, dones, log_pis, next_images, next_states, images_exp, states_exp,
                dones_exp, log_pis_exp, next_images_exp, next_states_exp, writer
            )

        # We don't use reward signals here,
        images, states, actions, _, dones, log_pis, next_images, next_states = self.buffer.get()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            images, states, dones, log_pis, next_images, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(
            images, states, actions, rewards, dones, log_pis, next_images, next_states, writer)

    def update_disc(self, images, states, dones, log_pis, next_images, next_states,
                    images_exp, states_exp, dones_exp, log_pis_exp,
                    next_images_exp, next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(images, states, dones,
                              log_pis, next_images, next_states)
        logits_exp = self.disc(
            images_exp, states_exp, dones_exp, log_pis_exp, next_images_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
