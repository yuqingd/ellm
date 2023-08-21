from turtle import pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

import utils

from dqn import DQNAgent

class ICM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, apt_rep_dim, device):
        super().__init__()
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, apt_rep_dim))

        self.forward_net = nn.Sequential(
            nn.Linear(apt_rep_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, apt_rep_dim))

        self.backward_net = nn.Sequential(
            nn.Linear(2 * apt_rep_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        embed_obs = self.encoder(obs)
        embed_next_obs = self.encoder(next_obs)
        next_obs_hat = self.forward_net(torch.cat([embed_obs, action.unsqueeze(-1)], dim=-1))
        action_hat = self.backward_net(torch.cat([embed_obs, embed_next_obs], dim=-1))

        forward_error = torch.norm(embed_next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error

########### 
# APT/ICM online + DQN 

class ICMAPTAgent(DQNAgent):
    def __init__(self, icm_scale=1.0, knn_rms=True, knn_k=12, knn_avg=True, knn_clip=0.0001, apt_rep_dim=512, apt_lr=1e-4, icm_hidden_dim=1024, 
                 use_language=False, l_lambda=1, **kwargs):
        super().__init__(**kwargs)
        self.icm_scale = icm_scale
        self.use_language = use_language
        if use_language:
            assert kwargs['use_goal'] or kwargs['use_language_state']
        self.l_lambda = l_lambda
        encoder_dim = self.critic.encoder.hidden_dim
        # action_dim is 1 
        self.icm = ICM(encoder_dim, 1, icm_hidden_dim, apt_rep_dim, self.device).to(self.device)
        if use_language:
            self.l_icm = ICM(encoder_dim, 1, icm_hidden_dim, apt_rep_dim, self.device).to(self.device)
        
        # optimizers
        if use_language:
            self.icm_optimizer = torch.optim.Adam(list(self.icm.parameters()) + list(self.l_icm.parameters()),
                                              lr=apt_lr)
        else:
            self.icm_optimizer = torch.optim.Adam(self.icm.parameters(),
                                              lr=apt_lr)
        
        self.icm.train()
        
        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)
    
    def update_icm(self, batch):
        metrics = dict()
        obs, action, reward, discount, next_obs = batch

        if self.use_language:
            forward_error, backward_error = self.icm(self.critic.encoder.img_encoder(obs['obs']), action, self.critic.encoder.img_encoder(next_obs['obs']))
            l_forward_error, l_backward_error = self.l_icm(self.critic.encoder.lang_state_encoder(obs['text_obs']), action, self.critic.encoder.lang_state_encoder(next_obs['text_obs']))
            forward_error = forward_error + l_forward_error
            backward_error = backward_error + l_backward_error
        else:
            forward_error, backward_error = self.icm(self.critic.encoder(obs), action, self.critic.encoder(next_obs))
        
        loss = forward_error.mean() + backward_error.mean()

        self.icm_optimizer.zero_grad()
        loss.backward()
        self.icm_optimizer.step()

        if self.log:
            metrics['icm_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, next_obs, step):
        # entropy reward
        if self.use_language:
            with torch.no_grad():
                rep = self.icm.encoder(self.critic.encoder.img_encoder(next_obs['obs']))
                l_rep = self.icm.encoder(self.critic.encoder.lang_state_encoder(next_obs['text_obs']))
            reward = self.pbe(rep) + self.l_lambda * self.pbe(l_rep)
        else:
            with torch.no_grad():
                rep = self.icm.encoder(self.critic.encoder(next_obs).detach())
            reward = self.pbe(rep)
        reward = reward.reshape(-1, 1)
        return reward
    
    def update_critic(self, batch, step, use_extr_rew=False):
        metrics = dict()
        obs, action, reward, discount, next_obs = batch
        
        # use intrinsic reward
        extr_reward = reward.clone()
        if use_extr_rew:
            reward = extr_reward
        else:   
            reward = self.compute_intr_reward(obs, next_obs, step).squeeze(-1)
        
        with torch.no_grad():
            next_action = self.critic(next_obs).argmax(dim=1).unsqueeze(1)
            next_Qs = self.critic_target(next_obs)
            next_Q = next_Qs.gather(1, next_action).squeeze(1)
            target_Q = reward + discount * next_Q

        # get current Q estimates
        Qs = self.critic(obs)
        Q = Qs.gather(1, action.unsqueeze(1)).squeeze(1)
        critic_loss = F.smooth_l1_loss(Q, target_Q)

        if self.log:
            metrics['q'] = Q.mean().item()
            metrics['batch_reward'] = reward.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['extr_reward'] = extr_reward.mean().item()

        self.opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.opt.step()

        return metrics
    
    def update(self, replay_iter, step, use_extr_rew=False):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        batch = utils.to_torch(batch, self.device)
        metrics.update(self.update_critic(batch, step, use_extr_rew=use_extr_rew))
        metrics.update(self.update_icm(batch))

        if step % self.critic_target_update_every_steps == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)

        return metrics
    