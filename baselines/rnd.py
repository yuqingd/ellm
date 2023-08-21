from turtle import pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

import utils

from dqn import DQNAgent

class RND(nn.Module):
    def __init__(self, obs_dim, hidden_dim, rnd_rep_dim, device, clip_val=5.):
        super().__init__()
        self.clip_val = clip_val
        self.device = device

        self.normalize_obs = nn.BatchNorm1d(obs_dim, affine=False)
        self.predictor = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rnd_rep_dim))
        self.target = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, rnd_rep_dim))

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = self.normalize_obs(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True)
        return prediction_error
    

########### 
# RND online + DQN 

class RNDAgent(DQNAgent):
    def __init__(self, rnd_hidden_dim=512, rnd_rep_dim=512, rnd_lr=1e-4, rnd_scale=1, use_language=True, l_lambda=.5, noveld=False, noveld_alpha=1,  **kwargs):
        super().__init__(**kwargs)
        self.rnd_scale = rnd_scale
        self.use_language = use_language
        self.noveld = noveld
        self.noveld_alpha = noveld_alpha

        if use_language:
            assert kwargs['use_goal'] or kwargs['use_language_state']
        self.l_lambda = l_lambda
        encoder_dim = self.critic.encoder.hidden_dim
        self.rnd = RND(encoder_dim, rnd_hidden_dim, rnd_rep_dim, self.device).to(self.device)
        if use_language:
            self.l_rnd = RND(encoder_dim, rnd_hidden_dim, rnd_rep_dim, self.device).to(self.device)
        
        # optimizers
        if use_language:
            self.rnd_optimizer = torch.optim.Adam(list(self.rnd.parameters()) + list(self.l_rnd.parameters()),
                                              lr=rnd_lr)
        else:
            self.rnd_optimizer = torch.optim.Adam(self.rnd.parameters(), lr=rnd_lr)
        
        self.intrinsic_reward_rms = utils.RMS_np(device=self.device)
        self.rnd.train()
    
    def update_rnd(self, batch):
        metrics = dict()
        obs = batch[0]
        
        if self.use_language:
            loss = self.rnd(self.critic.encoder.img_encoder(obs['obs'])) + self.l_rnd(self.critic.encoder.lang_state_encoder(obs['text_obs'])) 
        else:
            loss = self.rnd(self.critic.encoder(obs))

        loss = loss.mean()
        self.rnd_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_optimizer.step()

        metrics['rnd_loss'] = loss.item()
        return metrics

    def compute_intr_reward(self, obs, next_obs, step):
        if self.use_language:
            img_prediction_error = self.rnd(self.critic.encoder.img_encoder(obs['obs']))
            l_prediction_error = self.l_rnd(self.critic.encoder.lang_state_encoder(obs['text_obs']))
            
            if self.noveld:
                next_img_prediction_error = self.rnd(self.critic.encoder.img_encoder(next_obs['obs']))
                next_l_prediction_error = self.l_rnd(self.critic.encoder.lang_state_encoder(next_obs['text_obs']))
                noveld_img_error = next_img_prediction_error - self.noveld_alpha * img_prediction_error
                noveld_l_error = next_l_prediction_error - self.noveld_alpha * l_prediction_error
                prediction_error = torch.nn.functional.relu(noveld_img_error, inplace=True) + self.l_lambda * torch.nn.functional.relu(noveld_l_error, inplace=True)
            else:      
                prediction_error = img_prediction_error + self.l_lambda * l_prediction_error 
        else:
            prediction_error = self.rnd(self.critic.encoder(obs))
            if self.noveld:
                next_img_prediction_error = self.rnd(self.critic.encoder.img_encoder(next_obs['obs']))
                noveld_img_error = next_img_prediction_error - self.noveld_alpha * prediction_error
                prediction_error = torch.nn.functional.relu(noveld_img_error, inplace=True)
                
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        
        reward = self.rnd_scale * prediction_error / (np.sqrt(intr_reward_var[0]) + 1e-8)
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
        metrics.update(self.update_rnd(batch))

        if step % self.critic_target_update_every_steps == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)

        return metrics
    
    