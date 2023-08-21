import numpy as np
from encoder import SbertEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import utils


class WordGRU(nn.Module):
    def __init__(self, msg_hdim, msg_edim, device, vocab_size=1000):
        super().__init__()
        self.msg_hdim = msg_hdim
        self.msg_edim = msg_edim
        self.device = device

        self.emb = nn.Embedding(vocab_size, self.msg_edim, padding_idx=0)
        self.rnn = nn.GRU(self.msg_edim, self.msg_hdim, batch_first=True)

    def forward(self, messages, messages_len):  
        B, S = messages.shape
        
        embeds = torch.zeros(B, self.msg_hdim).to(self.device)
        zero_len = (messages_len == 0).squeeze()
        messages_emb = self.emb(messages)
        
        if B == 1:
            if zero_len:
                return embeds
            packed_input = pack_padded_sequence(messages_emb[~zero_len][0], messages_len[~zero_len, 0].cpu()[0], enforce_sorted=False, batch_first=True) 
        else:
            packed_input = pack_padded_sequence(messages_emb[~zero_len], messages_len[~zero_len, 0].cpu(), enforce_sorted=False, batch_first=True) 
        
        _, hidden = self.rnn(packed_input)
        
        embeds[~zero_len] = hidden[0]
        
        return embeds

class LanguageEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, device, type='gru', input_type='goal'):
        super().__init__()
        self.type = type
        self.vocab = vocab
        self.device = device
        
        assert input_type in ['goal', 'state'] # goal or state based encoder
        self.input_type = input_type
        
        if type == 'gru':  
            self.word_embedding = nn.Embedding(vocab, hidden_size)
            self.lang_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=False)
        elif type == 'pretrained_sbert':
            self.sbert_encoder = SbertEncoder(hidden_size, device)
            self.sbert_encoder.to(device)            
        elif type == 'wordgru':
            self.lang_encoder = WordGRU(hidden_size, msg_edim=64, vocab_size=vocab, device=device)
    

    def forward(self, obs):
        if self.type == 'gru':
            embedded_obs = self.word_embedding(obs)
            embedding, _ = self.lang_encoder(embedded_obs)  # shape is B x seq x hidden
            last_index = (obs != 0).sum(1).long() - 1 # How many non-padding indices are there?
            last_index = torch.maximum(last_index, torch.zeros_like(last_index))
            # TODO: THIS COMPUTATION IS WRONG IF YOU USE LANG OBS + GOAL
            B = len(last_index)
            embedding = embedding[range(B), last_index]  # Choose the embedding corresponding to the last non-padding input
        elif self.type == 'pretrained_sbert':
            embedding = self.sbert_encoder(obs)
        elif self.type == 'wordgru':
            message_len = (obs != 0).long().sum(-1, keepdim=True)
            embedding = self.lang_encoder(obs, message_len)
                              
        return embedding


class Encoder(nn.Module):

    def __init__(self, obs_shape, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim=None):
        super().__init__()
        # assert use_image or use_language, "must provide either img or lang obs"
        self.use_image = use_image
        self.use_language = use_language_state or use_goal
        self.use_language_state = use_language_state
        self.use_goal = use_goal
        # if other_dim == 0:
        #     other_dim += 1
        self.other_dim = other_dim 
        self.device = device
        self.hidden_dim = hidden_dim
        self.goal_encoder_type = goal_encoder_type
        obs_inputs = 0
        if self.use_image:
            obs_inputs += 1
            self.img_encoder = ImageEncoder(obs_shape, hidden_dim)
        if self.use_language:
            if self.use_goal: # Note: sbert 'text_obs' encoding includes goal already
                obs_inputs += 1
                if goal_encoder_type == 'sbert':
                    self.lang_goal_encoder = LanguageEncoder(vocab, hidden_dim, device, type='pretrained_sbert', input_type='goal')
                else:
                    self.lang_goal_encoder = LanguageEncoder(vocab, hidden_dim, device, type=goal_encoder_type, input_type='goal') 
            if self.use_language_state == 'sbert':
                if self.use_goal and goal_encoder_type == 'sbert':
                    pass
                else:
                    self.lang_state_encoder = LanguageEncoder(vocab, hidden_dim, device, type='pretrained_sbert', input_type='state')
                    obs_inputs += 1
            elif self.use_language_state == 'conv':
                obs_inputs += 1
                self.lang_state_encoder = ImageEncoder((1,7,9), hidden_dim, semantic=True, vocab=vocab)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * obs_inputs + int(other_dim), hidden_dim), nn.ReLU())

    def forward(self, obs):
        obs_encoded = []
        if self.use_image:
            try:
                obs_encoded.append(self.img_encoder((obs['obs'])))
            except:
                import pdb; pdb.set_trace()
        if self.use_language:
            if self.use_goal and self.goal_encoder_type == 'sbert' and self.use_language_state == 'sbert':
                state_and_goal = torch.cat([obs['text_obs'], obs['goal']], dim=-1)  
                obs_encoded.append(self.lang_goal_encoder(state_and_goal))  
            else:
                if self.use_goal:
                    obs_encoded.append(self.lang_goal_encoder(obs['goal']))
                if self.use_language_state:
                    obs_encoded.append(self.lang_state_encoder(obs['text_obs'])) 
        if self.other_dim > 0:
            if len(obs['other'].shape) == 1:
                obs_encoded.append(obs['other'].unsqueeze(-1))
            else:
                obs_encoded.append(obs['other'])
        
        obs_encoded = torch.cat(obs_encoded, -1)
        obs_encoded = self.mlp(obs_encoded)
        return obs_encoded


class ImageEncoder(nn.Module):
    def __init__(self, obs_shape, out_dim, semantic=False, vocab=None):
        super().__init__()
        self.embedding = None
        self.semantic = semantic # language embeddings instead of pixels
        if semantic:
            self.vocab = vocab
            self.feature_dim = 64 * 4 * 6
            self.convnet = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 2, stride=1), nn.ReLU(),
                nn.Conv2d(32, 64, 2, stride=1), nn.ReLU(),
                nn.Conv2d(64, 64, 2, stride=1), nn.ReLU())
        else:
            if obs_shape[-1] == 84:
                self.feature_dim = 64 * 7 * 7
            elif obs_shape[-1] == 128:
                self.feature_dim = 128 * 72
            else:
                raise NotImplementedError

            self.convnet = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU())
        self.linear = nn.Linear(self.feature_dim, out_dim)
        

    def forward(self, obs):
        if self.semantic:
            obs = obs[..., :63].view(obs.shape[0], 1, 7, 9) # obs is 7 x 9 tiles 
            # Normalize by vocab size
            obs = obs / self.vocab
        elif obs.dtype == torch.uint8:
            # Pixels
            obs = obs.type(torch.float32)
            obs = obs / 255.
        if self.embedding is not None:
            B, C, H, W = obs.shape
            obs = self.embedding(obs.int().flatten(1)).reshape((B, C, H, W, -1))
            obs = obs.moveaxis(-1, 2)
            obs = obs.reshape((B, -1, H, W))
        h = self.convnet(obs)
        h = h.reshape(h.shape[0], -1)
        h = F.normalize(h, p=2, dim=1)
        h = self.linear(h)
        return h


class Critic(nn.Module):
    def __init__(self, encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim):
        super().__init__()

        self.encoder = encoder

        self.V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, 1))
        
        self.A = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, num_actions))

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.encoder(obs)
        v = self.V(h)
        a = self.A(h)
        q = v + a - a.mean(1, keepdim=True)
        return q
    
    def forward_encoded(self, h):
        v = self.V(h)
        a = self.A(h)
        q = v + a - a.mean(1, keepdim=True)
        return q


class DQNAgent:
    def __init__(self, obs_shape, num_actions,  device, lr,
                 critic_target_tau, critic_target_update_every_steps, train_eps_min, train_eps_max,
                 train_eps_decay_steps, reward_scale, eval_eps, update_every_steps,
                 use_tb, use_wandb,  hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, other_dim, finetune_settings, 
                 other_model_prob, debug, **kwargs):
        self.num_actions = num_actions
        self.critic_target_update_every_steps = critic_target_update_every_steps
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.train_eps_min = train_eps_min
        self.train_eps_max = train_eps_max
        self.train_eps_decay_steps = train_eps_decay_steps
        self._reward_scale = reward_scale
        self.eval_eps = eval_eps
        self.device = device
        self.log = use_tb or use_wandb
        self.other_model_prob = other_model_prob
        self.metrics = {}
        self.encoder = Encoder(obs_shape, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim)
        self.critic = Critic(self.encoder, obs_shape, num_actions, hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        self.critic_target = Critic(self.encoder, obs_shape, num_actions,
                                    hidden_dim, use_image, use_language_state, use_goal, vocab, goal_encoder_type, device, other_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.decay_started = False
        self.decay_started_step = None

        if not finetune_settings == 'train_all':
            if 'linear' in finetune_settings:
                layers = [4]
            elif 'critic' in finetune_settings:
                layers = [0, 2, 4] 
            finetune_params = []
            for layer in layers:
                finetune_params += [self.critic.get_parameter(f'V.{layer}.weight'), self.critic.get_parameter(f'V.{layer}.bias'), self.critic.get_parameter(f'A.{layer}.weight'), self.critic.get_parameter(f'A.{layer}.bias')]
            self.opt = torch.optim.Adam(finetune_params, lr=lr)
        else:
            self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target.train()
        self.train()

    def new_opt(self, lr):
        self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def new_tau(self, tau):
        self.critic_target_tau = tau

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def compute_train_eps(self, step):
        if self.decay_started:
            step_unit = (self.train_eps_max - self.train_eps_min) / self.train_eps_decay_steps
            step_since_decay = step - self.decay_started_step
            train_eps = max(0, self.train_eps_max - step_unit * step_since_decay)
            return max(self.train_eps_min, train_eps)
        else:
            return self.train_eps_max

    def preprocess_obs(self, obs):
        """
        Input obs is a dictionary, with an image at 'obs' and language at 'goal'
        """
        preprocessed_obs = {}
        for k, v in obs.items():
            preprocessed_obs[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
        return preprocessed_obs

    
    def act(self, obs, step, eval_mode, other_model=None):
        # For calling act on 'other model' 
        if other_model == -1:
            obs = self.preprocess_obs(obs)
            Qs = self.critic(obs)
            action = Qs.argmax(dim=1).item()
            return action
            
        train_eps = self.compute_train_eps(step)
        eps = self.eval_eps if eval_mode else train_eps
        if np.random.rand() < eps:
            if other_model is not None and np.random.rand() < self.other_model_prob:
                action = other_model.act(obs, step, eval_mode=eval_mode, other_model=-1)
                self.rand_action = 'other'
            else:
                action = np.random.randint(self.num_actions)
                self.rand_action = 'random'
        else:
            self.rand_action = 'policy'
            obs = self.preprocess_obs(obs)
            Qs = self.critic(obs)
            action = Qs.argmax(dim=1).item()
        return action

    def update_critic(self, batch, step):
        metrics = dict()
        obs, action, reward, discount, next_obs = batch
        if not hasattr(self, '_reward_scale'):
            self._reward_scale = 1
        reward *= self._reward_scale
        
        with torch.no_grad():
            # share critic, critic_target encoder
            encoder_output = self.encoder(next_obs)
                
            next_action = self.critic.forward_encoded(encoder_output).argmax(dim=1).unsqueeze(1)
            next_Qs = self.critic_target.forward_encoded(encoder_output)
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
            metrics['train_eps'] = self.compute_train_eps(step)
            try:
                new_metrics_dict = self.log_time()
                metrics.update(new_metrics_dict)
            except Exception as e:
                pass
        
        self.opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        
        # Log the gradient norms
        if self.log:
            for name, param in self.critic.named_parameters():
                if param.grad is not None:
                    metrics['grad_norm_' + name] = param.grad.norm().item()
        
        self.opt.step()

        return metrics

    def log_time(self):
        if hasattr(self.encoder, 'lang_goal_encoder') and hasattr(self.encoder.lang_goal_encoder, 'sbert_encoder'):
            return self.encoder.lang_goal_encoder.sbert_encoder.log()
        if hasattr(self.encoder, 'lang_state_encoder') and hasattr(self.encoder.lang_state_encoder, 'sbert_encoder'):
            return self.encoder.lang_state_encoder.sbert_encoder.log()

    def update(self, replay_iter, step, use_extr_rew=False):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
            
        batch = next(replay_iter)
        batch = utils.to_torch(batch, self.device)

        self.metrics = self.update_critic(batch, step)

        if step % self.critic_target_update_every_steps == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)

        return self.metrics

    def load_and_save_cache(self):
        if hasattr(self.encoder, 'lang_goal_encoder') and hasattr(self.encoder.lang_goal_encoder, 'sbert_encoder'):
            return self.encoder.lang_goal_encoder.sbert_encoder.load_and_save_cache()
        if hasattr(self.encoder, 'lang_state_encoder') and hasattr(self.encoder.lang_state_encoder, 'sbert_encoder'):
            return self.encoder.lang_state_encoder.sbert_encoder.load_and_save_cache()
