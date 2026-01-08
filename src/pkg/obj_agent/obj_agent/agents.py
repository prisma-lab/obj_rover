import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from obj_agent.minigrid_obss_utils import *

#WORD_EMBEDDING_SIZE = 1024
max_sentence_length = 15
vocab_max_size = 100
WORD_EMBEDDING_SIZE = 1024
and_token = -1

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init_no_bias(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    return layer



class ObsMissionAttentionLayer(nn.Module):
    # We query the obs (image + hidden state of lstm) on the mission in order to change the focus on the different subtask w.r.t. what we see on the environment
    def __init__(self, word_embedding_size, d_k, obs_size):
        super(ObsMissionAttentionLayer, self).__init__()
        self.scores = None
        self.word_embedding_size = word_embedding_size
        self.obs_size = obs_size
        self.d_k = d_k
        self.key = layer_init(nn.Linear(self.obs_size, self.d_k))
        self.query = layer_init(nn.Linear(self.word_embedding_size, self.d_k))

    def get_word_attention_maps(self, state, mission, mask=None):
        #img_reshape = torch.reshape(state, (state.shape[0], state.shape[1], state.shape[2]*state.shape[3]))
        #img = img_reshape.transpose(1, 2)
        #print(f"mission {mission.shape}")
        #print(query_hidden[0].shape)
        #print(state.shape)
        img_reshape = torch.reshape(state, (state.shape[0], state.shape[1], state.shape[2]*state.shape[3]))
        img = img_reshape.transpose(1, 2)

        queries = self.query(mission)
        keys = self.key(img)
        #values = self.value(img)
        #self.cached_keys = keys
        self.scores = torch.matmul(queries, keys.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        #if mask is not None:
        #    self.scores = self.scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attention_weights = F.softmax(self.scores, dim=-1)
        #attention_weights = attention_weights * mask.unsqueeze(-1).expand(mask.shape[0], mask.shape[1], attention_weights.shape[-1])
        attention_entropies = -Categorical(probs=attention_weights).entropy()
        attention_entropies = attention_entropies.masked_fill(mask == 0, -1e9)
        word_attention_weights = F.softmax(attention_entropies, dim=-1)

        return attention_weights, word_attention_weights

    def get_attention_weights(self, state, mission, mask=None):
        img_reshape = torch.reshape(state, (state.shape[0], state.shape[1], state.shape[2]*state.shape[3]))
        img = img_reshape.transpose(1, 2)
        return self.get_word_attention_maps(img, mission, mask)

    # TODO: Questa funzione non è più necessaria, ma la lascio per ora per comodità
    def img_to_text_attention(self, state, mission, mask=None, completed_tasks=None, task_attention_module=None):
        img_reshape = torch.reshape(state, (state.shape[0], state.shape[1], state.shape[2]*state.shape[3]))
        img = img_reshape.transpose(1, 2)

        attention_weights, word_attention_weights = self.get_word_attention_maps(img, mission, mask)

        task_attention_weights = None
        #print(self.task_attention_module)
        if task_attention_module and completed_tasks:
            task_attention_weights = task_attention_module(mission, word_attention_weights, completed_tasks)
            word_attention_weights = word_attention_weights * task_attention_weights
        attention_mask = torch.sum(attention_weights * word_attention_weights.unsqueeze(-1), dim=1)

        output = img * attention_mask.unsqueeze(-1)
        return output, attention_weights, word_attention_weights, task_attention_weights

class InternalMemoryModule(nn.Module):
    def __init__(self, obs_size, lstm_emb_size = 128, num_layers = 1):
        super(InternalMemoryModule, self).__init__()
        self.obs_size = obs_size
        self.lstm_emb_size = lstm_emb_size
        self.num_layers = num_layers
        self.state_lstm = nn.LSTM(self.obs_size, self.lstm_emb_size, batch_first=False, num_layers=self.num_layers)
        for name, param in self.state_lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def forward(self, state, done, internal_memory_hidden):
        batch_size = internal_memory_hidden[0].shape[1]
        state_reshape = torch.reshape(state, (-1, batch_size, self.obs_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(state_reshape, done):
            h, internal_memory_hidden = self.state_lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * internal_memory_hidden[0],
                    (1.0 - d).view(1, -1, 1) * internal_memory_hidden[1],
                ),
            )
            new_hidden += [h]
        
        #print(f"new_hidden {torch.cat(new_hidden).shape}")
        state = torch.flatten(torch.cat(new_hidden), 0, 1)
        #print(state.shape)
        return state, internal_memory_hidden

def init_params(m):
    #classname = m.__class__.__name__
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class Agent(nn.Module):
    def __init__(self, n_actions=7, kernel_size=1, stride=1, padding=0, include_3x3_network=True, use_direction=False):
        super().__init__()
        self.feature_maps_size = 7*7#((n-1)//2-2)*((m-1)//2-2)
        self.K_out = 128
        self.image_embedding_size = self.feature_maps_size*self.K_out
        self.d_k = 32
        self.obj_task_attention_weights = None
        self.color_task_attention_weights = None
        self.include_3x3_network = include_3x3_network
        self.use_direction = use_direction

        if use_direction:
            self.dir_emb_dim = 8
            #self.dir_embedding = nn.Sequential(nn.Embedding(4, self.dir_emb_dim), nn.LayerNorm(self.dir_emb_dim))
        '''
        self.conv_network = nn.Sequential(layer_init_no_bias(nn.Conv2d(3, 32, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                                         nn.BatchNorm2d(32),
                                         #nn.GroupNorm(num_groups=4, num_channels=32),
                                         nn.ReLU(),
                                         layer_init_no_bias(nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                                         nn.BatchNorm2d(64),
                                         #nn.GroupNorm(num_groups=4, num_channels=64),
                                         nn.ReLU(),
                                         layer_init_no_bias(nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                                         nn.BatchNorm2d(64),
                                         #nn.GroupNorm(num_groups=4, num_channels=64),
                                         nn.ReLU(),
                                         layer_init_no_bias(nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                                         nn.BatchNorm2d(64),
                                         #nn.GroupNorm(num_groups=4, num_channels=64),
                                         nn.ReLU(),
                                         layer_init_no_bias(nn.Conv2d(64, self.K_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                                         nn.BatchNorm2d(self.K_out),
                                         #nn.GroupNorm(num_groups=4, num_channels=self.K_out),
                                         nn.ReLU(),
        )
        '''
        self.conv_network = nn.Sequential(layer_init_no_bias(nn.Conv2d(3, 128, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                                         nn.BatchNorm2d(128),
                                         #nn.LayerNorm([128, 7, 7]),
                                         #nn.ReLU(),
                                         nn.SiLU(),
                                         #nn.GroupNorm(num_groups=4, num_channels=32),
                                         layer_init_no_bias(nn.Conv2d(128, 128, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                                         nn.BatchNorm2d(128),
                                         #nn.GroupNorm(num_groups=4, num_channels=64),
                                         #nn.LayerNorm([128, 7, 7]),
                                         #nn.ReLU(),
                                         nn.SiLU(),
                                         #layer_init_no_bias(nn.Conv2d(128, 128, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                                         #nn.BatchNorm2d(128),
                                         #nn.GroupNorm(num_groups=4, num_channels=64),
                                         #nn.ReLU(),
                                         layer_init_no_bias(nn.Conv2d(128, self.K_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                                         nn.BatchNorm2d(self.K_out),
                                         #nn.GroupNorm(num_groups=4, num_channels=self.K_out),
                                         #nn.LayerNorm([self.K_out, 7, 7]),
                                         #nn.ReLU(),
                                         nn.SiLU(),
        )
        
        '''
        self.conv_network = nn.Sequential(
            layer_init(nn.Conv2d(3, 128, (3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            layer_init(nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            #nn.MaxPool2d((2, 2), stride=2),
            layer_init(nn.Conv2d(128, self.K_out, (3, 3), padding=1)),
            nn.BatchNorm2d(self.K_out),
            nn.ReLU(),
            #nn.MaxPool2d((2, 2), stride=2),
        )
        '''
        if include_3x3_network:
            self.conv_embedding_network = nn.Sequential(layer_init_no_bias(nn.Conv2d(self.K_out, 128, kernel_size=3, stride=1, padding=1, bias=False)),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(),
                                            #layer_init_no_bias(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)),
                                            #nn.BatchNorm2d(128),
                                            #nn.ReLU(),
                                            #layer_init_no_bias(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                                            #nn.BatchNorm2d(64),
                                            #nn.ReLU(),
                                            layer_init_no_bias(nn.Conv2d(128, self.K_out, kernel_size=3, stride=1, padding=1, bias=False)),
                                            nn.BatchNorm2d(self.K_out),
                                            nn.ReLU(),
            )
            
        
        self.task_attention_module = None

        self.word_embedding_size = WORD_EMBEDDING_SIZE
        self.text_dim = self.word_embedding_size
        self.GRU_hidden_size = 256

        self.word_embedding = nn.Sequential(nn.Embedding(vocab_max_size, self.word_embedding_size), nn.LayerNorm(self.word_embedding_size))
        #self.word_embedding = nn.Embedding(vocab_max_size, self.word_embedding_size)
        self.film_blocks = []

        self.obs_mission_attention_obs_dim = self.K_out
        
        self.attention = ObsMissionAttentionLayer(word_embedding_size=self.text_dim, d_k=self.d_k, obs_size=self.obs_mission_attention_obs_dim)

        self.internal_memory_input_size = self.image_embedding_size + self.dir_emb_dim if self.use_direction else self.image_embedding_size
        self.internal_memory_hidden_size = self.actor_critic_in_dim = 128
        self.internal_memory = InternalMemoryModule(obs_size=self.internal_memory_input_size, lstm_emb_size=self.internal_memory_hidden_size)

        
        self.text_rnn = nn.GRU(self.text_dim, self.GRU_hidden_size, batch_first=True, bidirectional=False)

        self.actor = nn.Sequential(layer_init(nn.Linear(self.actor_critic_in_dim + self.GRU_hidden_size, 64)),
                                   nn.Tanh(),
                                   layer_init(nn.Linear(64, n_actions)))
        self.critic = nn.Sequential(layer_init(nn.Linear(self.actor_critic_in_dim + self.GRU_hidden_size, 64)),
                                    nn.Tanh(),
                                    layer_init(nn.Linear(64, 1)))

        self.obj_text_attention_weights = None
        self.obj_word_attention_weights = None
        self.color_text_attention_weights = None
        self.color_word_attention_weights = None

    def get_word_embedding(self, text):
        return self.word_embedding(text)

    def get_states(self, x, state_hidden, done, mask=None, completed_tasks=None, **kwargs):
        img, mission = x['image'], x['text']
        emb = self.word_embedding(mission)

        img_for_conv = img.transpose(1, 3).transpose(2, 3).float()
        
        conv = self.conv_network(img_for_conv)

        self.text_attention_weights, self.word_attention_weights = self.attention.get_word_attention_maps(conv, emb, mask)#self.obj_attention.img_to_text_attention(obj_conv, emb, mask, completed_tasks=completed_tasks)
        
        self.task_attention_weights = None
        if self.task_attention_module:
            self.task_attention_weights = self.task_attention_module(emb, self.word_attention_weights, completed_tasks)
            self.word_attention_weights = self.word_attention_weights * self.task_attention_weights

        attention_map = torch.sum(self.text_attention_weights * self.word_attention_weights.unsqueeze(-1), dim=1)

        img_reshape = torch.reshape(conv, (conv.shape[0], conv.shape[1], conv.shape[2]*conv.shape[3]))
        img_reshape = img_reshape.transpose(1, 2)

        F_masked = F_refined = img_reshape * attention_map.unsqueeze(-1)
        #F_masked = torch.reshape(F_masked, (combined.shape[0], combined.shape[1], combined.shape[2], combined.shape[3]))
        #F_refined = self.combined_network_residual(F_masked)
        #F_refined = torch.reshape(F_refined, (F_refined.shape[0], F_refined.shape[1], F_refined.shape[2]*F_refined.shape[3]))
        #F_refined = F_refined.transpose(1, 2)
        output = img_reshape + F_refined
        
        if self.include_3x3_network:
            output = output.transpose(1, 2)
            output = self.conv_embedding_network(torch.reshape(output, ((output.shape[0], conv.shape[1], conv.shape[2], conv.shape[3]))))
        '''
        else:
            B, C, H, W = F_masked.shape

            x_coords = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(B, 1, H, W)
            y_coords = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, 1, H, W)

            output = torch.cat([F_masked, x_coords.to(F_masked.device), y_coords.to(F_masked.device)], dim=1)
        '''
        output = output.reshape((output.shape[0], -1))
        if self.use_direction:
            direction = x['direction']
            #dir_embed = F.one_hot(direction.long(), num_classes=4).float()
            #dir_embed = self.dir_embedding(x['direction'].long())
            dir_embed = direction.unsqueeze(1).repeat(1, self.dir_emb_dim)
            output = torch.cat((output, dir_embed), dim=-1)
            
            
            #output = torch.cat((output, dir_embed), dim=-1)

        output, state_hidden = self.internal_memory(output, done, state_hidden)
        
        _, rnn_embeddings = self.text_rnn(emb)
        return output, state_hidden, emb, rnn_embeddings.squeeze(0)

    def get_value(self, x, state_hidden, done, mask=None, completed_tasks=None, **kwargs):
        hidden, _, _, rnn_embeddings = self.get_states(x, state_hidden, done, mask, completed_tasks=completed_tasks)
        return self.critic(torch.cat((hidden, rnn_embeddings), dim=-1))

    def get_action_and_value(self, x, state_hidden, done, action=None, mask=None, completed_tasks=None, action_mask=None, **kwargs):
        hidden, state_hidden, context, rnn_embeddings = self.get_states(x, state_hidden, done, mask, completed_tasks=completed_tasks)
        logits = self.actor(torch.cat((hidden, rnn_embeddings), dim=-1)).squeeze()
        #print(action_mask)
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, device=logits.device, dtype=torch.int32)
            logits[action_mask == 0] = float('-inf')
        probs = Categorical(logits=logits)
        print(probs.probs)
        if action is None:
            action = probs.sample()

        return torch.tensor([action]), probs.log_prob(action), probs.entropy(), self.critic(torch.cat((hidden, rnn_embeddings), dim=-1)), state_hidden#, task_finished


class ObsMissionTaskAttentionLayer(ObsMissionAttentionLayer):
    def __init__(self, word_embedding_size, d_k, obs_size):
        super(ObsMissionTaskAttentionLayer, self).__init__(word_embedding_size, d_k, obs_size)
        self.task_attention_module = None
    
    def img_to_text_attention(self, state, mission, mask=None, completed_tasks=None):
        img_reshape = torch.reshape(state, (state.shape[0], state.shape[1], state.shape[2]*state.shape[3]))
        img = img_reshape.transpose(1, 2)

        attention_weights, word_attention_weights = self.get_word_attention_maps(img, mission, mask)

        task_attention_weights = None
        #print(self.task_attention_module)
        if self.task_attention_module is not None:
            task_attention_weights = self.task_attention_module(mission, word_attention_weights, completed_tasks)
            word_attention_weights = word_attention_weights * task_attention_weights
        attention_mask = torch.sum(attention_weights * word_attention_weights.unsqueeze(-1), dim=1)

        output = img * attention_mask.unsqueeze(-1)
        return output, attention_weights, word_attention_weights, task_attention_weights


class TaskAttentionModule(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks
        self.task_attention = nn.Sequential(layer_init(nn.Linear(WORD_EMBEDDING_SIZE + self.n_tasks, 128)),
                                            nn.ReLU(),
                                            layer_init(nn.Linear(128, 128)),
                                            nn.ReLU(),
                                            layer_init(nn.Linear(128, max_sentence_length))
                                            )
        #self.apply(init_params)
    
    def forward(self, word_embeddings, word_attention_weights, completed_tasks):
        weighted_sum = torch.sum(word_embeddings * word_attention_weights.unsqueeze(-1), dim=1)
        model_output = self.task_attention(torch.cat((weighted_sum, completed_tasks), dim=-1))
        return F.sigmoid(model_output)
    