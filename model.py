import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from tqdm import tqdm
import copy
from tqdm.auto import tqdm
import numpy as np

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, t):
        device = t.device
        half_dim = self.feature_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        scaled_t = t * 1000.0 
        embeddings = scaled_t.unsqueeze(1) * embeddings
        sin_embeddings = torch.sin(embeddings)
        cos_embeddings = torch.cos(embeddings)
        full_embeddings = torch.cat((sin_embeddings, cos_embeddings), dim=-1)
        if self.feature_dim % 2 == 1:
            full_embeddings = torch.cat([full_embeddings, torch.zeros_like(full_embeddings[:, :1])], dim=-1)
            
        return full_embeddings

class RandomFeatureModel(nn.Module):
    def __init__(self, input_dim, feature_dim, K_t, T=1.0, activation='relu'):
        super().__init__()
        self.input_dim = input_dim  
        self.feature_dim = feature_dim  
        self.K_t = K_t
        self.T = T
        self.activation_name = activation
        self.A = nn.Parameter(torch.randn(input_dim, feature_dim))
        self.register_buffer('W_x', torch.randn(feature_dim, input_dim) / math.sqrt(input_dim))
        self.register_buffer('W_t', torch.randn(feature_dim, 2 * K_t + 1) / math.sqrt(2 * K_t + 1))
        self.register_buffer('b', torch.randn(feature_dim))
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            self.activation = F.relu  # Default to ReLU
            
    def time_parametrization(self, t):
        t_normalized = t / self.T
        batch_size = t.shape[0]
        phi = torch.zeros(batch_size, 2 * self.K_t + 1, device=t.device)
        phi[:, 0] = 1
        for k in range(1, self.K_t + 1):
            phi[:, 2*k-1] = torch.sin(2 * math.pi * k * t_normalized)
            phi[:, 2*k] = torch.cos(2 * math.pi * k * t_normalized)
            
        return phi

    def forward(self, x, t):
        batch_size = x.shape[0]
        phi = self.time_parametrization(t)  
        x_proj = torch.matmul(x, self.W_x.t())   
        t_proj = torch.matmul(phi, self.W_t.t()) 
        pre_activation = x_proj + t_proj + self.b.unsqueeze(0).expand(batch_size, -1)
        activated_features = self.activation(pre_activation)
        output = torch.matmul(activated_features, self.A.t()) / math.sqrt(self.feature_dim)
        
        return output

## 使用正弦编码的 RFM
class RandomFeatureModelSine(nn.Module):
    def __init__(self, input_dim, feature_dim, time_emb_dim, scaling=1):
        super().__init__()
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
        self.feature_dim = feature_dim
        self.scaling = scaling
        random_weights = torch.randn(input_dim + time_emb_dim, feature_dim) * scaling
        self.register_buffer('random_weights', random_weights)
        self.output_layer = nn.Linear(feature_dim, input_dim, bias=False)
        self.time_embed_net = SinusoidalTimeEmbedding(time_emb_dim)
        self.b = nn.Linear(input_dim, input_dim, bias=False)
        self.c = nn.Parameter(torch.zeros(input_dim))

    def get_random_features(self, x, t):
        t_emb = self.time_embed_net(t)
        combined_input = torch.cat([x, t_emb], dim=-1)
        projection = torch.matmul(combined_input, self.random_weights)
        features = F.relu(projection) * math.sqrt(1.0 / self.feature_dim)
        
        return features

    def forward(self, x, t):
        features = self.get_random_features(x, t)
        score = self.output_layer(features)
        return score 