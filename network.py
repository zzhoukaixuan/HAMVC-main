import torch.nn as nn
from torch.nn.functional import normalize
import torch
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, dims, bn = False):
        super(Encoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
        self.models = nn.Sequential(*models)
    def forward(self, X):
        features = []
        for layer in self.models:
            X = layer(X)
            if isinstance(layer, nn.Linear):
                features.append(X)
        return features
    
class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
        self.models = nn.Sequential(*models)
    
    def forward(self, X):
        return self.models(X)
    
class MyNet(nn.Module):
    def __init__(self, input_dims, embedding_dim, view_num, h_dims, fusion_indices, steps, alpha, temperature, device):
        super().__init__()
        self.input_dims = input_dims
        self.view = view_num
        self.embedding_dim = embedding_dim
        self.h_dims = h_dims
        self.device = device
        self.fusion_indices = fusion_indices
        self.steps = steps
        self.encoders = []
        self.decoders = []
        self.alpha = alpha
        self.temperature = temperature
        
        h_dims_reverse = list(reversed(h_dims))
        for v in range(self.view):
            self.encoders.append(Encoder([input_dims[v]] + h_dims + [embedding_dim], bn=False).to(device))
            self.decoders.append(Decoder([embedding_dim] + h_dims_reverse + [input_dims[v]]).to(device)) 
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        

    def forward(self, xs):
        xrs = []
        zs = []
        selected_features_list = []      
        for v in range(self.view):
            x = xs[v]
            encoder_features = self.encoders[v](x)
            selected_features = [encoder_features[idx] for idx in self.fusion_indices]
            z = self.cross_layer_fusion(selected_features, steps=self.steps, alpha=self.alpha, temperature=self.temperature)
            xr = self.decoders[v](encoder_features[-1])

            selected_features_list.append(selected_features)
            xrs.append(xr)
            zs.append(z)

        return xrs, zs, selected_features_list
    
    def cross_layer_fusion(self, selected_features, steps, alpha, temperature):
        fused_feature = selected_features[0]
        for i in range(1, len(selected_features)):
            current_feature = selected_features[i]
            diffuse_feat = self.affinity_diffuse(current_feature, fused_feature, steps, temperature) @ current_feature
            fused_feature = alpha * diffuse_feat + (1 - alpha) * current_feature        
        return fused_feature
    
    def affinity_diffuse(self, target, source, steps, temperature=0.1):
        target = F.normalize(target, p=2, dim=1)
        source = F.normalize(source, p=2, dim=1)
        dist = 2 - 2 * (target @ source.t())
        G = torch.exp(-dist / temperature)
        G = G / G.sum(dim=1, keepdim=True)
        G = torch.matrix_power(G, steps)
        
        return G
