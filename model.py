import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
import math
from kmeans import cached_kmeans

import numpy as np

class FixupResNetCNN(nn.Module):
    """source: https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/model.py"""

    class _FixupResidual(nn.Module):
        def __init__(self, depth, num_residual):
            super().__init__()
            self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            for p in self.conv1.parameters():
                p.data.mul_(1 / math.sqrt(num_residual))
            for p in self.conv2.parameters():
                p.data.zero_()
            self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

        def forward(self, x):
            x = F.relu(x)
            out = x + self.bias1
            out = self.conv1(out)
            out = out + self.bias2
            out = F.relu(out)
            out = out + self.bias3
            out = self.conv2(out)
            out = out * self.scale
            out = out + self.bias4
            return out + x

    def __init__(self, input_channels, double_channels=False):
        super().__init__()
        depth_in = input_channels

        layers = []
        if not double_channels:
            channel_sizes = [32, 64, 64]
        else:
            channel_sizes = [64, 64, 128]
        for depth_out in channel_sizes:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self._FixupResidual(depth_out, 8),
                self._FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            self._FixupResidual(depth_in, 8),
            self._FixupResidual(depth_in, 8),
        ])
        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.output_size = math.ceil(64 / 8) ** 2 * depth_in

    def forward(self, x):
        return self.conv_layers(x)


class InputProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layers = FixupResNetCNN(3,double_channels=True)
        self.spatial_reshape = nn.Sequential(nn.Linear(128*8*8, 448),nn.ReLU(),nn.LayerNorm(448)) ##
        self.nonspatial_reshape = nn.Sequential(nn.Linear(66,64),nn.ReLU(),nn.LayerNorm(64))

    def forward(self, spatial, nonspatial):
        shape = spatial.shape
        spatial = spatial.view((shape[0]*shape[1],)+shape[2:])/255.0
        #print(spatial.shape)
        spatial = self.conv_layers(spatial)
        #print(spatial.shape)
        new_shape = spatial.shape
        spatial = spatial.view(shape[:2]+(-1,))
        nonspatial = self.nonspatial_reshape(nonspatial)

        #print(spatial.shape)

        spatial = self.spatial_reshape(spatial)

        #print(spatial.shape)

        #print(nonspatial.shape)

        return torch.cat([spatial, nonspatial],dim=-1)


class Core(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.input_proc = InputProcessor()
        self.lstm = nn.LSTM(512, 512, 1)
        
    def forward(self, spatial, nonspatial, state):
        #print('start forward')
        processed = self.input_proc.forward(spatial, nonspatial)
        #print('Processed shape: ', processed.shape)
        #print(state[0].shape)
        #exit(0)
        lstm_output, new_state = self.lstm(processed, state)
        #print('lstm end')
        # exit(0)
        return lstm_output+processed, new_state


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.kmeans = cached_kmeans("train","MineRLObtainDiamondVectorObf-v0")
        self.core = Core()
        self.selector = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 150))

    def get_zero_state(self, batch_size, device="cuda"):
        return (torch.zeros((1, batch_size, 512), device=device), torch.zeros((1, batch_size, 512), device=device))

    def compute_front(self, spatial, nonspatial, state):
        #print('start self core')
        hidden, new_state = self.core(spatial, nonspatial, state)
        #print('end self core')
        return hidden, self.selector(hidden), new_state

    def forward(self, spatial, nonspatial, state, target):
        pass

    def get_loss(self, spatial, nonspatial, prev_action, state, target, point):
        loss = nn.CrossEntropyLoss()
        #print('loss done')
        hidden, d, state = self.compute_front(spatial, nonspatial, state)
        #print('compute front done')
        l1 = loss(d.view(-1, d.shape[-1]), point.view(-1))
        return l1, {"action":l1.item()}, state

    def sample(self, spatial, nonspatial, prev_action, state, target):
        hidden, d, state = self.compute_front(spatial, nonspatial, state)
        dist = D.Categorical(logits = d)
        s = dist.sample()
        s = s.squeeze().cpu().numpy()
        return self.kmeans.cluster_centers_[s], state
