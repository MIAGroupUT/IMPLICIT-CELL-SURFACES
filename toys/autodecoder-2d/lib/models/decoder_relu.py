#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
import numpy as np
from lib.utils import *


class DeepSDF(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xy_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        positional_encoding = False,
        fourier_degree = 1
    ):
        super(DeepSDF, self).__init__()

        def make_sequence():
            return []
        if positional_encoding is True:
            dims = [latent_size + 2*fourier_degree*3] + dims + [1]
        else:
            dims = [latent_size + 2] + dims + [1]

        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.theta = () #torch.tensor(0.33, dtype=torch.float32, requires_grad=True).cuda()
        #if not torch.is_tensor(theta):
        #    self.theta = torch.tensor(theta, dtype=torch.float32).cuda()
        #else:
        #    self.theta = theta.cuda()
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xy_in_all = xy_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xy_in_all and layer != self.num_layers - 2:
                    out_dim -= 2

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()
    
    # get rotation matrix
    def R(self, theta):
        
        return torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta), torch.cos(theta)]],
                            dtype=torch.float32).cuda()
        #print('THETA', self.theta)
    
    # input: N x (L+3)
    def forward(self, latent, theta, xy):
        
        # rotate a set of coordinates by theta
        # (inverse rotation by transposed matrix
        # of coordinates results in desired counterclockwise
        # rotation of the shape sdf)
        #self.theta = theta
        xy = xy @ self.R(theta).T

        if self.positional_encoding:
            xy = fourier_transform(xy, self.fourier_degree)
        input = torch.cat([latent, xy.cuda()], dim=1)

        if input.shape[1] > 2 and self.latent_dropout:
            latent_vecs = input[:, :-2]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xy], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xy_in_all:
                x = torch.cat([x, xy], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x