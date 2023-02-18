import escnn
from escnn.group import so3_group
from escnn.gspaces import no_base_space
from escnn.nn import FieldType, SequentialModule, QuotientFourierELU, Linear, IIDBatchNorm1d, GeometricTensor, tensor_directsum


class EquiCoder(escnn.nn.EquivariantModule):
    def __init__(self, latent_code_length, num_layers=4, num_latent_channels=4):
        super().__init__()
        nl, nlc = num_layers, num_latent_channels

        symmetry_group = so3_group()
        symmetry_group_space = no_base_space(symmetry_group)

        '''
        We divide the latent code into 3-vectors and remaining scalars. Let, e.g.,

            latent_code = (c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8).

        Dividing its elements into vectors of three and remainders, we get

            (c_1, c_2, c_3), (c_4, c_5, c_6), c_7, c_8.

        Vectors can be rotated by elements of SO(3) and scalars stay unaffected.
        '''

        self.in_type = FieldType(symmetry_group_space, [
            *latent_code_length // 3 * [symmetry_group.irrep(1)],  # latent code 3-vectors
            *latent_code_length % 3 * [symmetry_group.irrep(0)],  # latent code remaining scalars
            symmetry_group.irrep(1),  # spatial coordinates
            symmetry_group.irrep(0)  # temporal coordinate
        ])
        self.out_type = FieldType(symmetry_group_space, [symmetry_group.irrep(0)])  # signed distance value

        # SO(3)-equivariant multilayer perceptron
        self.layers = SequentialModule()
        layer_in_type = self.in_type

        for num_channels, rep_band_limit_frequency, num_sphere_samples in zip(nl * [nlc], nl * [3], nl * [128]):
            act = QuotientFourierELU(
                symmetry_group_space,
                subgroup_id=(False, -1),
                channels=num_channels,
                irreps=symmetry_group.bl_sphere_representation(L=rep_band_limit_frequency).irreps,
                grid=symmetry_group.sphere_grid(type='thomson', N=num_sphere_samples),
                inplac= True
            )

            self.layers.append(SequentialModule(Linear(layer_in_type, act.in_type), IIDBatchNorm1d(act.in_type), act))
            layer_in_type = self.layers[-1].out_type

        self.layers.append(Linear(self.layers[-1].out_type, self.out_type))

    # Abstract method in "escnn.nn.EquivariantModule"
    def evaluate_output_shape(self, input_shape):
        shape = list(input_shape)

        assert len(shape) == 2, shape
        assert shape[1] == self.in_type.size, shape

        shape[1] = self.out_type.size

        return shape

    def forward(self, latent_code, xyzt):
        latent_code = GeometricTensor(latent_code, FieldType(self.in_type.gspace, self.in_type.representations[:-2]))
        xyzt = GeometricTensor(xyzt, FieldType(self.in_type.gspace, self.in_type.representations[-2:]))

        return self.layers(tensor_directsum((latent_code, xyzt))).tensor


# import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
# 
# import torch
# from emlp.reps import Vector, Scalar
# from emlp.groups import SO
# from emlp.nn.pytorch import EMLP
# 
# 
# class EquiCoder(torch.nn.Module):
#     def __init__(self, latent_code_length=4, num_layers=4, num_hidden_channels=26):
#         super().__init__()
#         self.latent_code_length = latent_code_length
# 
#         '''
#         We divide the latent code into 3-vectors and remaining scalars. Let, e.g.,
# 
#             latent_code = (a, b, c, d, e, f, g, h).
# 
#         Dividing its elements into vectors of three and remainders, we get
# 
#             (a, b, c), (d, e, f), g, h.
# 
#         Vectors can be rotated by elements of SO(3) and scalars stay unaffected.
#         '''
# 
#         in_rep = sum((
#             self.latent_code_length // 3 * Vector,  # latent code 3-vectors
#             self.latent_code_length % 3 * Scalar,  # latent code remaining scalars
#             Vector,  # spatial coordinates
#             Scalar  # temporal coordinate
#         ))
#         out_rep = Scalar  # signed distance value
# 
#         self.mlp = EMLP(
#             in_rep,
#             out_rep,
#             group=SO(3),
#             ch=num_hidden_channels,  # must be larger than six for non-trivial representation to appear in hidden layers
#             num_layers=num_layers
#         ).to(torch.device('cuda'))
# 
#     def forward(self, latent_code, xyzt):
#         return self.mlp(torch.cat((latent_code, xyzt), dim=-1))


# Copied and adapted from "neural implicit reconstruction with vector neurons" (https://github.com/FlyingGiraffe/vnn-neural-implicits)
import torch
import torch.nn as nn
import torch.nn.functional as F


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.actvn(self.fc_0(x))  # [CHANGED] layer order
        dx = self.fc_1(net)  # [CHANGED] layer order

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return self.actvn(x_s + dx)  # [CHANGED] layer order


class DecoderInner(nn.Module):
    ''' Decoder class.
    It does not perform any form of normalization.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=128, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if z_dim > 0:
            self.z_in = VNLinear(z_dim//dim, z_dim//dim)
        if c_dim > 0:
            self.c_in = VNLinear(c_dim//dim, c_dim//dim)

        self.fc_in = nn.Linear(z_dim//dim*2+c_dim//dim*2+1 + 1, hidden_size)  # [CHANGED] include temporal coordinate

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, z, p, c=None, **kwargs):  # [CHANGED] p <-> z
        p, t = p[:, None, :3], p[:, None, -1:]  # [ADDED] expand dimensions and separate temporal coordinate
        batch_size, T, D = p.size()

        z = z[:, slice(z.size(-1) // D * D)]  # [ADDED] discard remaining scalars in latent code

        net = (p * p).sum(2, keepdim=True)

        if self.z_dim != 0:
            z = z.view(batch_size, -1, D).contiguous()
            net_z = torch.einsum('bmi,bni->bmn', p, z)
            z_dir = self.z_in(z)
            z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_z, z_inv], dim=2)

        if self.c_dim != 0:
            c = c.view(batch_size, -1, D).contiguous()
            net_c = torch.einsum('bmi,bni->bmn', p, c)
            c_dir = self.c_in(c)
            c_inv = (c * c_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_c, c_inv], dim=2)

        net = self.actvn(self.fc_in(torch.cat((net, t), dim=-1)))  # [CHANGED] include temporal coordinate | layer order

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(net)  # [CHANGED] layer order
        out = out.squeeze(-1)

        return out
