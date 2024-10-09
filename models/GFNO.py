"""
Adopted from:
@author: Jacob Helwig, Xuan Zhang, AIRS Lab TAMU
https://github.com/divelab/AIRS/tree/main/OpenPDE/G-FNO
"""

import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import math
from .utils import format_tensor_size

class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, first_layer=False, last_layer=False,
                 spectral=False, Hermitian=False, reflection=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = torch.cfloat if spectral else torch.float
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.Hermitian = Hermitian
        if first_layer or last_layer:
            self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.kernel_size_Y, self.kernel_size_X, dtype=dtype))
        else:
            if self.Hermitian:
                self.W = nn.ParameterDict({
                    'y0_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_X - 1, 1, dtype=dtype)),
                    'yposx_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X - 1, dtype=dtype)),
                    '00_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, 1, 1, dtype=torch.float))
                })
            else:
                self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X, dtype=dtype))
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1)) if bias else None
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.Hermitian:
            self.weights = torch.cat([self.W["y0_modes"], self.W["00_modes"].cfloat(), self.W["y0_modes"].flip(dims=(-2, )).conj()], dim=-2)
            self.weights = torch.cat([self.weights, self.W["yposx_modes"]], dim=-1)
            self.weights = torch.cat([self.weights[..., 1:].conj().rot90(k=2, dims=[-2, -1]), self.weights], dim=-1)
        else:
            self.weights = self.W[:]

        if self.first_layer or self.last_layer:
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1)
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, dims=[-2, -1])
            if self.reflection:
                self.weights[:, self.rt_group_size:] = self.weights[:, :self.rt_group_size].flip(dims=[-2])
            if self.first_layer:
                self.weights = self.weights.view(-1, self.in_channels, self.kernel_size_Y, self.kernel_size_Y)
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(self.out_channels, -1, self.kernel_size_Y, self.kernel_size_Y)
                self.bias = self.B

        else:
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-2, -1])

                if self.reflection:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                                                    self.weights[:, k, :, :(self.rt_group_size - 1)],
                                                    self.weights[:, k, :, (self.rt_group_size + 1):],
                                                    self.weights[:, k, :, self.rt_group_size].unsqueeze(2)], dim=2)
                else:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, -1].unsqueeze(2), self.weights[:, k, :, :-1]], dim=2)

            if self.reflection:
                self.weights[:, self.rt_group_size:] = torch.cat(
                    [self.weights[:, :self.rt_group_size, :, self.rt_group_size:],
                     self.weights[:, :self.rt_group_size, :, :self.rt_group_size]], dim=3).flip([-2])
            self.weights = self.weights.view(self.out_channels * self.group_size, self.in_channels * self.group_size,
                                             self.kernel_size_Y, self.kernel_size_Y)
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_X:]

    def forward(self, x):

        self.get_weight()
        x = nn.functional.conv2d(input=x, weight=self.weights)
        if self.B is not None:
            x = x + self.bias
        return x


class GSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, reflection=False):
        super(GSpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.conv = GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * modes - 1,
                            reflection=reflection, bias=False, spectral=True, Hermitian=True)
        self.get_weight()

    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()

        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes]
        out_ft = torch.zeros(batchsize, self.weights.shape[0], x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes] = \
            self.compl_mul2d(x_ft, self.weights)

        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))
        return x

class GMLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, reflection=False, last_layer=False):
        super(GMLP2d, self).__init__()
        self.mlp1 = GConv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, reflection=reflection)
        self.mlp2 = GConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, reflection=reflection,
                            last_layer=last_layer)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class GNorm(nn.Module):
    def __init__(self, width, group_size):
        super().__init__()
        self.group_size = group_size
        self.norm = torch.nn.InstanceNorm3d(width)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.group_size, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return x
        
        
class grid(torch.nn.Module):
    def __init__(self, grid_type):
        super(grid, self).__init__()
        assert grid_type in ["cartesian", "symmetric"], "Invalid grid type"
        self.symmetric = grid_type == "symmetric"
        self.grid_dim = (1 + (not self.symmetric))
        self.get_grid = self.twoD_grid
    def forward(self, x):
        return self.get_grid(x)

    def twoD_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=x.device)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=x.device)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy), dim=1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = gridx + gridy
        grid = grid
        return torch.cat((x, grid), dim=1)
        

class GFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_layers=4, in_channel_dim=1, out_channel_dim=1, reflection=0, grid_type="None"):
        super(GFNO2d, self).__init__()
        self.modes = modes1
        self.width = width
        self.num_layers = num_layers
        self.padding = 9  # Pad the domain if input is non-periodic
        self.grid_type = grid_type
        if grid_type != "None":
            self.grid = grid(grid_type=grid_type)
            grid_dim = self.grid.grid_dim
        else:
            grid_dim = 0
        
        self.p = GConv2d(in_channels=in_channel_dim + grid_dim, out_channels=self.width, kernel_size=1, reflection=reflection, first_layer=True)

        self.convs = nn.ModuleList([GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection) for _ in range(num_layers)])
        self.ws = nn.ModuleList([GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection) for _ in range(num_layers)])
        self.norm = GNorm(self.width, group_size=4 * (1 + reflection))
        self.q = GMLP2d(in_channels=self.width, out_channels=out_channel_dim, mid_channels=self.width * 4, reflection=reflection, last_layer=True)

    def forward(self, x):
        if self.grid_type != "None":
            x = self.grid(x)
        x = self.p(x)
        x = F.pad(x, [0, self.padding, 0, self.padding])
        for i in range(self.num_layers):
            x1 = self.norm(self.convs[i](self.norm(x)))
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()

            # Check if the tensor is complex
            if param.is_complex():
                nbytes += param.data.element_size() * param.numel() * 2  # Multiply by 2 for complex numbers
                nparams += param.numel()
            else:
                nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters in Group-FNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams