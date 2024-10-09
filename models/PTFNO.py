"""
PTFNO performs polar transform in particular. General coordinate transform can be adopted for different symmetries.
FNO Backbone dopted from:
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import math
from .utils import format_tensor_size

class Spectral_weights(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        dtype = torch.cfloat
        self.kernel_size_Y = 2*modes1 - 1
        self.kernel_size_X = modes2
        self.W = nn.ParameterDict({
            'y0_modes': torch.nn.Parameter(torch.empty(in_channels, out_channels, modes1 - 1, 1, dtype=dtype)),
            'yposx_modes': torch.nn.Parameter(torch.empty(in_channels, out_channels, self.kernel_size_Y, self.kernel_size_X - 1, dtype=dtype)),
            '00_modes': torch.nn.Parameter(torch.empty(in_channels, out_channels, 1, 1, dtype=torch.float))
        })
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        for v in self.W.values():
            nn.init.kaiming_uniform_(v, a=math.sqrt(5))
            
    def get_weight(self):
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        self.weights = torch.cat([self.W["y0_modes"], self.W["00_modes"].cfloat(), self.W["y0_modes"].flip(dims=(-2, )).conj()], dim=-2)
        self.weights = torch.cat([self.weights, self.W["yposx_modes"]], dim=-1)
        self.weights = self.weights.view(self.in_channels, self.out_channels,
                                         self.kernel_size_Y, self.kernel_size_X)
        

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.spectral_weight = Spectral_weights(in_channels=in_channels, out_channels=out_channels, modes1=modes1, modes2=modes2)
        self.get_weight()

    def get_weight(self):
        self.spectral_weight.get_weight()
        self.weights = self.spectral_weight.weights
        
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2]
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)              
        out_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2] = \
            self.compl_mul2d(x_ft, self.weights)


        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
        
        
class PolarLayer(nn.Module):
    def __init__(self, polar_size):
        super(PolarLayer, self).__init__()
        self.polar_size = polar_size
        # change np.pi to torch.pi
        phi = torch.linspace(0, 2*torch.pi, steps=self.polar_size[1])
        r = torch.linspace(0, 2**0.5, steps=self.polar_size[0])  # Torch assumes the domain of the input 2D feature to be [0,1] x [0,1]

        grid_r, grid_phi = torch.meshgrid(r, phi)
        self.grid_x = grid_r * torch.cos(grid_phi)
        self.grid_y = grid_r * torch.sin(grid_phi)
        self.grid_x = self.grid_x.unsqueeze(0).unsqueeze(0)
        self.grid_y = self.grid_y.unsqueeze(0).unsqueeze(0)

    def forward(self, input):
        batch = input.shape[0]
        sample_grid = torch.cat([self.grid_x.cuda(), self.grid_y.cuda()], dim=1)
        sample_grid = sample_grid.permute(0,2,3,1).contiguous()
        sample_grid = sample_grid.repeat(batch, 1, 1, 1)
        resampled_input = F.grid_sample(input, sample_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        return resampled_input
        
        
class InversePolarLayer(nn.Module):
    def __init__(self, planar_size):
        super(InversePolarLayer, self).__init__()
        self.planar_size = planar_size
        x = torch.linspace(-1 + 1 / planar_size[1], 1 - 1 / planar_size[1], planar_size[1])
        y = torch.linspace(-1 + 1 / planar_size[0], 1 - 1 / planar_size[0], planar_size[0])
        # Torch assumes the domain of the input 2D feature to be [0,1] x [0,1]
        grid_y, grid_x = torch.meshgrid(y, x)
        self.grid_r = (grid_x ** 2 + grid_y ** 2) ** 0.5
        self.grid_theta = torch.arccos(grid_x / self.grid_r)
        self.grid_theta[grid_y < 0] = torch.pi * 2 - self.grid_theta[grid_y < 0]
        self.grid_theta = self.grid_theta/(torch.pi) - 1
        self.grid_r = self.grid_r*(2**0.5) -1
        self.grid_r = self.grid_r.unsqueeze(0).unsqueeze(0)
        self.grid_theta = self.grid_theta.unsqueeze(0).unsqueeze(0)

    def forward(self, input):
        batch = input.shape[0]
        sample_grid = torch.cat([self.grid_theta.cuda(), self.grid_r.cuda()], dim=1)
        sample_grid = sample_grid.permute(0,2,3,1).contiguous()
        sample_grid = sample_grid.repeat(batch, 1, 1, 1)
        resampled_input = F.grid_sample(input, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return resampled_input

class PTFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_layers=4, in_channel_dim=1, out_channel_dim=1, grid=False, in_size = [49,49], polar_size = [20,120]):
        super(PTFNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.in_channel_dim = in_channel_dim
        self.out_channel_dim = out_channel_dim
        self.grid = grid
        self.padding = 9 # pad the domain if input is non-periodic
        self.polar = PolarLayer(polar_size)
        self.p = nn.Conv2d(in_channel_dim + (2 if grid else 0), self.width, 1) # input channel is 3: (a(x, y), x, y)
        self.conv_layers = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(num_layers)])
        self.mlp_layers = nn.ModuleList([MLP(self.width, self.width, self.width) for _ in range(num_layers)])
        self.w_layers = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(num_layers)])
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)
        self.ipolar = InversePolarLayer(in_size)
    def forward(self, x):
        x = self.polar(x)
        if self.grid:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)
        x = self.p(x)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x1 = self.mlp_layers[i](x1)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = self.ipolar(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1)
        
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

        print(f'Total number of model parameters in Coordinate(Polar) Transform FNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams