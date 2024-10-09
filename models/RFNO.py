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

class radialSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, reflection):
        super(radialSpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.reflection = reflection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = 1 / (in_channels * out_channels)
        self.dtype = torch.float

        if reflection:
            # get indices of lower triangular part of a matrix that is of shape (modes x modes)
            self.inds_lower = torch.tril_indices(self.modes + 1, self.modes + 1)

            # init weights
            self.W = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.inds_lower.shape[1], dtype=self.dtype))
        else:
            # lower center component of weights; [in_channels, out_channels, modes, 1]
            self.W_LC = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes + 1, 1, dtype=self.dtype))

            # lower right component of weights; [in_channels, out_channels, modes, modes]
            self.W_LR = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=self.dtype))
        self.eval_build = True
        self.get_weight()

    # Building the weight
    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.reflection:

            # construct the lower right part of the parameter matrix; this matrix is symmetric
            W_LR = torch.zeros(self.in_channels, self.out_channels, self.modes + 1, self.modes + 1,
                               dtype=self.dtype).to(self.W.device)
            W_LR[..., self.inds_lower[0], self.inds_lower[1]] = self.W
            W_LR.transpose(-1, -2)[..., self.inds_lower[0], self.inds_lower[1]] = self.W

            # construct the right part of the parameter matrix
            self.weights = torch.cat([W_LR[..., 1:, :].flip((-2)), W_LR], dim=-2).cfloat()

        else:

            # Build the right half of the weight by first constructing the lower and upper parts of the right half
            W_LR = torch.cat([self.W_LC[:, :, 1:], self.W_LR], dim=-1)
            W_UR = torch.cat([self.W_LC.flip(-2), W_LR.rot90(dims=[-2, -1])], dim=-1)
            self.weights = torch.cat([W_UR, W_LR], dim=-2).cfloat()

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
        x_ft = x_ft[..., (freq0_y - self.modes):(freq0_y + self.modes + 1), :(self.modes + 1)]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[..., (freq0_y - self.modes):(freq0_y + self.modes + 1), :(self.modes + 1)] = \
            self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))

        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
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
        

class RFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_layers=4, in_channel_dim=1, out_channel_dim=1, reflection=0, grid_type="None"):
        super(RFNO2d, self).__init__()
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
            
        self.norm = nn.InstanceNorm2d(width)
        self.p = nn.Conv2d(in_channels=in_channel_dim + grid_dim, out_channels=self.width, kernel_size=1)
        self.convs = nn.ModuleList([radialSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([MLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width,) for _ in range(num_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=1) for _ in range(num_layers)])

        self.q = MLP2d(in_channels=self.width, out_channels=out_channel_dim, mid_channels=self.width * 4)

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

        print(f'Total number of model parameters in Radial (Isotropic Kernel) FNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams