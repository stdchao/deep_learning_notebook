#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ddpm.py
@Paper   :   Denoising Diffusion Probabilistic Models
@Desc    :   learn from https://github.com/w86763777/pytorch-ddpm
'''

# here put the import lib
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    v = v.to(t.device)
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class Swish(nn.Module):
    '''
    activation
    '''
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    '''
    as position embedding
    '''
    def __init__(self, T, in_dim, out_dim):
        super().__init__()
        emb = torch.arange(0, in_dim, step=2) 
        emb = torch.exp(- emb * math.log(10000) / in_dim)
        pos = torch.arange(T)
        emb = pos[:, None] * emb[None, :] # [T, in_dim/2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1) # [T, in_dim, 2]
        emb = emb.view(T, in_dim)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_dim, out_dim),
            Swish(),
            nn.Linear(out_dim, out_dim)
        )
        self.initialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
 
    def forward(self, t):
        emb_t = self.time_embedding(t)
        return emb_t

class AttnBlock(nn.Module):
    '''
    self-attention for 2D
    '''
    def __init__(self, channel):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channel)
        self.proj_q = nn.Conv2d(channel, channel, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(channel, channel, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(channel, channel, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(channel, channel, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
            init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        q = q.permute(0, 2, 3, 1).view(B, H*W, C)
        k = k.view(B, C, H*W)
        w = torch.bmm(q, k) * (C ** (-0.5))
        w = F.softmax(w, dim=-1) # [B, H*W, H*W]

        v = v.permute(0, 2, 3, 1).view(B, H*W, C)
        h = torch.bmm(w, v) # [B, H*W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x+h

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            Swish(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_channel)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channel),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

        if in_channel != out_channel:
            self.shorcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        else:
            self.shorcut = nn.Identity()
        
        if attn:
            self.attn = AttnBlock(out_channel)
        else:
            self.attn = nn.Identity()
        
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        '''
        x + time emb
        '''
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h += self.shorcut(x)
        h = self.attn(h)
        return h

class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.main = nn.Conv2d(channel, channel, 3, stride=2, padding=1) # downsample
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x

class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.main = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        B, C, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.main(x)
        return x

class UNet(nn.Module):
    '''
    UNet backbone
    '''
    def __init__(self, T, channel, channel_mult, attn, num_res_blocks, dropout):
        super().__init__()
        tdim = channel * 4
        self.time_embedding = TimeEmbedding(T, channel, tdim)
        
        # head
        self.head = nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1)
        
        # downblocks
        self.downblocks = nn.ModuleList()
        now_channel = channel
        channel_list = [now_channel]
        for i, mult in enumerate(channel_mult):
            out_channel = channel * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(in_channel=now_channel, out_channel=out_channel, tdim=tdim, dropout=dropout, attn=(i in attn))
                )
                now_channel = out_channel
                channel_list.append(now_channel)
            if i != len(channel_mult) - 1:
                self.downblocks.append(DownSample(now_channel))
                channel_list.append(now_channel)
        
        # middle
        self.middleblocks = nn.ModuleList([
            ResBlock(in_channel=now_channel, out_channel=now_channel, tdim=tdim, dropout=dropout, attn=True),
            ResBlock(in_channel=now_channel, out_channel=now_channel, tdim=tdim, dropout=dropout, attn=False),
        ])
        
        # upblocks
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channel = channel * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(in_channel=channel_list.pop() + now_channel,
                             out_channel=out_channel, tdim=tdim, dropout=dropout, attn=(i in attn))
                )
                now_channel = out_channel
            if i != 0:
                self.upblocks.append(UpSample(now_channel))

        # tail
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_channel),
            Swish(),
            nn.Conv2d(now_channel, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # time emb
        temb = self.time_embedding(t) # [128, 512]
        ho = self.head(x) # [128, 128, 32, 32]
        # downsample
        hs = [ho]
        for layer in self.downblocks:
            ho = layer(ho, temb) # [128,128,32,32]x2 -> [..16,16] -> [128,256,16,16]x2 -> [..8,8]x3 -> [..4,4]x3
            hs.append(ho)
        # middle
        for layer in self.middleblocks:
            ho = layer(ho, temb) # [128, 256, 4, 4]x2
        # upsample
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                ho = torch.cat([ho, hs.pop()], dim=1) # [128,512,4,4]x3 -> [] -> [128,512,8,8]x3 -> [] -> [128,512,16,16]x2 -> [128,384,16,16] -> [] -> [128,384,32,32] -> [128,256,32,32]x2
            ho = layer(ho, temb) # [128,256,4,4]x3 -> [128,256,8,8] -> [128,256,8,8]x3 -> [128,256,16,16] -> [128,256,16,16]x2 -> [128,256,16,16] -> [128,256,32,32] -> [128,128,32,32] -> [128,128,32,32]x2
        ho = self.tail(ho) # [128, 3, 32, 32]
        return ho

class DDPM(nn.Module):
    '''
    DDPM
    '''
    def __init__(self, T, beta_1, beta_T, denosise_model):
        super().__init__()
        self.T = T
        self.betas = torch.linspace(beta_1, beta_T, T).double() # [0.001, 0.002, ..., 0.01]
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)
        #
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1,0], value=1)[:T] # [1] + alphas_bar[:T]
        self.sqrt_recip_alphas_bar = torch.sqrt(1./self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1./self.alphas_bar - 1)
        self.posterior_var = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar) # [0, x, ...]
        self.posterior_log_var_clipped = torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])) # log([x, x, ...])
        self.posterior_mean_coef1 = torch.sqrt(self.alphas_bar_prev) * self.betas / (1. - self.alphas_bar)
        self.posterior_mean_coef2 = torch.sqrt(self.alphas) * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)
        # UNet
        self.denoise_model = denosise_model

    def forward(self, x_0):
        '''
        Algorithm 1.
        '''
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.rand_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        y_noise = self.denoise_model(x_t, t)
        x_t_denoise = x_t - y_noise
        loss = F.mse_loss(y_noise, noise, reduction='mean')
        return x_t_denoise, loss
    
    def p_mean_variable(self, x_t, t):
        '''
        mean & var of p(x) distribution
        '''
        model_log_var = extract(self.posterior_log_var_clipped, t, x_t.shape) # equations 7
        eps = self.denoise_model(x_t, t)
        x_0 = extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t - \
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps # quations 15
        model_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 + \
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t # equations 7
        return model_mean, model_log_var

    def sampler(self, x_T):
        '''
        Algorithm 2.
        '''
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step  # [t] * batch_size
            mean, log_var = self.p_mean_variable(x_t=x_t, t=t)
            
            if time_step > 0:
                noise = torch.randn_like(x_t) 
            else:
                noise = 0
            
            x_t = mean + torch.exp(0.5 * log_var) * noise # reparameterize
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
