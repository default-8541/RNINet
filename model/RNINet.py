import torch
import torch.nn as nn
import model.basicblock as B
import numpy as np


class NoiseInjection(nn.Module):

    def __init__(self,apply_prob=0.5,ep=1e-6):
        super(NoiseInjection, self).__init__()
        self.apply_prob=apply_prob
        self.ep=ep

    def _noise_injection(self, mean, variance):
        noise = torch.randn_like(variance)
        return mean + noise * variance


    def _compute_stats(self, features):
        mean = features.mean(dim=[2, 3], keepdim=True)
        variance = torch.sqrt(features.var(dim=[2, 3], keepdim=True) + self.ep)
        return mean, variance

    def _duplicate_for_batch(self, stat):
        dstat = (stat.var(dim=0, keepdim=True) + self.ep).sqrt()
        dstat = dstat.repeat(stat.shape[0], 1,1,1)
        return dstat

    def forward(self, features):

        if (not self.training) or (np.random.random()) > self.apply_prob:
            return features

        mean, variance = self._compute_stats(features)

        noised_mean = self._noise_injection(mean, self._duplicate_for_batch(mean))
        noised_variance = self._noise_injection(variance, self._duplicate_for_batch(variance))

        normalized_features = (features - mean) / variance
        noised_features = normalized_features * noised_variance + noised_mean
        return noised_features


class NIBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2, stride=2, padding=0, bias=True,apply_prob=0.5,ep=1e-6):
        super(NIBlock, self).__init__()
        self.conv=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
              padding=padding, bias=bias)
        self.bn=nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True)
        self.relu=nn.ReLU(inplace=True)
        self.noise=NoiseInjection(apply_prob,ep)

    def forward(self, x0):
        x0 = self.conv(x0)
        x0 = self.bn(x0)
        x0 = self.relu(x0)
        x0 = self.noise(x0)
        return x0

    
class RNINet(nn.Module):
    def __init__(self, apply_prob=0.5,ep=1e-6,in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(RNINet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], mode='C' + act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C' + act_mode) for _ in range(nb)],
                                    NIBlock(nc[0], nc[1],apply_prob=apply_prob,ep=ep))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C' + act_mode) for _ in range(nb)],
                                    NIBlock(nc[1], nc[2],apply_prob=apply_prob,ep=ep))
        self.m_down3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C' + act_mode) for _ in range(nb)],
                                    NIBlock(nc[2], nc[3],apply_prob=apply_prob,ep=ep))
        self.m_body = B.sequential(*[B.conv(nc[3], nc[3], mode='C' + act_mode) for _ in range(nb)],
                                   NIBlock(nc[3], nc[3],kernel_size=3, stride=1, padding=1, bias=True,apply_prob=apply_prob,ep=ep))

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2' + act_mode),
                                  *[B.conv(nc[2], nc[2], mode='C' + act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2' + act_mode),
                                  *[B.conv(nc[1], nc[1], mode='C' + act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2' + act_mode),
                                  *[B.conv(nc[0], nc[0], mode='C' + act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=True, mode='C')

    def forward(self, x0):

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1) + x0

        return x
    
###
#Some codes are borrowed from https://github.com/xinntao/BasicSR, https://github.com/lixiaotong97/DSU and https://github.com/Boyiliee/MoEx. Thanks to the #original authors for their work.
###