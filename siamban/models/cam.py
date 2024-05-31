from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.conv1 = ConvBlock(961, 31, 1)  # xf
        self.conv2 = nn.Conv2d(31, 961, 1, stride=1, padding=0)
        self.conv3 = ConvBlock(49, 7, 1)  # zf
        self.conv4 = nn.Conv2d(7, 49, 1, stride=1, padding=0)

        # self.conv1 = ConvBlock(49, 7, 1)
        # self.conv2 = nn.Conv2d(7, 49, 1, stride=1, padding=0)
        # self.conv3 = ConvBlock(49, 7, 1)
        # self.conv4 = nn.Conv2d(7, 49, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def get_attention(self, a):
        input_a = a

        a = a.mean(3)  # GAP xyl20230227
        a = a.transpose(1, 3)
        a = F.relu(self.conv1(a))  # meta-learner xyl20230227
        a = self.conv2(a)  # meta-learner xyl20230227
        a = a.transpose(1, 3)
        a = a.unsqueeze(3)

        a = torch.mean(input_a * a, -1)
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a

    def get_attention1(self, a):
        input_a = a

        a = a.mean(3)  # GAP xyl20230227
        a = a.transpose(1, 3)
        a = F.relu(self.conv3(a))  # meta-learner xyl20230227
        a = self.conv4(a)  # meta-learner xyl20230227
        a = a.transpose(1, 3)
        a = a.unsqueeze(3)

        a = torch.mean(input_a * a, -1)
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a

    def forward(self, zf1, xf2):
        out_tensor1 = []
        out_tensor2 = []
        for (f1, f2) in zip(zf1, xf2):  # zf_st and xf xyl20230303
            f1 = torch.stack((f1, f1), dim=1)
            f2 = torch.stack((f2, f2), dim=1)

            b, n1, c, h, w = f1.size()
            b2, n2, c2, h2, w2 = f2.size()

            f1 = f1.view(b, n1, c, -1)  # f1 = (1, 2, 10, 625)
            f2 = f2.view(b, n2, c, -1)

            f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
            f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)

            f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
            f2_norm = f2_norm.unsqueeze(1)

            a1 = torch.matmul(f1_norm, f2_norm)
            a2 = a1.transpose(3, 4)

            a1 = self.get_attention(a1)  # a1 = (1, 2, 2, 625, 625) --> a1 = (1, 2, 2, 625). Meta Fusion Layer xyl20230227
            a2 = self.get_attention1(a2)

            f1 = f1.unsqueeze(2) * a1.unsqueeze(3)  # f1 = (1, 2, 2, 10, 625)
            f1 = f1.view(b, n1, n2, c, h, w)  # f1 = (1, 2, 2, 10, 25, 25)
            f2 = f2.unsqueeze(1) * a2.unsqueeze(3)
            f2 = f2.view(b, n1, n2, c, h2, w2)
            f1 = f1.transpose(1, 2)
            f2 = f2.transpose(1, 2)
            f1 = f1.mean(2)
            f1 = f1.mean(1)
            f2 = f2.mean(2)
            f2 = f2.mean(1)
            out_tensor1.append(f1)
            out_tensor2.append(f2)
        return out_tensor1, out_tensor2


if __name__ == "__main__":
    in_tensor1 = torch.ones((1, 256, 31, 31))
    in_tensor2 = torch.ones((1, 256, 7, 7))
    print('in_tensor1:', in_tensor1.shape)
    in_tensor1 = torch.stack((in_tensor1, in_tensor1), dim=1)  # stack是新的维度上进行堆叠，cat是直接在原来维度上进行拼接
    in_tensor2 = torch.stack((in_tensor2, in_tensor2), dim=1)

    cam = CAM()
    out_tensor1, out_tensor2 = cam(in_tensor1, in_tensor2)

    out_tensor1 = out_tensor1.mean(2)
    out_tensor1 = out_tensor1.mean(1)
    out_tensor2 = out_tensor2.mean(2)
    out_tensor2 = out_tensor2.mean(1)
    print('out_tensor1:', out_tensor1.shape)  # torch.Size([1, 2, 256, 31, 31])
    print('out_tensor2:', out_tensor2.shape)  # torch.Size([1, 2, 256, 7, 7])
