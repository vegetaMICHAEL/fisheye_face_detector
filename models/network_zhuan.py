import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from utils.s2cnn.soft.s2_fft import S2_fft_real
from utils.s2cnn.soft.so3_fft import SO3_ifft_real, SO3_fft_real
from utils.s2cnn import s2_mm
from utils.s2cnn import s2_rft
from utils.s2cnn import so3_mm
from utils.s2cnn import so3_rft
from utils.s2cnn import so3_near_identity_grid, s2_near_identity_grid

import sys
sys.path.append('/nfs/volume-365-3/miaojingbo/facedata/IDnet_v2/models')
from non_local import NONLocalBlock2D_embedded_gaussian, NONLocalBlock2D_dot_product, NONLocalBlock2D_concatenation


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

'''
class S2FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(S2FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.s2net2 = S2Net(in_channel=out_channels, f1=1, f2=out_channels, b_in=90, b_l1=20, b_l2=20)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.s2net3 = S2Net(in_channel=out_channels, f1=1, f2=out_channels, b_in=60, b_l1=10, b_l2=10)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        # print('o1.shape')
        #print(output1.shape)
        output2 = self.output2(input[1])
        # print('o2.shape')
        # print(output2.shape)
        output2 = self.s2net2(output2)
        output3 = self.output3(input[2])
        # print('o3.shape')
        # print(output3.shape)
        output3 = self.s2net3(output3)

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        # print(output2.shape)
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        # print('merge bias grad:', self.merge1.bias.grad)
        # param = list(self.merge1.parameters())
        # print(len(param))
        # print(param[0].size())
        # print(output1.shape)
        # print('output1 grad:',output1.grad)

        out = [output1, output2, output3]
        return out
'''


# s2cnn_80_non
class S2FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(S2FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

        self.FPN1 = nn.Sequential(
            conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky),
            S2Net(in_channel=out_channels, f1=1, f2=out_channels, b_in=90, b_l1=20, b_l2=40),
            NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)
        )
        self.FPN2 = nn.Sequential(
            conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky),
            # S2Net(in_channel=out_channels, f1=1, f2=out_channels, b_in=90, b_l1=20, b_l2=20),

        )
        self.FPN3 = nn.Sequential(
            conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        )

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.FPN1(input[0])  # 输出为[8,64,80,80]
        output2 = self.FPN2(input[1])  # 输出为[8,64,40,40]
        output3 = self.FPN3(input[2])  # 输出为[8,64,20,20]

        # 用来上采样或下采样，可以给定size或者scale_factor来进行上下采样。同时支持3D、4D、5D的张量输入
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3  # 输出为[8,64,40,40]
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)  # 输出为[8,64,80,80]
        # 保留test1 ，加上non-local # test3
        out = [output1, output2, output3]
        return out

'''
class S2FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(S2FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

        self.FPN1 = nn.Sequential(
            conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky),
            S2Net(in_channel=out_channels, f1=1, f2=out_channels, b_in=90, b_l1=20, b_l2=40),
            NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)
        )
        self.FPN2 = nn.Sequential(
            conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky),
            # S2Net(in_channel=out_channels, f1=1, f2=out_channels, b_in=90, b_l1=20, b_l2=20),
            # NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)

        )
        self.FPN3 = nn.Sequential(
            conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        )

        # self.merge_Non_local1 = nn.Sequential(
        #     conv_bn(out_channels, out_channels, leaky=leaky),
        #     NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)
        # )
        #
        # self.merge_Non_local2 = nn.Sequential(
        #     conv_bn(out_channels, out_channels, leaky=leaky),
        #     NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)
        # )

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.FPN1(input[0])  # 输出为[8,64,80,80]
        output2 = self.FPN2(input[1])  # 输出为[8,64,40,40]
        output3 = self.FPN3(input[2])  # 输出为[8,64,20,20]

        # 用来上采样或下采样，可以给定size或者scale_factor来进行上下采样。同时支持3D、4D、5D的张量输入
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3  # 输出为[8,64,40,40]
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)  # 输出为[8,64,80,80]
        # 保留test1 ，加上non-local # test3
        out = [output1, output2, output3]
        return out
'''
class New_S2FPN(nn.Module):
    """
        FPN1:添加S2CNN、Non-local模块；
        添加input1的下采样，从80尺度分别AdaptivePool到40尺度和20尺度，并在最后输出时和对应特征进行融合，并加入Non-local
    """

    def __init__(self, in_channels_list, out_channels):
        super(New_S2FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge3 = conv_bn(out_channels, out_channels, leaky=leaky)

        self.FPN1 = nn.Sequential(
            conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky),
            S2Net(in_channel=out_channels, f1=1, f2=out_channels, b_in=90, b_l1=20, b_l2=40),
            NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)
        )
        self.FPN2 = nn.Sequential(
            conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky),
            # S2Net(in_channel=out_channels, f1=1, f2=out_channels, b_in=90, b_l1=20, b_l2=20),
            # NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)

        )
        self.FPN3 = nn.Sequential(
            conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        )

        self.merge_Non_local1 = nn.Sequential(
            conv_bn(out_channels, out_channels, leaky=leaky),
            NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)
        )

        self.merge_Non_local2 = nn.Sequential(
            conv_bn(out_channels, out_channels, leaky=leaky),
            NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)
        )

        self.merge_Non_local3 = nn.Sequential(
            conv_bn(out_channels, out_channels, leaky=leaky),
            NONLocalBlock2D_embedded_gaussian(in_channels=out_channels)
        )

        self.AdaptivePool1 = nn.Sequential(
            nn.AdaptiveMaxPool2d((40, 40))
        )

        self.AdaptivePool2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((20, 20))
        )

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.FPN1(input[0])  # 输出为[8,64,80,80]
        output2 = self.FPN2(input[1])  # 输出为[8,64,40,40]
        output3 = self.FPN3(input[2])  # 输出为[8,64,20,20]

        output1_down = self.AdaptivePool1(output1)  # 输出为[8,64,40,40]
        output2_down = self.AdaptivePool2(output1)  # 输出为[8,64,20,20]

        # 用来上采样或下采样，可以给定size或者scale_factor来进行上下采样。同时支持3D、4D、5D的张量输入
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3  # 输出为[8,64,40,40]
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)  # 输出为[8,64,80,80]

        output2 = output2 + output1_down
        output2 = self.merge_Non_local2(output2)

        output3 = output3 + output2_down
        output3 = self.merge_Non_local3(output3)  # 输出为[8,64,20,20]

        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class S2Conv(nn.Module):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid):
        '''
        :param nfeature_in: number of input fearures
        :param nfeature_out: number of output features
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        :param grid: points of the sphere defining the kernel, tuple of (alpha, beta)'s
        '''
        super(S2Conv, self).__init__()
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in = b_in
        self.b_out = b_out
        self.grid = grid
        self.kernel = Parameter(torch.empty(nfeature_in, nfeature_out, len(grid)).uniform_(-1, 1))
        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 4.) / (self.b_in ** 2.))
        self.bias = Parameter(torch.zeros(1, nfeature_out, 1, 1, 1))

    def forward(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha](32, 1, 60, 60)
        :return: [batch, feature_out, beta, alpha, gamma]
        '''

        # x-feature map 由panar转sphere
        # assert x.size(1) == self.nfeature_in
        # assert x.size(2) == 2 * self.b_in
        # assert x.size(3) == 2 * self.b_in
        x = S2_fft_real.apply(x, self.b_out)  # [l * m, batch, feature_in, complex](100,32,1,2)
        y = s2_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m, feature_in, feature_out, complex](100, 1, 20, 2)
        z = s2_mm(x, y)  # [l * m * n, batch, feature_out, complex](1330, 32, 20, 2)
        z = SO3_ifft_real.apply(z)  # [batch, feature_out, beta, alpha, gamma](32, 20, 20, 20, 20)

        z = z + self.bias

        return z

class SO3Conv(nn.Module):
    def __init__(self, nfeature_in, nfeature_out, b_in, b_out, grid):
        '''
        :param nfeature_in: number of input fearures
        :param nfeature_out: number of output features
        :param b_in: input bandwidth (precision of the input SOFT grid)
        :param b_out: output bandwidth
        :param grid: points of the SO(3) group defining the kernel, tuple of (alpha, beta, gamma)'s
        '''
        super(SO3Conv, self).__init__()
        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.b_in = b_in
        self.b_out = b_out
        self.grid = grid
        self.kernel = Parameter(torch.empty(nfeature_in, nfeature_out, len(grid)).uniform_(-1, 1))
        self.bias = Parameter(torch.zeros(1, nfeature_out, 1, 1, 1))

        # When useing ADAM optimizer, the variance of each componant of the gradient
        # is normalized by ADAM around 1.
        # Then it is suited to have parameters of order one.
        # Therefore the scaling, needed for the proper forward propagation, is done "outside" of the parameters
        self.scaling = 1. / math.sqrt(len(self.grid) * self.nfeature_in * (self.b_out ** 3.) / (self.b_in ** 3.))

    def forward(self, x):  # pylint: disable=W
        '''
        :x:      [batch, feature_in,  beta, alpha, gamma]
        :return: [batch, feature_out, beta, alpha, gamma]
        '''
        assert x.size(1) == self.nfeature_in
        assert x.size(2) == 2 * self.b_in
        assert x.size(3) == 2 * self.b_in
        assert x.size(4) == 2 * self.b_in

        x = SO3_fft_real.apply(x, self.b_out)  # [l * m * n, batch, feature_in, complex]
        y = so3_rft(self.kernel * self.scaling, self.b_out, self.grid)  # [l * m * n, feature_in, feature_out, complex]
        assert x.size(0) == y.size(0)
        assert x.size(2) == y.size(1)
        z = so3_mm(x, y)  # [l * m * n, batch, feature_out, complex]
        assert z.size(0) == x.size(0)
        assert z.size(1) == x.size(1)
        assert z.size(2) == y.size(2)
        z = SO3_ifft_real.apply(z)  # [batch, feature_out, beta, alpha, gamma]

        z = z + self.bias

        return z


class S2Net(nn.Module):

    def __init__(self, in_channel, f1=20, f2=40, b_in=30, b_l1=10, b_l2=6):
        super(S2Net, self).__init__()

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.convred_dim = nn.Conv2d(in_channel, 1, 3, stride=1, padding=1, bias=False)
        self.conv1 = S2Conv(
            nfeature_in=1,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)

        self.conv2 = SO3Conv(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3)
        # class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.conv3d = nn.Conv3d(in_channels=f2, out_channels=1, kernel_size=1)
        self.pool3d = nn.MaxPool3d(kernel_size=(1, 1, 2*b_l2))
        self.bn3d = nn.BatchNorm2d(f2)
        self.lrelu3d = nn.LeakyReLU(negative_slope=0, inplace=True)

    def forward(self, x):

        x = self.convred_dim(x)
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)  # 32,20,40,40,40
        #print(x.shape)
        x = self.conv2(x)
        x = F.relu(x)  # 32,40, 12, 12, 12

        # x = so3_integrate(x)  # 32, 40
        # print(x.shape)
        x = self.pool3d(x)
        # print(x.shape)
        x = x.squeeze(-1)
        # print(x.shape)
        x = self.bn3d(x)
        x = self.lrelu3d(x)

        return x


