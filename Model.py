import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from mamba_ssm import Mamba
from torchsummary import summary
from thop import profile

torch.manual_seed(970530)
torch.cuda.manual_seed_all(970530)

class GroupBatchnorm2d(nn.Module):
    def __init__(self,
                 # 特征通道数
                 c_num: int,
                 # 分组数
                 group_num: int = 16,
                 # 归一化过程中防止分母为零的小数
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        # 确保特征通道数不小于分组数
        assert c_num >= group_num
        self.group_num = group_num
        # 用于归一化后的特征进行缩放
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        # 用于归一化后的特征进行偏移
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        # N:batch size    C:通道数   H，W:高度和宽度
        N, C, H, W = x.size()
        print(N,C,H,W)
        # 将特征张量x重塑成为(N, group_num, -1)的形状
        x = x.view(N, self.group_num, -1)
        # 在第三个维度上计算分组的平均值
        mean = x.mean(dim=2, keepdim=True)
        # 在第三个维度上计算分组的标准差
        std = x.std(dim=2, keepdim=True)
        # 归一化
        x = (x - mean) / (std + self.eps)
        # 将归一化后的特征张量重塑回原始形状(N, C, H, W)
        x = x.view(N, C, H, W)
        # 归一化后的特征进行缩放和偏移，然后返回结果，这一步缩放和偏移应用于每个通道的特征
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 # 输出通道数
                 oup_channels: int,
                 # 分组数
                 group_num: int = 1,
                 # 门限值
                 gate_treshold: float = 0.5,
                 # bool值，用于显示nn.GroupNorm是否进行分组归一化
                 torch_gn: bool = True
                 ):
        super().__init__()
        # 分组归一化层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        # 门的阈值
        self.gate_treshold = gate_treshold

        # sigmoid函数
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        # 归一化处理输入特征x
        # print(self.gate_treshold)
        gn_x = self.gn(x)
        # 归一化后的权重参数计算得到加权参数
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        # 门控系数，通过加权特征进行sigmoid函数处理得到
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate 门控系数将输入特征x进行分组
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        # 通过重构函数 reconstruct 对分组后的特征进行融合得到最终输出
        y = self.reconstruct(x_1, x_2)
        # print(x_1.shape)

        return y

    def reconstruct(self, x_1, x_2):
        # 将门控后的输入特征x_1和x_2进行分组
        # torch.split 函数：按照通道数的一半将特征张量进行拆分
        a=x_1.size(1) // 2
        b=x_2.size(1) // 2
        x_11, x_12 = torch.split(x_1, [a,a+1], dim=1)
        x_21, x_22 = torch.split(x_2, b+1, dim=1)
        # torch.cat 函数：将拆分后的特征张量按照指定维度进行拼接，形成最终的输出特征
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)




class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 # 输出通道数
                 op_channel: int,
                 # 控制通道分配的参数，表示高层特征和低层特征的通道比例
                 alpha: float = 1 / 2,
                 # 压缩比例
                 squeeze_radio: int = 2,
                 # 分组卷积的组数
                 group_size: int = 1,
                 # 分组卷积核大小
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        # 高层通道数和底层通道数
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        # 对高层和底层特征进行压缩的1*1卷积层
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up 高层特征的组卷积层，用于对压缩后的高层特征进行重构
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        # 高层和底层特征的1*1卷积层，用于通道的变换和压缩
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        # 自适应平均池化层，用于在通道维度上进行全局平均池化
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split 将输入特征张量x按照通道分割为高层和低层特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        # 分别对高层和底层特征进行压缩
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        # 对压缩后的高层特征进行组卷积和通道变换
        Y1 = self.GWC(up) + self.PWC1(up)
        # 对低层特征进行通道变换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        # 对变换后的特征按通道维度进行拼接
        out = torch.cat([Y1, Y2], dim=1)
        # 对拼接后的特征进行softmax操作，用于学习通道权重
        out = F.softmax(self.advavg(out), dim=1) * out
        # 将softmax后的特征按照通道分割为两部分
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        # 返回两部分元素级加和
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 # 输出通道数
                 op_channel: int,
                 # SRU模块参数，控制门控特征融合行为
                 group_num: int = 1,
                 gate_treshold: float = 0.5,
                 # CRU模块参数，控制通道重构单元行为
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 1,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)
        self.frequencyAttention = frequencyAttention(op_channel)
        self.spatialAttention = spatialAttention(op_channel)

    def forward(self, x):
        #x, spaAtten = self.spatialAttention(x)
        #print(1)
        x,x_1,x_2 = self.SRU(x)
        # print(x_1.shape)
        # USC = self.CRU(x)
        # x=USS+USC
        x, freqAtten = self.frequencyAttention(x)
        return x


# This is two parts of the attention module:
## Spatial_Attention in attention module
# 空间注意力机制，卷积核1*1*1，sigmoid激活，之后和原始信号混合
class spatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)
        spaAtten = q
        spaAtten = torch.squeeze(spaAtten, 1)
        q = self.norm(q)
        # In addition, return to spaAtten for visualization
        return U * q, spaAtten


## Frequency Attention in attention module
# 频率注意力机制：利用五个频带的时间切片去产生关联权重
# 从6*9*5转换到1*1*5
class frequencyAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2,
                                      kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels,
                                         kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)
        z = self.Conv_Squeeze(z)
        z = self.Conv_Excitation(z)
        freqAtten = z
        freqAtten = torch.squeeze(freqAtten, 3)
        z = self.norm(z)
        # In addition, return to freqAtten for visualization
        return U * z.expand_as(U), freqAtten


# Attention module:
# spatial-frequency attention
# 经过注意力机制之后，空间和频率信息进行混合，返回新的四维数据，6*9*5*16
class sfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.frequencyAttention = frequencyAttention(in_channels)
        self.spatialAttention = spatialAttention(in_channels)

    def forward(self, U):
        U_sse, spaAtten = self.spatialAttention(U)
        U_cse, freqAtten = self.frequencyAttention(U)
        # Return new 4D features
        # and the Frequency Attention and Spatial_Attention
        return U_cse + U_sse, spaAtten, freqAtten


# depthwise separable convolution(DS Conv):
# depthwise conv + pointwise conv + bn + relu
# 主要用于从特征图当中提取出空间特征
# 只采用了一个average pooling池化层，达到降采样的同时也没有丢失空间信息太多

"""
和mobilnet相比，在3*3深度分离卷积之后没有使用归一化和ReLU激活
添加后导致训练结果准确率下降
"""
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernal_size = kernel_size
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depth_conv(x)

        x = self.point_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Context module in DSC module
# 在减少计算消耗的同时，保证模型的性能
class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3x3BNReLU, self).__init__()
        self.conv3x3 = depthwise_separable_conv(in_channel, out_channel, 3)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv3x3(x)))

# 用于混合两个分支层的信息
class ContextModule(nn.Module):
    def __init__(self, in_channel):
        super(ContextModule, self).__init__()
        self.stem = Conv3x3BNReLU(in_channel, in_channel // 2)
        self.branch1_conv3x3 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
        self.branch2_conv3x3_1 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
        self.branch2_conv3x3_2 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)

    def forward(self, x):
        x = self.stem(x)
        # branch1
        x1 = self.branch1_conv3x3(x)
        # branch2
        x2 = self.branch2_conv3x3_1(x)
        x2 = self.branch2_conv3x3_2(x2)
        # concat
        return torch.cat([x1, x2], dim=1)


# SCM-NET:
# Attention module + DSC module + LSTM module
class SCM_NET(nn.Module):
    def __init__(self, num_classes=1):
        super(SFT_Net, self).__init__()
        # self.Atten = sfAttention(in_channels=5)
        self.ScConv=ScConv(5)
        # self.SRU=SRU(5)
        # self.CRU = CRU(5)
        self.bneck = nn.Sequential(
            #  begin x = [32, 16, 5, 6, 9], in fact x1 = [32, 5, 6, 9]
            # 主要用于从特征图当中提取出空间特征
            depthwise_separable_conv(5, 32, 3),
            depthwise_separable_conv(32, 64, 3),
            # default dropout
            nn.Dropout2d(0.3),
            depthwise_separable_conv(64, 128, 3),
            # Context Module  在减少计算消耗的同时，保证模型的性能
            ContextModule(128),
            depthwise_separable_conv(128, 64, 3),
            # default dropout
            nn.Dropout2d(0.3),
            depthwise_separable_conv(64, 32, 3),
            # 平均池化层减小过度混合的现象，同时提高鲁棒性
            # 只采用了一个average pooling池化层，达到降采样的同时也没有丢失空间信息太多
            nn.AdaptiveAvgPool2d((2, 2))  # [batch, 32, 2, 2]
        )
        self.block=nn.Sequential(
            nn.Conv2d(5,32, kernel_size=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.Conv2d(32, 5, kernel_size=1, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True)
        )
        # 最后通过线性层输出，得到64*16大小，输入到LSTM当中提取时间特征  # cat 16 * [batch, 1, 32] -> [batch, 16, 32]
        #self.linear = nn.Linear(5*6*9, 64)
        self.linear = nn.Linear(32 * 2 * 2, 64)
        # self.linear = nn.Linear(5 * 6 * 9, 64)
        # LSTM提取时间特征
        self.rnn=nn.RNN(input_size=64, hidden_size=32, num_layers=2, batch_first=True)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)  # [batch, input_size, -]
        # [32,16,64]
        # 87.9653% 64 32 4 1
        # 87.7% 64 32 4 1
        self.mamba=Mamba(d_model=64,d_state=32,d_conv=4,expand=1)
        # 经过最后两层线性层提取分类信息
        self.linear1 = nn.Linear(64 * 16, 120)
        # self.linear1 = nn.Linear(64 * 8, 120)
        #self.linear0 = nn.Linear(256, 120)
        self.dropout = nn.Dropout(0.4)  # default dropout
        self.linear2 = nn.Linear(120, num_classes)

    # 神经网络模型的参数初始化函数
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用了Kaiming正态分布初始化权重，fan_out是一种特定的初始化模式
                init.kaiming_normal_(m.weight, mode='fan_out')
                # 卷积层的bias项设置为0
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 批量归一化层，权重初始化为1，偏量初始化为常数0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层，使用了正态分布初始化权重，标准差为0.001
                # 全连接层有偏置项初始化为0
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def call_back(self):
        return
    def forward(self, x):
    
        # x1 - x16 [batch, 16, 5, 6, 9]
        x1 = torch.squeeze(x[:, 0, :, :, :], 1)  # [batch, 5, 6, 9]
        x2 = torch.squeeze(x[:, 1, :, :, :], 1)
        x3 = torch.squeeze(x[:, 2, :, :, :], 1)
        x4 = torch.squeeze(x[:, 3, :, :, :], 1)
        x5 = torch.squeeze(x[:, 4, :, :, :], 1)
        x6 = torch.squeeze(x[:, 5, :, :, :], 1)
        x7 = torch.squeeze(x[:, 6, :, :, :], 1)
        x8 = torch.squeeze(x[:, 7, :, :, :], 1)
        x9 = torch.squeeze(x[:, 8, :, :, :], 1)
        x10 = torch.squeeze(x[:, 9, :, :, :], 1)
        x11 = torch.squeeze(x[:, 10, :, :, :], 1)
        x12 = torch.squeeze(x[:, 11, :, :, :], 1)
        x13 = torch.squeeze(x[:, 12, :, :, :], 1)
        x14 = torch.squeeze(x[:, 13, :, :, :], 1)
        x15 = torch.squeeze(x[:, 14, :, :, :], 1)
        x16 = torch.squeeze(x[:, 15, :, :, :], 1)
     
        #
        x1 = self.ScConv(x1)  # [batch, 5, 6, 9], [batch, 6, 9], [batch, 5, 1]
        x2 = self.ScConv(x2)
        x3 = self.ScConv(x3)
        x4 = self.ScConv(x4)
        x5 = self.ScConv(x5)
        x6 = self.ScConv(x6)
        x7 = self.ScConv(x7)
        x8 = self.ScConv(x8)
        x9 = self.ScConv(x9)
        x10 = self.ScConv(x10)
        x11 = self.ScConv(x11)
        x12 = self.ScConv(x12)
        x13 = self.ScConv(x13)
        x14 = self.ScConv(x14)
        x15 = self.ScConv(x15)
        x16 = self.ScConv(x16)
        # sscout1 = torch.cat()
        # print(x1.shape)
        # print(x1_1.shape)
        # # print(x16.shape)
        # scconv_x1=torch.cat((x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1, x8_1, x9_1, x10_1, x11_1, x12_1, x13_1, x14_1, x15_1, x16_1), dim=1)
        #
        #
        #
        #
        # # bneck
        x1 = self.bneck(x1)
        x2 = self.bneck(x2)
        x3 = self.bneck(x3)
        x4 = self.bneck(x4)
        x5 = self.bneck(x5)
        x6 = self.bneck(x6)
        x7 = self.bneck(x7)
        x8 = self.bneck(x8)
        x9 = self.bneck(x9)
        x10 = self.bneck(x10)
        x11 = self.bneck(x11)
        x12 = self.bneck(x12)
        x13 = self.bneck(x13)
        x14 = self.bneck(x14)
        x15 = self.bneck(x15)
        x16 = self.bneck(x16)
        # #print(x1.shape)
        #
        # #x1 = self.block(x1)
        # #x2 = self.block(x2)
        # #x3 = self.block(x3)
        # #x4 = self.block(x4)
        # #x5 = self.block(x5)
        # #x6 = self.block(x6)
        # #x7 = self.block(x7)
        # #x8 = self.block(x8)
        # #x9 = self.block(x9)
        # #x10 = self.block(x10)
        # #x11 = self.block(x11)
        # #x12 = self.block(x12)
        # #x13 = self.block(x13)
        # #x14 = self.block(x14)
        # #x15 = self.block(x15)
        # #x16 = self.block(x16)
        #
        # # print(x16.shape)
        # # print(x1.view(x1.shape[0], 1, -1).shape)
        # # bneck
        #
        #
        x1 = self.linear(x1.view(x1.shape[0], 1, -1))  # [batch, 1, 32*2*2] -> [batch, 1, 64]
        x2 = self.linear(x2.view(x2.shape[0], 1, -1))
        x3 = self.linear(x3.view(x3.shape[0], 1, -1))
        x4 = self.linear(x4.view(x4.shape[0], 1, -1))
        x5 = self.linear(x5.view(x5.shape[0], 1, -1))
        x6 = self.linear(x6.view(x6.shape[0], 1, -1))
        x7 = self.linear(x7.view(x7.shape[0], 1, -1))
        x8 = self.linear(x8.view(x8.shape[0], 1, -1))
        x9 = self.linear(x9.view(x9.shape[0], 1, -1))
        x10 = self.linear(x10.view(x10.shape[0], 1, -1))
        x11 = self.linear(x11.view(x11.shape[0], 1, -1))
        x12 = self.linear(x12.view(x12.shape[0], 1, -1))
        x13 = self.linear(x13.view(x13.shape[0], 1, -1))
        x14 = self.linear(x14.view(x14.shape[0], 1, -1))
        x15 = self.linear(x15.view(x15.shape[0], 1, -1))
        x16 = self.linear(x16.view(x16.shape[0], 1, -1))
 
        # # cat 16 * [batch, 1, 32] -> [batch, 16, 32]
        out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), dim=1)

        #
        # # after MAMBA                [batch, 16, 64]

        #
        # #out,_=self.rnn(out)
        out=self.mamba(out)
 
        #
        # # flatten 通常用于将多维的输入张量（比如卷积层的输出）转换为一维张量，以便连接到全连接层或者输出层
        # # [batch, 16*120]
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
        #
        # # first linear                  [batch, 120]
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out



if __name__ == '__main__':


    input = torch.rand((32, 16, 5, 6, 9))
    input=input.cuda("cuda:1")
    net = SCM_NET().cuda("cuda:1")
    a,output,ssc_x1,ssc_x2 = net(input)
    print(a.shape)
    print("mamba")
    print(ssc_x1.shape)

    #
    # print(output.device)
    # print(torch.version.cuda)
    # print(torch.cuda.is_available())
    # print(net)
    print("Input shape     : ", input.shape)
    print("Output shape    : ", output.shape)

