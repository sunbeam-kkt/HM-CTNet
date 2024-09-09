import math
import pdb
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange, repeat

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1, loop_times=4, fuse_alpha=0.15):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        self.loop_times = loop_times
        self.fuse_alpha = fuse_alpha

    def k_hop(self, A):
        # A: N, C, V, V
        N, C, V, _ = A.shape
        # A0: 1, 1, V, V
        A0 = torch.eye(V, dtype=A.dtype).to(A.device).unsqueeze(0).unsqueeze(0) * self.fuse_alpha
        
        A_power = torch.eye(V, dtype=A.dtype).to(A.device).unsqueeze(0).unsqueeze(0)
        for i in range(1, self.loop_times + 1):   # i=1,2,3,4
            A_power = torch.einsum('ncuv,ncvw->ncuw', A, A_power)
            A0 = A_power * (self.fuse_alpha * (1 - self.fuse_alpha) ** i) + A0

        return A0
            

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V

        if self.loop_times != 0:
            fuse_A = self.k_hop(x1)
            out_k = torch.einsum('ncuv,nctv->nctu', fuse_A, x3)
            return out_k
        else:
            x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
            return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True, **kwargs):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.test = 1
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels, **kwargs))
        # self.convs = CTRGC(in_channels, out_channels, **kwargs)
        
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32))) 
            # nn.init.constant_(self.PA, 1e-6)
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        N, C, T, V = x.size()
        # print(x.size())       # 16, 3, 300, 25
        # if self.adaptive:
        #     A = self.PA + self.A.cuda(x.get_device())
        # else:
        #     A = self.A.cuda(x.get_device())
        # A = self.A.cuda(x.get_device())
        PA = self.PA                               # 改了这里，加了self.A
        # nn.init.constant_(self.PA, 1e-6)
        # PA = PA.mean(0).view(V, V)
        # for i in range(self.num_subset):
        #     z = self.convs[i](x, A[i], self.alpha)
        #     y = z + y if y is not None else z
        # print(self.A.size())



        A_class = []
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, self.inter_c, V, T) 
            A1 = A1.mean(-1)
            A1 = A1.view(N, self.inter_c, V)
            A2 = self.conv_b[i](x).view(N, V, self.inter_c, T)        
            A2 = A2.mean(-1)
            A2 = A2.view(N, V, self.inter_c)
            A1 = self.soft(torch.matmul(A2, A1) / A1.size(-1))  # N V V  
            A_i = A1 + self.A[i].cuda(x.get_device())

            A_class.append(A_i)
        # A_out = (A_class[0] + A_class[1] + A_class[2])/3
        # A_out = torch.cat((A_class[0], A_class[1], A_class[2]), dim=0)
        # A_out = A_out.mean(0)




        # A_out = A_out + PA
        # A_in = x.view(N, C * T, V).view(N, C, T, V)

        # z = self.convs(x, A_out, self.alpha)

        z = None 
        for i in range(self.num_subset):
            # A_out = A_out + self.A[i].cuda()  
            # y = self.convs[i](x, A_out, self.alpha)

            # y = self.convs[i](x, PA[i] + A_out, self.alpha)
            y = self.convs[i](x, PA[i] + A_class[i].mean(0), self.alpha)

            z = y + z if z is not None else y
        # z = self.convs(torch.matmul(A_in, A_out).view(N, C, T, V))
        z = self.bn(z)
        z += self.down(x)
        z = self.relu(z)
        return z
    # def forward(self, x):
    #     y = None
    #     if self.adaptive:
    #         A = self.PA + self.A.cuda(x.get_device())
    #     else:
    #         A = self.A.cuda(x.get_device())
    #     for i in range(self.num_subset):
    #         z = self.convs[i](x, A[i], self.alpha)
    #         y = z + y if y is not None else z
    #     y = self.bn(y)
    #     y += self.down(x)
    #     y = self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, loop_times=0, kernel_size=7, dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y, A


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        A1 = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A1)

        self.num_class = num_class
        self.num_point = num_point
        # self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.feature_bn = nn.BatchNorm1d(in_channels * num_point)
        base_channel = 64
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)


        # self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        # self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        # self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        # self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        # self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        # self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        # self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        # self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        # self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        # self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.st_gcn_networks = nn.ModuleList((
            TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive, **kwargs),
            TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, loop_times=0, **kwargs),
            TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, loop_times=0, **kwargs),
            TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, loop_times=0, **kwargs),
            TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive, loop_times=0, **kwargs),
            TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive, loop_times=0, **kwargs),
            TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive, loop_times=0, **kwargs),
            TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive, loop_times=0, **kwargs),
            TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive, loop_times=0, **kwargs),
            TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive, loop_times=0, **kwargs),
        ))

        # self.st_gcn_networks_extract = nn.ModuleList((
        #     TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive, **kwargs),
        #     TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, loop_times=0, **kwargs),
        #     TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, loop_times=0, **kwargs),
        #     # TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, loop_times=0, **kwargs),
        #     TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive, loop_times=0, **kwargs),
        #     TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive, loop_times=0, **kwargs),
        #     TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive, loop_times=0, **kwargs),
        #     TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive, loop_times=0, **kwargs),
        #     TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive, loop_times=0, **kwargs),
        #     TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive, loop_times=0, **kwargs),
        # ))

        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.A.size()))
            for i in self.st_gcn_networks
            ])

        self.fc = nn.Linear(base_channel*4, num_class)
        # self.aux_fc = nn.Conv2d(base_channel*4, 1, 1, 1)
        # self.fc1 = nn.Conv2d(256, num_class, kernel_size=1)
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        # self.fcn = nn.Linear(25, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        # x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        # x = self.to_joint_embedding(x)
        # x += self.pos_embedding[:, :self.num_point]
        # x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        # x = self.data_bn(x)
        # x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        
        # N*M,C,T,V
        c_new = x.size(1)

        # # aux branch
        # # N * M, C, T, V -> N * M, C, V
        # aux_x = x.mean(2)
        # # N * M, C, V -> N * M, C, n_cls, V
        # aux_x = torch.einsum('nmv,cvu->nmcu', aux_x, self.examplar)
        # #  N * M, C, n_cls, V ->  N * M, n_cls, V
        # aux_x = self.aux_fc(aux_x)
        # aux_x = aux_x.squeeze(1)
        # # N * M, n_cls, V -> N * M, n_cls
        # aux_x = aux_x.mean(2)

        # aux_x = aux_x.reshape(N, M, self.num_class)
        # aux_x = aux_x.mean(dim=1)

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        x = self.fc(x)

        return x
    
    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        # N, C, T, V, M to N, M, V, C, T
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        # N, C, T, V, M to N * M, V * C, T
        x = x.view(N * M, V * C, T)
        x = self.feature_bn(x)
        # N * M, V * C, T to N, M, V, C, T
        x = x.view(N, M, V, C, T)
        # N, M, V, C, T to N, M, C, T, V
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)    # N, M, C, T, V -> N, C, T, V, M (N, [C, T, V], M)

        # prediction
        x = self.fcn(x)                                           # N, 60, T, V, M
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)    # N, 60, T, V, M to N, T, V, M, 60
        OUTPUT = x

        return output, feature, OUTPUT


