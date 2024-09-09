import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .resnet import resnet18 as ResNet
import numpy as np
import sys

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
# model.eval()

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args, **kwargs):
        super().__init__()

        self.resnet = ResNet(pretrained=True)
        # self.efficientnet = model
        self.FC = nn.Conv2d(1, 3, kernel_size=1)
        self.fc1 = nn.Linear(16, 225)
        # self.efficientnet.eval()
        # self.efficientnet.fc = nn.Linear(512, num_class)
        self.fc = nn.Linear(512, num_class)  # resnet18
        self.softmax = nn.Softmax()
        self.stgcn = ''
        self.temporal_positions = 15
        self.temporal_rgb_frames = 1

    def forward(self, x_, x_rgb):
    #def forward(self, x_rgb):

        # predict, feature = self.stgcn.extract_feature(x_)
        # intensity_s = (feature*feature).sum(dim=1)**0.5   # 身体区域重要关节的关节权重计算公式

        # intensity_s = intensity_s.cpu().detach().numpy()

        # feature_s = np.abs(intensity_s)

        # feature_s = 255 * (feature_s-feature_s.min()) / (feature_s.max()-feature_s.min())  # feature_s归一化操作，乘以255，转换为灰度值[0,255]
        # N, C, T, V, M = x_.size()

        # weight = np.full((N, 1, 225, 45*self.temporal_rgb_frames), 0)  # full_rgb_crop_sklweight_auto_1
        # for n in range(N):
        #     if feature_s[n, :, :, 0].mean(1).mean(0) > feature_s[n, :, :, 1].mean(1).mean(0):
        #         for j, v in enumerate([11, 7]):  # 代表了两个关键关节点
        #             # use TOP 10 values along the temporal dimension
        #             feature = feature_s[n, :, v, 0]
        #             #print('0 node: '+ str(v)+'\n', feature)
        #             temp = np.partition(-feature, self.temporal_positions)  # 将-feature按self.temporal_positions分成self.temporal_positions之前和之后的两部分
        #             #print('feature ', v, ' ', feature, -temp[:15].mean())
        #             feature = -temp[:self.temporal_positions].mean()
        #             weight[n, 0, 45*j:45*(j+1), :] = feature[np.newaxis, np.newaxis]
        #     else:
        #         for j, v in enumerate([11, 7]):
        #             # use TOP 10 values along the temporal dimension
        #             feature = feature_s[n, :, v, 1]
        #             #print('1 node: '+ str(v)+'\n', feature)
        #             temp = np.partition(-feature, self.temporal_positions)
        #             #print('feature ', v, ' ', feature, -temp[:15].mean())
        #             feature = -temp[:self.temporal_positions].mean()
        #             weight[n, 0, 45*j:45*(j+1), :] = feature[np.newaxis, np.newaxis]

        # weight_cuda = torch.from_numpy(weight).float().cuda()
        # weight_cuda = torch.from_numpy(weight).float().cuda()
        # N, M, C, T = weight_cuda.size()
        # weight_cuda = weight_cuda.reshape(N, M, T, C)
        # weight_cuda = self.FC(weight_cuda)
        # weight_cuda = weight_cuda / 127
        #print('weight_cuda',weight_cuda[:,0,0,0].cpu().numpy())
        # x_rgb = self.fc1(x_rgb)
        # print(x_rgb.size(), weight_cuda.size())
        # print(weight_cuda.size())
        # rgb_weighted = x_rgb * weight_cuda   # 得到focused ST-ROI
        
        # N, C, T, V = rgb_weighted.size()
        # rgb_weighted = rgb_weighted.reshape(N, T, V, C)
        # rgb_weighted = self.fc1(rgb_weighted)
        # rgb_weighted = rgb_weighted.reshape(N, 3, T, V)
        #'''

        x = self.resnet.conv1(x_rgb)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)

        x = self.resnet.layer2(x)

        x = self.resnet.layer3(x)

        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        # x = self.efficientnet(rgb_weighted)
        x = torch.flatten(x, 1)  # 将输入X进行扁平化操作
        # out = self.resnet.fc(x)
        # print(x.size())
        out = self.fc(x)

        return out
