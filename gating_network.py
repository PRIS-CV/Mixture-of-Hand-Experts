import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 定义全连接层
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 3)  # 输出三类

    def forward(self, x):
        # 卷积层1
        x = F.relu(self.conv1(x))
        # 最大池化层
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 卷积层2
        x = F.relu(self.conv2(x))
        # 最大池化层
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 将特征图展平为一维向量
        x = x.view(-1, 32 * 32 * 32)
        # 全连接层1
        x = F.relu(self.fc1(x))
        # 全连接层2
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


