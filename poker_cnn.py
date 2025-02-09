import torch.nn as nn
import torch.nn.functional as F

class PokerCNN(nn.Module):
    def __init__(self, num_classes=52):
        super(PokerCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 卷积块1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积块2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积块3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 卷积块4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # 展平
        x = x.view(-1, 256 * 4 * 4)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 