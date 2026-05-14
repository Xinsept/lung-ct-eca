import torch
import torch.nn as nn
import torchvision.models as models

class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)              # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)                     # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y


class ResNet50_ECA(nn.Module):
    """ResNet50 with ECA after layer4"""
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet50_ECA, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = self.backbone.fc.in_features
        
        # Remove original fc
        self.backbone.fc = nn.Identity()
        
        # ECA after layer4 (before avgpool)
        self.eca = ECA(2048, kernel_size=3)  # layer4 output = 2048 channels
        
        # New classifier
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.eca(x)              # ECA applied here
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet50_Baseline(nn.Module):
    """Plain ResNet50 for baseline"""
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet50_Baseline, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)