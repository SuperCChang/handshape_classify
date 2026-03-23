import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp_arcface import ArcFace

class VGG_ArcFace(nn.Module):
    def __init__(self, num_classes=116, input_feat_dim=1, s=30.0, m=0.5):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(input_feat_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.arcface = ArcFace(in_features=256, out_features=num_classes, s=s, m=m)

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)

        if self.training:
            if labels is None:
                raise ValueError("训练模式必须输入标签！")
            return self.arcface(feat, labels)
        else:
            normalized_feat = F.normalize(feat)
            normalized_weight = F.normalize(self.arcface.weight)
            return F.linear(normalized_feat, normalized_weight) * self.arcface.s
