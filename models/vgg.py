import torch
import torch.nn as nn
import torch.nn.functional as F
from .heads import ArcFace

class VGG(nn.Module):
    def __init__(self, num_classes=116, input_feat_dim=1, head='arcface', s=64.0, m=0.5):
        super().__init__()
        self.head_type = head.lower()
        
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

        feat_dim = 256 # 上一层输出的通道数
        
        if self.head_type == 'arcface':
            self.head = ArcFace(in_features=feat_dim, out_features=num_classes, s=s, m=m)
        elif self.head_type == 'linear':
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            raise ValueError(f"不支持的分类头类型: {self.head_type}")

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)

        if self.head_type == 'arcface':
            if self.training:
                if labels is None:
                    raise ValueError("ArcFace 训练模式必须输入标签！")
                return self.head(feat, labels)
            else:
                normalized_feat = F.normalize(feat)
                normalized_weight = F.normalize(self.head.weight)
                return F.linear(normalized_feat, normalized_weight) * self.head.s
                
        elif self.head_type == 'linear':
            return self.head(feat)
