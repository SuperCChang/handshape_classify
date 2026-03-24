# models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .heads import ArcFace

class ResNet(nn.Module):
    def __init__(self, num_classes=116, input_feat_dim=1, head='arcface', s=64.0, m=0.5):
        super().__init__()
        self.head_type = head.lower()  # 统一转小写，防止配置里大小写写错
        
        # 1. 骨干网络 (Backbone)
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(input_feat_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 2. 动态挂载分类头 (Head)
        if self.head_type == 'arcface':
            self.head = ArcFace(in_features=feat_dim, out_features=num_classes, s=s, m=m)
        elif self.head_type == 'linear':
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            raise ValueError(f"不支持的分类头类型: {self.head_type}")

    def forward(self, x, labels=None):
        feat = self.backbone(x) 
        feat = torch.flatten(feat, 1) 
        
        # 3. 动态路由转发逻辑
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
