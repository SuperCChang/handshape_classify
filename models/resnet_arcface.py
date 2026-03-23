import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .mlp_arcface import ArcFace

class ResNet_ArcFace(nn.Module):
    def __init__(self, num_classes=116, input_feat_dim=1, s=30.0, m=0.5):
        super().__init__()
        
        self.backbone = models.resnet18(weights=None)
        
        self.backbone.conv1 = nn.Conv2d(input_feat_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.arcface = ArcFace(in_features=feat_dim, out_features=num_classes, s=s, m=m)

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
