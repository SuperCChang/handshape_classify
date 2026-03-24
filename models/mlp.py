import torch
import torch.nn as nn
import torch.nn.functional as F
from .heads import ArcFace

class MLP(nn.Module):
    def __init__(self, num_classes=116, input_feat_dim=105, hidden_dim=512, head='arcface', s=64.0, m=0.5):
        super().__init__()
        self.head_type = head.lower()
        
        self.backbone = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        if self.head_type == 'arcface':
            self.head = ArcFace(in_features=hidden_dim, out_features=num_classes, s=s, m=m)
        elif self.head_type == 'linear':
            self.head = nn.Linear(hidden_dim, num_classes)
        else:
            raise ValueError(f"不支持的分类头类型: {self.head_type}")

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        
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
