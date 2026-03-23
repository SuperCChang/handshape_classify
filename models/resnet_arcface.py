import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .mlp_arcface import ArcFace

class ResNet_ArcFace(nn.Module):
    def __init__(self, num_classes=116, input_feat_dim=105, s=30.0, m=0.5):
        super().__init__()
        
        # 官方 ResNet18
        self.backbone = models.resnet18(pretrained=False)
        
        # 魔改第一层：接收单通道，小卷积核
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.arcface = ArcFace(in_features=feat_dim, out_features=num_classes, s=s, m=m)

    def _restore_matrix(self, vec, n=15):
        """
        【魔法函数】：将 105 维无损还原为 (Batch, 1, 15, 15) 的对称矩阵
        完全不需要改动你的 DataLoader！
        """
        batch_size = vec.shape[0]
        # 初始化 15x15 的全零矩阵
        mat = torch.zeros((batch_size, n, n), device=vec.device)
        
        # 获取上三角索引
        triu_indices = torch.triu_indices(n, n, offset=1)
        
        # 填入上三角
        mat[:, triu_indices[0], triu_indices[1]] = vec
        # 镜像填入下三角，对角线保持为 0
        mat[:, triu_indices[1], triu_indices[0]] = vec
        
        # 增加通道维度使其变成 "单通道图像"
        return mat.unsqueeze(1) # 输出: (Batch, 1, 15, 15)

    def forward(self, x, labels=None):
        # x 的输入形状依然是你原本的 (Batch, 105)
        
        # 1. 动态还原成 2D 矩阵图像
        img_x = self._restore_matrix(x, n=15)
        
        # 2. 喂给 ResNet 提取高阶拓扑特征
        feat = self.backbone(img_x) 
        
        # 3. 走 ArcFace 分类流程
        if self.training:
            if labels is None:
                raise ValueError("训练模式必须输入标签！")
            logits = self.arcface(feat, labels)
            return logits
        else:
            normalized_feat = F.normalize(feat)
            normalized_weight = F.normalize(self.arcface.weight)
            return F.linear(normalized_feat, normalized_weight) * self.arcface.s
