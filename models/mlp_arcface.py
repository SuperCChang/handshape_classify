import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features=115, s=30.0, m=0.50):
        """
        Args:
            in_features: 输入特征的维度
            out_features: 类别数
            s: 缩放尺度 (Scale)。通常设为 30 到 64 之间。
            m: 角度裕度 (Margin)。控制类间推开的力度，通常 0.3~0.5。
        """
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)       
        output *= self.s
        
        return output


class MLP_ArcFace(nn.Module):
    def __init__(self, num_classes=115, input_feat_dim=105, hidden_dim=256, s=30.0, m=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.arcface = ArcFace(in_features=hidden_dim, out_features=num_classes, s=s, m=m)


    def forward(self, x, labels=None):

        feat = self.encoder(x) # 输出形状: (batch_size, hidden_dim)
        
        if self.training:
            if labels is None:
                raise ValueError("训练模式必须输入标签！")
            # 将提取出的 feat 送给 arcface
            logits = self.arcface(feat, labels)
            return logits
        else:
            # 将提取出的 feat 用于计算余弦相似度
            normalized_feat = F.normalize(feat)
            normalized_weight = F.normalize(self.arcface.weight)
            cosine_similarity = F.linear(normalized_feat, normalized_weight)
            return cosine_similarity * self.arcface.s