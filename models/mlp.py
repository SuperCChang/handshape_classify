import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes=116, input_feat_dim=45, hidden_dim=256, device='cpu'):
        """
        全连接分类模组 (MLP)，专门用于单帧静态手形分类。
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_feat_dim = input_feat_dim
        self.device = device

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
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    

    def forward(self, x, labels=None):
        """
        前向传播
        Args:
            x: (batch_size, input_feat_dim) 纯粹的独立单帧特征矩阵
        """
        feat = self.encoder(x)
        
        logits = self.classifier(feat) # (batch_size, num_classes)
        
        return logits
