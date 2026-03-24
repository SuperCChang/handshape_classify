# models/__init__.py
from .mlp import MLP
from .resnet import ResNet
from .vgg import VGG

def build_model(model_name, num_classes=116, input_feat_dim=1, hidden_dim=256, head='arcface', **kwargs):
    if model_name == 'mlp':
        return MLP(num_classes=num_classes, input_feat_dim=input_feat_dim, hidden_dim=hidden_dim, head=head, **kwargs)
    elif model_name == 'resnet':
        return ResNet(num_classes=num_classes, input_feat_dim=input_feat_dim, head=head, **kwargs)
    elif model_name == 'vgg':
        return VGG(num_classes=num_classes, input_feat_dim=input_feat_dim, head=head, **kwargs)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}。")