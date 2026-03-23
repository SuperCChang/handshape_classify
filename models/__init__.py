from .mlp import MLP
from .mlp_arcface import MLP_ArcFace
from .resnet_arcface import ResNet_ArcFace

def build_model(model_name, num_classes=115, input_feat_dim=210, hidden_dim=256, **kwargs):
    if model_name == 'mlp':
        return MLP(num_classes=num_classes, input_feat_dim=input_feat_dim, hidden_dim=hidden_dim)
    elif model_name == 'mlp_arcface':
        return MLP_ArcFace(num_classes=num_classes, input_feat_dim=input_feat_dim, hidden_dim=hidden_dim, **kwargs)
    elif model_name == 'resnet_arcface':
        return ResNet_ArcFace(num_classes=num_classes, input_feat_dim=input_feat_dim, **kwargs)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}。请检查配置文件！")
