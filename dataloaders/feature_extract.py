import torch
import numpy as np
import smplx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 底层数学计算辅助函数
# ==========================================

def _compute_distance_flatten(joints_coord):
    """辅助函数 1：计算距离矩阵，并截取上三角展平为 105 维"""
    num_joints = joints_coord.shape[1]
    dist_matrix = torch.cdist(joints_coord, joints_coord, p=2)
    triu_indices = torch.triu_indices(num_joints, num_joints, offset=1)
    return dist_matrix[:, triu_indices[0], triu_indices[1]] # 输出形状: (F, 105)

def _compute_distance_matrix(joints_coord):
    """辅助函数 2：计算完整距离矩阵，并补充通道维度变为图像"""
    dist_matrix = torch.cdist(joints_coord, joints_coord, p=2)
    return dist_matrix.unsqueeze(1) # 输出形状: (F, 1, 15, 15)


def _get_3d_joints_from_smplx(raw_data_list, smplx_model_path):
    """核心通用引擎：将 raw_data 转为真实的 3D 关节坐标集合"""
    if smplx_model_path is None:
        raise ValueError("使用距离特征时必须提供 smplx_model_path")
        
    smplx_model = smplx.create(
        model_path=str(smplx_model_path), model_type='smplx',
        gender='neutral', use_pca=False, flat_hand_mean=True, ext='pkl'
    ).to(device)
    smplx_model.eval()

    joint_sequences = [] # 存放提取出的有效 3D 关节 (F, 15, 3)
    labels = []
    
    key_mapping = {
        'smplx_root_pose': 'global_orient', 'smplx_body_pose': 'body_pose',
        'smplx_lhand_pose': 'left_hand_pose', 'smplx_rhand_pose': 'right_hand_pose',
        'smplx_jaw_pose': 'jaw_pose', 'smplx_shape': 'betas', 'smplx_expr': 'expression'
    }

    official_dims = {
        'global_orient': 3, 'body_pose': 63, 'jaw_pose': 3,
        'leye_pose': 3, 'reye_pose': 3, 'left_hand_pose': 45, 'right_hand_pose': 45,
        'betas': 10, 'expression': 10
    }

    for item in raw_data_list:
        smplx_numpy_list, side, label = item['npy_list'], item['side'], item['label']
        frame_num = len(smplx_numpy_list)
        kwargs_for_smplx = {}
        
        for npy_key, official_key in key_mapping.items():
            if npy_key in smplx_numpy_list[0]:
                stacked_array = np.stack([d[npy_key] for d in smplx_numpy_list])
                kwargs_for_smplx[official_key] = torch.tensor(stacked_array).float().view(frame_num, -1).to(device)
        
        for off_key, dim in official_dims.items():
            if off_key not in kwargs_for_smplx:
                kwargs_for_smplx[off_key] = torch.zeros((frame_num, dim), dtype=torch.float32, device=device)

        with torch.no_grad():
            smplx_output = smplx_model(**kwargs_for_smplx)
            
        all_joints = smplx_output.joints
        
        if 'L' in side:
            lhand = all_joints[:, 25:40, :].view(-1, 15, 3)
            joint_sequences.append(lhand)
            labels.append(label)
        if 'R' in side:
            rhand = all_joints[:, 40:55, :].view(-1, 15, 3)
            joint_sequences.append(rhand)
            labels.append(label)

    return joint_sequences, labels


def extract_axis_angle(raw_data_list, **kwargs):
    """特征方案 1：直接使用 45 维轴角"""
    features = []
    labels = []
    for item in raw_data_list:
        smplx_numpy_list, side, label = item['npy_list'], item['side'], item['label']
        if 'L' in side:
            lhand = torch.tensor(np.stack([d['smplx_lhand_pose'] for d in smplx_numpy_list])).float().reshape(-1, 45)
            features.append(lhand)
            labels.append(label)
        if 'R' in side:
            rhand = torch.tensor(np.stack([d['smplx_rhand_pose'] for d in smplx_numpy_list])).float().reshape(-1, 45)
            features.append(rhand)
            labels.append(label)
    return features, labels


def extract_distance_flatten(raw_data_list, smplx_model_path=None, **kwargs):
    """
    特征方案 2：105 维 3D 关节欧氏距离展平向量
    -> 适用模型：MLP 系列 (由于是一维向量，直接丢给全连接层)
    """
    joint_seqs, labels = _get_3d_joints_from_smplx(raw_data_list, smplx_model_path)
    features = []
    for joints in joint_seqs:
        # 将 (F, 15, 3) 转化为 (F, 105)
        features.append(_compute_distance_flatten(joints).cpu())
    return features, labels


def extract_distance_matrix(raw_data_list, smplx_model_path=None, **kwargs):
    """
    特征方案 3：完整的 1x15x15 关节距离热力图 (伪装成单通道灰度图)
    -> 适用模型：CNN, VGG, ResNet 系列 (由于是二维矩阵，直接丢给 Conv2d)
    """
    joint_seqs, labels = _get_3d_joints_from_smplx(raw_data_list, smplx_model_path)
    features = []
    for joints in joint_seqs:
        # 将 (F, 15, 3) 转化为 (F, 1, 15, 15)
        features.append(_compute_distance_matrix(joints).cpu())
    return features, labels


FEATURE_EXTRACTORS = {
    "axis_angle": extract_axis_angle,
    "distance_flatten": extract_distance_flatten,  # <--- 修改了名字
    "distance_matrix": extract_distance_matrix     # <--- 新的二维提取方式
}
