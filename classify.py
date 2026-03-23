import os
import torch
import numpy as np
import logging
import json
from pathlib import Path

from cnn import CNN, CNN_ArcFace
from load_data import FEATURE_EXTRACT

with open('config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

CHECKPOINT_PATH = CONFIG['paths']['checkpoint_save_dir']
SMPLX_MODEL_PATH = CONFIG['paths']['smplx_dir']
best_model_path = os.path.join(CHECKPOINT_PATH, f'best_model_{CONFIG["model"]["feature"]}.pth')

NUM_CLASSES = CONFIG['model']['num_classes']
HIDDEN_DIM = CONFIG['model']['hidden_dim']


def save_sequence_to_npy(sequence_array, save_path):
    """
    将标签序列保存为 .npy 文件的独立工具函数
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, sequence_array)
    logging.info(f"标签序列已成功保存至: {save_path}")


def smooth_mode(bimanual_sequence, window=5):
    """
    对双边标签序列进行滑动窗口众数平滑
    
    Args:
        bimanual_sequence: (F, 2) 的 NumPy 数组，包含左右手预测标签
        window: 滑动窗口大小，奇数
        
    Returns:
        sequence_smoothed: 平滑后的 (F, 2) 数组
    """
    logging.info(f"正在对标签使用窗口大小为{window}的众数平滑。")
    sequence_smoothed = np.zeros_like(bimanual_sequence)
    
    F = bimanual_sequence.shape[0]
    half_w = window // 2
    
    for i in range(F):
        start_idx = max(0, i - half_w)
        end_idx = min(F, i + half_w + 1)
        
        for hand_idx in range(bimanual_sequence.shape[1]):
            window_data = bimanual_sequence[start_idx:end_idx, hand_idx]
            
            vals, counts = np.unique(window_data, return_counts=True)
            
            mode_val = vals[np.argmax(counts)]
            sequence_smoothed[i, hand_idx] = mode_val
            
    return sequence_smoothed

def classify(input_dir, model_weights_path, device, smooth=True, save_path=None):
    """
    双手同步序列推理引擎：输入 F 帧，输出 (F, 2) 的 NumPy 数组
    其中第 0 列为左手预测标签，第 1 列为右手预测标签。

    Args:
    """
    logging.info(f"读取文件：{input_dir}")
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"找不到输入文件夹: {input_path}")
        return None

    smplx_list = sorted(input_path.glob('*.npy'))   # , key=lambda x: int(x.stem))
    smplx_data_list = [np.load(f, allow_pickle=True).item() for f in smplx_list]
    
    if not smplx_data_list:
        logging.error("输入的文件夹为空！")
        return None

    dummy_dict = {"infer_sample": (smplx_data_list, 'LR', 0)}
    
    feature_kwargs = {
        'smplx_dict': dummy_dict,
        'smplx_dir': CONFIG['paths']['smplx_dir'],
    }

    extract_feature = CONFIG['model']['feature']
    logging.info(f"正在使用特征提取方法: {extract_feature}")
    feature_list = FEATURE_EXTRACT[extract_feature](**feature_kwargs)
    
    feat_tensor_l = feature_list[0][0].to(device)
    feat_tensor_r = feature_list[1][0].to(device)
    
    F = feat_tensor_l.shape[0]
    feature_dim = feat_tensor_l.shape[1]
    logging.info(f"特征提取完毕，特征维度：{feature_dim}，序列长度: {F}帧。")

    model = CNN(
        num_classes=NUM_CLASSES, 
        input_feat_dim=feature_dim, 
        hidden_dim=HIDDEN_DIM, 
        device=device
    ).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval() 

    with torch.no_grad():
        logits_l = model(feat_tensor_l)
        preds_l = torch.argmax(logits_l, dim=1).cpu().numpy()  # (F,)
        
        logits_r = model(feat_tensor_r)
        preds_r = torch.argmax(logits_r, dim=1).cpu().numpy()  # (F,)

    bimanual_preds = np.stack((preds_l, preds_r), axis=1)

    if smooth:
        bimanual_preds = smooth_mode(bimanual_preds)

    if save_path is not None:
        save_sequence_to_npy(bimanual_preds, save_path)
        logging.info(f"预测标签已保存至: {save_path}")

    logging.info("推理完成！")
    return bimanual_preds


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    name = "舍己救人"
    first_letter = 'S'

    test_input_path = f"Z:/98/jz/shouyu/smplx_GJ_ccbr/{first_letter}/{name}/numpy_result"   
    save_path = rf"C:\Users\capg303\Desktop\Project\手形\smpl-x解析\output\{name}.npy"

    original_sequence = classify(
        input_dir=test_input_path, 
        model_weights_path=best_model_path, 
        device=device,
        smooth=False,
        save_path=None
    )
    logging.info(f"原始预测标签（未平滑）:\n{original_sequence[:, 0]}\n{original_sequence[:, 1]}")

    predicted_sequence = classify(
        input_dir=test_input_path, 
        model_weights_path=best_model_path, 
        device=device,
        save_path=save_path
    )
    logging.info(f"左右手的预测标签为:\n{predicted_sequence[:, 0]}\n{predicted_sequence[:, 1]}")
