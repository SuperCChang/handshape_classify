import os
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import yaml

from models import build_model
from dataloaders.feature_extract import FEATURE_EXTRACTORS
from utils.core import get_target_run_dirs

def save_sequence_to_npy(sequence_array, save_path):
    """
    将标签序列保存为 .npy 文件的独立工具函数
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, sequence_array)
    logging.info(f"预测标签序列已成功保存至: {save_path}")


def smooth_mode(bimanual_sequence, window=5):
    """
    对双边标签序列进行滑动窗口众数平滑
    """
    logging.info(f"正在应用滑动窗口众数平滑 (Window Size = {window})...")
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


def classify(input_dir, model, cfg, device, smooth_window=5):
    """
    双手同步序列推理引擎：输入 F 帧目录，输出 (F, 2) 的预测标签 NumPy 数组。
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"找不到输入文件夹: {input_path}")
        return None

    # 1. 严格按文件名数字顺序读取序列
    smplx_list = sorted(input_path.glob('*.npy'), key=lambda x: int(x.stem))
    smplx_data_list = [np.load(f, allow_pickle=True).item() for f in smplx_list]
    
    if not smplx_data_list:
        logging.error("输入的文件夹为空！")
        return None

    # 2. 伪造 Dataset 所需的基础数据结构
    raw_data = [{'npy_list': smplx_data_list, 'side': 'LR', 'label': 0}]
    
    # 3. 动态调用特征提取器
    feature_type = cfg['data']['feature_type']
    smplx_model_path = cfg['data']['smplx_model_path']
    logging.info(f"正在提取序列特征 (提取策略: {feature_type})...")
    
    extractor_func = FEATURE_EXTRACTORS[feature_type]
    features, _ = extractor_func(raw_data, smplx_model_path=smplx_model_path)
    
    # 提取出的 features 列表中：[0] 为左手序列，[1] 为右手序列
    feat_tensor_l = features[0].to(device)
    feat_tensor_r = features[1].to(device)
    
    F = feat_tensor_l.shape[0]
    logging.info(f"特征提取完毕，序列长度: {F} 帧。")

    # 4. 模型前向推理
    model.eval()
    with torch.no_grad():
        logits_l = model(feat_tensor_l)
        preds_l = torch.argmax(logits_l, dim=1).cpu().numpy()  # (F,)
        
        logits_r = model(feat_tensor_r)
        preds_r = torch.argmax(logits_r, dim=1).cpu().numpy()  # (F,)

    bimanual_preds = np.stack((preds_l, preds_r), axis=1)

    # 5. 平滑处理
    if smooth_window > 0:
        bimanual_preds = smooth_mode(bimanual_preds, window=smooth_window)

    return bimanual_preds


def main():
    parser = argparse.ArgumentParser(description="手形序列单次推理脚本 (完全数据驱动版)")
    parser.add_argument('--input', type=str, required=True, help="输入文件夹路径 (包含多帧 SMPLX 的 .npy)")
    parser.add_argument('--output', type=str, required=True, help="预测结果保存路径 (如 xxx/output.npy)")
    parser.add_argument('--window', type=int, default=5, help="平滑窗口大小，设为0则不进行平滑")
    
    parser.add_argument('--weight', type=str, default='best_model.pth')
    parser.add_argument('--runs', type=str, default=None, help="指定实验标号 (默认自动寻址最新的一次)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    global_cfg_path = Path('configs/global_config.yaml')
    if not global_cfg_path.exists():
        logging.error("错误！找不到configs/global_config.yaml文件，请检查路径。")
        return
        
    with open(global_cfg_path, 'r', encoding='utf-8') as f:
        global_cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = global_cfg['train']['save_dir']

    run_dirs = get_target_run_dirs(save_dir, method_name="", run_id_str=args.runs)
    if not run_dirs:
        logging.error("错误！未找到任何符合条件的训练记录，无法加载权重。")
        return
        
    target_dir = run_dirs[-1]
    config_path = target_dir / "config_backup.yaml"
    weight_path = target_dir / args.weight
    
    if not weight_path.exists():
        logging.error(f"错误！找不到权重文件: {weight_path}")
        return
    if not config_path.exists():
        logging.error(f"错误！找不到配置文件: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    logging.info("="*60)
    logging.info(f"开始手形序列预测 (Inference)")
    logging.info(f"加载实验环境: {target_dir.name}")
    logging.info(f"模型架构: {cfg['model']['name']} | 分类头: {cfg['model']['head']}")
    logging.info(f"特征类型: {cfg['data']['feature_type']}")
    logging.info(f"目标输入路径: {args.input}")
    logging.info("="*60)

    feat_type = cfg['data']['feature_type']
    dynamic_feat_dim = 105 if feat_type == "distance_flatten" else 1
    if feat_type == "axis_angle":
        dynamic_feat_dim = 45

    model = build_model(
        model_name=cfg['model']['name'],
        head=cfg['model'].get('head', 'linear'),
        num_classes=cfg['data']['num_classes'],
        input_feat_dim=dynamic_feat_dim,
        hidden_dim=cfg['model'].get('hidden_dim', 256),
        s=cfg['model'].get('s', 30.0),
        m=cfg['model'].get('margin', 0.5)
    ).to(device)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    # 预测
    predicted_sequence = classify(
        input_dir=args.input,
        model=model,
        cfg=cfg,
        device=device,
        smooth_window=args.window
    )
    
    if predicted_sequence is not None:
        logging.info(f"输出预测矩阵形状: {predicted_sequence.shape}")
        save_sequence_to_npy(predicted_sequence, args.output)
        
    logging.info("预测成功！")

if __name__ == '__main__':
    main()