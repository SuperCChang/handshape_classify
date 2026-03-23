# utils/core.py
import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch

class TqdmLoggingHandler(logging.Handler):
    """解决 logging 和 tqdm 进度条冲突的 Handler"""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(log_file_path):
    """配置双重输出的 Logger (文件 + 控制台)"""
    logger = logging.getLogger('project_logger')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_new_run_dir(base_dir, method_name):
    """Train 专用：动态创建新的运行目录 (如 20260323_01_resnet_...)"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    existing_dirs = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith(date_str)]
    
    idx = 0
    if existing_dirs:
        indices = []
        for d in existing_dirs:
            parts = d.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                indices.append(int(parts[1]))
        if indices:
            idx = max(indices) + 1
            
    run_name = f"{date_str}_{idx:02d}_{method_name}"
    return base_path / run_name


def get_latest_run_dir(base_dir, method_name):
    """Evaluate/Infer 专用：自动寻找最近一次的实验目录"""
    base_path = Path(base_dir)
    existing_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and method_name in d.name])
    if not existing_dirs:
        raise FileNotFoundError(f"未找到关于 {method_name} 的历史实验记录！")
    return existing_dirs[-1]


def load_merged_config(model_name):
    with open('configs/global_config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    model_cfg_path = Path(f'configs/{model_name}.yaml')
    if model_cfg_path.exists():
        with open(model_cfg_path, 'r', encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
        
        for key, value in model_cfg.items():
            if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg

def generate_synthetic_negatives(num_negatives, feature_type="distance_flatten"):
    """
    根据特征类型自动生成负样本。
    - distance_flatten: 生成 (num_negatives, 105) 的 1D 向量
    - distance_matrix: 生成 (num_negatives, 1, 15, 15) 的 2D 图像
    """
    if feature_type == "distance_flatten":
        # 生成 105 维的高斯噪声向量
        neg_feats = torch.randn(num_negatives, 105) * 0.5 
        neg_labels = torch.full((num_negatives,), 0, dtype=torch.long)
    
    elif feature_type == "distance_matrix":
        # 生成 1x15x15 的高斯噪声图像
        neg_feats = torch.randn(num_negatives, 1, 15, 15) * 0.5
        neg_labels = torch.full((num_negatives,), 0, dtype=torch.long)
    
    else:
        # 兼容 axis_angle 等其他 45 维特征
        neg_feats = torch.randn(num_negatives, 45) * 0.5
        neg_labels = torch.full((num_negatives,), 0, dtype=torch.long)

    return neg_feats, neg_labels
