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
    """
    配置双重输出的 Logger (文件 + 控制台)
    """
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
    """
    Train 专用：动态创建新的运行目录 (如 20260323_01_resnet_...)
    """
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


def parse_indices(indices_str):
    """
    辅助函数：将 '1-3,5' 这种字符串解析为集合 {1, 2, 3, 5}
    """
    if not indices_str:
        return set()
    indices = set()
    # 去除空格并按逗号分割
    for part in indices_str.replace(' ', '').split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices

def get_target_run_dirs(base_dir, method_name, run_name=None, run_id_str=None):
    """
    终极寻址引擎：支持跨日期精确寻址、最新日自动推断、编号范围解析
    """
    base_path = Path(base_dir)
    
    if run_name:
        target_path = base_path / run_name
        return [target_path] if target_path.exists() else []

    all_dirs = sorted([
        d for d in base_path.iterdir() 
        if d.is_dir() and d.name[0].isdigit()
    ])
    
    if not all_dirs:
        raise FileNotFoundError(f"在{base_dir}下未找到任何有效的实验目录。")

    if not run_id_str:
        if not method_name:
            return [all_dirs[-1]]
        filtered = [d for d in all_dirs if method_name in d.name]
        return [filtered[-1]] if filtered else []

    if '_' in run_id_str:
        target_date, indices_str = run_id_str.split('_', 1)
    else:
        target_date = all_dirs[-1].name.split('_')[0]
        indices_str = run_id_str
        
    target_indices = parse_indices(indices_str)
    
    found_dirs = []
    found_indices = set()
    
    for d in all_dirs:
        # 必须匹配指定的日期（或者推断出的最新日期）
        if d.name.startswith(target_date):
            parts = d.name.split('_')
            # 防止带有下划线但不是编号的错误文件夹解析
            if len(parts) >= 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx in target_indices:
                    found_dirs.append(d)
                    found_indices.add(idx)

    # 空缺与跨天检查报警
    missing = target_indices - found_indices
    if missing:
        print(f"警告：在日期 [{target_date}] 下，未能找到以下编号的实验目录: {sorted(list(missing))}")
        
    return found_dirs


def load_merged_config(model_name):
    """
    读取模型对应的配置文件。优先级：(cmd > )model > global
    """
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

def generate_synthetic_negatives(num_negatives, sample_shape):
    """
    根据真实数据的特征维度，全自适应生成负样本。
    
    Args:
        num_negatives (int): 需要生成的负样本数量。
        sample_shape (torch.Size 或 tuple): 单个样本特征的真实形状。
                                            例如 (105,), (1, 15, 15) 或 (3, 15, 15)。
    """
    neg_feats = torch.randn(num_negatives, *sample_shape) * 0.5 
    neg_labels = torch.full((num_negatives,), 0, dtype=torch.long)

    return neg_feats, neg_labels
