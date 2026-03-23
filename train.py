import os
import yaml
import argparse
import logging
import shutil
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloaders.dataset import HandshapeDataset
from torch.utils.data import DataLoader
from models import build_model


class TqdmLoggingHandler(logging.Handler):
    """
    自定义的 Logging Handler，解决标准 logging 输出会打断 tqdm 进度条的问题。
    """
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
    """配置日志记录器，同时输出到文件和控制台(适配tqdm)"""
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # 清除默认的 handlers

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_run_dir(base_dir, method_name):
    """
    动态生成运行目录，格式为: {Date}_{Daily_Run_Index}_{Method_Name}
    例如: runs/20260323_00_mlp_arcface
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
            # 假设文件夹名称格式严格为 YYYYMMDD_XX_method_name
            if len(parts) >= 2 and parts[1].isdigit():
                indices.append(int(parts[1]))
        if indices:
            idx = max(indices) + 1
            
    run_name = f"{date_str}_{idx:02d}_{method_name}"
    return base_path / run_name


def load_config():
    """解析参数并合并配置"""
    parser = argparse.ArgumentParser(description="手形分类训练脚本")
    parser.add_argument('--model', type=str, default='mlp_arcface', help='要训练的模型名称(对应configs/下的yaml)')
    args = parser.parse_args()

    # 载入全局配置
    with open('configs/global_config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 载入并合并模型配置
    with open(f'configs/{args.model}.yaml', 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)

    for key, value in model_cfg.items():
        if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key].update(value)
        else:
            cfg[key] = value

    return cfg


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    专门负责一个 Epoch 内的训练逻辑。
    为保持进度条清爽，这里内部不打印任何 log，仅返回最终计算指标。
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs, labels=labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    acc = 100. * correct / total
    return avg_loss, acc


def main():
    # 1. 初始化配置与运行目录
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    method_name = f"{cfg['model']['name']}_{cfg['data']['feature_type']}"
    run_dir = get_run_dir(cfg['train']['save_dir'], method_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. 配置 Logger
    log_file = run_dir / "train.log"
    logger = setup_logger(log_file)

    try:
        logger.info(f"============== 实验初始化 ==============")
        logger.info(f"运行目录：{run_dir}")
        logger.info(f"计算设备：{device}")
        logger.info(f"特征提取：{cfg['data']['feature_type']}")

        with open(run_dir / "config_backup.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True)

        logger.info("正在构建数据集")
        dataset = HandshapeDataset(
            train_path=cfg['data']['train_path'],
            which_side_path=cfg['data']['which_side_path'],
            feature_type=cfg['data']['feature_type'],
            smplx_model_path=cfg['data']['smplx_model_path']
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg['train']['batch_size'], 
            shuffle=True, 
            pin_memory=True
        )
        logger.info(f"数据集构建完毕，共计{len(dataset)}个样本，划分为{len(dataloader)}个Batch。")

        sample_inputs, _ = next(iter(dataloader))
        dynamic_feat_dim = sample_inputs.shape[1]
        logger.info(f"[*] 动态推断特征维度为: {dynamic_feat_dim}")

        logger.info(f"正在构建模型 [{cfg['model']['name']}]...")
        model = build_model(
            model_name=cfg['model']['name'],
            num_classes=cfg['data']['num_classes'],
            input_feat_dim=dynamic_feat_dim,
            hidden_dim=cfg['model'].get('hidden_dim', 256),
            s=cfg['model'].get('s', 30.0),
            m=cfg['model'].get('margin', 0.5)
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'])

        epochs = cfg['train']['epochs']
        logger.info(f"开始训练，共计{epochs}Epochs...")
        
        best_loss = float('inf')
        history_loss = []
        history_acc = []

        with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
            for epoch in range(1, epochs + 1):
                
                train_loss, train_acc = train_epoch(model, dataloader, criterion, optimizer, device)
                
                history_loss.append(train_loss)
                history_acc.append(train_acc)
                
                pbar.set_postfix({'Loss': f'{train_loss:.4f}', 'Acc': f'{train_acc:.2f}%'})
                pbar.update(1)
                
                logger.info(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
                
                # 保存逻辑
                torch.save(model.state_dict(), run_dir / f"last_model.pth")
                if train_loss < best_loss:
                    best_loss = train_loss
                    torch.save(model.state_dict(), run_dir / f"best_model.pth")
                    logger.info(f"--> [发现新最优模型]Loss降至{best_loss:.4f}，已保存至best_model.pth")

        logger.info("训练结束！正在生成并保存训练曲线...")

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), history_loss, marker='', linestyle='-', color='r', label='Train Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('CrossEntropy Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), history_acc, marker='', linestyle='-', color='b', label='Train Acc')
        plt.title('Training Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(run_dir / 'training_curves.png', dpi=300)
        plt.close()
        
        logger.info(f"所有训练结果及曲线图已成功保存至: {run_dir}")

    except Exception as e:
        logger.error(f"训练发生意外崩溃: {e}")
        logger.error(traceback.format_exc())
        
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
            
        if not (run_dir / "last_model.pth").exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            print(f"\n[清理机制触发]: 检测到实验未完成即崩溃，已自动删除无效目录 -> {run_dir}")
            
        raise e


if __name__ == '__main__':
    main()
