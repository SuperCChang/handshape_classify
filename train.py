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

from utils.core import setup_logger, get_new_run_dir, load_merged_config, generate_synthetic_negatives


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    专门负责一个 Epoch 内的训练逻辑。
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
    parser = argparse.ArgumentParser(description="手形分类训练脚本")
    parser.add_argument('--model', type=str, default='resnet_arcface')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    cfg = load_merged_config(args.model)

    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    if args.lr is not None:
        cfg['train']['lr'] = args.lr
    if args.batch_size is not None:
        cfg['train']['batch_size'] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    method_name = f"{cfg['model']['name']}_{cfg['data']['feature_type']}"
    run_dir = get_new_run_dir(cfg['train']['save_dir'], method_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "train.log")

    try:
        logger.info(f"{'='*60}")
        logger.info(f"实验初始化")
        logger.info(f"{'='*60}")
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

        sample_feat = dataset[0][0]
        dynamic_feat_dim = sample_feat.shape[0]
        logger.info(f"[*] 动态推断特征维度为: {dynamic_feat_dim}")

        logger.info(f"检测到类别数为 {cfg['data']['num_classes']}，正在生成第 117 类(负样本)...")
        neg_count = int(len(dataset) * 0.08) 
        neg_feats, neg_labels = generate_synthetic_negatives(
            num_negatives=neg_count,
            feature_type=cfg['data']['feature_type']
        )
        dataset.append_samples(neg_feats, neg_labels)
        logger.info(f"已注入{neg_count}条合成负样本。")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg['train']['batch_size'], 
            shuffle=True, 
            pin_memory=True
        )
        logger.info(f"数据集构建完毕，共计{len(dataset)}个样本，划分为{len(dataloader)}个Batch。")
        
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
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
            
        if not (run_dir / "last_model.pth").exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            print(f"\n[清理机制触发]: 检测到实验未完成即崩溃，已自动删除无效目录 -> {run_dir}")
            
        error_dir = Path(cfg['train']['save_dir']) / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        
        error_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_file = error_dir / f"error_{error_time}_{method_name}.log"
        
        with open(error_log_file, 'w', encoding='utf-8') as ef:
            ef.write(f"========== 崩溃时间: {error_time} ==========\n")
            ef.write(f"实验名称: {method_name}\n")
            ef.write(f"关键配置: Batch={cfg['train']['batch_size']}, LR={cfg['train']['lr']}\n")
            ef.write(f"报错简述: {e}\n\n")
            ef.write("========== 完整堆栈 (Traceback) ==========\n")
            ef.write(error_traceback)
            
        print(f"[错误日志记录]: 本次崩溃的详细堆栈已独立封存至 -> {error_log_file}")
        raise e


if __name__ == '__main__':
    main()
