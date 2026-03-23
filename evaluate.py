import argparse
import yaml
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from dataloaders.dataset import HandshapeDataset
from torch.utils.data import DataLoader
from models import build_model


def load_config():
    """解析参数并合并配置"""
    parser = argparse.ArgumentParser(description="手形分类评估脚本")
    parser.add_argument('--model', type=str, default='mlp_arcface', help='要评估的模型名称')
    parser.add_argument('--weight', type=str, default='best_model.pth', help='要加载的权重文件名')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='评估的数据集划分')
    args = parser.parse_args()

    with open('configs/global_config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    with open(f'configs/{args.model}.yaml', 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)

    for key, value in model_cfg.items():
        if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key].update(value)
        else:
            cfg[key] = value

    cfg['experiment_name'] = f"{cfg['experiment_base_name']}_{args.model}_{cfg['data']['feature_type']}"
    return cfg, args


def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """计算 Top-K 准确率"""
    maxk = max(topk)
    batch_size = target.size(0)

    # 获取预测概率最大的前 maxk 个类别的索引
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def plot_confusion_matrix(cm, save_path, num_classes):
    """绘制并保存混淆矩阵热力图"""
    plt.figure(figsize=(24, 20)) # 115类矩阵非常大，需要高分辨率画板
    sns.heatmap(cm, annot=False, cmap='OrRd', cbar=True)
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def calculate_overfitting_gap(train_metrics, val_metrics):
    """
    预留接口：计算过拟合指标 (Generalization Gap)
    通过对比训练集和验证集的性能落差来量化过拟合程度。
    """
    if val_metrics is None:
        return "无法计算 (缺失验证集数据)"
    
    acc_gap = train_metrics['top1'] - val_metrics['top1']
    # 如果验证集准确率比训练集低超过 5%，通常被视为过拟合的危险信号
    status = "严重过拟合" if acc_gap > 5.0 else "正常"
    
    return f"Acc Gap: {acc_gap:.2f}% ({status})"


def evaluate(model, dataloader, device, num_classes):
    """核心推理与评估逻辑"""
    model.eval()
    
    all_targets = []
    all_outputs = []
    
    top1_sum = 0.0
    top5_sum = 0.0
    total_batches = len(dataloader)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 推理：ArcFace 在 eval 模式下直接输出余弦相似度分数
            outputs = model(inputs)
            
            # 收集用于计算全局指标的数据
            all_outputs.append(outputs.cpu())
            all_targets.append(labels.cpu())
            
            # 计算当前 Batch 的 Top-K
            acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
            top1_sum += acc1
            top5_sum += acc5

    # 聚合所有数据
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    _, all_preds = all_outputs.max(dim=1)

    # 1. 计算全局 Top-1 和 Top-5
    avg_top1 = top1_sum / total_batches
    avg_top5 = top5_sum / total_batches

    # 2. 计算 Macro F1 分数
    macro_f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average='macro')

    # 3. 计算混淆矩阵
    cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=list(range(num_classes)), normalize='true')

    # 4. 生成详细的分类报告 (按需获取每个类的 Precision, Recall, F1)
    class_report = classification_report(all_targets.numpy(), all_preds.numpy(), zero_division=0)

    metrics = {
        'top1': avg_top1,
        'top5': avg_top5,
        'macro_f1': macro_f1 * 100,
        'cm': cm,
        'report': class_report
    }
    return metrics

def main():
    cfg, args = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg['data']['num_classes']

    # 动态寻址：找到最近一次运行该模型的文件夹
    base_dir = Path(cfg['train']['save_dir'])
    method_name = f"{cfg['model']['name']}_{cfg['data']['feature_type']}"
    existing_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and method_name in d.name])
    
    if not existing_dirs:
        raise FileNotFoundError(f"未找到关于 {method_name} 的训练记录文件夹！")
    
    run_dir = existing_dirs[-1] # 默认取最新的一次实验
    weight_path = run_dir / args.weight

    logging.info(f"========== 开始评估 ==========")
    logging.info(f"目标实验目录: {run_dir}")
    logging.info(f"加载权重文件: {weight_path}")
    logging.info(f"评估数据划分: {args.split} 集")

    # 1. 准备数据 (预留了拆分接口，目前直接读取你唯一的路径)
    dataset = HandshapeDataset(
        train_path=cfg['data']['train_path'], # 未来如果有 val_path，可以通过 args.split 判断传入
        which_side_path=cfg['data']['which_side_path'],
        feature_type=cfg['data']['feature_type'],
        smplx_model_path=cfg['data']['smplx_model_path']
    )
    dataloader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=False, pin_memory=True)

    # 2. 动态推断特征维度并构建模型
    sample_inputs, _ = next(iter(dataloader))
    dynamic_feat_dim = sample_inputs.shape[1]
    
    model = build_model(
        model_name=cfg['model']['name'],
        num_classes=num_classes,
        input_feat_dim=dynamic_feat_dim,
        hidden_dim=cfg['model'].get('hidden_dim', 256),
        s=cfg['model'].get('s', 30.0),
        m=cfg['model'].get('margin', 0.5)
    ).to(device)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    logging.info("模型权重加载成功。")

    metrics = evaluate(model, dataloader, device, num_classes)

    logging.info("\n" + "="*40)
    logging.info("             评估结果报告             ")
    logging.info("="*40)
    logging.info(f"Top-1 Accuracy : {metrics['top1']:.2f}%")
    logging.info(f"Top-5 Accuracy : {metrics['top5']:.2f}%")
    logging.info(f"Macro F1-Score : {metrics['macro_f1']:.2f}%")
    
    # 【预留过拟合计算接口】
    # val_metrics = evaluate(model, val_dataloader, device, num_classes)
    # gap_info = calculate_overfitting_gap(train_metrics=metrics, val_metrics=val_metrics)
    # logging.info(f"Overfitting Gap: {gap_info}")
    logging.info("Overfitting Gap: [等待引入验证集后计算]")
    logging.info("="*40)

    cm_path = run_dir / f"confusion_matrix_{args.split}.png"
    plot_confusion_matrix(metrics['cm'], cm_path, num_classes)
    
    report_path = run_dir / f"classification_report_{args.split}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(metrics['report'])

    logging.info(f"\n混淆矩阵已保存至: {cm_path}")
    logging.info(f"详细各类别召回率报告已保存至: {report_path}")

if __name__ == '__main__':
    main()
