import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from dataloaders.dataset import HandshapeDataset
from torch.utils.data import DataLoader
from models import build_model

from utils.core import load_merged_config, get_latest_run_dir, setup_logger


def calculate_topk_accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def plot_confusion_matrix(cm, save_path, num_classes):
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap='OrRd', cbar=True)
    plt.title('Normalized Confusion Matrix Heatmap')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_single_split(model, dataloader, device, num_classes, split_name):
    """评估单个数据集的内部逻辑"""
    model.eval()
    all_targets = []
    all_outputs = []
    top1_sum = 0.0
    top5_sum = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating [{split_name}]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            all_outputs.append(outputs.cpu())
            all_targets.append(labels.cpu())
            
            acc1, acc5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
            top1_sum += acc1
            top5_sum += acc5

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    _, all_preds = all_outputs.max(dim=1)

    avg_top1 = top1_sum / len(dataloader)
    avg_top5 = top5_sum / len(dataloader)
    macro_f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average='macro')
    cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=list(range(num_classes)), normalize='true')
    class_report = classification_report(all_targets.numpy(), all_preds.numpy(), zero_division=0)

    return {
        'top1': avg_top1,
        'top5': avg_top5,
        'macro_f1': macro_f1 * 100,
        'cm': cm,
        'report': class_report
    }


def main():
    parser = argparse.ArgumentParser(description="手形分类评估脚本 (包含过拟合分析)")
    parser.add_argument('--model', type=str, default='resnet_arcface')
    parser.add_argument('--weight', type=str, default='best_model.pth')
    args = parser.parse_args()

    cfg = load_merged_config(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg['data']['num_classes']

    method_name = f"{cfg['model']['name']}_{cfg['data']['feature_type']}"
    run_dir = get_latest_run_dir(cfg['train']['save_dir'], method_name)
    weight_path = run_dir / args.weight

    log_file = run_dir / "evaluation_report.log"
    logger = setup_logger(log_file)

    logger.info("开始评估")
    logger.info(f"目标实验目录: {run_dir}")
    logger.info(f"加载权重文件: {weight_path}")

    # 1. 准备 Train 和 Val 数据集
    logger.info("加载训练集...")
    train_dataset = HandshapeDataset(
        train_path=cfg['data']['train_path'],
        which_side_path=cfg['data']['which_side_path'],
        feature_type=cfg['data']['feature_type'],
        smplx_model_path=cfg['data']['smplx_model_path']
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    
    # 动态推断特征维度并构建模型
    sample_inputs, _ = next(iter(train_loader))
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
    logger.info("模型权重加载成功。")

    train_metrics = evaluate_single_split(model, train_loader, device, num_classes, "Train Set")
    
    val_metrics = None
    if 'val_path' in cfg['data'] and cfg['data']['val_path']:
        logger.info("加载验证集...")
        val_dataset = HandshapeDataset(
            train_path=cfg['data']['val_path'],
            which_side_path=cfg['data']['which_side_path'],
            feature_type=cfg['data']['feature_type'],
            smplx_model_path=cfg['data']['smplx_model_path']
        )
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        val_metrics = evaluate_single_split(model, val_loader, device, num_classes, "Validation Set")
    else:
        logger.warning("未在配置中找到val_path，将跳过过拟合分析。")

    logger.info("="*60)
    logger.info("               评估报告                ")
    logger.info("="*60)
    logger.info(f"{'指标 (Metrics)':<20} | {'训练集 (Train)':<15} | {'验证集 (Val)':<15}")
    logger.info("-" * 60)
    
    val_top1_str = f"{val_metrics['top1']:.2f}%" if val_metrics else "N/A"
    val_top5_str = f"{val_metrics['top5']:.2f}%" if val_metrics else "N/A"
    val_f1_str = f"{val_metrics['macro_f1']:.2f}%" if val_metrics else "N/A"

    logger.info(f"{'Top-1 Accuracy':<20} | {train_metrics['top1']:<15.2f}% | {val_top1_str}")
    logger.info(f"{'Top-5 Accuracy':<20} | {train_metrics['top5']:<15.2f}% | {val_top5_str}")
    logger.info(f"{'Macro F1-Score':<20} | {train_metrics['macro_f1']:<15.2f}% | {val_f1_str}")
    
    logger.info("-" * 60)
    if val_metrics:
        acc_gap = train_metrics['top1'] - val_metrics['top1']
        f1_gap = train_metrics['macro_f1'] - val_metrics['macro_f1']
        
        status = "健康 (拟合良好)"
        if acc_gap > 10.0:
            status = "严重过拟合 (建议增加正则化或检查数据漏泄)"
        elif acc_gap > 4.0:
            status = "轻微过拟合 (处于正常警戒边缘)"
            
        logger.info(f"泛化差距 (Acc Gap) : {acc_gap:.2f}% -> {status}")
        logger.info(f"F1 泛化差距 (F1 Gap): {f1_gap:.2f}%")
    else:
        logger.info("过拟合状态: [需配置验证集数据方可计算]")
    logger.info("="*60)

    plot_confusion_matrix(train_metrics['cm'], run_dir / "confusion_matrix_train.png", num_classes)
    with open(run_dir / "train_report.txt", 'w', encoding='utf-8') as f:
        f.write(train_metrics['report'])

    if val_metrics:
        plot_confusion_matrix(val_metrics['cm'], run_dir / "confusion_matrix_val.png", num_classes)
        with open(run_dir / "validate_report.txt", 'w', encoding='utf-8') as f:
            f.write(val_metrics['report'])
            
    logger.info(f"评估完成！评估报告已保存至: {log_file}")
    logger.info(f"混淆矩阵及详细分类报告已保存至: {run_dir}")

if __name__ == '__main__':
    main()
