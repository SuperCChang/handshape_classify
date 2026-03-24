import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from dataloaders.dataset import HandshapeDataset
from torch.utils.data import DataLoader
from models import build_model

from utils.core import get_target_run_dirs, setup_logger


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


def analyze_top_confusions(cm, logger, split_name="Train Set", top_k=15):
    """找出最容易混淆的类别对并记录到日志中"""
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0.0)
    top_indices = np.argsort(cm_off_diag, axis=None)[::-1][:top_k]
    num_classes = cm.shape[1]
    
    logger.info(f"\n[{split_name}] 最易混淆的 Top-{top_k} 类别对：")
    logger.info("-" * 60)
    logger.info(f"{'真实标签 (True)':<15} -> {'错判为 (Pred)':<15} | {'错判比例 (Error Rate)':<15}")
    logger.info("-" * 60)
    
    for idx in top_indices:
        true_label = idx // num_classes
        pred_label = idx % num_classes
        error_rate = cm_off_diag[true_label, pred_label]
        if error_rate <= 0.0: break
        logger.info(f"Class {true_label:<10} -> Class {pred_label:<10} | {error_rate * 100:>5.2f}%")
    logger.info("-" * 60)


def evaluate_single_split(model, dataloader, device, num_classes, split_name):
    """评估单个数据集的内部逻辑"""
    model.eval()
    all_targets = []
    all_outputs = []
    top1_sum = 0.0
    top5_sum = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating [{split_name}]", leave=False):
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


def run_evaluation_for_dir(run_dir, cfg, args, device, num_classes, train_loader, val_loader, dynamic_feat_dim):
    """处理单一目录的评估流程，保持原始的详尽输出，并隔离日志器"""
    log_file = run_dir / "evaluation_report.log"
    logger = setup_logger(log_file)
    weight_path = run_dir / args.weight

    logger.info(f"\n{'='*60}")
    logger.info(f"正在评估实验: {run_dir.name}")
    logger.info(f"模型架构: {cfg['model']['name']} | 分类头: {cfg['model'].get('head', 'linear')}")
    logger.info(f"特征类型: {cfg['data']['feature_type']}")
    logger.info(f"{'='*60}")
    
    if not weight_path.exists():
        logger.error(f"[致命错误] 权重文件不存在，跳过该目录: {weight_path}")
        return

    # 从当前文件夹专属的 cfg 动态构建模型
    model = build_model(
        model_name=cfg['model']['name'],
        head=cfg['model'].get('head', 'linear'),
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
    if val_loader:
        val_metrics = evaluate_single_split(model, val_loader, device, num_classes, "Validation Set")
    else:
        logger.warning("未在配置中找到 val_path，将跳过验证集评估。")

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

    # 打印错题本
    analyze_top_confusions(train_metrics['cm'], logger, "Train Set", top_k=10)
    if val_metrics:
        analyze_top_confusions(val_metrics['cm'], logger, "Validation Set", top_k=10)

    # 保存文件
    plot_confusion_matrix(train_metrics['cm'], run_dir / "confusion_matrix_train.png", num_classes)
    with open(run_dir / "train_report.txt", 'w', encoding='utf-8') as f:
        f.write(train_metrics['report'])

    if val_metrics:
        plot_confusion_matrix(val_metrics['cm'], run_dir / "confusion_matrix_val.png", num_classes)
        with open(run_dir / "validate_report.txt", 'w', encoding='utf-8') as f:
            f.write(val_metrics['report'])
            
    logger.info(f"当前实验评估完成！详细报告已存入目录。")
    
    # 释放当前日志句柄，防止污染下一个批次
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description="手形分类评估脚本 (完全数据驱动版)")
    parser.add_argument('--weight', type=str, default='best_model.pth')
    parser.add_argument('--run_name', type=str, default=None, help="直接指定文件夹全名")
    parser.add_argument('--runs', type=str, default=None, help="指定实验标号，支持范围如 '1-3,5'")
    args = parser.parse_args()

    # 读取全局配置获取基础存储路径 (save_dir)
    global_cfg_path = Path('configs/global_config.yaml')
    if not global_cfg_path.exists():
        raise FileNotFoundError("找不到 configs/global_config.yaml 文件，请检查路径。")
    
    with open(global_cfg_path, 'r', encoding='utf-8') as f:
        global_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = global_cfg['train']['save_dir']

    # 智能寻找目标文件夹：传入 method_name="" 以获取所有符合日期的目录
    run_dirs = get_target_run_dirs(save_dir, method_name="", run_name=args.run_name, run_id_str=args.runs)
    
    if not run_dirs:
        print("\n未找到任何需要评估的目录，退出程序。")
        return
        
    print(f"\n寻址完毕，共有 {len(run_dirs)} 个实验目录等待评估。")

    # ==========================================
    # 🌟 特征缓存池 (Feature Cache Pool)
    # 避免不同目录使用相同特征时重复提取和加载数据
    # 字典格式 -> {"distance_matrix": (train_loader, val_loader, feat_dim)}
    # ==========================================
    dataset_cache = {}

    for run_dir in run_dirs:
        config_path = run_dir / "config_backup.yaml"
        if not config_path.exists():
            print(f"[警告] 文件夹 {run_dir.name} 中缺少 config_backup.yaml")
            continue
            
        with open(config_path, 'r', encoding='utf-8') as f:
            current_cfg = yaml.safe_load(f)

        feature_type = current_cfg['data']['feature_type']
        num_classes = current_cfg['data']['num_classes']

        # 如果这种特征是首次遇见，则执行加载
        if feature_type not in dataset_cache:
            print(f"\n检测到新特征类型 [{feature_type}]，正在加载数据集至内存...")
            
            train_dataset = HandshapeDataset(
                train_path=current_cfg['data']['train_path'],
                which_side_path=current_cfg['data']['which_side_path'],
                feature_type=feature_type,
                smplx_model_path=current_cfg['data']['smplx_model_path']
            )
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
            
            val_loader = None
            print(f"正在加载 [{feature_type}] 的验证集...")
            if 'val_path' in current_cfg['data'] and current_cfg['data']['val_path']:
                val_dataset = HandshapeDataset(
                    train_path=current_cfg['data']['val_path'],
                    which_side_path=current_cfg['data']['which_side_path'],
                    feature_type=feature_type,
                    smplx_model_path=current_cfg['data']['smplx_model_path']
                )
                val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
                
            sample_feat = train_dataset[0][0]
            dynamic_feat_dim = sample_feat.shape[0]
            
            dataset_cache[feature_type] = (train_loader, val_loader, dynamic_feat_dim)

        train_loader, val_loader, dynamic_feat_dim = dataset_cache[feature_type]

        run_evaluation_for_dir(
            run_dir, 
            current_cfg,  
            args, 
            device, 
            num_classes, 
            train_loader, 
            val_loader, 
            dynamic_feat_dim
        )
        
    print(f"\n全部 {len(run_dirs)}个实验评估完成！")


if __name__ == '__main__':
    main()
