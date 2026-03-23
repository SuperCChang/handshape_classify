import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json

# 加载配置文件
with open('config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

def mirror_pose(pose_array):
    """
    SMPL-X 手部姿态镜像魔法：
    将 45 维度的单手轴角重塑为 (15, 3)，
    X轴不变，Y轴和Z轴取反，再展平回 45 维。
    """
    # 为了安全起见，拷贝一份防止修改原数据
    mirrored = np.copy(pose_array).reshape(15, 3)
    mirrored[:, 1] *= -1.0
    mirrored[:, 2] *= -1.0
    return mirrored.reshape(-1)

def extract_handshape_templates(raw_data_dir, csv_path, output_dir, strategy='middle'):
    """
    从原始 smplx 序列中提取指定帧，保存为标准手形模板。
    核心逻辑：同时生成双手的姿态。利用镜像反转，让左右手呈现完全对称的相同手形。
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    
    # 确保输出文件夹存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 读取标签文件
    try:
        df = pd.read_csv(csv_path, encoding='gbk', header=None)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='utf-8', header=None)
        
    # 去重：每个 label_id 我们只需要挑一个代表性的例词去提手形就够了
    unique_labels_df = df.drop_duplicates(subset=[0])
    
    success_count = 0
    missing_count = 0

    logging.info(f"🔍 开始提取双手对称模板，共有 {len(unique_labels_df)} 个独立标签待处理...")
    
    for _, row in unique_labels_df.iterrows():
        label_id = int(row.iloc[0])
        word_name = str(row.iloc[1]).strip()
        side_info = str(row.iloc[2]).strip()
        
        # 组装原始序列所在的文件夹路径
        smplx_path = raw_data_dir / str(label_id) / word_name / 'numpy_result'
        
        if not smplx_path.exists():
            logging.warning(f"⚠️ 跳过: 找不到对应文件夹 -> {smplx_path}")
            missing_count += 1
            continue
            
        # 获取所有帧
        smplx_files = sorted(smplx_path.glob('*.npy'), key=lambda x: int(x.stem))
        
        if not smplx_files:
            logging.warning(f"⚠️ 跳过: 文件夹为空 -> {smplx_path}")
            missing_count += 1
            continue
            
        # 2. 核心逻辑：决定提取哪一帧
        if strategy == 'middle':
            target_idx = len(smplx_files) // 2
        else:
            target_idx = 0
            
        target_file = smplx_files[target_idx]
        
        # 3. 读取该帧数据
        frame_data = np.load(target_file, allow_pickle=True).item()
        
        template_dict = {}
        target_pose = None
        source_side = None  # 记录捞上来的数据到底属于左边还是右边
        
        # 🌟 核心规范化逻辑：根据 side_info 精准捞取手形
        if side_info == '右':
            target_pose = frame_data.get('smplx_rhand_pose')
            source_side = 'R'
            # 兜底
            if target_pose is None:
                target_pose = frame_data.get('smplx_lhand_pose')
                source_side = 'L'
                
        elif side_info in ['左', '左右']:
            # 注意：按照你的要求，'左右'情况也直接拿左手，抛弃原右手的差异，强制复制左手
            target_pose = frame_data.get('smplx_lhand_pose')
            source_side = 'L'
            # 兜底
            if target_pose is None:
                target_pose = frame_data.get('smplx_rhand_pose')
                source_side = 'R'
                
        # 极端缺失情况兜底
        if target_pose is None:
            logging.warning(f"⚠️ 标签 {label_id} ({word_name}) 缺失手形数据，已使用全0兜底。")
            target_pose = np.zeros(45, dtype=np.float32)
            source_side = 'L'

        # 🎯 终极映射：生成对称的双手模板
        if source_side == 'L':
            # 来源是左手：左手原样写入，右手使用左手的镜像
            template_dict['smplx_lhand_pose'] = target_pose
            template_dict['smplx_rhand_pose'] = mirror_pose(target_pose)
        else:
            # 来源是右手：右手原样写入，左手使用右手的镜像
            template_dict['smplx_rhand_pose'] = target_pose
            template_dict['smplx_lhand_pose'] = mirror_pose(target_pose)
            
        # 4. 保存文件
        save_path = output_dir / f"{label_id}.npy"
        np.save(save_path, template_dict)
        success_count += 1

    logging.info("="*40)
    logging.info(f"🎉 模板提取大功告成！")
    logging.info(f"✅ 成功生成双面对称模板: {success_count} 个")
    logging.info(f"❌ 缺失/跳过标签: {missing_count} 个")
    logging.info(f"📂 模板统一存放在: {output_dir.absolute()}")


def fix_specific_template(label_id, word_name, frame_filename, raw_data_dir, csv_path, output_dir):
    """
    【人工精准修正补丁】
    读取指定的例词和指定的 .npy 帧文件，重新生成双手对称模板并覆盖。
    
    Args:
        label_id (int): 错误的标签 ID (例如: 62)
        word_name (str): 重新指定的例词名字 (例如: '高兴')
        frame_filename (str): 你人工确认过正确的帧文件名 (例如: '0035.npy')
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    
    # 1. 查 CSV 获取这只手到底是左手词还是右手词
    try:
        df = pd.read_csv(csv_path, encoding='gbk', header=None)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='utf-8', header=None)
        
    row = df[df[0] == label_id]
    if row.empty:
        logging.error(f"❌ 找不到标签 {label_id} 的 side_info，请检查 CSV。")
        return
        
    side_info = str(row.iloc[0, 2]).strip()
    
    # 2. 定位到你指定的那个确切的 .npy 文件
    target_file = raw_data_dir / str(label_id) / word_name / 'numpy_result' / frame_filename
    if not target_file.exists():
        logging.error(f"❌ 找不到指定的文件: {target_file}")
        return
        
    # 3. 读取数据
    frame_data = np.load(target_file, allow_pickle=True).item()
    template_dict = {}
    target_pose = None
    source_side = None
    
    # 4. 根据 side_info 捞取手形
    if side_info == '右':
        target_pose = frame_data.get('smplx_rhand_pose')
        source_side = 'R'
    elif side_info in ['左', '左右']:
        target_pose = frame_data.get('smplx_lhand_pose')
        source_side = 'L'

            
    if target_pose is None:
        logging.error(f"❌ 文件 {target_file} 中没有手部姿态数据！")
        return

    # 5. 再次施展镜像对称魔法
    if source_side == 'L':
        template_dict['smplx_lhand_pose'] = target_pose
        template_dict['smplx_rhand_pose'] = mirror_pose(target_pose)
    else:
        template_dict['smplx_rhand_pose'] = target_pose
        template_dict['smplx_lhand_pose'] = mirror_pose(target_pose)
        
    # 6. 暴力覆盖原来的模板文件
    save_path = output_dir / f"{label_id}.npy"
    np.save(save_path, template_dict)
    logging.info(f"✅ 成功修正！标签 {label_id} 的模板已被 {word_name} 的 {frame_filename} 帧覆盖。")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # =============== 路径配置自动读取 ===============
    RAW_DATA_DIR = CONFIG['paths']['dataset_root']
    CSV_PATH = CONFIG['paths']['which_side_root']
    OUTPUT_DIR = CONFIG['paths']['handshape_templates_dir']
    
    # 开始提取
    extract_handshape_templates(RAW_DATA_DIR, CSV_PATH, OUTPUT_DIR, strategy='middle')

    fixes = [
        (4, "谷子(谷)", "000025.npy"),
        (6, "哥哥", "000016.npy"),
        (8, "脏", "000034.npy"),
        (20, "药(服药、吃药)", "000020.npy"),
        (23, "辍学(肄业)", "000035.npy"),
        (26, "发祥地(发源地)", "000006.npy"),
        (29, "借代", "000013.npy"),
        (42, "鹤", "000070.npy"),
        (44, "闽(福建②)", "000040.npy"),
        (51, "水龙头", "000043.npy"),
        (64, "3(三、叁)", "000020.npy"),
        (65, "30(三十)", "000036.npy"),
        (71, "40(四十)", "000036.npy"),
        (77, "激光", "000038.npy"),
        (99, "标枪(投掷)", "000012.npy"),
        (107, "缆车(天车)", "000040.npy")
    ]
    
    logging.info("🔧 开始执行人工打补丁流程...")
    for label_id, word_name, frame_name in fixes:
        fix_specific_template(
            label_id=label_id,
            word_name=word_name,
            frame_filename=frame_name,
            raw_data_dir=RAW_DATA_DIR,
            csv_path=CSV_PATH,
            output_dir=OUTPUT_DIR
        )