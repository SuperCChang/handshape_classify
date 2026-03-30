import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import torch
import yaml
import smplx
import trimesh
import pyrender
import argparse
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import trimesh.transformations as tf

def load_hand_templates(template_dir):
    templates = {}
    template_path = Path(template_dir)
    if not template_path.exists():
        logging.warning(f"警告：未找到手形模板库目录: {template_path}，将无法进行手形替换！")
        return templates

    for npy_file in template_path.glob("*.npy"):
        if not npy_file.stem.isdigit():
            continue
            
        label_id = int(npy_file.stem)
        data = np.load(npy_file, allow_pickle=True).item()
        
        templates[label_id] = {
            'left_hand_pose': torch.tensor(data.get('smplx_lhand_pose', np.zeros(45))).float(),
            'right_hand_pose': torch.tensor(data.get('smplx_rhand_pose', np.zeros(45))).float()
        }
        
    logging.info(f"成功加载 {len(templates)} 个手形模板。")
    return templates


def render_smplx_to_video(
    base_seq_dir, 
    smplx_model_path, 
    output_mp4_path, 
    predicted_labels_path=None, 
    template_dir=None,
    fps=25,
    device='cpu'
):
    smplx_model = smplx.create(
        model_path=smplx_model_path, 
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True,
        ext='pkl'
    ).to(device)
    smplx_model.eval()
    
    seq_path = Path(base_seq_dir)
    import re
    def extract_number(fpath):
        nums = re.findall(r'\d+', fpath.stem)
        return int(nums[-1]) if nums else 0

    smplx_files = sorted(seq_path.glob('*.npy'), key=extract_number)
    F = len(smplx_files)
    if F == 0:
        logging.error(f"在{base_seq_dir}中未找到任何 .npy 序列文件！")
        return

    replace_hands = False
    if predicted_labels_path and template_dir:
        pred_path = Path(predicted_labels_path)
        if pred_path.exists():
            bimanual_labels = np.load(pred_path)
            templates = load_hand_templates(template_dir)
            
            if len(bimanual_labels) != F:
                logging.warning(f"预测标签长度 ({len(bimanual_labels)}) 与序列帧数 ({F}) 不匹配！自动取消手形注入。")
            else:
                replace_hands = True
        else:
            logging.warning(f"未找到预测标签文件 {pred_path}，将渲染原始动作。")

    width, height = 800, 800
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    
    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  0.5], 
        [0.0,  0.0,  1.0,  2.5], 
        [0.0,  0.0,  0.0,  1.0]
    ])

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)

    key_mapping = {
        'smplx_root_pose': 'global_orient', 'smplx_body_pose': 'body_pose',
        'smplx_lhand_pose': 'left_hand_pose', 'smplx_rhand_pose': 'right_hand_pose',
        'smplx_jaw_pose': 'jaw_pose'
    }

    logging.info(f"正在渲染视频 -> {Path(output_mp4_path).name}")
    for i in tqdm(range(F), desc="Rendering Frames", unit="frame"):
        npy_file = smplx_files[i]
        data = np.load(npy_file, allow_pickle=True).item()
        kwargs_for_smplx = {}
        
        for npy_key, official_key in key_mapping.items():
            if npy_key in data:
                kwargs_for_smplx[official_key] = torch.tensor(data[npy_key]).float().view(1, -1).to(device)

        if replace_hands:
            left_label, right_label = bimanual_labels[i]
            if left_label in templates:
                kwargs_for_smplx['left_hand_pose'] = templates[left_label]['left_hand_pose'].view(1, -1).to(device)
            if right_label in templates:
                kwargs_for_smplx['right_hand_pose'] = templates[right_label]['right_hand_pose'].view(1, -1).to(device)

        official_dims = {'global_orient': 3, 'body_pose': 63, 'left_hand_pose': 45, 'right_hand_pose': 45, 'jaw_pose': 3}
        for off_key, dim in official_dims.items():
            if off_key not in kwargs_for_smplx:
                kwargs_for_smplx[off_key] = torch.zeros((1, dim), dtype=torch.float32, device=device)

        with torch.no_grad():
            smplx_output = smplx_model(**kwargs_for_smplx)
            vertices = smplx_output.vertices[0].cpu().numpy()
        
        mesh = trimesh.Trimesh(vertices, smplx_model.faces, vertex_colors=[200, 200, 200, 255])
        rot = tf.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose) 

        color, _ = renderer.render(scene)
        bgr_color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_color)

    video_writer.release()
    renderer.delete()
    logging.info(f"渲染成功！文件存至：{output_mp4_path}\n")


def main():
    parser = argparse.ArgumentParser(description="SMPL-X序列渲染可视化工具")
    # --- 核心 IO 参数 ---
    parser.add_argument('--origin', type=str, required=True, help="原始身体序列的 .npy *文件夹*")
    parser.add_argument('--labels', type=str, default=None, help="预测的手形标签 .npy *文件*")
    parser.add_argument('--output_dir', type=str, default='./output', help="视频保存目录")
    parser.add_argument('--name', type=str, default=None, help="生成的视频文件名前缀 (默认自动读取 labels 或 input 的名称)")
    parser.add_argument('--only_replace', action='store_true', help="加入此参数后，将跳过原版渲染，仅渲染换手版")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 【智能内部推断逻辑：模式判定】
    if args.labels is None:
        internal_mode = 'original'
        if args.only_replace:
            logging.warning("⚠️ 你勾选了 --only_replace，但没有传 --labels！已自动切回渲染原版。")
    else:
        internal_mode = 'replace' if args.only_replace else 'both'

    # 【智能内部推断逻辑：名称提取】
    if args.name is not None:
        video_name = args.name
    elif args.labels is not None:
        video_name = Path(args.labels).stem  # 读取标签文件的名字，如 "舍己救人"
    else:
        video_name = Path(args.origin).name   # 兜底：读取动作文件夹的名字

    global_cfg_path = Path('configs/global_config.yaml')
    if not global_cfg_path.exists():
        logging.error("❌ 找不到 configs/global_config.yaml 文件。")
        return
        
    with open(global_cfg_path, 'r', encoding='utf-8') as f:
        global_cfg = yaml.safe_load(f)
        
    smplx_model_path = global_cfg['data']['smplx_model_path']
    template_dir = global_cfg['data'].get('handshape_templates_dir', './templates') 

    os.makedirs(args.output_dir, exist_ok=True)
    out_original = os.path.join(args.output_dir, f"{video_name}_original.mp4")
    out_replace = os.path.join(args.output_dir, f"{video_name}_replace.mp4")

    logging.info("="*60)
    logging.info(f"🎥 启动 3D 视频渲染管线")
    logging.info(f"身体基底输入: {args.origin}")
    if args.labels:
        logging.info(f"手形标签注入: {args.labels}")
    logging.info(f"执行模式: {internal_mode}")
    logging.info(f"视频命名前缀: {video_name}")
    logging.info("="*60)

    # 渲染原始序列
    if internal_mode in ['both', 'original']:
        logging.info("▶️ 任务: 开始渲染 [原始动作] 视频...")
        render_smplx_to_video(
            base_seq_dir=args.origin,
            smplx_model_path=smplx_model_path,
            output_mp4_path=out_original,
            predicted_labels_path=None,  
            template_dir=None,
            device=device
        )

    # 渲染替换序列
    if internal_mode in ['both', 'replace']:
        logging.info("▶️ 任务: 开始渲染 [注入新手形] 视频...")
        render_smplx_to_video(
            base_seq_dir=args.origin,
            smplx_model_path=smplx_model_path,
            output_mp4_path=out_replace,
            predicted_labels_path=args.labels,
            template_dir=template_dir,
            device=device
        )


if __name__ == '__main__':  
    main()