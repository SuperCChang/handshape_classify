import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import cv2
import torch
import json
import smplx
import trimesh
import pyrender
import numpy as np
import logging
from pathlib import Path

import trimesh.transformations as tf

with open('config.json', 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)

def load_hand_templates(template_dir):
    """
    加载手形模板字典：根据标签 ID，预先将 45维的手部姿态存入内存。
    假设你的模板也是 .npy 文件，名字是 "标签ID.npy" (例如 62.npy)
    """
    templates = {}
    template_path = Path(template_dir)
    if not template_path.exists():
        logging.warning("未找到手形模板库，将无法进行替换！")
        return templates

    for npy_file in template_path.glob("*.npy"):
        label_id = int(npy_file.stem)
        data = np.load(npy_file, allow_pickle=True).item()
        
        # 提取 45 维轴角特征作为模板
        templates[label_id] = {
            'left_hand_pose': torch.tensor(data.get('smplx_lhand_pose', np.zeros(45))).float(),
            'right_hand_pose': torch.tensor(data.get('smplx_rhand_pose', np.zeros(45))).float()
        }
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
    """
    渲染引擎核心：将 SMPL-X 参数序列渲染为 .mp4 视频，并支持手形动态注入。
    
    Args:
        base_seq_dir: 基础动作序列的 .npy 文件夹
        smplx_model_path: SMPL-X 官方模型路径
        output_mp4_path: 视频保存路径
        predicted_labels_path: 你刚刚推理出来的 (F, 2) 预测标签 .npy 文件路径 (可选)
        template_dir: 存放手形参数模板的文件夹 (可选)
    """
    logging.info("初始化 3D 渲染引擎...")
    
    # 1. 初始化 SMPL-X 模型
    smplx_model = smplx.create(
        model_path=smplx_model_path, 
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True,
        ext='pkl'
    ).to(device)
    smplx_model.eval()
    
    # 2. 读取基础序列
    seq_path = Path(base_seq_dir)
    smplx_files = sorted(seq_path.glob('*.npy'), key=lambda x: int(x.stem))
    F = len(smplx_files)
    if F == 0:
        logging.error("基础序列为空！")
        return

    # 3. 读取预测标签与模板 (如果启用了替换)
    replace_hands = False
    if predicted_labels_path and template_dir:
        bimanual_labels = np.load(predicted_labels_path) # 形状 (F, 2)
        templates = load_hand_templates(template_dir)
        
        if len(bimanual_labels) != F:
            logging.warning(f"⚠️ 标签长度 ({len(bimanual_labels)}) 与序列长度 ({F}) 不等！将不进行替换。")
        else:
            replace_hands = True
            logging.info("✨ 手形注入功能已激活！")

    # 4. 配置 Pyrender 渲染器
    width, height = 800, 800
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    
    # 初始化视频写入器
    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))

    # 你需要根据你的坐标系适当调整相机位置。这里给出一个能看清全身的经典视角
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  0.5], # 上下平移 (负值让人体往上移)
        [0.0,  0.0,  1.0,  3], # 摄像机距离 (拉远看全身)
        [0.0,  0.0,  0.0,  1.0]
    ])

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)

    # 5. 逐帧渲染循环
    key_mapping = {
        'smplx_root_pose': 'global_orient', 'smplx_body_pose': 'body_pose',
        'smplx_lhand_pose': 'left_hand_pose', 'smplx_rhand_pose': 'right_hand_pose',
        'smplx_jaw_pose': 'jaw_pose'
    }

    logging.info("正在逐帧渲染视频")
    for i, npy_file in enumerate(smplx_files):
        data = np.load(npy_file, allow_pickle=True).item()
        kwargs_for_smplx = {}
        
        # 组装基础身体参数
        for npy_key, official_key in key_mapping.items():
            if npy_key in data:
                # 转换形状为 (1, 维度)
                kwargs_for_smplx[official_key] = torch.tensor(data[npy_key]).float().view(1, -1).to(device)

        # 💉 核心替换逻辑：注入新的手部姿态参数！
        if replace_hands:
            left_label, right_label = bimanual_labels[i]
            
            # 只有当该标签真的在模板库里时才替换，否则保留原始动作
            if left_label in templates:
                kwargs_for_smplx['left_hand_pose'] = templates[left_label]['left_hand_pose'].view(1, -1).to(device)
            if right_label in templates:
                kwargs_for_smplx['right_hand_pose'] = templates[right_label]['right_hand_pose'].view(1, -1).to(device)

        # 缺失部位兜底补零 (防崩溃机制)
        official_dims = {'global_orient': 3, 'body_pose': 63, 'left_hand_pose': 45, 'right_hand_pose': 45, 'jaw_pose': 3}
        for off_key, dim in official_dims.items():
            if off_key not in kwargs_for_smplx:
                kwargs_for_smplx[off_key] = torch.zeros((1, dim), dtype=torch.float32, device=device)

        # 前向传播，生成 3D 网格顶点
        with torch.no_grad():
            smplx_output = smplx_model(**kwargs_for_smplx)
            vertices = smplx_output.vertices[0].cpu().numpy()
        
        # 构建 Trimesh 对象 (赋予一个基础的灰色)
        mesh = trimesh.Trimesh(vertices, smplx_model.faces, vertex_colors=[200, 200, 200, 255])
        
        rot = tf.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        # 构建 Pyrender 场景
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose) # 把灯光绑在摄像机上

        # 渲染一帧图像
        color, _ = renderer.render(scene)
        
        # OpenCV 写入视频 (需要从 RGB 转为 BGR)
        bgr_color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_color)

    # 6. 收尾清理
    video_writer.release()
    renderer.delete()
    logging.info(f"🎉 视频渲染完成！已保存至: {output_mp4_path}")

if __name__ == '__main__':  
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    name = "前言(序言)"
    first_letter = 'Q'

    BASE_SEQ_DIR = f"/98/jz/shouyu/smplx_GJ_ccbr/{first_letter}/{name}/numpy_result"  # 你的基础动作序列 .npy 文件夹路径
    SMPLX_MODEL_PATH = CONFIG['paths']['smplx_dir']
    OUTPUT_ORIGINAL = f"/101/xcj/project/handshape/output/{name}_original.mp4"
    OUTPUT_REPLACE = f"/101/xcj/project/handshape/output/{name}_replace.mp4"
    
    PRED_LABELS_PATH = f"/101/xcj/project/handshape/output/{name}.npy"
    
    TEMPLATE_DIR = CONFIG['paths']['handshape_templates_dir']

    # 运行渲染！
    render_smplx_to_video(
        base_seq_dir=BASE_SEQ_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        output_mp4_path=OUTPUT_ORIGINAL,
        predicted_labels_path=None,
        template_dir=TEMPLATE_DIR,
        device=DEVICE
    )

    render_smplx_to_video(
        base_seq_dir=BASE_SEQ_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        output_mp4_path=OUTPUT_REPLACE,
        predicted_labels_path=PRED_LABELS_PATH,
        template_dir=TEMPLATE_DIR,
        device=DEVICE
    )
