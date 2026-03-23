import os
# 🌟 必须放在最前面：强制使用 CPU 纯软件渲染，绕过服务器显卡权限限制
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import cv2
import torch
import smplx
import json
import trimesh
import pyrender
import numpy as np
import logging
import trimesh.transformations as tf
from pathlib import Path

with open('config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

def batch_render_multiview_templates(template_dir, smplx_model_path, output_dir, device='cpu'):
    """
    【多视角拼接版】批量渲染手形模板，一次生成正视、俯视、侧视三视角拼接图。
    聚焦放大左手，打上绿色标签，用于极速人工质检参考。
    """
    template_path = Path(template_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(template_path.glob("*.npy"))
    if not npy_files:
        logging.error(f"在 {template_dir} 中没有找到任何 .npy 模板文件！")
        return

    logging.info(f"🔍 找到 {len(npy_files)} 个手形模板，准备开始批量拼接渲染...")

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

    # 2. 初始化 Pyrender 渲染器 (800x800 高清正方形)
    # 🌟 将单个视角的尺寸调小一点，防止三拼接后图片太大，建议 600x600 或 700x700
    sub_width, sub_height = 700, 700 
    renderer = pyrender.OffscreenRenderer(viewport_width=sub_width, viewport_height=sub_height)

    # ========================== 3. 核心：设置三个视角的相机位姿矩阵 ==========================
    # 🌟 微调策略：针对 SMPL-X 左手腕位置 [0.43, 0.0, 0.0] 进行聚焦贴脸放大。
    
    # --- 视角 A: 正视特写 (FRONT) ---
    camera_front = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    # 稍微拉远 Z 轴看全身手部，调大 X 轴居中
    pose_front = tf.translation_matrix([0.75, 0.05, 0.25])
    
    # --- 视角 B: 俯视特写 (TOP) ---
    camera_top = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    # 镜头朝下旋转 90 度
    rot_down = tf.rotation_matrix(np.radians(90), [1, 0, 0])
    # 降低高度 Y 轴，调大 X 轴居中手腕上方。Z轴不偏。
    # Y = 0.2 左右能实现微距特写
    trans_top = tf.translation_matrix([0.75, -0.3, -0.05])
    pose_top = np.dot(trans_top, rot_down)
    
    # --- 视角 C: 侧视特写 (SIDE) ---
    camera_side = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    # 镜头绕 Y 轴旋转 -90 度，从小人左侧看过来
    rot_side = tf.rotation_matrix(np.radians(90), [0, 1.1, 0])
    # 移动到小人左侧，高度调整对齐手腕
    trans_side = tf.translation_matrix([1, 0, -0.05])
    pose_side = np.dot(trans_side, rot_side)

    # 俯视时需要更强的光照
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)

    success_count = 0
    
    # 4. 遍历所有模板
    for npy_file in npy_files:
        label_id = npy_file.stem
        data = np.load(npy_file, allow_pickle=True).item()
        
        # 提取双手姿态
        left_pose = data.get('smplx_lhand_pose', np.zeros((1, 45)))
        right_pose = data.get('smplx_rhand_pose', np.zeros((1, 45)))
        
        # 转换为 Tensor
        kwargs_for_smplx = {
            'left_hand_pose': torch.tensor(left_pose).float().view(1, -1).to(device),
            'right_hand_pose': torch.tensor(right_pose).float().view(1, -1).to(device),
            # 强制清零躯干和朝向，确保稳定
            'body_pose': torch.zeros((1, 63)).float().to(device),
            'global_orient': torch.zeros((1, 3)).float().to(device),
            'transl': torch.zeros((1, 3)).float().to(device)
        }

        # 5. 前向传播生成 3D 网格
        with torch.no_grad():
            smplx_output = smplx_model(**kwargs_for_smplx)
            vertices = smplx_output.vertices[0].cpu().numpy()
            
        # 6. 3D 场景组装
        mesh = trimesh.Trimesh(vertices, smplx_model.faces, vertex_colors=[200, 200, 200, 255])
        
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        
        # 🌟 正确做法：将灯光和相机先加入场景，并保存它们对应的 Node 对象
        light_node = scene.add(light, pose=pose_front)
        
        # 为了避免每次都创建新相机，我们先用正视角创建一个相机节点
        cam_node = scene.add(camera_front, pose=pose_front)

        # ========================== 7. 核心：多视角顺序渲染与拼接 ==========================
        view_images = []
        
        # --- 渲染正视 (A) ---
        # 此时相机节点已经在 pose_front 位置了，直接渲染
        color_a, _ = renderer.render(scene)
        view_images.append(cv2.cvtColor(color_a, cv2.COLOR_RGB2BGR))
        
        # --- 渲染俯视 (B) ---
        # 🌟 正确做法：修改已有相机的类型和位姿矩阵 (matrix)
        scene.set_pose(cam_node, pose=pose_top)
        scene.set_pose(light_node, pose=pose_top) # 灯光也跟着移动，避免侧视时太暗
        cam_node.camera = camera_top # 切换相机的内参属性
        
        color_b, _ = renderer.render(scene)
        view_images.append(cv2.cvtColor(color_b, cv2.COLOR_RGB2BGR))
        
        # --- 渲染侧视 (C) ---
        scene.set_pose(cam_node, pose=pose_side)
        scene.set_pose(light_node, pose=pose_side) 
        cam_node.camera = camera_side
        
        color_c, _ = renderer.render(scene)
        view_images.append(cv2.cvtColor(color_c, cv2.COLOR_RGB2BGR))

        # 🌟 水平拼接图片 -> 得到 2100x700 的拼接大图
        canvas = np.hstack(view_images)
        
        # 8. 打上标签水印
        # 在拼接大图的左上角打上标签
        text = f"Label: {label_id}"
        cv2.putText(canvas, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        
        # 在三张子图下面打上小视角说明（可选）
        # sub_text_y = sub_height - 30
        # cv2.putText(canvas, "FRONT", (30, sub_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # cv2.putText(canvas, "TOP", (sub_width + 30, sub_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # cv2.putText(canvas, "SIDE", (2*sub_width + 30, sub_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 9. 保存拼接后的图片
        save_path = output_path / f"template_{label_id}.png"
        cv2.imencode('.png', canvas)[1].tofile(str(save_path))
        
        success_count += 1
        if success_count % 10 == 0:
            logging.info(f"已渲染 {success_count} / {len(npy_files)} 张拼接图...")

    renderer.delete()
    logging.info(f"🎉 批量渲染大功告成！拼接图存放在: {output_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    TEMPLATE_DIR = CONFIG['paths']['handshape_templates_dir']
    SMPLX_MODEL_PATH = CONFIG['paths']['smplx_dir']
    OUTPUT_DIR = r"C:\Users\capg303\Desktop\Project\手形\smpl-x解析\data\verify_images"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_render_multiview_templates(
        template_dir=TEMPLATE_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )