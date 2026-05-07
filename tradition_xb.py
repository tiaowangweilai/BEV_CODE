import numpy as np
import plotly.graph_objects as go
import os
import re
from scipy.ndimage import label, find_objects, binary_closing, zoom # 🌟 引入 zoom 用于缩放

# ==========================================
# ⚙️ 配置区
# ==========================================
FILE_LIST = ["1.1.txt", "1.2.txt", "1.3.txt", "1.4.txt", "1.5.txt", "1.6.txt"]
# FILE_LIST = ["1.1.txt", "1.2.txt", "1.3.txt"]
RES_X, RES_Y, RES_Z = 1.0, 2.0, 0.15 

# 🌟 视图控制开关：
# "FULL"        -> 显示所有原始数据（包括表面波、底波、噪声）
# "DEFECT_ONLY" -> 只显示提取出的缺陷实体
RENDER_MODE = "DEFECT_ONLY" 

def load_and_preprocess(filepath, rotate_k=0):
    """读取并预处理数据"""
    if not os.path.exists(filepath): return None
    content = ""
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
        try:
            with open(filepath, 'r', encoding=enc) as f: content = f.read()
            break
        except UnicodeDecodeError: continue

    parts = re.split(r'={5,}', content)
    header_str, data_str = parts[0], parts[1]
    nums = re.findall(r'\d+', header_str)
    ny, nx, nz = (int(nums[0]), int(nums[1]), int(nums[2])) if len(nums) >= 3 else (31, 300, 339)

    data_list = []
    for item in data_str.split():
        try:
            val = float(item)
            energy = 255.0 - val 
            data_list.append(energy if energy > 35 else 0.0) 
        except ValueError: continue
            
    vol_yxz = np.array(data_list, dtype=np.float32).reshape((ny, nx, nz))
    
    # 物理盲区屏蔽 (保留原始值用于 FULL 模式显示，但检测时只看内部)
    # 此处我们不直接清空数据，而是返回原始处理后的矩阵
    if rotate_k != 0:
        vol_yxz = np.rot90(vol_yxz, k=rotate_k, axes=(0, 1))
    return vol_yxz

def main_process(file_list):
    n = len(file_list)
    
    # 1. 前 n-2 个文件拼接
    base_vols = [load_and_preprocess(file_list[i]) for i in range(n-2)]
    main_block = np.concatenate([v for v in base_vols if v is not None], axis=1)
    
    # 2. 最后两个块旋转并预拼接
    ccw_block = load_and_preprocess(file_list[n-2], rotate_k=1)
    cw_block = load_and_preprocess(file_list[n-1], rotate_k=-1)
    rotated_row = np.concatenate([ccw_block, cw_block], axis=1)

    # ==========================================
    # 📏 🌟 核心改进 1：等比例放缩 (Scaling)
    # ==========================================
    nx_main = main_block.shape[1]
    nx_rot = rotated_row.shape[1]
    
    if nx_main != nx_rot:
        zoom_factor_x = nx_main / nx_rot
        print(f"🔄 检测到长度不一致：主块X={nx_main}, 旋转块X={nx_rot}")
        print(f"🔄 正在执行等比例缩放，缩放系数: {zoom_factor_x:.4f}")
        
        # 使用双线性插值进行缩放，Z轴(axis=2)保持 1.0 不变
        # 我们同时缩放 Y 和 X 轴以保持比例，或者只缩放 X 轴。
        # 按照用户要求“放缩或扩大”，我们对 X 轴进行精准匹配
        rotated_row = zoom(rotated_row, (1.0, zoom_factor_x, 1.0), order=1)

    # 3. Y 轴追加拼接
    final_vol_yxz = np.concatenate([main_block, rotated_row], axis=0)
    volume_3d = final_vol_yxz.transpose((1, 0, 2))
    nx, ny, nz = volume_3d.shape

    # ==========================================
    # 🧠 缺陷提取引擎
    # ==========================================
    # 内部切片用于检测（避开表面波）
    z_s, z_e = int(nz*0.12), int(nz*0.88)
    detect_vol = np.zeros_like(volume_3d)
    detect_vol[:, :, z_s:z_e] = volume_3d[:, :, z_s:z_e]
    
    vol_norm = detect_vol / (np.max(detect_vol) + 1e-5)
    binary_mask = vol_norm > 0.45
    stitched = binary_closing(binary_mask, iterations=2)
    labeled, num_features = label(stitched)
    
    # 提取纯缺陷体素
    defect_only_vol = np.zeros_like(volume_3d)
    defect_only_vol[labeled > 0] = volume_3d[labeled > 0]
    
    objects = find_objects(labeled)
    defects_boxes = [obj for obj in objects if obj is not None]

    # ==========================================
    # 🌟 🌟 核心改进 2：视图模式选择渲染
    # ==========================================
    print(f"🚀 渲染引擎启动，当前模式: {RENDER_MODE}")
    
    if RENDER_MODE == "FULL":
        display_vol = volume_3d
        isomin_val = 45.0  # 全量显示时，稍微过滤掉极低分底噪
        opacity_val = 0.2
        title_suffix = "(Full View)"
    else:
        display_vol = defect_only_vol
        isomin_val = 1.0   # 只显示提取出的物体
        opacity_val = 0.8
        title_suffix = "(Defects Only)"

    # 降采样防卡死
    sx, sy, sz = max(1, nx//120), 1, max(1, nz//80)
    vol_ds = display_vol[::sx, ::sy, ::sz]
    gx, gy, gz = np.mgrid[0:nx:sx, 0:ny:sy, 0:nz:sz]

    fig = go.Figure()

    # 绘制外壳参考线
    fig.add_trace(go.Scatter3d(
        x=[0,nx,nx,0,0,0,nx,nx,0,0,nx,nx,nx,nx,0,0],
        y=[0,0,ny,ny,0,0,0,ny,ny,0,0,0,ny,ny,ny,ny],
        z=[0,0,0,0,0,nz,nz,nz,nz,nz,nz,0,0,nz,nz,0],
        mode='lines', line=dict(color='gray', width=1), name="Bounds"
    ))

    fig.add_trace(go.Volume(
        x=gx.flatten(), y=gy.flatten(), z=gz.flatten(),
        value=vol_ds.flatten(), isomin=isomin_val, isomax=255, 
        opacity=opacity_val, surface_count=8, colorscale='Turbo'
    ))

    for i, obj in enumerate(defects_boxes):
        sx, sy, sz = obj
        x0, x1, y0, y1, z0, z1 = sx.start, sx.stop, sy.start, sy.stop, sz.start, sz.stop
        bx = [x0, x1, x1, x0, x0, x0, x1, x1, x0, x0, x1, x1, x1, x1, x0, x0]
        by = [y0, y0, y1, y1, y0, y0, y0, y1, y1, y0, y0, y0, y1, y1, y1, y1]
        bz = [z0, z0, z0, z0, z0, z1, z1, z1, z1, z1, z1, z0, z0, z1, z1, z0]
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='magenta', width=5)))

    fig.update_layout(
        title=f"Adaptive Stitching Result {title_suffix}",
        template='plotly_dark',
        scene=dict(aspectratio=dict(x=1, y=ny/nx*3, z=0.4))
    )
    fig.write_html("Proportional_Merged_Result.html", auto_open=True)

if __name__ == "__main__":
    main_process(FILE_LIST)