import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re

FILE_NAME = "Synthetic_2026_Raw.txt"

def load_raw_tensor_to_physbev(filepath):
    print(f"📂 [PhysBEV Engine] 正在挂载原始硬件数据: {filepath} ...")
    
    if not os.path.exists(filepath):
        print("❌ 找不到文件，请检查路径！")
        return

    # 1. 强力读取与自适应切分
    content = ""
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    parts = re.split(r'={5,}', content)
    header_str = parts[0] if len(parts) >= 2 else ""
    data_str = parts[1] if len(parts) >= 2 else content

    # 2. 物理坐标系解析
    nums = re.findall(r'\d+', header_str)
    if len(nums) >= 3:
        ny, nx, nz = int(nums[0]), int(nums[1]), int(nums[2])
    else:
        ny, nx, nz = 20, 300, 339
    
    # 3. 数据提取与反演
    cleaned_data = data_str.strip()
    data_list = []
    for item in cleaned_data.split():
        try:
            data_list.append(float(item))
        except ValueError:
            continue
            
    raw_values = np.array(data_list, dtype=np.float32)
    
    # 【物理第一性原理 1：信号极性反演】
    energy_values = 255.0 - raw_values 
    
    total_expected = nx * ny * nz
    if len(energy_values) < total_expected:
        padded = np.zeros(total_expected)
        padded[:len(energy_values)] = energy_values
        energy_values = padded
    elif len(energy_values) > total_expected:
        energy_values = energy_values[:total_expected]

    # 将 1D 数据重塑为 (X, Y, Z) 的 3D 张量
    volume_3d = energy_values.reshape((ny, nx, nz))
    volume_3d = volume_3d.transpose((1, 0, 2)) 

    # ==========================================
    # 🌟 PhysBEV 核心处理引擎
    # ==========================================
    print("⏳ [PhysBEV Engine] 正在执行物理场重构 (Physics-Guided Reconstruction)...")

    # 【物理第一性原理 2：指数级深度增益补偿】
    # 生成一个从 1.0 逐渐增加到 e^(alpha) 的补偿矩阵
    alpha = 1.2 # 介质衰减系数 (可调)
    z_indices = np.arange(nz)
    compensation_curve = np.exp(alpha * (z_indices / nz))
    
    # 对整个 3D 空间进行深度补偿
    volume_3d = volume_3d * compensation_curve.reshape(1, 1, nz)

    # 归一化为 0~1 的占据概率 (Occupancy Probability)
    volume_3d = np.clip(volume_3d / np.max(volume_3d), 0, 1.0)
    
    # 滤除微小的底噪杂波 (比如低于 0.15 的直接清零)，让缺陷更锐利
    volume_3d[volume_3d < 0.15] = 0

    # 【多模态融合：注入表面视觉感知】
    # 模拟一张带有生锈/划痕的表面视觉图 (尺寸正好等于 nx, ny)
    np.random.seed(42)
    visual_surface = np.random.uniform(0.1, 0.4, (nx, ny))
    # 在表面人为加两条高亮“划痕”语义
    visual_surface[50:150, 8:12] = 0.8 
    visual_surface[200:250, 15:18] = 0.7
    
    # 将视觉特征强制钉死在 Z=0 到 Z=2 的表层空间
    for zi in range(3):
        volume_3d[:, :, zi] = np.maximum(volume_3d[:, :, zi], visual_surface)

    # 【BEV 鸟瞰投影】
    # 提取内部缺陷，投影到 2D 鸟瞰图。
    # 故意避开表层 (z<10) 和底波层 (z>nz-40)，只捕捉纯粹的内部缺陷！
    bev_projection = np.max(volume_3d[:, :, 10:nz-40], axis=2)

    # ==========================================
    # 🌟 动态降维与 WebGL 双视窗渲染
    # ==========================================
    print("🚀 [PhysBEV Engine] 正在生成双模态交互仪表盘...")
    step_x, step_y, step_z = 3, 1, 3
    vol_downsampled = volume_3d[::step_x, ::step_y, ::step_z]
    grid_x, grid_y, grid_z = np.mgrid[0:nx:step_x, 0:ny:step_y, 0:nz:step_z]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.25, 0.75], 
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("PhysBEV: Internal Defect Projection (Top-Down)", "PhysBEV: 3D Multi-modal Voxel Field")
    )

    # 左侧：2D BEV 热力图
    fig.add_trace(go.Heatmap(
        z=bev_projection.T, 
        colorscale='Jet', zmin=0.0, zmax=1.0, showscale=False 
    ), row=1, col=1)

    # 右侧：3D 体素图
    fig.add_trace(go.Volume(
        x=grid_x.flatten(), 
        y=grid_y.flatten(), 
        z=grid_z.flatten(),
        value=vol_downsampled.flatten(),
        isomin=0.2, # 仅显示能量 > 0.2 的实体
        isomax=1.0,
        opacity=0.35, 
        surface_count=8, 
        colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title=dict(text="Occupancy<br>Probability", font=dict(color='white')), x=1.02)
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text='Raw Data + PhysBEV: Physics-Guided Perception Dashboard', font=dict(color='white', size=20)),
        scene=dict(
            xaxis=dict(title='Scan X', color='white', showbackground=False),
            yaxis=dict(title='Index Y', color='white', showbackground=False, autorange='reversed'),
            zaxis=dict(title='Depth Z', color='white', showbackground=False, autorange='reversed'),
            aspectratio=dict(x=1, y=(ny/nx)*4, z=0.5), # 自动拉伸过窄的Y轴
            camera=dict(eye=dict(x=1.3, y=-1.5, z=0.7))
        ),
        xaxis=dict(color='white'), yaxis=dict(color='white', autorange="reversed"),
        dragmode='turntable',
        template='plotly_dark',
        margin=dict(l=20, r=80, b=50, t=80)
    )
    
    output_html = "RawData_PhysBEV_Dashboard.html"
    fig.write_html(output_html, auto_open=True)
    print(f"✅ 渲染完美结束！请双击查看终极融合态: {os.path.abspath(output_html)}")

if __name__ == "__main__":
    load_raw_tensor_to_physbev(FILE_NAME)