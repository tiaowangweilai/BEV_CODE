# =====================================================================
# 脚本名称 : pinclie_dibo.py
# 修改日期 : 2026-05-07
# 作    者 : Antigravity & USER
# 脚本作用 : 仿真可视化引擎 (早期版本)。加入了底波和表面波的物理效应模拟。
# =====================================================================

import numpy as np
import plotly.graph_objects as go
import os
from scipy.spatial.transform import Rotation as R_scipy
from scipy.ndimage import gaussian_filter

TXT_FILENAME = "2026.3.17.txt"

def create_sample_txt():
    """自动生成包含全域面积和细长缺陷的配置文件"""
    if not os.path.exists(TXT_FILENAME):
        # 注意：这里的 sx (半宽) 已经修改为 2mm，还原了真实的细窄裂纹形态，防止相互粘连
        sample_content = """# 绘图区域物理尺寸 (单位: mm)
Area 1100 2600 50

# 缺陷配置 (长条形 strip)
# 类型 中心X 中心Y 中心Z 半长(sx) 半宽(sy) 半高(sz) Roll Pitch Yaw
strip 98 120 17.76 2 110 10 0 0 0
strip 255.5 140 17.76 2 110 10 0 0 0
strip 110 150 17.52 2 110 10 0 0 0
strip 267 153 17.88 2 110 10 0 0 0
strip 176.5 98 17.52 2 110 10 0 0 0
"""
        with open(TXT_FILENAME, "w", encoding="utf-8") as f:
            f.write(sample_content)
        print(f"已自动生成配置文件: {TXT_FILENAME}")

def render_full_area_physics_field(txt_filepath):
    # ==========================================
    # 1. 严格读取 txt 定义的面积尺寸
    # ==========================================
    Lx, Ly, Lz = 1100.0, 2600.0, 50.0 # 默认兜底值
    defects = []
    
    with open(txt_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            
            if parts[0] == 'Area' and len(parts) >= 4:
                Lx, Ly, Lz = float(parts[1]), float(parts[2]), float(parts[3])
                continue
            if len(parts) >= 10:
                defects.append({
                    'type': parts[0],
                    'cx': float(parts[1]), 'cy': float(parts[2]), 'cz': float(parts[3]),
                    'sx': float(parts[4]), 'sy': float(parts[5]), 'sz': float(parts[6]),
                    'rx': float(parts[7]), 'ry': float(parts[8]), 'rz': float(parts[9])
                })

    # ==========================================
    # 2. 自适应防卡死全图网格 (保持真实 X/Y 比例)
    # ==========================================
    nx = 120 
    ny = int(nx * (Ly / Lx)) # 自动计算 Y 轴点数，防止网格拉伸
    nz = 35 # Z轴点数，兼顾流畅度与底波细节
    
    print(f"正在构建全局尺寸 {Lx}x{Ly}x{Lz} (mm) 的空间网格: {nx}x{ny}x{nz}...")
    X, Y, Z = np.mgrid[0:Lx:complex(nx), 0:Ly:complex(ny), 0:Lz:complex(nz)]
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)
    
    global_defect_intensity = np.zeros_like(X)

    # ==========================================
    # 3. 计算缺陷能量场 (使用高阶超高斯，边缘锐利)
    # ==========================================
    print("正在计算空间缺陷能量场...")
    for d in defects:
        cx, cy, cz = d['cx'], d['cy'], d['cz']
        # 为了极细的缺陷也能被网格捕捉，设置一个最小尺寸保护
        sx, sy, sz = max(d['sx'], 2.0), max(d['sy'], 2.0), max(d['sz'], 2.0)
        rx, ry, rz = d['rx'], d['ry'], d['rz']
        
        # 6DoF 旋转变换
        rotation = R_scipy.from_euler('xyz', [rx, ry, rz], degrees=True)
        R_inv = rotation.inv().as_matrix()
        
        points_shifted = points - np.array([[cx], [cy], [cz]])
        points_local = R_inv @ points_shifted
        
        X_local = points_local[0, :].reshape(X.shape)
        Y_local = points_local[1, :].reshape(Y.shape)
        Z_local = points_local[2, :].reshape(Z.shape)
        
        # 超高斯分布
        local_intensity = np.exp(-((X_local/sx)**6 + (Y_local/sy)**6 + (Z_local/sz)**6))
        global_defect_intensity = np.maximum(global_defect_intensity, local_intensity)

    # ==========================================
    # 4. 【核心重构】深蓝色高低起伏底波与声影遮罩
    # ==========================================
    print("正在生成自然起伏底波与物理声影...")
    np.random.seed(42) # 固定随机地形种子
    
    # 4.1 生成纯天然平滑地形起伏
    raw_noise = np.random.normal(0, 1, (nx, ny))
    smoothed_noise = gaussian_filter(raw_noise, sigma=2.0)
    max_noise = np.max(np.abs(smoothed_noise))
    # 制造 ±4.5mm 的剧烈物理起伏
    z_fluctuation_mm = (smoothed_noise / max_noise) * 4.5 
    z_fluctuation_3d = z_fluctuation_mm[:, :, np.newaxis]

    # 4.2 计算带起伏的底波 Z 坐标与几何厚度
    bottom_z_base = Lz - 3.0 
    actual_bottom_z = bottom_z_base + z_fluctuation_3d  
    bottom_thickness = 3.5 
    base_bottom_geometry = np.exp(-((Z - actual_bottom_z) / bottom_thickness)**2)
    
    # 4.3 渲染为蓝色调 (强度控制在 0.35~0.5 之间)
    intensity_noise = gaussian_filter(np.random.normal(0, 1, (nx, ny)), sigma=2.0)
    intensity_map = 0.35 + 0.15 * (intensity_noise / np.max(np.abs(intensity_noise)))
    intensity_3d = intensity_map[:, :, np.newaxis]

    # 4.4 垂直投影声影遮罩 (Acoustic Shadowing)
    defect_projection = np.max(global_defect_intensity, axis=2, keepdims=True)
    # 乘以 1.2 模拟声束遮挡的发散效应
    shadow_mask = np.clip(1.0 - defect_projection * 1.2, 0.0, 1.0)
    
    # 4.5 物理场融合
    bottom_echo_field = base_bottom_geometry * intensity_3d * shadow_mask
    final_intensity = np.maximum(global_defect_intensity, bottom_echo_field)

    # ==========================================
    # 5. Plotly 全局渲染与轨道控制
    # ==========================================
    print("启动 3D 全局渲染引擎...")
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=final_intensity.flatten(),
        isomin=0.25,      # 必须低于底波最低强度(0.35)，否则底波变透明
        isomax=1.0, 
        opacity=0.6,      
        surface_count=10,  
        colorscale='Jet', 
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    # 绘制全局大边框
    edges = [
        ([0,Lx,Lx,0,0], [0,0,Ly,Ly,0], [0,0,0,0,0]),
        ([0,Lx,Lx,0,0], [0,0,Ly,Ly,0], [Lz,Lz,Lz,Lz,Lz]),
        ([0,0], [0,0], [0,Lz]), ([Lx,Lx], [0,0], [0,Lz]),
        ([Lx,Lx], [Ly,Ly], [0,Lz]), ([0,0], [Ly,Ly], [0,Lz])
    ]
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=edge[0], y=edge[1], z=edge[2], mode='lines',
            line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'),
            showlegend=False
        ))

    # Z 轴视觉放大：保证大图下能看清内部细节
    z_visual_scale = (Lz / Lx) * 6.0 
    
    fig.update_layout(
        title=f'全景超声 BEV 空间重构 - 缺陷聚类与底波声影',
        scene=dict(
            xaxis_title='X 轴 (mm)', yaxis_title='Y 轴 (mm)', zaxis_title='深度 Z (mm)',
            aspectmode='manual', 
            aspectratio=dict(x=1, y=Ly/Lx, z=z_visual_scale), 
            
            # 【解锁仰视】：初始镜头沉入钢板下方 (z=-0.6) 往上看
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.2, y=-1.5, z=-0.6) 
            )
        ),
        # 【解锁 360 度翻滚】：打破地平线限制
        dragmode='orbit', 
        template='plotly_dark', margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.show()

if __name__ == "__main__":
    create_sample_txt()
    if os.path.exists(TXT_FILENAME):
        render_full_area_physics_field(TXT_FILENAME)