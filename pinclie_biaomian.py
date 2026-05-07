import numpy as np
import plotly.graph_objects as go
import os
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from scipy.ndimage import gaussian_filter

TXT_FILENAME = "1.txt"

def create_sample_txt():
    """如果当前目录下没有 1.txt，则自动生成极细的长条形缺陷"""
    if not os.path.exists(TXT_FILENAME):
        sample_content = """# 绘图区域物理尺寸 (单位: mm)
Area 1100 2600 50
strip 98 120 17.76 2 110 10 0 0 0
strip 255.5 140 17.76 2 110 10 0 0 0
strip 110 150 17.52 2 110 10 0 0 0
strip 267 153 17.88 2 110 10 0 0 0
strip 176.5 98 17.52 2 110 10 0 0 0
"""
        with open(TXT_FILENAME, "w", encoding="utf-8") as f:
            f.write(sample_content)

def get_blended_surface_colors(Lx, Ly, nx, ny, defects):
    """
    【修复版】：精确映射钢板与石灰板的物理坐标区间，消除纹理畸变
    """
    if os.path.exists('1.jpg') and os.path.exists('2.jpg'):
        img_steel = cv2.cvtColor(cv2.imread('1.jpg'), cv2.COLOR_BGR2RGB)
        img_lime = cv2.cvtColor(cv2.imread('2.jpg'), cv2.COLOR_BGR2RGB)
    else:
        print("警告：未找到图片")
        return []

    blended_img = np.zeros((ny, nx, 3), dtype=np.float32)
    y_coords = np.linspace(0, Ly, ny)
    
    # 钢板与石灰板的物理交界线
    steel_length = 300.0
    lime_length = Ly - steel_length
    
    for j, y in enumerate(y_coords):
        # --- 1. 计算当前 Y 坐标在钢板图片中的像素行 ---
        y_steel_ratio = np.clip(y / steel_length, 0, 1)
        row_s = int(y_steel_ratio * (img_steel.shape[0] - 1))
        col_s_idx = np.linspace(0, img_steel.shape[1] - 1, nx).astype(int)
        row_steel_colors = img_steel[row_s, col_s_idx, :]

        # --- 2. 计算当前 Y 坐标在石灰板图片中的像素行 ---
        y_lime_ratio = np.clip((y - steel_length) / lime_length, 0, 1)
        row_l = int(y_lime_ratio * (img_lime.shape[0] - 1))
        col_l_idx = np.linspace(0, img_lime.shape[1] - 1, nx).astype(int)
        row_lime_colors = img_lime[row_l, col_l_idx, :]

        # --- 3. 边界羽化融合 (在 280mm 到 320mm 之间平滑过渡) ---
        if y <= 280.0:
            alpha = 1.0
        elif y >= 320.0:
            alpha = 0.0
        else:
            alpha = 1.0 - (y - 280.0) / 40.0

        blended_img[j, :, :] = row_steel_colors * alpha + row_lime_colors * (1.0 - alpha)

    blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
    
    # ==========================================
    # 在表面绘制对应的视觉检测特征 (绿色框与损伤)
    # ==========================================
    for d in defects:
        if d['cy'] > 400: continue 
        
        px = int((d['cx'] / Lx) * nx)
        py = int((d['cy'] / Ly) * ny)
        pw_half = max(2, int((d['sx'] / Lx) * nx)) 
        ph_half = max(2, int((d['sy'] / Ly) * ny))
        
        pt1 = (px - pw_half, py - ph_half)
        pt2 = (px + pw_half, py + ph_half)
        
        cv2.rectangle(blended_img, pt1, pt2, (40, 30, 30), -1) 
        cv2.rectangle(blended_img, (pt1[0]-2, pt1[1]-2), (pt2[0]+2, pt2[1]+2), (0, 255, 0), 2)

    # ==========================================
    # 转换为 Plotly 识别的 rgba 点云
    # ==========================================
    colors = []
    for i in range(nx):
        for j in range(ny):
            r, g, b = blended_img[j, i]
            # opacity 设为 0.9，稍微透出一点点底波
            colors.append(f'rgba({int(r)},{int(g)},{int(b)}, 0.9)')
            
    return colors

def render_multimodal_bev(txt_filepath):
    # 1. 读取基础配置
    Lx, Ly, Lz = 1100.0, 2600.0, 50.0 
    defects = []
    with open(txt_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if parts[0] == 'Area':
                Lx, Ly, Lz = float(parts[1]), float(parts[2]), float(parts[3])
                continue
            if len(parts) >= 10:
                defects.append({
                    'type': parts[0],
                    'cx': float(parts[1]), 'cy': float(parts[2]), 'cz': float(parts[3]),
                    'sx': float(parts[4]), 'sy': float(parts[5]), 'sz': float(parts[6]),
                    'rx': float(parts[7]), 'ry': float(parts[8]), 'rz': float(parts[9])
                })

    # 2. 生成空间网格 (兼顾流畅与精细度)
    nx = 140 
    ny = int(nx * (Ly / Lx))
    nz = 35
    
    print(f"正在构建全局 BEV 网格: {nx}x{ny}x{nz}...")
    X, Y, Z = np.mgrid[0:Lx:complex(nx), 0:Ly:complex(ny), 0:Lz:complex(nz)]
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)
    global_defect_intensity = np.zeros_like(X)

    # 3. 计算超声缺陷能量场 
    print("正在融合超声缺陷特征...")
    for d in defects:
        if d['cy'] > 400: continue
        cx, cy, cz = d['cx'], d['cy'], d['cz']
        sx, sy, sz = max(d['sx'], 1.5), max(d['sy'], 1.5), max(d['sz'], 1.5)
        rx, ry, rz = d['rx'], d['ry'], d['rz']
        
        rotation = R_scipy.from_euler('xyz', [rx, ry, rz], degrees=True)
        R_inv = rotation.inv().as_matrix()
        
        points_shifted = points - np.array([[cx], [cy], [cz]])
        points_local = R_inv @ points_shifted
        
        X_local = points_local[0, :].reshape(X.shape)
        Y_local = points_local[1, :].reshape(Y.shape)
        Z_local = points_local[2, :].reshape(Z.shape)
        
        local_intensity = np.exp(-((X_local/sx)**6 + (Y_local/sy)**6 + (Z_local/sz)**6))
        global_defect_intensity = np.maximum(global_defect_intensity, local_intensity)

    # 4. 底波计算与声影遮挡
    print("正在计算自然底波起伏及物理声影...")
    np.random.seed(42)
    smoothed_noise = gaussian_filter(np.random.normal(0, 1, (nx, ny)), sigma=2.0)
    z_fluctuation_3d = ((smoothed_noise / np.max(np.abs(smoothed_noise))) * 4.5)[:, :, np.newaxis]
    
    actual_bottom_z = (Lz - 3.0) + z_fluctuation_3d  
    base_bottom = np.exp(-((Z - actual_bottom_z) / 3.5)**2)
    
    intensity_map = 0.35 + 0.15 * (gaussian_filter(np.random.normal(0, 1, (nx, ny)), sigma=2.0) / 3.0)
    
    defect_projection = np.max(global_defect_intensity, axis=2, keepdims=True)
    shadow_mask = np.clip(1.0 - defect_projection * 1.5, 0.0, 1.0)
    
    final_intensity = np.maximum(global_defect_intensity, base_bottom * intensity_map[:, :, np.newaxis] * shadow_mask)

    # ==========================================
    # 5. 渲染引擎组合 (表面 + 内部)
    # ==========================================
    fig = go.Figure()

    # A. 渲染底层物理场 (Volume)
    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=final_intensity.flatten(),
        isomin=0.25, isomax=1.0, opacity=0.5, surface_count=10, colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False 
    ))

    # B. 渲染顶层带有视觉特征的表面融合图像 (Z=0)
    print("正在映射表面融合纹理与视觉检测标记...")
    surface_colors = get_blended_surface_colors(Lx, Ly, nx, ny, defects)
    X_surf, Y_surf = np.mgrid[0:Lx:complex(nx), 0:Ly:complex(ny)]
    Z_surf = np.zeros_like(X_surf) # 表面锁定在 Z=0 处
    
    fig.add_trace(go.Scatter3d(
        x=X_surf.flatten(), y=Y_surf.flatten(), z=Z_surf.flatten(),
        mode='markers',
        marker=dict(
            size=4, 
            color=surface_colors,
            symbol='square',
            opacity=0.9
        ),
        hoverinfo='skip'
    ))

    # 边框辅助线
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

    z_visual_scale = (Lz / Lx) * 6.0 
    fig.update_layout(
        title='多模态 BEV 感知：无缝表面视觉 + 内部声学物理场',
        scene=dict(
            xaxis_title='X 轴 (mm)', yaxis_title='Y 轴 (mm)', zaxis_title='深度 Z (mm)',
            aspectmode='manual', aspectratio=dict(x=1, y=Ly/Lx, z=z_visual_scale), 
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=-1.5, z=-0.4) 
            )
        ),
        dragmode='orbit', 
        template='plotly_dark', margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.show()

if __name__ == "__main__":
    create_sample_txt()
    render_multimodal_bev(TXT_FILENAME)
