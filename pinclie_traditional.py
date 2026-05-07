import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from scipy.ndimage import gaussian_filter

TXT_FILENAME = "1.txt"

def get_noisy_surface_intensity(Lx, Ly, nx, ny):
    """提取表面能量，加入较多噪点"""
    if os.path.exists('1.jpg') and os.path.exists('2.jpg'):
        img_steel = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
        img_lime = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        np.random.seed(42)
        img_steel = np.random.randint(20, 90, (100, 100), dtype=np.uint8)
        img_lime = np.random.randint(150, 230, (100, 100), dtype=np.uint8)

    steel_h = int((300 / Ly) * ny)
    s_part = cv2.resize(img_steel, (nx, steel_h))
    l_part = cv2.resize(img_lime, (nx, ny - steel_h))
    full_surface = np.vstack([s_part, l_part])
    
    surface_intensity = 1.0 - (full_surface.astype(float) / 255.0)
    return np.clip(surface_intensity * 0.6 + 0.15, 0.1, 0.8)

def generate_mid_tier_dashboard(txt_filepath):
    Lx, Ly, Lz = 1100.0, 2600.0, 50.0 
    defects = []
    if os.path.exists(txt_filepath):
        with open(txt_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split()
                if parts[0] == 'Area': Lx, Ly, Lz = float(parts[1]), float(parts[2]), float(parts[3])
                elif len(parts) >= 10:
                    defects.append({'cx': float(parts[1]), 'cy': float(parts[2]), 'cz': float(parts[3]),
                                    'sx': float(parts[4]), 'sy': float(parts[5]), 'sz': float(parts[6]),
                                    'rx': float(parts[7]), 'ry': float(parts[8]), 'rz': float(parts[9])})

    nx = 75; ny = int(nx * (Ly / Lx)); nz = 35 
    X, Y, Z = np.mgrid[0:Lx:complex(nx), 0:Ly:complex(ny), 0:Lz:complex(nz)]
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)
    
    background_val = 0.2 
    combined_field = np.full_like(X, background_val)

    surf_data = get_noisy_surface_intensity(Lx, Ly, nx, ny).T
    for zi in range(2): combined_field[:, :, zi] = np.maximum(combined_field[:, :, zi], surf_data)

    for d in defects:
        if d['cy'] > 600: continue
        cx, cy, cz = d['cx'], d['cy'], d['cz']
        
        # ==========================================
        # 🌟 破绽制造区：模拟超大发散角的探头
        # ==========================================
        # 1. 随着深度急剧扩大的发散因子
        divergence_factor = 1.0 + (cz * 0.04) 
        
        # 2. 强制将横向影响范围 (sx, sy) 放大 2.5 倍，导致物理边界极其模糊
        sx = max(d['sx'] * divergence_factor * 1.5, 6.0)
        sy = max(d['sy'] * divergence_factor * 1.5, 6.0)
        sz = max(d['sz'] * 1.5, 3.0)
        
        rotation = R_scipy.from_euler('xyz', [d['rx'], d['ry'], d['rz']], degrees=True)
        R_inv = rotation.inv().as_matrix()
        points_local = R_inv @ (points - np.array([[cx], [cy], [cz]]))
        X_l = points_local[0, :].reshape(X.shape); Y_l = points_local[1, :].reshape(X.shape); Z_l = points_local[2, :].reshape(X.shape)
        
        # 3. 基础 2次幂高斯（拖尾长，导致重影） + 无严格增益补偿（深处能量低）
        depth_penalty = max(1.0 - (cz / Lz) * 0.5, 0.45) 
        local_def = np.exp(-((X_l/sx)**2 + (Y_l/sy)**2 + (Z_l/sz)**2)) * depth_penalty
        
        # 4. 使用能量叠加而不是单纯的 maximum，让两个靠近的缺陷之间的区域能量累积，产生明显的“桥接/粘连”效应
        combined_field = np.clip(combined_field + local_def * 0.6, 0, 1.0)

    np.random.seed(10)
    noise = gaussian_filter(np.random.normal(0, 1, (nx, ny)), sigma=3.0)
    actual_bottom_z = (Lz - 3.0) + ((noise / np.max(np.abs(noise))) * 6.0)[:, :, np.newaxis]
    bottom_base = np.exp(-((Z - actual_bottom_z) / 4.0)**2)
    combined_field = np.maximum(combined_field, bottom_base * 0.35)

    bev_projection = np.max(combined_field[:, :, 3:], axis=2)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.25, 0.75], 
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("2D BEV Projection (Ghosting Artifacts)", "PhysBEV Voxel Space (Uncalibrated)")
    )

    fig.add_trace(go.Heatmap(
        z=bev_projection.T, 
        colorscale='Jet', zmin=0.1, zmax=1.0, showscale=False 
    ), row=1, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=combined_field.flatten(),
        isomin=0.25, isomax=1.0, opacity=0.25, surface_count=10, colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=True, 
        colorbar=dict(title=dict(text="Energy", font=dict(color='white')), thickness=15, len=0.8, x=1.02, tickfont=dict(color='white'))
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text='PhysBEV Architecture: V1.0 Baseline (Acoustic Blurring Issue)', font=dict(color='white', size=20)),
        scene=dict(
            xaxis=dict(showbackground=False, color='white'), 
            yaxis=dict(showbackground=False, color='white'), 
            zaxis=dict(showbackground=False, color='white'),
            aspectratio=dict(x=1, y=Ly/Lx, z=0.5),
            camera=dict(eye=dict(x=1.3, y=-1.5, z=0.7))
        ),
        xaxis=dict(color='white'), yaxis=dict(color='white', autorange="reversed"),
        template='plotly_dark', margin=dict(l=20, r=80, b=50, t=80)
    )
    
    output_html = "PhysBEV_Ghosting_Test.html"
    fig.write_html(output_html, auto_open=True)

if __name__ == "__main__":
    generate_mid_tier_dashboard(TXT_FILENAME)