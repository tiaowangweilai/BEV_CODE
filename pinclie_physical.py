import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from scipy.ndimage import gaussian_filter

TXT_FILENAME = "1.txt"

def get_volumetric_surface_intensity(Lx, Ly, nx, ny):
    """
    【视觉语义能量提取】：模拟表面不连续性
    """
    if os.path.exists('1.jpg') and os.path.exists('2.jpg'):
        img_steel = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
        img_lime = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        np.random.seed(0)
        img_steel = np.random.randint(40, 70, (100, 100), dtype=np.uint8)
        img_lime = np.random.randint(180, 210, (100, 100), dtype=np.uint8)

    steel_h = int((300 / Ly) * ny)
    s_part = cv2.resize(img_steel, (nx, steel_h))
    l_part = cv2.resize(img_lime, (nx, ny - steel_h))
    full_surface = np.vstack([s_part, l_part])
    
    surface_intensity = 1.0 - (full_surface.astype(float) / 255.0)
    surface_intensity = np.clip(surface_intensity * 0.7 + 0.1, 0.1, 0.9)
    return surface_intensity

def generate_physics_dashboard(txt_filepath):
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

    nx = 85; ny = int(nx * (Ly / Lx)); nz = 40 
    X, Y, Z = np.mgrid[0:Lx:complex(nx), 0:Ly:complex(ny), 0:Lz:complex(nz)]
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)
    
    background_val = 0.15
    combined_field = np.full_like(X, background_val)

    surf_data = get_volumetric_surface_intensity(Lx, Ly, nx, ny).T
    for zi in range(3): combined_field[:, :, zi] = np.maximum(combined_field[:, :, zi], surf_data)

    alpha = 0.002 
    for d in defects:
        if d['cy'] > 600: continue
        cx, cy, cz = d['cx'], d['cy'], d['cz']
        
        divergence_factor = 1.0 + (cz * 0.005) 
        sx = max(d['sx'] * 0.6 * divergence_factor, 1.5)
        sy = max(d['sy'] * 0.6 * divergence_factor, 1.5)
        sz = max(d['sz'] * 0.6, 1.5)
        
        rotation = R_scipy.from_euler('xyz', [d['rx'], d['ry'], d['rz']], degrees=True)
        R_inv = rotation.inv().as_matrix()
        points_local = R_inv @ (points - np.array([[cx], [cy], [cz]]))
        X_l = points_local[0, :].reshape(X.shape); Y_l = points_local[1, :].reshape(X.shape); Z_l = points_local[2, :].reshape(X.shape)
        
        local_def = np.exp(-((X_l/sx)**4 + (Y_l/sy)**4 + (Z_l/sz)**4)) * np.exp(alpha * cz)
        local_def = np.clip(local_def, 0, 1.0)
        combined_field = np.maximum(combined_field, local_def)

    hull_val = 0.22 
    combined_field[0,:,:] = np.maximum(combined_field[0,:,:], hull_val)
    combined_field[-1,:,:] = np.maximum(combined_field[-1,:,:], hull_val)
    combined_field[:,0,:] = np.maximum(combined_field[:,0,:], hull_val)
    combined_field[:,-1,:] = np.maximum(combined_field[:,-1,:], hull_val)

    np.random.seed(42)
    noise = gaussian_filter(np.random.normal(0, 1, (nx, ny)), sigma=2.0)
    actual_bottom_z = (Lz - 3.0) + ((noise / np.max(np.abs(noise))) * 4.5)[:, :, np.newaxis]
    bottom_base = np.exp(-((Z - actual_bottom_z) / 3.5)**2)
    proj = np.max(combined_field[:, :, 3:], axis=2, keepdims=True)
    mask = np.clip(1.0 - proj * 1.5, 0.0, 1.0) 
    combined_field = np.maximum(combined_field, bottom_base * 0.4 * mask)

    bev_projection = np.max(combined_field[:, :, 5:], axis=2)

    print("启动物理感知场联合渲染...")
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.3, 0.7], 
        specs=[[{"type": "heatmap"}, {"type": "volume"}]],
        subplot_titles=("Z-Axis Max Energy Projection (2D BEV)", "Unified Physics Voxel Field (3D)")
    )

    fig.add_trace(go.Heatmap(
        z=bev_projection.T, 
        colorscale='Jet', zmin=0.1, zmax=1.0,
        showscale=False 
    ), row=1, col=1)

    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=combined_field.flatten(),
        isomin=0.1, isomax=1.0, opacity=0.35, surface_count=20, colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=True, 
        colorbar=dict(
            title=dict(text="Energy (E)", font=dict(color='white', size=14)),
            thickness=15, len=0.8, x=1.02, tickfont=dict(color='white')
        )
    ), row=1, col=2)

    # ==========================================
    # 🌟 修复：去除 3D 标注中不支持的 xref/yref/zref
    # ==========================================
    annotations_3d = [
        dict(
            x=Lx*0.5, y=100, z=0,
            text="视觉表层高能区<br>(语义引导)",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="white",
            ax=-50, ay=-50, font=dict(color="cyan", size=12)
        ),
        dict(
            x=Lx*0.5, y=400, z=Lz,
            text="底层物理声影<br>(遮挡投影)",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="white",
            ax=-50, ay=50, font=dict(color="orange", size=12)
        )
    ]

    fig.update_layout(
        title=dict(text='受物理方程约束的多模态 BEV 孪生感知场', font=dict(color='white', size=24)),
        scene=dict(
            xaxis=dict(showbackground=False, color='white', title='X(mm)'), 
            yaxis=dict(showbackground=False, color='white', title='Y(mm)'), 
            zaxis=dict(showbackground=False, color='white', title='Depth Z(mm)'),
            aspectratio=dict(x=1, y=Ly/Lx, z=0.5),
            camera=dict(eye=dict(x=1.3, y=-1.5, z=0.7)),
            annotations=annotations_3d # 🌟 修复：将标注正确挂载到 3D scene 内部
        ),
        xaxis=dict(color='white'), yaxis=dict(color='white', autorange="reversed"),
        dragmode='orbit', template='plotly_dark', margin=dict(l=20, r=80, b=20, t=80)
    )
    
    output_html = "Physics_Twin_Dashboard.html"
    fig.write_html(output_html, auto_open=True)
    print(f"✅ 生成完毕！请双击打开: {os.path.abspath(output_html)}")

if __name__ == "__main__":
    generate_physics_dashboard(TXT_FILENAME)