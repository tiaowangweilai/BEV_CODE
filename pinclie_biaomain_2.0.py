import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from scipy.ndimage import gaussian_filter, label, find_objects

TXT_FILENAME = "1.txt"

# ================= 恢复：原始的表面生成逻辑 =================
def get_volumetric_surface_intensity(Lx, Ly, nx, ny):
    if os.path.exists('1.jpg') and os.path.exists('2.jpg'):
        img_steel = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
        img_lime = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        img_steel = np.full((100, 100), 50, dtype=np.uint8)
        img_lime = np.full((100, 100), 200, dtype=np.uint8)

    steel_h = int((300 / Ly) * ny)
    s_part = cv2.resize(img_steel, (nx, steel_h))
    l_part = cv2.resize(img_lime, (nx, ny - steel_h))
    full_surface = np.vstack([s_part, l_part])
    surface_intensity = 1.0 - (full_surface.astype(float) / 255.0)
    surface_intensity = np.clip(surface_intensity * 0.7 + 0.1, 0.1, 0.9)
    return surface_intensity

def render_unified_physics_bev(txt_filepath):
    # 1. 物理参数配置
    Lx, Ly, Lz = 1100.0, 2600.0, 50.0 
    defects_data = []
    if os.path.exists(txt_filepath):
        with open(txt_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split()
                if parts[0] == 'Area':
                    Lx, Ly, Lz = float(parts[1]), float(parts[2]), float(parts[3])
                elif len(parts) >= 10:
                    defects_data.append({
                        'cx': float(parts[1]), 'cy': float(parts[2]), 'cz': float(parts[3]),
                        'sx': float(parts[4]), 'sy': float(parts[5]), 'sz': float(parts[6]),
                        'rx': float(parts[7]), 'ry': float(parts[8]), 'rz': float(parts[9])
                    })

    # 2. 网格与坐标标定
    nx, nz = 80, 35
    ny = int(nx * (Ly / Lx))
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz 
    
    print(f"⏳ 正在重构清透版 BEV 空间...")
    X, Y, Z = np.mgrid[0:Lx:complex(nx), 0:Ly:complex(ny), 0:Lz:complex(nz)]
    combined_field = np.full_like(X, 0.15) # 基础底噪

    # ================= 恢复：原始的表面、缺陷、底波注入逻辑 =================
    # 注入表面
    surf_2d = get_volumetric_surface_intensity(Lx, Ly, nx, ny)
    surf_data = surf_2d.T 
    for zi in range(3): combined_field[:, :, zi] = np.maximum(combined_field[:, :, zi], surf_data)

    # 注入缺陷
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)
    for d in defects_data:
        if d['cy'] > 500: continue 
        cx, cy, cz = d['cx'], d['cy'], d['cz']
        sx, sy, sz = max(d['sx'], 2.0), max(d['sy'], 2.0), max(d['sz'], 2.0)
        rotation = R_scipy.from_euler('xyz', [d['rx'], d['ry'], d['rz']], degrees=True)
        R_inv = rotation.inv().as_matrix()
        points_local = R_inv @ (points - np.array([[cx], [cy], [cz]]))
        X_l, Y_l, Z_l = points_local[0,:].reshape(X.shape), points_local[1,:].reshape(X.shape), points_local[2,:].reshape(X.shape)
        combined_field = np.maximum(combined_field, np.exp(-((X_l/sx)**6 + (Y_l/sy)**6 + (Z_l/sz)**6)))

    # 侧墙与薄底波填充 (恢复原貌)
    hull_val = 0.22 
    combined_field[0, :, :] = np.maximum(combined_field[0, :, :], hull_val)
    combined_field[-1, :, :] = np.maximum(combined_field[-1, :, :], hull_val)
    combined_field[:, 0, :] = np.maximum(combined_field[:, 0, :], hull_val)
    combined_field[:, -1, :] = np.maximum(combined_field[:, -1, :], hull_val)
    
    np.random.seed(42)
    noise = gaussian_filter(np.random.normal(0, 1, (nx, ny)), sigma=2.0)
    z_fluct = ((noise / np.max(np.abs(noise))) * 4.5)[:, :, np.newaxis]
    bottom_base = np.exp(-((Z - ((Lz - 3.0) + z_fluct)) / 3.5)**2)
    combined_field = np.maximum(combined_field, bottom_base * 0.4)

    # ================= 3. 拓扑量化与分析 =================
    internal_field = combined_field.copy()
    internal_field[:, :, :5] = 0 # 屏蔽表面
    internal_field[0,:,:] = 0; internal_field[-1,:,:] = 0 
    
    binary_mask = internal_field > 0.6
    labeled_array, num_features = label(binary_mask)
    objects = find_objects(labeled_array)
    
    bev_projection = np.max(internal_field, axis=2)

    # --- 优化：去掉缺陷类型，精简表格 ---
    table_headers = ['ID', '中心 (X,Y,Z)', '尺寸 L*W*H', '状态']
    table_rows = []
    defects_info = []
    
    for i, obj in enumerate(objects):
        sx, sy, sz = obj
        cx_mm = ((sx.start + sx.stop) / 2) * dx
        cy_mm = ((sy.start + sy.stop) / 2) * dy
        cz_mm = ((sz.start + sz.stop) / 2) * dz
        len_x = (sx.stop - sx.start) * dx
        len_y = (sy.stop - sy.start) * dy
        len_z = (sz.stop - sz.start) * dz
        
        status = "🔴 危急" if max(len_x, len_y) > 50 else "🟡 警告"
        
        # 数据转整
        coord_str = f"{cx_mm:.0f}, {cy_mm:.0f}, {cz_mm:.0f}"
        size_str = f"{len_x:.0f} x {len_y:.0f} x {len_z:.0f}"
        table_rows.append([i+1, coord_str, size_str, status])
        
        defects_info.append({
            'id': i+1,
            'x_range': (sx.start * dx, sx.stop * dx),
            'y_range': (sy.start * dy, sy.stop * dy),
            'z_range': (sz.start * dz, sz.stop * dz)
        })

    # ================= 4. 大屏布局渲染 =================
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.20, 0.48, 0.32],
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "table"}]],
        subplot_titles=("2D BEV 投影", "3D 物理场 (带拓扑追踪)", "AI 自动量化检测报告")
    )

    # 左侧 BEV
    fig.add_trace(go.Heatmap(
        z=bev_projection.T, x=np.linspace(0, Lx, nx), y=np.linspace(0, Ly, ny),
        colorscale='Jet', zmin=0.1, zmax=1.0, showscale=False
    ), row=1, col=1)

    # 中间 3D (恢复 20 层高密度，清透底噪)
    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=combined_field.flatten(),
        isomin=0.12, isomax=1.0, 
        opacity=0.35, surface_count=20, colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=True,
        colorbar=dict(title="Energy", thickness=15, x=0.66)
    ), row=1, col=2)

    # --- 优化：洋红色高强对比检测框 ---
    for d in defects_info:
        x0, x1 = d['x_range']
        y0, y1 = d['y_range']
        z0, z1 = d['z_range']
        bx = [x0, x1, x1, x0, x0, x0, x1, x1, x0, x0, x1, x1, x1, x1, x0, x0]
        by = [y0, y0, y1, y1, y0, y0, y0, y1, y1, y0, y0, y0, y1, y1, y1, y1]
        bz = [z0, z0, z0, z0, z0, z1, z1, z1, z1, z1, z1, z0, z0, z1, z1, z0]
        
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode='lines', line=dict(color='magenta', width=6, dash='solid'),
            name=f"Defect {d['id']}"
        ), row=1, col=2)

    # 右侧表格 (去掉缺陷类型，排版更宽敞)
    if not table_rows: table_rows = [["-"]*4]
    fig.add_trace(go.Table(
        columnwidth=[0.8, 2.0, 2.0, 1.2], 
        header=dict(values=table_headers, fill_color='#1f77b4', font=dict(color='white', size=13), align='center'),
        cells=dict(values=list(zip(*table_rows)), fill_color='#2c3e50', font=dict(color='white', size=12), align='center', height=30)
    ), row=1, col=3)

    # ================= 5. 全局配置 =================
    fig.update_layout(
        template='plotly_dark',
        title=dict(text='PhysBEV 工业级多模态探伤与量化系统', font=dict(color='white', size=22)),
        scene=dict(
            aspectratio=dict(x=1, y=Ly/Lx, z=0.4),
            xaxis=dict(title="X (mm)", showbackground=False), 
            yaxis=dict(title="Y (mm)", showbackground=False), 
            zaxis=dict(title="Depth (mm)", showbackground=False, autorange='reversed'),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))
        ),
        dragmode='orbit', 
        margin=dict(l=20, r=10, b=40, t=80)
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)

    output_file = "PhysBEV_Pro_Dashboard.html"
    fig.write_html(output_file, auto_open=True)
    print(f"✅ 工业级大屏生成完毕！请在浏览器查看: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    render_unified_physics_bev(TXT_FILENAME)

