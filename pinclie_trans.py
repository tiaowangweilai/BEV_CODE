# =====================================================================
# 脚本名称 : pinclie_levelset_4.1.py
# 修改日期 : 2026-05-07
# 作    者 : Antigravity & USER
# 脚本作用 : 根据绘画txt文件，生成原始超声数据文件。
# =====================================================================

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from scipy.ndimage import gaussian_filter, label, find_objects
import matplotlib.pyplot as plt

TXT_FILENAME = "1.txt"
RAW_OUTPUT_FILENAME = "raw_ultrasound_data.txt" # 导出的原始数据文件名

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

# ================= 新增：原始超声数据导出引擎 =================
def export_to_raw_txt(combined_field, nx, ny, nz, output_filename):
    print(f"\n💾 正在将三维能量场降维并导出为采集卡 TXT 格式...")
    
    # 1. 坐标轴转换：从 (nx, ny, nz) 转为 (ny, nx, nz) 
    # 匹配 "步进(y) x 扫查(x) x 声程(z)" 的工业标准存储顺序
    volume = combined_field.transpose(1, 0, 2)
    
    # 2. 物理量纲映射：PhysBEV (0.15~1.0) 反相映射到 8-bit (255~0)
    # 工业格式中，255 代表静音(无能量)，数值越小代表能量越强
    mapped_volume = np.clip((1.0 - volume) * 255, 0, 255).astype(np.int32)
    
    # 3. 展平并流式写入防爆内存
    flattened_data = mapped_volume.flatten()
    total_points = len(flattened_data)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        # 严格写入文件头
        f.write(f"步进步数：{ny}，扫查步数：{nx}，声程点数：{nz}\n")
        f.write("====================\n")
        
        # 分块写入，防止字符串过长导致内存溢出
        chunk_size = 200000 
        for i in range(0, total_points, chunk_size):
            chunk = flattened_data[i : i + chunk_size]
            chunk_str = " ".join(chunk.astype(str))
            if i > 0: f.write(" ") 
            f.write(chunk_str)
            
            # 进度提示
            progress = min(100, int((i + chunk_size) / total_points * 100))
            print(f"\r   写入进度: {progress}%", end="")
            
    print(f"\n✅ 原始超声数据已成功保存至: {os.path.abspath(output_filename)}")

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
    combined_field = np.full_like(X, 0.15) 

    # 注入表面、缺陷、底波
    surf_2d = get_volumetric_surface_intensity(Lx, Ly, nx, ny)
    surf_data = surf_2d.T 
    for zi in range(3): combined_field[:, :, zi] = np.maximum(combined_field[:, :, zi], surf_data)

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
    internal_field[:, :, :5] = 0 
    internal_field[0,:,:] = 0; internal_field[-1,:,:] = 0 
    
    binary_mask = internal_field > 0.6
    labeled_array, num_features = label(binary_mask)
    objects = find_objects(labeled_array)

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
        coord_str = f"{cx_mm:.0f}, {cy_mm:.0f}, {cz_mm:.0f}"
        size_str = f"{len_x:.0f} x {len_y:.0f} x {len_z:.0f}"
        table_rows.append([i+1, coord_str, size_str, status])
        
        defects_info.append({
            'id': i+1,
            'x_range': (sx.start * dx, sx.stop * dx),
            'y_range': (sy.start * dy, sy.stop * dy),
            'z_range': (sz.start * dz, sz.stop * dz)
        })

    # ================= 4. 保存实体文件到本地文件夹 =================
    slice_dir = "slices2"
    if not os.path.exists(slice_dir):
        os.makedirs(slice_dir)
        
    print(f"🚀 正在将切片实体图片保存至 '{slice_dir}' 文件夹...")
    
    steps = []
    initial_z_idx = nz // 2  
    
    import base64
    initial_data_uri = ""
    for i in range(nz):
        depth_val = i * dz
        slice_data = combined_field[:, :, i].T
        
        img_path = f"{slice_dir}/slice_{i}.png"
        plt.imsave(img_path, slice_data, cmap='jet', format='png', origin='upper', vmin=0.1, vmax=1.0)
        
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        data_uri = f"data:image/png;base64,{encoded_string}"
        
        if i == initial_z_idx:
            initial_data_uri = data_uri
            
        step = dict(
            method="relayout",
            args=[{"images[0].source": data_uri}], 
            label=f"{depth_val:.1f}"
        )
        steps.append(step)

    sliders = [dict(
        active=initial_z_idx,
        currentvalue={"prefix": "当前深度: ", "suffix": " mm"},
        pad={"t": 30},
        x=0.01,
        y=-0.15,
        xanchor='left',
        yanchor='top',
        len=0.20,
        steps=steps,
        font=dict(color='white')
    )]

    # ================= 5. 大屏布局渲染 =================
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.20, 0.48, 0.32],
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "table"}]],
        subplot_titles=("2D 深度切片 (拖动下方滑块)", "3D 物理场 (带拓扑追踪)", "AI 自动量化检测报告")
    )

    fig.add_trace(go.Scatter(
        x=[0, Lx, Lx, 0, 0], y=[0, 0, Ly, Ly, 0],
        mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=1),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    fig.add_layout_image(
        dict(
            source=initial_data_uri,
            xref="x1", yref="y1",
            x=0, y=0,               
            xanchor="left",         
            yanchor="top",          
            sizex=Lx, sizey=Ly,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=combined_field.flatten(),
        isomin=0.12, isomax=1.0, 
        opacity=0.35, 
        surface_count=12,
        colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=True,
        colorbar=dict(title="Energy", thickness=15, x=0.66)
    ), row=1, col=2)

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

    if not table_rows: table_rows = [["-"]*4]
    fig.add_trace(go.Table(
        columnwidth=[0.8, 2.0, 2.0, 1.2], 
        header=dict(values=table_headers, fill_color='#1f77b4', font=dict(color='white', size=13), align='center'),
        cells=dict(values=list(zip(*table_rows)), fill_color='#2c3e50', font=dict(color='white', size=12), align='center', height=30)
    ), row=1, col=3)

    fig.update_layout(
        template='plotly_dark',
        title=dict(text='PhysBEV 工业级多模态探伤与量化系统', font=dict(color='white', size=22)),
        sliders=sliders,
        scene=dict(
            aspectratio=dict(x=1, y=Ly/Lx, z=0.4),
            xaxis=dict(title="X (mm)", showbackground=False), 
            yaxis=dict(title="Y (mm)", showbackground=False), 
            zaxis=dict(title="Depth (mm)", showbackground=False, autorange='reversed'),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))
        ),
        dragmode='orbit', 
        margin=dict(l=20, r=10, b=100, t=80) 
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)

    output_file = "PhysBEV_Pro_Dashboard.html"
    fig.write_html(output_file, auto_open=True)
    print(f"✅ HTML生成完毕！切片文件已储存于 '{slice_dir}' 文件夹，请在浏览器中打开: {os.path.abspath(output_file)}")

    # ================= 6. 核心修改：触发原始超声数据导出 =================
    export_to_raw_txt(combined_field, nx, ny, nz, RAW_OUTPUT_FILENAME)

if __name__ == "__main__":
    render_unified_physics_bev(TXT_FILENAME)