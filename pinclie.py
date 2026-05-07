# =====================================================================
# 脚本名称 : pinclie.py
# 修改日期 : 2026-05-07
# 作    者 : Antigravity & USER
# 脚本作用 : 仿真可视化引擎 (早期版本)。用于基础的三维点云包围盒绘制与渲染。
# =====================================================================

import numpy as np
import plotly.graph_objects as go
import os
from scipy.spatial.transform import Rotation as R_scipy

TXT_FILENAME = "2026.3.17.txt"

def load_and_render_strip_defects(txt_filepath):
    Lx, Ly, Lz = 1100.0, 2600.0, 50.0 # 默认值，将被 TXT 第一行覆盖
    defects = []
    
    # ==========================================
    # 1. 解析 TXT 文件
    # ==========================================
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
    # 2. 自适应网格分辨率 (解决大尺寸下的锯齿问题)
    # ==========================================
    # 按照实际物理比例分配采样点，保证 Y 轴方向的平滑度
    nx = 110
    ny = int(nx * (Ly / Lx)) # 260
    nz = 50
    
    print(f"正在生成 {nx}x{ny}x{nz} 的高精度空间网格...")
    X, Y, Z = np.mgrid[0:Lx:complex(nx), 0:Ly:complex(ny), 0:Lz:complex(nz)]
    global_intensity = np.zeros_like(X)
    
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)

    # ==========================================
    # 3. 6D 姿态计算与场融合
    # ==========================================
    for d in defects:
        cx, cy, cz = d['cx'], d['cy'], d['cz']
        sx, sy, sz = max(d['sx'], 0.1), max(d['sy'], 0.1), max(d['sz'], 0.1)
        rx, ry, rz = d['rx'], d['ry'], d['rz']
        
        # 旋转矩阵
        rotation = R_scipy.from_euler('xyz', [rx, ry, rz], degrees=True)
        R_inv = rotation.inv().as_matrix()
        
        # 坐标变换
        points_shifted = points - np.array([[cx], [cy], [cz]])
        points_local = R_inv @ points_shifted
        
        X_local = points_local[0, :].reshape(X.shape)
        Y_local = points_local[1, :].reshape(Y.shape)
        Z_local = points_local[2, :].reshape(Z.shape)
        
        # 核心：长条形缺陷的高阶超高斯分布 (4次方)
        local_intensity = np.exp(-((X_local/sx)**4 + (Y_local/sy)**4 + (Z_local/sz)**4))
        
        global_intensity = np.maximum(global_intensity, local_intensity)

    # ==========================================
    # 4. Plotly 渲染
    # ==========================================
    print("正在渲染 3D 体积...")
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=global_intensity.flatten(),
        isomin=0.15, isomax=1.0, opacity=0.35, surface_count=20, colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    # 绘制外边框
    edges = [
        ([0,Lx,Lx,0,0], [0,0,Ly,Ly,0], [0,0,0,0,0]),
        ([0,Lx,Lx,0,0], [0,0,Ly,Ly,0], [Lz,Lz,Lz,Lz,Lz]),
        ([0,0], [0,0], [0,Lz]), ([Lx,Lx], [0,0], [0,Lz]),
        ([Lx,Lx], [Ly,Ly], [0,Lz]), ([0,0], [Ly,Ly], [0,Lz])
    ]
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=edge[0], y=edge[1], z=edge[2], mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'),
            showlegend=False
        ))

    # Z 轴视觉放大系数：因为 50mm 相对 2600mm 太薄了，视觉上放大 10 倍以看清内部
    z_visual_scale = (Lz / Lx) * 10.0 
    
    fig.update_layout(
        title='1.1m x 2.6m 大区域超声感知场 - 多倾角长条形缺陷',
        scene=dict(
            xaxis_title='X 轴 (mm)', yaxis_title='Y 轴 (mm)', zaxis_title='深度 Z (mm)',
            aspectmode='manual', 
            aspectratio=dict(x=1, y=Ly/Lx, z=z_visual_scale), # 保持 X 和 Y 的 1.1:2.6 真实比例
            camera=dict(eye=dict(x=1.5, y=-1.0, z=0.8))
        ),
        template='plotly_dark', margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.show()

if __name__ == "__main__":
    if not os.path.exists(TXT_FILENAME):
        print("请先创建对应的 txt 配置文件。")
    else:
        load_and_render_strip_defects(TXT_FILENAME)