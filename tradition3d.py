import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
from scipy.ndimage import label, find_objects, binary_closing # 🌟 新增形态学库
import matplotlib.pyplot as plt
import base64

FILE_NAME = "2026.4.21_normalized.txt"

def parse_render_and_quantify_raw(filepath):
    print(f"📂 正在解析真实采集卡数据: {filepath} ...")
    
    if not os.path.exists(filepath):
        print("❌ 找不到文件，请确保文件名正确！")
        return

    # 1. 终极防爆读取
    content = ""
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError: continue

    if not content: return

    # 2. 动态自适应切分
    parts = re.split(r'={5,}', content)
    header_str = parts[0] if len(parts) >= 2 else ""
    data_str = parts[1] if len(parts) >= 2 else content

    # 3. 提取那三个核心物理坐标轴大小
    nums = re.findall(r'\d+', header_str)
    if len(nums) >= 3:
        ny, nx, nz = int(nums[0]), int(nums[1]), int(nums[2])
    else:
        ny, nx, nz = 20, 300, 339

    # 4. 暴力榨取所有数据点
    print("⏳ 正在进行底层信号提取与转换，请稍候...")
    cleaned_data = data_str.strip()
    data_list = []
    for item in cleaned_data.split():
        try: data_list.append(float(item))
        except ValueError: continue
            
    raw_values = np.array(data_list, dtype=np.float32)
    
    # 5. 核心物理反演：适配新数据极性
    # 注意：新的归一化数据背景已经是0左右，回波是高值，所以不能再用 255 相减了！
    energy_values = raw_values
    
    total_expected = nx * ny * nz
    if len(energy_values) < total_expected:
        padded = np.zeros(total_expected)
        padded[:len(energy_values)] = energy_values
        energy_values = padded
    elif len(energy_values) > total_expected:
        energy_values = energy_values[:total_expected]

    # 6. 张量重塑
    volume_3d = energy_values.reshape((ny, nx, nz))
    volume_3d = volume_3d.transpose((1, 0, 2)) 
    # 🌟 核心改进 1：智能极性识别 (自动适配正相/反相数据)
    # 某些采集数据背景是 0(低值)，回波是高值；但比如 1.1.txt，背景却是 255(高值)，回波由于衰减/吸波变成了 82(低值)。
    # 我们通过统计学中位数来判断背景极性：如果中位数偏大，说明高值是背景，需要反转极性！
    v_min = np.min(volume_3d)
    v_max = np.max(volume_3d)
    v_median = np.median(volume_3d)
    
    if v_median > (v_min + v_max) / 2.0:
        print(f"🔄 检测到反相超声信号 (中位数 {v_median:.1f} 偏向高值)，正在执行极性自动反转...")
        volume_3d = v_max - volume_3d
        
    # 🌟 核心改进 2：在送入算法前，强制对整个体积数据进行 0-255 的极值拉伸统一！
    # 这样无论是哪种传感器采集的几十或一百多的范围，都会被标准化为背景 0，缺陷 255
    v_min = np.min(volume_3d)
    v_max = np.max(volume_3d)
    volume_3d = (volume_3d - v_min) / (v_max - v_min + 1e-5) * 255.0
    print(f"🔄 已对输入数据进行动态极值拉伸: 统一映射至 0-255 范围")

    # ==========================================
    # 🌟 智能缺陷检测算法 (Smart Defect Extraction)
    # ==========================================
    print("\n" + "="*50)
    print("📊 启动智能缺陷检测算法 (针对底部缺陷优化)")
    print("="*50)

    res_x, res_y, res_z = 1.0, 2.0, 0.15 
    
    # 1. 设置物理盲区 (过滤掉顶部极强的表面波干扰，约 15mm)
    surface_blind_mm = 15.0
    z_start_idx = int(surface_blind_mm / res_z)
    z_end_idx = nz - int(3.0 / res_z) # 底部也稍微留一点盲区
    
    internal_raw = np.zeros_like(volume_3d)
    internal_raw[:, :, z_start_idx:z_end_idx] = volume_3d[:, :, z_start_idx:z_end_idx]
    
    max_internal = np.max(internal_raw)
    print(f"🔬 内部区域提取完成，内部最大能量峰值: {max_internal:.2f}")
    
    # 2. 绝对能量阈值分割
    # 数据已经强制拉伸到 0-255。背景往往是低频散斑，缺陷则是 120-255 之间的高亮信号
    threshold_energy = 60.0 
    binary_mask = internal_raw > threshold_energy
    
    # 3. 形态学定向聚类：解决Z轴方向的断裂
    print(f"⏳ 正在进行高能区域聚类 (Threshold={threshold_energy})...")
    # ⚠️ 修复：两个包围盒是因为Z轴方向有微小断层。我们专门使用 Z 轴方向的结构元素进行闭运算，
    # 这样既能完美粘合上下两个包围盒，又能彻底避免边界效应把缺陷“腐蚀”掉
    z_struct = np.zeros((3, 3, 3))
    z_struct[1, 1, :] = 1  # 纯 Z 轴方向的 3 像素长条
    binary_mask = binary_closing(binary_mask, structure=z_struct, iterations=4)
    
    # 4. 连通域提取
    labeled_array, num_features = label(binary_mask)
    print(f"DEBUG: num_features = {num_features}")
    
    defects_info = []
    target_slice_z = int(nz * 0.8) # 默认切片看较深的位置
    
    valid_count = 0
    if num_features > 0:
        objects = find_objects(labeled_array)
        for i, obj in enumerate(objects):
            slice_x, slice_y, slice_z = obj
            
            phys_lx = (slice_x.stop - slice_x.start) * res_x
            phys_ly = (slice_y.stop - slice_y.start) * res_y
            phys_lz = (slice_z.stop - slice_z.start) * res_z
            
            # 计算物理中心坐标用于过滤
            cx = (slice_x.start + slice_x.stop) / 2 * res_x
            cy = (slice_y.start + slice_y.stop) / 2 * res_y
            cz = (slice_z.start + slice_z.stop) / 2 * res_z

            # --- 核心过滤逻辑 ---
            # 规则 A: 过滤太碎的单像素噪点 (保留大于 3mm 的实体)
            if phys_lx < 3.0 and phys_ly < 3.0:
                continue
                
            # 规则 B: 过滤大面积的层状伪影/底波 (如果缺陷面积铺满了整个扫描面的一半，那它大概率是底波或表面波残留)
            if phys_lx > (nx * res_x * 0.4) or phys_ly > (ny * res_y * 0.4):
                continue
                
            # 规则 C: 我们只关心较深位置的缺陷 (避免表面浅层杂波干扰)
            if cz < 30.0:
                continue
                
            valid_count += 1
            cx = (slice_x.start + slice_x.stop) / 2 * res_x
            cy = (slice_y.start + slice_y.stop) / 2 * res_y
            cz = (slice_z.start + slice_z.stop) / 2 * res_z
            
            defects_info.append({
                'id': valid_count, 'cx': cx, 'cy': cy, 'cz': cz,
                'lx': phys_lx, 'ly': phys_ly, 'lz': phys_lz,
                'x_range': (slice_x.start, slice_x.stop),
                'y_range': (slice_y.start, slice_y.stop),
                'z_range': (slice_z.start, slice_z.stop)
            })
            
            print(f" 🚩 发现底部有效缺陷 {valid_count} 号:")
            print(f"    - 中心坐标 (X,Y,Z): ({cx:.1f}, {cy:.1f}, {cz:.1f}) mm")
            print(f"    - 物理尺寸: 长 {phys_lx:.1f}mm × 宽 {phys_ly:.1f}mm × 深 {phys_lz:.1f}mm")
            
            target_slice_z = int((slice_z.start + slice_z.stop) / 2)
            
    print(f"✅ 系统总计精准捕获 {valid_count} 处局部有效缺陷！")
    print("="*50 + "\n")

    # 执行切片
    depth_slice = volume_3d[:, :, target_slice_z]

    # ==========================================
    # 🌟 渲染 WebGL 仪表盘 (升级为动态切片滑块)
    # ==========================================
    print("🚀 正在生成量化可视化仪表盘(带有图片切片)...")
    
    # 1. 预先生成所有 Z 层的切片并转换为 Base64
    slice_dir = "slices_tradition"
    if not os.path.exists(slice_dir):
        os.makedirs(slice_dir)
        
    print(f"🚀 正在保存并硬编码 {nz} 张切片图像，请稍候...")
    steps = []
    initial_z_idx = target_slice_z
    initial_data_uri = ""
    
    for i in range(nz):
        depth_val = i * res_z
        slice_data = volume_3d[:, :, i].T
        
        img_path = f"{slice_dir}/slice_{i}.png"
        plt.imsave(img_path, slice_data, cmap='jet', format='png', origin='upper', vmin=0, vmax=255)
        
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        data_uri = f"data:image/png;base64,{encoded_string}"
        
        if i == initial_z_idx:
            initial_data_uri = data_uri
            
        step = dict(
            method="relayout",
            args=[{"images[0].source": data_uri}, {"annotations[0].text": f"Raw Tomographic Slice (Z={depth_val:.1f}mm)"}],
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
        len=0.30,
        steps=steps,
        font=dict(color='white')
    )]

    step_x, step_y, step_z = 3, 1, 3
    vol_downsampled = volume_3d[::step_x, ::step_y, ::step_z]
    grid_x, grid_y, grid_z = np.mgrid[0:nx:step_x, 0:ny:step_y, 0:nz:step_z]

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.20, 0.45, 0.35], 
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "table"}]],
        subplot_titles=(f"Raw Tomographic Slice (Z={target_slice_z*res_z:.1f}mm)", "Raw 3D Field with Defect Bounding Boxes", "AI 自动量化检测报告")
    )

    # 左侧：用隐形边框撑起坐标系
    fig.add_trace(go.Scatter(
        x=[0, nx, nx, 0, 0], y=[0, 0, ny, ny, 0],
        mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=1),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

    # 将首张图片注入 Layout
    fig.add_layout_image(
        dict(
            source=initial_data_uri,
            xref="x1", yref="y1",
            x=0, y=0,               # <--- 反转的Y轴，0为最上方
            xanchor="left",
            yanchor="top",
            sizex=nx, sizey=ny,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # 右侧：原始 3D 体素图
    iso_min = 40.0 # 原始数据的杂波底噪
    iso_max = np.max(vol_downsampled)

    fig.add_trace(go.Volume(
        x=grid_x.flatten(), y=grid_y.flatten(), z=grid_z.flatten(),
        value=vol_downsampled.flatten(),
        isomin=iso_min, isomax=iso_max, opacity=0.3, surface_count=6, colorscale='Jet',
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title=dict(text="Raw<br>Amplitude", font=dict(color='white')), x=1.02)
    ), row=1, col=2)

    # 绘制 3D 物理包围盒
    for d in defects_info:
        x0, x1 = d['x_range']
        y0, y1 = d['y_range']
        z0, z1 = d['z_range']
        bx = [x0, x1, x1, x0, x0, x0, x1, x1, x0, x0, x1, x1, x1, x1, x0, x0]
        by = [y0, y0, y1, y1, y0, y0, y0, y1, y1, y0, y0, y0, y1, y1, y1, y1]
        bz = [z0, z0, z0, z0, z0, z1, z1, z1, z1, z1, z1, z0, z0, z1, z1, z0]
        
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode='lines', line=dict(color='white', width=4, dash='solid'),
            name=f"Raw Box {d['id']}"
        ), row=1, col=2)

    # ==========================
    # 右侧：添加缺陷量化数据表格
    # ==========================
    table_headers = ['ID', '中心 (X,Y,Z)', '尺寸 L*W*H', '状态']
    table_rows = []
    for d in defects_info:
        coord_str = f"{d['cx']:.1f}, {d['cy']:.1f}, {d['cz']:.1f}"
        size_str = f"{d['lx']:.1f} x {d['ly']:.1f} x {d['lz']:.1f}"
        status = "🔴 危急" if max(d['lx'], d['ly']) > 20 else "🟡 警告"
        table_rows.append([d['id'], coord_str, size_str, status])
    if not table_rows: table_rows = [["-"]*4]
    
    fig.add_trace(go.Table(
        columnwidth=[0.8, 2.2, 2.2, 1.2], 
        header=dict(values=table_headers, fill_color='#1f77b4', font=dict(color='white', size=13), align='center'),
        cells=dict(values=list(zip(*table_rows)), fill_color='#2c3e50', font=dict(color='white', size=12), align='center', height=30)
    ), row=1, col=3)

    fig.update_layout(
        title=dict(text='Raw Data Quantification (Baseline / No Physics Compensation)', font=dict(color='white', size=20)),
        sliders=sliders,
        scene=dict(
            xaxis=dict(title='Scan X', color='white', showbackground=False),
            yaxis=dict(title='Index Y', color='white', showbackground=False, autorange='reversed'),
            zaxis=dict(title='Depth Z', color='white', showbackground=False, autorange='reversed'),
            aspectratio=dict(x=1, y=(ny/nx)*4, z=0.5), 
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        xaxis=dict(color='white'), yaxis=dict(color='white', autorange="reversed"),
        dragmode='turntable', template='plotly_dark', margin=dict(l=20, r=80, b=50, t=80)
    )
    
    output_html = "Raw_Quantification_Dashboard.html"
    fig.write_html(output_html, auto_open=True)
    print(f"✅ 生成完毕！请查看原始状态下的夸张包围盒: {os.path.abspath(output_html)}")

if __name__ == "__main__":
    parse_render_and_quantify_raw(FILE_NAME)