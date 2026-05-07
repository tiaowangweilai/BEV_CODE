import numpy as np
import os
import re
from scipy.ndimage import gaussian_filter

# ==========================================
# ⚙️ 配置：分辨率设置 (根据 Area 映射格点)
# ==========================================
# 假设 X 和 Y 方向 1mm 对应多少个像素，Z 方向由于厚度小，分辨率通常更高
RES_X = 1.0  # 1mm = 1 pixel
RES_Y = 0.5  # 2mm = 1 pixel (对应你之前 ny 比较小的情况)
RES_Z = 10.0 # 0.1mm = 1 pixel

def generate_ultrasound_from_config(config_text):
    # 1. 解析 Area 物理尺寸
    area_match = re.search(r"Area\s+(\d+)\s+(\d+)\s+(\d+)", config_text)
    if not area_match:
        print("❌ 未找到 Area 配置！")
        return
    
    phys_x, phys_y, phys_z = map(float, area_match.groups())
    
    # 映射为网格维度
    nx = int(phys_x * RES_X)
    ny = int(phys_y * RES_Y)
    nz = int(phys_z * RES_Z)
    
    print(f"🚀 启动物理场模拟: Area({phys_x}x{phys_y}x{phys_z}) -> Grid({ny}x{nx}x{nz})")

    # 2. 初始化背景噪声 (结构性斑点噪声)
    np.random.seed(42)
    base_noise = np.random.normal(0, 1, (ny, nx, nz))
    speckle = gaussian_filter(base_noise, sigma=(1, 2, 2))
    noise_intensity = np.abs(speckle) / np.max(np.abs(speckle)) * 30
    volume = 255.0 - noise_intensity - np.random.randint(0, 5, (ny, nx, nz))

    # 3. 边界效应 (表面回波与底波)
    print("⏳ 注入边界物理特征...")
    front_wall = 10 # 表面层
    volume[:, :, front_wall:front_wall+3] -= 120 # 主反射
    volume[:, :, front_wall+3:front_wall+12] -= np.random.randint(20, 50, (ny, nx, 9)) # 振铃
    volume[:, :, nz-20:nz-10] -= 60 # 底波

    # 4. 动态解析缺陷并植入
    print("⏳ 正在根据配置植入旋转缺陷(Strips)...")
    grid_y, grid_x, grid_z = np.mgrid[0:ny, 0:nx, 0:nz]
    
    # 衰减系数
    alpha = 0.003

    # 正则提取所有 strip 缺陷
    # 格式: strip cx cy cz sx sy sz roll pitch yaw
    strips = re.findall(r"strip\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", config_text)
    
    for i, s in enumerate(strips):
        cx_mm, cy_mm, cz_mm, sx_mm, sy_mm, sz_mm, roll, pitch, yaw_deg = map(float, s)
        
        # 转换为像素坐标
        cx, cy, cz = cx_mm * RES_X, cy_mm * RES_Y, cz_mm * RES_Z
        sx, sy, sz = sx_mm * RES_X, sy_mm * RES_Y, sz_mm * RES_Z
        
        # 旋转逻辑 (Yaw 偏转角 - 绕 Z 轴旋转)
        theta = np.radians(yaw_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # 坐标平移到原点并旋转
        dx = grid_x - cx
        dy = grid_y - cy
        rot_x = dx * cos_t + dy * sin_t
        rot_y = -dx * sin_t + dy * cos_t
        dz = grid_z - cz

        # 计算椭球体/长条体分布 (Gaussian Strip)
        # 公式: exp( -(x_rot^2/sx^2 + y_rot^2/sy^2 + z^2/sz^2) )
        dist = (rot_x**2 / sx**2) + (rot_y**2 / sy**2) + (dz**2 / sz**2)
        defect_shape = np.exp(-dist)
        
        # 强度与衰减补偿
        intensity = 180 * np.exp(-alpha * cz)
        
        # 植入数据体
        volume -= defect_shape * intensity
        print(f"   ✅ 已植入缺陷 {i+1}: 中心({cx_mm},{cy_mm}), 偏转 {yaw_deg}°")

    # 5. 格式化输出
    volume = np.clip(np.round(volume), 0, 255).astype(np.int32)
    output_filename = "Synthetic_2026_Raw.txt"
    
    print(f"⏳ 正在执行流式导出: {output_filename} ...")
    flattened = volume.flatten()
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"步进步数：{ny}，扫查步数：{nx}，声程点数：{nz}\n")
        f.write("====================\n")
        
        chunk_size = 500000
        for i in range(0, len(flattened), chunk_size):
            chunk = flattened[i : i + chunk_size]
            f.write(" ".join(chunk.astype(str)) + " ")
            
    print(f"🎉 模拟数据合成成功！输出文件: {os.path.abspath(output_filename)}")

# ==========================================
# 📄 你的配置文本
# ==========================================
my_config = """
# 绘图区域物理尺寸 (单位: mm)
Area 1100 2600 30

# 缺陷配置 (全部为长条形 strip)
# 格式: 类型 中心X 中心Y 中心Z 半长(sx) 半宽(sy) 半高(sz) Roll Pitch Yaw
strip 98 120 17.76 20 100 2 0 0 30
strip 255.5 140 17.76 20 100 2 0 0 35
strip 470 150 17.52 20 100 2 0 0 40
strip 627 153 17.88 20 100 2 0 0 45
strip 896.5 98 17.52 20 100 2 0 0 50
"""

if __name__ == "__main__":
    generate_ultrasound_from_config(my_config)