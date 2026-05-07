# =====================================================================
# 脚本名称 : generate_synthetic_raw.py
# 修改日期 : 2026-05-07
# 作    者 : Antigravity & USER
# 脚本作用 : 综合物理合成系统。读取独立缺陷参数配置文件(1.txt)，逆向融合物理退化规律，流式生成逼真的超声原始点阵矩阵。
# =====================================================================

import numpy as np
import os
import re
from scipy.ndimage import gaussian_filter

# ==========================================
# ⚙️ 配置：分辨率设置 (物理尺寸 mm -> 像素网格)
# ==========================================
RES_X = 1.0  # X方向: 1mm = 1 pixel (扫查步数 nx)
RES_Y = 0.2  # Y方向: 5mm = 1 pixel (步进步数 ny)
RES_Z = 10.0 # Z方向: 0.1mm = 1 pixel (声程点数 nz)

def generate_ultrasound_from_txt(config_filepath, output_filename="Synthetic_1.txt"):
    print(f"🚀 启动前向建模引擎：正在读取配置 {config_filepath}...")
    
    if not os.path.exists(config_filepath):
        print(f"❌ 找不到配置文件：{config_filepath}")
        return

    with open(config_filepath, "r", encoding="utf-8") as f:
        config_text = f.read()

    # 1. 解析 Area 物理尺寸
    area_match = re.search(r"Area\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", config_text)
    if not area_match:
        print("❌ 未找到 Area 配置，格式应为：Area 1100 2600 30")
        return
    
    phys_x, phys_y, phys_z = map(float, area_match.groups())
    
    # 映射为网格维度
    nx = int(phys_x * RES_X)
    ny = int(phys_y * RES_Y)
    nz = int(phys_z * RES_Z)
    
    print(f"📊 空间维度映射: Area({phys_x}x{phys_y}x{phys_z} mm) -> Grid(Y:{ny} x X:{nx} x Z:{nz})")

    # ==========================================
    # 2. 制造高真实度的“结构性斑点噪声” (Structural Speckle Noise)
    # ==========================================
    print("⏳ 正在注入晶粒散射与结构噪声...")
    np.random.seed(42)
    # 生成基础高斯白噪声并进行平滑，形成“云团状”的斑点噪声
    base_noise = np.random.normal(0, 1, (ny, nx, nz))
    speckle_noise = gaussian_filter(base_noise, sigma=(1, 3, 3)) 
    
    # 转换为 8-bit 采集卡的反向底噪 (255 是绝对安静，大部分在 220-255 波动)
    noise_intensity = np.abs(speckle_noise) / np.max(np.abs(speckle_noise)) * 35
    volume = 255.0 - noise_intensity
    
    # 加入高频电平抖动
    volume -= np.random.randint(0, 8, (ny, nx, nz))

    # ==========================================
    # 3. 铸造粗糙的物理边界 (带振铃拖尾)
    # ==========================================
    print("⏳ 正在注入物理边界特征 (表面回波 + 底波)...")
    # 表面回波：表面粗糙导致回波不齐，且带有振铃 (Ringing)
    front_wall_depths = np.random.randint(8, 12, size=(ny, nx))
    for i in range(ny):
        for j in range(nx):
            d = front_wall_depths[i, j]
            # 主反射
            volume[i, j, d:d+4] -= np.random.randint(100, 150)
            # 振铃拖尾 (Ringing artifact)
            volume[i, j, d+4:d+15] -= np.random.randint(20, 60)

    # 底波：声束经历全壁厚衰减，到达底面时较弱且模糊
    volume[:, :, nz-25:nz-15] -= np.random.randint(40, 80, size=(ny, nx, 10))

    # ==========================================
    # 4. 动态解析缺陷并植入 (带衰减与波束发散)
    # ==========================================
    print("⏳ 正在根据配置植入物理退化缺陷...")
    grid_y, grid_x, grid_z = np.mgrid[0:ny, 0:nx, 0:nz]
    
    # 衰减系数
    alpha = 0.004

    # 提取所有 strip 缺陷
    # 格式: strip cx cy cz sx sy sz roll pitch yaw
    strips = re.findall(r"strip\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)", config_text)
    
    if not strips:
        print("⚠️ 警告：配置文本中未找到任何 strip 缺陷定义！")

    for i, s in enumerate(strips):
        cx_mm, cy_mm, cz_mm, sx_mm, sy_mm, sz_mm, roll, pitch, yaw_deg = map(float, s)
        
        # 转换为像素坐标 (网格位置)
        cx, cy, cz = cx_mm * RES_X, cy_mm * RES_Y, cz_mm * RES_Z
        # 缺陷物理尺寸 -> 像素尺寸 (由于声束发散，这里做基础拉伸)
        sx, sy, sz = sx_mm * RES_X, sy_mm * RES_Y, sz_mm * RES_Z
        
        # 【物理现象：波束发散】深度越深，横向越模糊发散
        spread_x = sx + cz * 0.03
        spread_y = sy + cz * 0.03
        spread_z = sz + 5.0  # 轴向拉长效应
        
        # 旋转逻辑 (Yaw 偏转角)
        theta = np.radians(yaw_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # 坐标平移到原点并旋转
        dx = grid_x - cx
        dy = grid_y - cy
        rot_x = dx * cos_t + dy * sin_t
        rot_y = -dx * sin_t + dy * cos_t
        dz_dist = grid_z - cz

        # 高斯包络模拟声压分布
        dist = (rot_x**2 / spread_x**2) + (rot_y**2 / spread_y**2) + (dz_dist**2 / spread_z**2)
        defect_shape = np.exp(-dist)
        
        # 伴随的振铃拖尾 (声压往后延续)
        ringing_dist = (rot_x**2 / spread_x**2) + (rot_y**2 / spread_y**2) + ((dz_dist - 8)**2 / spread_z**2)
        defect_ringing = np.exp(-ringing_dist) * 0.4
        
        # 物理强度公式：初始强度 * 深度指数衰减
        intensity = 180 * np.exp(-alpha * cz)
        
        # 将缺陷和振铃注入数据体
        volume -= (defect_shape + defect_ringing) * intensity
        
        print(f"   ✅ 已植入缺陷 {i+1}: 中心({cx_mm},{cy_mm},{cz_mm}mm), 偏转 {yaw_deg}°, 信号强度衰减至 {intensity:.1f}")

    # ==========================================
    # 5. 裁剪量化与流式输出 (防爆内存)
    # ==========================================
    # 确保数值在 8-bit 合法区间
    volume = np.clip(np.round(volume), 0, 255).astype(np.int32)
    
    print(f"⏳ 正在流式写入 {output_filename} ...")
    flattened_data = volume.flatten()
    total_points = len(flattened_data)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"步进步数：{ny}，扫查步数：{nx}，声程点数：{nz}\n")
        f.write("====================\n")
        
        chunk_size = 100000 
        for i in range(0, total_points, chunk_size):
            chunk = flattened_data[i : i + chunk_size]
            chunk_str = " ".join(chunk.astype(str))
            if i > 0: f.write(" ") 
            f.write(chunk_str)
            
            # 打印进度
            percent = min(100, int((i + chunk_size) / total_points * 100))
            print(f"\r   写入进度: {percent}%", end="")
            
    print(f"\n🎉 高保真模拟数据合成完毕！输出文件: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    # 读取同目录下的 1.txt 生成 Synthetic_from_1.txt
    INPUT_CONFIG = "1.txt"
    OUTPUT_FILE = "Synthetic_from_1.txt"
    generate_ultrasound_from_txt(INPUT_CONFIG, OUTPUT_FILE)
