import numpy as np
import os
from scipy.ndimage import gaussian_filter

def generate_messy_realistic_ultrasound(ny=50, nx=300, nz=300):
    print("🚀 正在启动带有【物理劣化约束】的前向建模引擎...")
    print(f"📊 空间维度: {ny}x{nx}x{nz}")

    # ==========================================
    # 1. 制造高真实度的“结构性斑点噪声” (Structural Speckle Noise)
    # 现实中不是均匀的白噪声，而是块状的晶粒散射杂波
    # ==========================================
    print("⏳ 正在注入晶粒散射与结构噪声...")
    np.random.seed(42)
    # 生成基础高斯白噪声并进行平滑，形成“云团状”的斑点噪声
    base_noise = np.random.normal(0, 1, (ny, nx, nz))
    speckle_noise = gaussian_filter(base_noise, sigma=(1, 3, 3)) 
    
    # 转换为 8-bit 采集卡的反向底噪 (255 是绝对安静，大部分在 230-255 波动)
    noise_intensity = np.abs(speckle_noise) / np.max(np.abs(speckle_noise)) * 35
    volume = 255.0 - noise_intensity
    
    # 加入高频电平抖动
    volume -= np.random.randint(0, 8, (ny, nx, nz))

    # ==========================================
    # 2. 铸造粗糙的物理边界 (带振铃拖尾)
    # ==========================================
    # 表面回波：表面粗糙导致回波不齐，且带有极长的振铃 (Ringing)
    front_wall_depths = np.random.randint(8, 12, size=(ny, nx))
    for i in range(ny):
        for j in range(nx):
            d = front_wall_depths[i, j]
            # 主反射
            volume[i, j, d:d+4] -= np.random.randint(100, 150)
            # 振铃拖尾 (Ringing artifact)
            volume[i, j, d+4:d+15] -= np.random.randint(20, 60)

    # 底波：由于声束经历全壁厚衰减，到达底面时已经模糊不堪
    volume[:, :, nz-25:nz-15] -= np.random.randint(40, 80, size=(ny, nx, 10))

    # ==========================================
    # 3. 植入带【波束发散】和【指数衰减】的缺陷
    # ==========================================
    print("⏳ 正在植入物理退化缺陷 (发散 + 衰减)...")
    grid_Y, grid_X, grid_Z = np.mgrid[0:ny, 0:nx, 0:nz]
    
    # 衰减系数
    alpha = 0.005 

    # 【缺陷 A：浅层缺陷】 (发散较小，信号较强)
    cx_A, cy_A, cz_A = 100, 10, 80
    # 动态发散角：深度越深，横向 (X) 和纵向 (Y) 糊得越厉害
    spread_x_A = 4.0 + cz_A * 0.03
    spread_y_A = 2.0 + cz_A * 0.03
    spread_z_A = 8.0 # 脉冲宽度
    
    # 物理强度公式：初始强度 * 衰减项
    intensity_A = 180 * np.exp(-alpha * cz_A)
    shape_A = np.exp(-((grid_X-cx_A)**2/(spread_x_A**2) + (grid_Y-cy_A)**2/(spread_y_A**2) + (grid_Z-cz_A)**2/(spread_z_A**2)))
    # 加上振铃拖尾 (声学现象：遇到缺陷后会有后续震荡)
    shape_A_ringing = np.exp(-((grid_X-cx_A)**2/(spread_x_A**2) + (grid_Y-cy_A)**2/(spread_y_A**2) + (grid_Z-(cz_A+12))**2/(spread_z_A**2))) * 0.4
    
    volume -= (shape_A + shape_A_ringing) * intensity_A

    # 【缺陷 B：深层缺陷】 (发散极其严重，信号极弱，几乎淹没在噪声中！)
    cx_B, cy_B, cz_B = 220, 10, 240
    spread_x_B = 4.0 + cz_B * 0.05 # 极度发散，造成严重的 Ghosting
    spread_y_B = 2.0 + cz_B * 0.05
    spread_z_B = 12.0
    
    intensity_B = 180 * np.exp(-alpha * cz_B) # 能量被严重吸干
    shape_B = np.exp(-((grid_X-cx_B)**2/(spread_x_B**2) + (grid_Y-cy_B)**2/(spread_y_B**2) + (grid_Z-cz_B)**2/(spread_z_B**2)))
    
    volume -= shape_B * intensity_B

    # 裁剪到 8-bit 合法区间
    volume = np.clip(np.round(volume), 0, 255).astype(np.int32)

    # ==========================================
    # 4. 工业级流式写入 (防爆内存)
    # ==========================================
    output_filename = "1.6.txt"
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
            print(f"\r写入进度: {min(100, int((i + chunk_size) / total_points * 100))}%", end="")
            
    print(f"\n✅ 劣质(高逼真)物理场合成完毕！")

if __name__ == "__main__":
    generate_messy_realistic_ultrasound()