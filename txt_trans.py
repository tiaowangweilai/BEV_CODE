# =====================================================================
# 脚本名称 : txt_trans.py
# 修改日期 : 2026-05-07
# 作    者 : Antigravity & USER
# 脚本作用 : 数据编码与格式转换工具。负责将超声数据的 UTF/GBK 编码处理及3D数据的坐标循环位移操作。
# =====================================================================

import numpy as np
import os
import re

# ==========================================
# ⚙️ 配置区
# ==========================================
INPUT_FILE = "2026.4.21.txt"            # 你的原始文件
OUTPUT_FILE = "2026.4.21_Centered.txt"  # 生成的中心化新文件

def shift_defect_to_center(input_path, output_path):
    print(f"📂 正在读取原始数据: {input_path} ...")
    
    if not os.path.exists(input_path):
        print(f"❌ 找不到文件: {input_path}，请检查路径！")
        return

    # 1. 无损读取
    content = ""
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
        try:
            with open(input_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError: continue

    parts = re.split(r'={5,}', content)
    if len(parts) < 2:
        print("❌ 文件格式异常，找不到分割线 (=====)！")
        return
        
    header_str = parts[0].strip()
    data_str = parts[1]

    # 2. 解析维度
    nums = re.findall(r'\d+', header_str)
    if len(nums) >= 3:
        ny, nx, nz = int(nums[0]), int(nums[1]), int(nums[2])
        print(f"✅ 成功读取物理维度: Y={ny}, X={nx}, Z={nz}")
    else: 
        print("❌ 表头缺失维度信息！")
        return

    # 3. 提取 100% 原始数值 (🌟 完美修复连字符 bug 🌟)
    print("⏳ 正在提取底层矩阵...")
    # lstrip('-') 会去掉左边的负号，然后再判断剩下的是不是纯数字，彻底防住 '-----------'
    data_list = [float(item) for item in data_str.strip().split() if item.lstrip('-').replace('.','',1).isdigit()]
    vol_raw_1d = np.array(data_list, dtype=np.float32)
    
    total_expected = nx * ny * nz
    if len(vol_raw_1d) != total_expected: 
        print(f"❌ 数据点数不匹配！预期 {total_expected} 个，实际提取到 {len(vol_raw_1d)} 个。")
        return

    # 4. 重新折叠为 3D 物理空间 (Y, X, Z)
    vol_3d = vol_raw_1d.reshape((ny, nx, nz))

    # ==========================================
    # 🪄 核心魔法：乾坤大挪移 (Circular Shift)
    # ==========================================
    shift_y = ny // 2
    shift_x = nx // 2
    
    print(f"🔄 正在启动空间循环平移...")
    print(f"   - X 轴滚动距离: {shift_x}")
    print(f"   - Y 轴滚动距离: {shift_y}")
    print(f"   - Z 轴锁定不动 (保护表面波物理意义)")

    vol_shifted = np.roll(vol_3d, shift=(shift_y, shift_x), axis=(0, 1))
    flat_shifted = vol_shifted.flatten()

    # ==========================================
    # 🛡️ 分块流式写入 (杜绝 MemoryError)
    # ==========================================
    print("⏳ 正在安全分块写入新文件 (零内存负担，请稍候)...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header_str + "\n")
        f.write("==============================================================\n")
        
        chunk_size = 100000
        for i in range(0, len(flat_shifted), chunk_size):
            chunk = flat_shifted[i:i+chunk_size]
            f.write('\t'.join([f"{x:g}" for x in chunk]) + '\t')

    print(f"\n✅ 大功告成！缺陷已被完美转移至中心！")
    print(f"   新文件已保存为: {os.path.abspath(output_path)}")
    print(f"💡 接下来，你可以将量化代码里的 FILE_NAME 改为 '{OUTPUT_FILE}' 继续进行检测！")

if __name__ == "__main__":
    shift_defect_to_center(INPUT_FILE, OUTPUT_FILE)