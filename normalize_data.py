# =====================================================================
# 脚本名称 : normalize_data.py
# 修改日期 : 2026-05-07
# 作    者 : Antigravity & USER
# 脚本作用 : 大文件处理与预处理工具。利用分块流式读写处理超大规模点云数据，防止内存溢出，并进行归一化。
# =====================================================================

import numpy as np
import os
import re

def normalize_ultrasound_data(input_filepath, output_filepath):
    print(f"\n⏳ 正在读取文件: {input_filepath} ...")
    if not os.path.exists(input_filepath):
        print(f"❌ 找不到文件: {input_filepath}")
        return

    # 1. 兼容多种编码读取 (由于工程中txt文件通常含有中文字符表头)
    content = ""
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
        try:
            with open(input_filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
            
    if not content:
        print("❌ 文件读取失败或文件为空！")
        return

    # 2. 正则分离表头和数据体
    parts = re.split(r'={5,}', content)
    if len(parts) < 2:
        print("❌ 文件格式不符合预期（未找到分隔符 '======'）")
        return
        
    header_str = parts[0]
    data_str = parts[1]

    # 3. 提取有效数据
    print("⏳ 正在解析数值...")
    data_list = []
    # 兼容工程中的原始读取方法，过滤换行和空格
    cleaned_items = data_str.split()
    for item in cleaned_items:
        try:
            data_list.append(float(item))
        except ValueError:
            continue
            
    if not data_list:
        print("❌ 未在文件中解析到有效的数值数据！")
        return

    # 转换为 numpy 数组进行快速向量化运算
    data_array = np.array(data_list, dtype=np.float32)
    
    # 4. 找到最大值和最小值
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    print(f"📊 原始数据统计: 最小值 = {min_val:.2f}, 最大值 = {max_val:.2f}")

    # 5. 线性拉伸到 0 - 255
    print("⏳ 正在进行 0-255 范围重构归一化...")
    if max_val > min_val:
        normalized_array = (data_array - min_val) / (max_val - min_val) * 255.0
    else:
        # 如果所有值都一样，防除零报错
        normalized_array = np.zeros_like(data_array)

    # 工业标准：一般 0-255 数据保存为整数
    normalized_array = np.clip(np.round(normalized_array), 0, 255).astype(np.int32)
    
    # 6. 流式写入新文件（防止大文件拼接字符串导致内存爆浆卡死）
    print(f"⏳ 正在写入重建后的文件至: {output_filepath} ...")
    with open(output_filepath, "w", encoding="utf-8") as f:
        # 原封不动写入原始表头
        if not header_str.endswith('\n'):
            header_str += '\n'
        f.write(header_str)
        f.write("====================\n")
        
        # 分块流式写入
        total_points = len(normalized_array)
        chunk_size = 100000 
        
        for i in range(0, total_points, chunk_size):
            chunk = normalized_array[i : i + chunk_size]
            chunk_str = " ".join(chunk.astype(str))
            
            if i > 0:
                f.write(" ") # 块与块之间保持空格
            f.write(chunk_str)
            
            # 打印进度
            progress = min(100, int((i + chunk_size) / total_points * 100))
            print(f"\r写入进度: {progress}%", end="")
            
    print(f"\n✅ 数据拉伸重构完成！保存为: {os.path.basename(output_filepath)}")

if __name__ == "__main__":
    # 配置你的工作目录
    base_dir = r"e:\data\bev_code"
    
    # 你可以手动指定单个文件
    # input_file = os.path.join(base_dir, "1.1.txt")
    # output_file = os.path.join(base_dir, "1.1_normalized.txt")
    # normalize_ultrasound_data(input_file, output_file)
    
    # 也可以一键批量处理所有相关 txt 拼接文件
    files_to_process = ["2026.4.21.txt"]
    
    for filename in files_to_process:
        in_path = os.path.join(base_dir, filename)
        # 生成带 _normalized 后缀的新文件，避免覆盖原始数据
        out_path = os.path.join(base_dir, filename.replace(".txt", "_normalized.txt"))
        
        if os.path.exists(in_path):
            normalize_ultrasound_data(in_path, out_path)
        else:
            print(f"\n⚠️ 跳过 {filename}: 文件不存在")
