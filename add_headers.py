import os
import datetime

# 定义文件及其作用
file_purposes = {
    'tradition3d.py': '核心超声缺陷检测引擎。对真实的超声原始采集数据进行自动极性判定、形态学闭运算和3D连通域缺陷量化分析。',
    'tradition_xb.py': '多块数据拼接引擎。负责将多个不同位置的原始超声扫描数据块进行平移、旋转、缩放插值与无缝拼接。',
    'normalize_data.py': '大文件处理与预处理工具。利用分块流式读写处理超大规模点云数据，防止内存溢出，并进行归一化。',
    'txt_trans.py': '数据编码与格式转换工具。负责将超声数据的 UTF/GBK 编码处理及3D数据的坐标循环位移操作。',
    'data_get_ultra.py': '高逼真物理前向建模引擎。带有散斑噪声、指数衰减、波束扩散和振铃拖尾等声学劣化效应的合成数据生成器。',
    'generate_synthetic_raw.py': '综合物理合成系统。读取独立缺陷参数配置文件(1.txt)，逆向融合物理退化规律，流式生成逼真的超声原始点阵矩阵。',
    'import numpy as np.py': '早期的参数化数据生成器。使用硬编码参数生成超声模拟原始数据(Synthetic_2026_Raw.txt)。',
    'bev_ultrasound.py': 'PhysBEV核心算法原型。展示声学物理原理指导的三维重构，包含深度增益补偿(TGC)和多模态特征融合。',
    'pinclie.py': '仿真可视化引擎 (早期版本)。用于基础的三维点云包围盒绘制与渲染。',
    'pinclie_dibo.py': '仿真可视化引擎 (早期版本)。加入了底波和表面波的物理效应模拟。',
    'pinclie_biaomian.py': '仿真可视化引擎 (V1.0)。支持自动生成基础缺陷配置，并初步融合表面图像纹理。',
    'pinclie_biaomain_2.0.py': '可视化引擎 (V2.0)。能够直接读取大规模原始采集数据，进行数据切片与三维交互展示。',
    'pinclie_biaomain_qiepain_3.0.py': '可视化引擎 (V3.0 Pro)。成熟的物理场仿真仪表盘，包含Z轴切片滑块交互、3D高亮追踪与自动缺陷量化报告(HTML导出)。',
    'pinclie_levelset_4.0.py': '高精度检测与物理反演系统 (V4.0)。集成水平集(Level Set)PDE演化算法实现高精度拓扑提取，并将渲染的物理场导出为超声原始数据。',
    'pinclie_levelset_4.1.py': '水平集拓扑演化量化系统 (V4.1)。V4.0的分支版本，进一步优化了缺陷体积计算与Marching Cubes高保真重构。',
    'pinclie_physical.py': '带有复杂声学物理模型的可视化渲染引擎。引入了波束发散拖尾、深度衰减惩罚等物理约束。',
    'pinclie_traditional.py': '传统的形态学高亮显示引擎。基于简单阈值和连通域算法的可视化展示。',
    'test_1_1.py': '临时测试脚本。用于部分算法原型的验证和文件IO测试。'
}

date_str = datetime.datetime.now().strftime('%Y-%m-%d')
author = 'Antigravity & USER'

for filename, purpose in file_purposes.items():
    if not os.path.exists(filename):
        continue
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有了头部注释
    if '# =====================================================================' in content[:200] and '脚本名称' in content[:200]:
        print(f'跳过: {filename} (已有头部)')
        continue
        
    header = f"""# =====================================================================
# 脚本名称 : {filename}
# 修改日期 : {date_str}
# 作    者 : {author}
# 脚本作用 : {purpose}
# =====================================================================
"""
    
    new_content = header + '\n' + content
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f'✅ 已成功添加头部注释: {filename}')
