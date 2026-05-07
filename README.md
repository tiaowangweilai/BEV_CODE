文件说明：
基于绘画txt文件
1.pinclie_biaomain_qiepain_3.0.py 用于切片和缺陷框出 1.txt（绘制txt文件） 是比较好的一个使用文件
2.pinclie_biaomain_2.0.py 是不带切片的一个版本
注：其他pinclie版本的文件，主要是按照txt文件进行缺陷映射
pinclie_biaomian.py 表面两张图和缺陷映射（基于绘制txt文件）
pinclie_trans.py 用于根据绘画txt文件，生成原始超声数据文件。

3.data_get_ultra.py 用于模拟生成获取超声数据

基于传统超声格式数据
4.tradition3d.py 用于传统3D重建与目标框选，包括切片
5.tradition_xb.py 用于多个txt的整体三维拼接（按照扫查顺序拼接）
6.tradition3d_BEV.py 用于传统3D重建与目标框选，包括切片，但是加入BEV的介质衰减等思想
7.tradition3d_BEV2.py 用于传统3D重建与目标框选，包括切片，但是加入BEV的介质衰减等思想

8.txt_trans.py 用于将txt文件转换为npy文件，方便后续处理

9.txt_trans.py主要是用于移动缺陷的位置，方便后续处理

10.normalize_data.py 用于将数据归一化，方便后续处理（用于颜色特别接近的数据）

11.bev_ultrasound.py 融入BEV的介质衰减等思想进行三维重构
