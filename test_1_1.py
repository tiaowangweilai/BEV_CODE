# =====================================================================
# 脚本名称 : test_1_1.py
# 修改日期 : 2026-05-07
# 作    者 : Antigravity & USER
# 脚本作用 : 临时测试脚本。用于部分算法原型的验证和文件IO测试。
# =====================================================================

import numpy as np
import io

def analyze():
    content = ""
    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
        try:
            with open('1.1.txt', 'r', encoding=enc) as f:
                content = f.read()
            break
        except Exception:
            pass
            
    # Assuming first 2 lines are headers
    lines = content.strip().split('\n')[2:]
    data = []
    for l in lines:
        for x in l.split():
            try:
                data.append(float(x))
            except:
                pass
                
    arr = np.array(data)
    print("Min:", np.min(arr))
    print("Max:", np.max(arr))
    print("Mean:", np.mean(arr))
    print("Median:", np.median(arr))

if __name__ == '__main__':
    analyze()
