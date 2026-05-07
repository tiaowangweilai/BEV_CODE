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
