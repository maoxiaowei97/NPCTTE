import numpy as np
import pandas as pd
# 假设存在重复的索引位置
array_index = [1, 2, 3, 2, 4, 4]

# 创建一个空的 img 数组
img = np.zeros((3, 6))

# 模拟 cell_group 的数据
cell_group = pd.DataFrame({
    'cell_index': [1, 2, 3, 2, 4, 5],
    'daytime_norm': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'offset': [1, 2, 3, 4, 5, 6]
})
#array_index中，1对应的daytime_norm是0.1，2对应的是0.4, 3对应的是0.3，4对应的是0.6
#array_index和cell_group['daytime_norm']的长度相同才行，最开始按顺序一一对应，由于array_index有重复值，则对应的元素都变成最后一个顺序位对应的元素

# 按照 array_index 给 img 的第二维通道赋值
img[1, array_index] = cell_group['daytime_norm']

print(img)