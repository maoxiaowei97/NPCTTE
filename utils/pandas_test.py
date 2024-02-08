import pandas as pd
img = pd.read_csv('/data/maodawei/DOT/data/chengdu/20181101/Modified_ALL_Traj_75.csv', names=['trip_id', 'driver_id', 'time', 'lat', 'lng', 'traj_id', 'timestamp']) # 250w
c = img
# 创建示例 DataFrame
data = {'Value': [0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2]}
df = pd.DataFrame(data)

# 使用 diff 方法标记连续变化的值
df['Group'] = (df['Value'].diff() != 0).cumsum()

# 对每个连续变化的值进行平均处理
# result = df.groupby('Group')['Value'].mean()

# 打印结果
# print(result)