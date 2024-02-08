import pandas as pd
from datetime import datetime
#加交通状态生成时间差，交通状态标准化到-1， 1，计算出发时刻前30分钟网格内轨迹的车速，先计算网格内距离，再计算时间
# start1 = datetime.now()
# df = pd.read_csv('/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181116_16d.csv', names=['trip_id', 'driver_id', 'time', 'lat', 'lng', 'traj_id', 'timestamp', 'hh', 'dd'])[:200]
df = pd.read_csv('/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181116.csv', names=['trip_id', 'driver_id', 'time', 'lat', 'lng', 'traj_id', 'timestamp', 'hh', 'dd'])
df = df.iloc[1:]

df.to_hdf('/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181116.hdf', 'df')

# start2 = datetime.now()
# df = pd.read_hdf('/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181116.hdf', 'df')
# end2 = datetime.now()
# print('read_csv: ', end1 - start1)
# print('read_hdf: ', end2 - start2)