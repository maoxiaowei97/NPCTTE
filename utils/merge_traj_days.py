import pandas as pd
import numpy as np
# df = pd.read_csv('/data/maodawei/DOT/data/chengdu/20181110/Modified_ALL_Traj.csv', names=['trip_id', 'driver_id', 'time', 'lat', 'lng', 'traj_id', 'timestamp'])[:30000]
# pdf = pd.read_csv('/data/maodawei/DOT/data/chengdu/20181101_20181110_test.csv')
# time = pd.to_datetime(df['time'])
# df['daytime'] = time.dt.hour * 3600 + time.dt.minute * 60 + time.dt.second  # s
# df['time'] = pd.to_datetime(df['time']).astype(int) // 10 ** 9

file_path = '/data/maodawei/DOT/data/chengdu/'
file_names = ['20181101', '20181102', '20181103', '20181104', '20181105',
              '20181106', '20181107', '20181108', '20181109', '20181110']
all_df = pd.DataFrame()
for file_name in file_names:
    file = file_path + file_name + '/Modified_ALL_Traj.csv'
    df = pd.read_csv(file, names=['trip_id', 'driver_id', 'time', 'lat', 'lng', 'traj_id', 'timestamp'])
    print('length: ', len(df))
    all_df = all_df.append(df, ignore_index=True)
print('total length: ',len(all_df))
all_df.to_csv('/data/maodawei/DOT/data/chengdu/20181101_20181110.csv', index=False)
print('saved')