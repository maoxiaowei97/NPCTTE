import pandas as pd
import numpy as np
from tqdm import tqdm
file_path = '/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181110.csv'
df = pd.read_csv(file_path).rename(columns={"col1": 'trip_id', "col2": 'driver_id', "col3": 'time', "col4": 'lat', "col5": 'lng', "col6": 'traj_id',  "col7": 'timestamp'})
print('length of df: ', len(df))
time = pd.to_datetime(df['time'])
df['hour'] = time.dt.hour
df['day'] = time.dt.day
select_day = [1, 2, 5, 6, 7, 8, 9] # 3 / 5
select_hour = [8, 9, 10] # 1 / 8
df = df[df['hour'].isin(select_hour)]
df = df[df['day'].isin(select_day)]
print('selected length of df: ', len(df))
df.to_csv('/data/XinZhi/ODUQ/DOT/data/chengdu/201811_d1256789_h8_10.csv', index=False)
print('saved...')

