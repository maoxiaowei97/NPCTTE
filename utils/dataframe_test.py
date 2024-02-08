import pandas as pd
import numpy as np
from tqdm import tqdm
file_path = '/data/XinZhi/ODUQ/DOT/utils/201811_d1289_h8_14.csv'
df = pd.read_csv(file_path).rename(columns={"col1": 'trip_id', "col2": 'driver_id', "col3": 'time', "col4": 'lat', "col5": 'lng', "col6": 'traj_id',  "col7": 'timestamp'})
print('length of df: ', len(df))
time = pd.to_datetime(df['time'])
df['hour'] = time.dt.hour
df['day'] = time.dt.day
select_day = [1, 8] # 1 / 5
select_hour = [8, 9, 10, 11, 12, 13] # 1/ 4
df = df[df['hour'].isin(select_hour)]
df = df[df['day'].isin(select_day)]
print('selected length of df: ', len(df))
df.to_csv('/data/XinZhi/ODUQ/DOT/data/chengdu/201811_d18_h8_13.csv', index=False)

