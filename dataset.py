import os
from collections import Counter
from copy import deepcopy
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

cur_path = os.path.abspath(__file__)
ws = os.path.dirname(cur_path)
pd.options.mode.chained_assignment = None

class TrajectoryDataset:
    def __init__(self, name, split, partial=1.0,  flat=False,  get_dataframe = False):

        self.split = split
        self.name = name
        self.partial = partial
        self.file_path = os.path.join('data', self.name)
        self.flat = flat
        self.get_dataframe = get_dataframe

        self.image_meta_path = os.path.join(ws + '/data/', self.name + f'_S{self.split}_F{self.flat}') + '_image.npz'
        self.traffic_condition_path = ws + '/data/' + self.name + f'_S{self.split}_F{self.flat}_trafficcondition.npy'

        if not self.get_dataframe:
            pass
        else:
            file_path = ws + '/data/sample.hdf'
            self.dataframe = pd.read_hdf(file_path, 'df')
            print(f"[Loaded dataset] {file_path}, number of travels {self.dataframe['traj_id'].drop_duplicates().shape[0]}")
            trip_len_counter = Counter(self.dataframe['traj_id'])
            self.max_len = max(trip_len_counter.values())
            num_trips = len(trip_len_counter)
            all_trips = list(trip_len_counter.keys())
            self.lng_min = self.dataframe['lng'].min()
            self.lng_max = self.dataframe['lng'].max()
            self.lat_min = self.dataframe['lat'].min()
            self.lat_max = self.dataframe['lat'].max()
            self.col_minmax = {}
            for col in ['lng', 'lat']:
                self.col_minmax[col + '_min'] = self.dataframe[col].min()
                self.col_minmax[col + '_max'] = self.dataframe[col].max()
                self.dataframe[col + '_norm'] = (self.dataframe[col] - self.dataframe[col].min()) / (self.dataframe[col].max() - self.dataframe[col].min())
                self.dataframe[col + '_norm'] = self.dataframe[col + '_norm'] * 2 - 1

            x_index = np.around((self.dataframe['lng'] - self.col_minmax['lng_min']) /
                                ((self.col_minmax['lng_max'] - self.col_minmax['lng_min']) / (self.split - 1)))
            y_index = np.around((self.dataframe['lat'] - self.col_minmax['lat_min']) /
                                ((self.col_minmax['lat_max'] - self.col_minmax['lat_min']) / (self.split - 1)))
            self.dataframe['cell_index'] = y_index * self.split + x_index

            train_val_split, val_test_split = int(num_trips * 0.8), int(num_trips * 0.9)
            self.split_df = [self.dataframe[self.dataframe['traj_id'].isin(select)]
                             for select in (all_trips[:train_val_split],
                                            all_trips[train_val_split:val_test_split],
                                            all_trips[val_test_split:])]
        self.images = {}
        self.trajs = {}
        self.fused_images = {}
        self.odt_dict = {}
        self.num_channel = 3
        self.time_interval = 10
        self.traffic_dict = {}
        self.Days = len(set(pd.to_datetime(self.dataframe['time']).dt.day))
        self.start_date = int((pd.to_datetime(self.dataframe['time']).dt.day).unique()[0])

    def get_images(self, df_index):
        lng_min, lng_max, lat_min, lat_max = self.lng_min, self.lng_max, self.lat_min, self.lat_max
        if df_index in self.images:
            images, ODTs, arrive_times = self.images[df_index]
            if 1:
                odt = deepcopy(ODTs)
                start_x = np.around((((((odt[:, 0] + 1) / 2) * (lng_max - lng_min) + lng_min) - lng_min) / (
                            (lng_max - lng_min) / (20 - 1))))
                start_y = np.around((((((odt[:, 1] + 1) / 2) * (lat_max - lat_min) + lat_min) - lat_min) / (
                            (lat_max - lat_min) / (20 - 1))))
                end_x = np.around((((((odt[:, 2] + 1) / 2) * (lng_max - lng_min) + lng_min) - lng_min) / (
                            (lng_max - lng_min) / (20 - 1))))
                end_y = np.around((((((odt[:, 3] + 1) / 2) * (lat_max - lat_min) + lat_min) - lat_min) / (
                            (lat_max - lat_min) / (20 - 1))))
                start_cell_index = start_y * 20 + start_x
                end_cell_index = end_y * 20 + end_x
                t_minute = (((odt[:, 4] + 1) / 2 * 24 * 60 * 60) // 60).astype(int)
                ts_10 = (t_minute // 10).astype(int)
                day_of_week = (odt[:, -2] + 3) % 7
                ODTs = np.concatenate([ODTs, day_of_week.reshape(-1, 1), start_cell_index.reshape(-1, 1), end_cell_index.reshape(-1, 1), t_minute.reshape(-1, 1), ts_10.reshape(-1, 1)], 1)
        else:
            selected_df = self.split_df[df_index].copy()
            time = pd.to_datetime(selected_df['time'])
            selected_df['date'] = time.dt.day
            selected_df['daytime'] = time.dt.hour * 3600 + time.dt.minute * 60 + time.dt.second
            selected_df['time'] = pd.to_datetime(selected_df['time']).astype(int) // 10 ** 9
            selected_df['daytime_norm'] = selected_df['daytime'] / 60 / 60 / 24 * 2 - 1
            images, ODTs, arrive_times = [], [], []
            for trip_id, group in tqdm(selected_df.groupby('traj_id'),
                                       desc="Gathering images",
                                       total=selected_df['traj_id'].drop_duplicates().shape[0]):
                if ((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60 > 60) or ((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60 < 5):
                    continue
                group['offset'] = (group['time'] - group['time'].iloc[0]) / ( group['time'].iloc[-1] - group['time'].iloc[0]) * 2 - 1
                ##各网格与出发位置时间差
                group['time_diff'] = ((group['daytime'] - group.iloc[0]['daytime']) / 60) / 60 * 2 - 1
                group['time_diff_min'] = deepcopy(group['time_diff'])
                group['time_diff_max'] = deepcopy(group['time_diff'])
                cell_group = group.groupby('cell_index').agg({'time_diff': np.mean, 'time_diff_min': np.min, 'time_diff_max':np.max}, as_index = False) # 网格最早访问时间，网格最晚访问时间
                cell_group['cell_index'] = cell_group.index

                #  make traffic condition
                traffic_group = deepcopy(group)
                traffic_group['ts'] = traffic_group[ 'daytime'] / 60 // self.time_interval  # self.time_interval分钟级时间片, 交通状况由10分钟级时间片构造得到
                traffic_group['daytime_diff'] = traffic_group['daytime'].diff()  # 每个轨迹点与上一轨迹点的时间差
                traffic_group['daytime_diff'].iloc[0] = 0
                traffic_group['sub_traj_id'] = (traffic_group['cell_index'].diff() != 0).cumsum()
                cell_daytimenorm_diff_ts = traffic_group.groupby('sub_traj_id').agg({'daytime_diff': np.sum, 'ts': np.max})  # 该小段轨迹计入的时间片，小段轨迹通行时间间隔
                cell_index_diff = group['cell_index'].shift(1) - group['cell_index']
                traffic_cell_group = traffic_group.loc[cell_index_diff != 0]
                traffic_cell_group['daytime_diff_min'] = np.array(
                    cell_daytimenorm_diff_ts['daytime_diff']) / 60  # 从上个格子到这个格子，加上子轨迹段在这个格子中的总停留时间, 秒转换为分钟
                traffic_cell_group['ts'] = np.array(cell_daytimenorm_diff_ts['ts'])  # 每个cell的通行时间片
                traffic_cell_group = traffic_cell_group.iloc[1: -1]  # 不考虑出发网格和到达网格的通行时间对交通状况的影响
                traffic_cell_group = traffic_cell_group.iloc[::-1]  #来回绕的网格是当作两段子轨迹
                traffic_cell_travel_time = traffic_cell_group[['ts', 'date', 'cell_index', 'daytime_diff_min']]  # 可能有来回绕的情况，分成两个cell和对应的通行时间
                for i in range(len(traffic_cell_travel_time)):
                    ts = int(traffic_cell_group['ts'].iloc[i])
                    day = int(traffic_cell_group['date'].iloc[i]) - self.start_date # start from 0
                    traffic_cell_index = int(traffic_cell_group['cell_index'].iloc[i])
                    traffic_daytime_diff = traffic_cell_group['daytime_diff_min'].iloc[i]
                    if traffic_daytime_diff < 0:
                        continue
                    if (day, ts) not in self.traffic_dict.keys():
                        self.traffic_dict[(day, ts)] = {}
                        self.traffic_dict[(day, ts)][traffic_cell_index] = []
                        self.traffic_dict[(day, ts)][traffic_cell_index].append(np.round(traffic_daytime_diff, 4))
                    else:
                        if traffic_cell_index not in self.traffic_dict[(day, ts)].keys():
                            self.traffic_dict[(day, ts)][traffic_cell_index] = []
                            self.traffic_dict[(day, ts)][traffic_cell_index].append(np.round(traffic_daytime_diff, 4))
                        else:
                            self.traffic_dict[(day, ts)][traffic_cell_index].append(np.round(traffic_daytime_diff, 4))

                array_index = np.array(cell_group.index).astype(int).tolist()
                img = np.ones((3, self.split * self.split)) * -1.0  # (C, W*H)
                img[0, array_index] = 1  # Mask
                img[1, array_index] = cell_group['time_diff_min']
                img[2, array_index] = cell_group['time_diff_max']
                img = img.reshape(img.shape[0], self.split, self.split)  # (C, W, H)
                images.append(img)
                ODTs.append((*group.iloc[0][['lng_norm', 'lat_norm']].to_list(),
                             *group.iloc[-1][['lng_norm', 'lat_norm']].to_list(),
                             group.iloc[0]['daytime_norm'],
                             int(group.iloc[0]['date']) - self.start_date,
                             int(group.iloc[0]['daytime'] / 60 // self.time_interval)))
                arrive_times.append((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60)
            images, ODTs, arrive_times = np.stack(images, 0), np.array(ODTs), np.array(arrive_times)
            self.images[df_index] = (images, ODTs, arrive_times)
            self.num_channel = images.shape[1]
        return (images, ODTs, arrive_times)

    def dump_images_state(self):
        for i in range(3):
            self.get_images(i)

        images = {}
        for i, label in enumerate(['train', 'val', 'test']):
            images.update({f'{label}_image': self.images[i][0],
                           f'{label}_odt': self.images[i][1],
                           f'{label}_arr': self.images[i][2]})
        np.savez(self.image_meta_path, **images)
        print(f'[Dumped meta] to {self.image_meta_path}')

        ##make traffic condition array
        traffic_condition_array = np.zeros([self.Days, 24 * 60 // self.time_interval, self.split * self.split])
        for key in self.traffic_dict.keys():
            day = key[0]
            ts = key[1]
            traffic_condition = self.traffic_dict[key]
            for grid in traffic_condition.keys():
                traffic_condition_array[day, ts, grid] = np.mean(traffic_condition[grid])
        traffic_condition_array = (traffic_condition_array / traffic_condition_array.max()) * 2 - 1
        np.save(self.traffic_condition_path, traffic_condition_array)
        print('traffic condition generated...')

    def load_images(self):
        if os.path.exists(self.image_meta_path):
            self.dataframe = 0
            loaded_images = np.load(self.image_meta_path)
            for i, label in enumerate(['train', 'val', 'test']):
                self.images[i] = (loaded_images[f'{label}_image'], loaded_images[f'{label}_odt'],
                                  loaded_images[f'{label}_arr'])
            self.num_channel = self.images[0][0].shape[1]
            print(f'[Loaded meta] from {self.image_meta_path}')
        else:
            for i in range(3):
                self.get_images(i)
                self.dataframe = 0
