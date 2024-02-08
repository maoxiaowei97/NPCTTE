import os
import math
from random import choices
from collections import Counter
from argparse import ArgumentParser
from copy import deepcopy
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

pd.options.mode.chained_assignment = None


class TrajectoryDataset:
    def __init__(self, name, split, partial=1.0, small=False, flat=False, traffic_states = False, is_test = False, get_dataframe = False):

        self.split = split
        self.name = name
        self.partial = partial
        self.file_path = os.path.join('data', self.name)
        self.flat = flat
        self.time_interval = 10
        self.traffic_states = traffic_states
        self.is_test = is_test
        self.get_dataframe = get_dataframe
        if is_test:
            self.image_meta_path = os.path.join('/data/XinZhi/ODUQ/DOT/data/',
                                                self.name + f'_S{self.split}_F{self.flat}') + '_image_test.npz'
        else:
            self.image_meta_path = os.path.join('/data/XinZhi/ODUQ/DOT/data/',
                                                self.name + f'_S{self.split}_F{self.flat}') + '_image.npz'
        self.traj_meta_path = os.path.join('/data/maodawei/DOT/data/cache_images/',
                                           self.name + f'_S{self.split}') + '_traj.npz'
        self.traffic_state_path = os.path.join('/data/maodawei/DOT/data/cache_traffic_state/',
                                            self.name + f'_S{self.split}_F{self.flat}') + '_trafficstate.npy'
        self.fuse_image_meta_path = '/data/MaoXiaowei/DOT/data/cache_images/20181101_all_time_interval_15s_S20_FFalse_fused_image.npz'
        #在不main_data.py的情况下，可以不加载原始csv
        if small:
            self.dataframe = pd.read_hdf(os.path.join('data', name) + '_small.h5', key='raw')
            self.dataframe['daytime'] = (self.dataframe['time'] % (24 * 60 * 60)) # 天中秒
        elif self.is_test:
            file_path = '/data/XinZhi/ODUQ/DOT/data/chengdu/test.csv'
            self.dataframe = pd.read_csv(file_path).rename( columns={"col1": 'trip_id', "col2": 'driver_id', "col3": 'time', "col4": 'lat', "col5": 'lng', "col6": 'traj_id', "col7": 'timestamp'})
            time = pd.to_datetime(self.dataframe['time'])
            self.dataframe['daytime'] = time.dt.hour * 3600 + time.dt.minute * 60 + time.dt.second  # s
            self.dataframe['time'] = pd.to_datetime(self.dataframe['time']).astype(int) // 10 ** 9
            print( f"[Loaded dataset] {file_path}, number of travels {self.dataframe['traj_id'].drop_duplicates().shape[0]}")
        elif not self.get_dataframe:
            pass
        else:
            # file_path = '/data/XinZhi/ODUQ/DOT/data/chengdu/201811_d1256789_h8_10.csv'
            # self.dataframe = pd.read_csv(file_path).rename(columns = {"col1": 'trip_id', "col2": 'driver_id', "col3": 'time', "col4": 'lat', "col5": 'lng', "col6":'traj_id', "col7":'timestamp' })
            # self.dataframe = pd.read_csv(file_path)
            file_path = '/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181116.hdf'
            self.dataframe = pd.read_hdf('/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181116.hdf', 'df')
            print('data read...')

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
            # for col in ['lng', 'lat', 'time']:
            for col in ['lng', 'lat']:
                self.col_minmax[col + '_min'] = self.dataframe[col].min()
                self.col_minmax[col + '_max'] = self.dataframe[col].max()
                self.dataframe[col + '_norm'] = (self.dataframe[col] - self.dataframe[col].min()) / (self.dataframe[col].max() - self.dataframe[col].min())
                self.dataframe[col + '_norm'] = self.dataframe[col + '_norm'] * 2 - 1
            # self.dataframe['daytime_norm'] = self.dataframe['daytime'] / 60 / 60 / 24 * 2 - 1
            # self.dataframe['norm_dt_from8'] = (self.dataframe['daytime'] // 60 - 8 * 60) / 180 * 2 - 1 # 从8点开始算，不超过3小时，值域为[0, 180], 规整到[-1, 1]，一分钟对应0.01
            x_index = np.around((self.dataframe['lng'] - self.col_minmax['lng_min']) /
                                ((self.col_minmax['lng_max'] - self.col_minmax['lng_min']) / (self.split - 1)))
            y_index = np.around((self.dataframe['lat'] - self.col_minmax['lat_min']) /
                                ((self.col_minmax['lat_max'] - self.col_minmax['lat_min']) / (self.split - 1)))
            self.dataframe['cell_index'] = y_index * self.split + x_index

            train_val_split, val_test_split = int(num_trips * 0.8), int(num_trips * 0.9)
            self.split_df = [self.dataframe[self.dataframe['traj_id'].isin(select)]
                             for select in (all_trips[:train_val_split],
                                            all_trips[train_val_split:],
                                            all_trips[train_val_split:])] # 验证集和测试集相同
        self.images = {}
        self.trajs = {}
        self.num_channel = 1
        self.num_feat = 1
        self.traffic_dict = {} # ts, cell_index, travel_time
        self.fused_images = {}

    def get_images(self, df_index):
        #已经load images, 扩充下ODT特征
        lng_min, lng_max, lat_min, lat_max = self.lng_min, self.lng_max, self.lat_min, self.lat_max
        if df_index in self.images:
            images, ODTs, arrive_times = self.images[df_index]

            if 1:#只在eta用，但是eta不能用网格特征，只能用出发时间相关特征，网格特征有问题，对应的网格需要重新算
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
                t_minute = (((odt[:, 4] + 1) / 2 * 24 * 60 * 60) // 60).astype(int)  # 一天中的多少分
                ts_10 = (t_minute // 10).astype(int)  # 10分钟级时间片
                ODTs = np.concatenate([ODTs, start_cell_index.reshape(-1, 1), end_cell_index.reshape(-1, 1), t_minute.reshape(-1, 1), ts_10.reshape(-1, 1)], 1)
                # ODTs = np.concatenate([ODTs, t_minute.reshape(-1, 1), ts_10.reshape(-1, 1)], 1)
        #未load images
        else:
            selected_df = self.split_df[df_index].copy()
            time = pd.to_datetime(selected_df['time'])
            selected_df['daytime'] = time.dt.hour * 3600 + time.dt.minute * 60 + time.dt.second  # 天中秒
            selected_df['time'] = pd.to_datetime(selected_df['time']).astype(int) // 10 ** 9
            selected_df['daytime_norm'] = selected_df['daytime'] / 60 / 60 / 24 * 2 - 1
            selected_df['norm_dt_from8'] = (selected_df['daytime'] // 60 - 8 * 60) / 180 * 2 - 1  # 从8点开始算，不超过3小时，值域为[0, 180], 规整到[-1, 1]，一分钟对应0.01
            images, ODTs, arrive_times = [], [], []
            traj_index = 0
            for trip_id, group in tqdm(selected_df.groupby('traj_id'),
                                       desc="Gathering images",
                                       total=selected_df['traj_id'].drop_duplicates().shape[0]):
                if ((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60 > 60) or ((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60 < 5): # 小于5分钟，大于60分钟，过滤
                    continue
                traj_index += 1
                if traj_index < 129:
                    continue
                ODTs.append((*group.iloc[0][['lng_norm', 'lat_norm']].to_list(),
                             *group.iloc[-1][['lng_norm', 'lat_norm']].to_list(),
                             group.iloc[0]['norm_dt_from8'])) # 修改T
                arrive_times.append((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60)
                group['offset'] = (group['time'] - group['time'].iloc[0]) / ( group['time'].iloc[-1] - group['time'].iloc[0]) * 2 - 1
                ##各网格与出发位置时间差
                group['time_diff'] = ((group['daytime'] - group.iloc[0]['daytime']) / 60) / 60 * 2 - 1 #与出发时间的秒级时间差转成分钟差(0, 60), 再规整到(-1, 1)
                group['time_diff_min'] = deepcopy(group['time_diff'])
                group['time_diff_max'] = deepcopy(group['time_diff'])
                cell_group = group.groupby('cell_index').agg({'time_diff': np.mean, 'time_diff_min':np.min, 'time_diff_max':np.max}, as_index = False)
                # cell_group['cell_time_diff'] = cell_group['time_diff_max'] - cell_group['time_diff_min']
                # cell_group[cell_group.cell_time_diff > ( 1/3)] = 1/3 # 60 min -> [-1, 1], 1min -> 1/30  每格最大-最小时间不超过10min，应对多次经过同一格的情况，最后一次到时间-第一次到时间很大，实际停留时间没这么大
                # cell_group['cell_time_diff'] = (cell_group['cell_time_diff']) / 2 * 60 # 格子内停留时间转成分钟
                # cell_group['cell_time_diff'] = (cell_group['cell_time_diff'] / 10) * 2 - 1 # 停留时间转成[0, 10]min -> [-1, 1], 1分钟对应间隔为0.2  只算了停留时间，没算转移时间，即当前格第一个时间点与上一格最后一个时间点之差
                #每个格子停留时间加上从上一格到该格的转移时间
                # cell_group['cell_time_diff'] += 0.25 * 0.2 # 每格加上转移时间，约为15s
                # cell_group[cell_group.index == group['cell_index'].iloc[0]]['cell_time_diff'] -= 0.25 * 0.2
                # transition_cell_index_diff = group['cell_index'].shift(1) - group['cell_index']
                # transition_cell_group = group.loc[transition_cell_index_diff != 0]

                array_index = np.array(cell_group.index).astype(int).tolist()
                img = np.ones((3, self.split * self.split)) * -1.0  # (C, W*H)
                img[0, array_index] = 1  # Mask
                # img[1, array_index] = cell_group['time_diff_min']  # Normalized time diff (min) from the start
                img[1, array_index] = cell_group['time_diff']  # Normalized time diff (mean) from the start
                img[2, array_index] = cell_group['time_diff_max']  # Normalized time diff (max) from the start
                # img[4, array_index] = cell_group['cell_time_diff']  # Normalized time diff (max) from the start
                img = img.reshape(img.shape[0], self.split, self.split)  # (C, W, H)
                #从起点到终点的分钟级时间差
                images.append(img)

            images, ODTs, arrive_times = np.stack(images, 0), np.array(ODTs), np.array(arrive_times)
            self.images[df_index] = (images, ODTs, arrive_times)
            self.num_channel = images.shape[1]

        return (images, ODTs, arrive_times)

    def dump_images_state(self):
        #先获取images
        for i in range(1, 2):
            self.get_images(i)

        images = {}
        for i, label in enumerate(['train', 'val', 'test']):
            images.update({f'{label}_image': self.images[i][0],
                           f'{label}_odt': self.images[i][1],
                           f'{label}_arr': self.images[i][2]})
        np.savez(self.image_meta_path, **images)
        print(f'[Dumped meta] to {self.image_meta_path}')

    def load_images(self):
        #load images
        # if self.is_test:
        #     self.image_meta_path = '/data/XinZhi/ODUQ/DOT/data/20181101_20181110_all_15s_S20_FFalse_image_test.npz'

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

