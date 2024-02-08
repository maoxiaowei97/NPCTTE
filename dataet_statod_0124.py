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
        self.time_interval = 10 #交通量的统计时间粒度
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
        # self.traffic_state_path = os.path.join('/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/',
        #                                     self.name + f'_S{self.split}_F{self.flat}') + '_trafficstate.npy' # _filtered表示用最大长度和最大时间进行了过滤
        # self.odt_path = os.path.join('/data/XinZhi/ODUQ/DOT/data/cache_odt/',
        #                                     self.name + f'_S{self.split}_F{self.flat}') + '_odt.npy' # _filtered表示用最大长度和最大时间进行了过滤
        self.traffic_state_path = os.path.join('/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/',
                                               self.name + f'_S{self.split}_F{self.flat}') + '_trafficstate_filtered.npy'  # _filtered表示用最大长度和最大时间进行了过滤
        self.odt_path = os.path.join('/data/XinZhi/ODUQ/DOT/data/cache_odt/',
                                     self.name + f'_S{self.split}_F{self.flat}') + '_odt_filtered.npy'  # _filtered表示用最大长度和最大时间进行了过滤，重新构造了一次
        self.fuse_image_meta_path = '/data/MaoXiaowei/DOT/data/cache_images/20181101_all_time_interval_15s_S20_FFalse_fused_image.npz'
        # self.odt_dict_path = '/data/XinZhi/ODUQ/DOT/data/cache_od_dict/1101_1116_15s_0117_3channel_S20_od_dict.npy'
        self.od_dict_path = '/data/XinZhi/ODUQ/DOT/data/cache_od_dict/1101_1116_15s_0125_3channel_S20_xian_od_dict_without_val_test.npy'
        self.od_dict = np.load(self.od_dict_path, allow_pickle=True).item()
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
            # file_path = '/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181116.hdf'
            file_path = '/data/XinZhi/ODUQ/DOT/data/xian/20181101_20181116_16d.hdf'
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
                                            all_trips[val_test_split:])]  # 验证集和测试集相同
            # self.split_df = [self.dataframe[self.dataframe['traj_id'].isin(select)]
            #                  for select in (all_trips[:train_val_split],
            #                                 all_trips[val_test_split:],
            #                                 all_trips[train_val_split:val_test_split])] # 验证集和测试集相同
        self.images = {}
        self.trajs = {}
        self.num_channel = 1
        self.num_feat = 1
        self.traffic_dict = {}
        self.fused_images = {}
        self.odt_dict = {}

    def get_images(self, df_index):
        #已经load images, 扩充下ODT特征
        lng_min, lng_max, lat_min, lat_max = self.lng_min, self.lng_max, self.lat_min, self.lat_max
        if df_index in self.images:
            images, ODTs, arrive_times = self.images[df_index]

            if 1:#只在eta用，但是eta不能用网格特征，只能用出发时间相关特征，网格特征有问题，对应的网格需要重新算 load dataset(images loaded), get_images
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
                day_of_week = (odt[:, -2] + 3) % 7
                # ODTs = np.concatenate([ODTs[:, [0, 1, 2, 3, 4]],  day_of_week.reshape(-1, 1), ts_10.reshape(-1, 1)], 1) # o, d, norm_t_min, day_of_week, ts_10
                ODTs = np.concatenate([ODTs, day_of_week.reshape(-1, 1), start_cell_index.reshape(-1, 1), end_cell_index.reshape(-1, 1), t_minute.reshape(-1, 1), ts_10.reshape(-1, 1)], 1) # diffusion
        #未load images
        else: # make dataset
            selected_df = self.split_df[df_index].copy()
            time = pd.to_datetime(selected_df['time'])
            selected_df['date'] = time.dt.day
            selected_df['daytime'] = time.dt.hour * 3600 + time.dt.minute * 60 + time.dt.second  # 天中秒, 8:30对应的天中秒为30600
            selected_df['time'] = pd.to_datetime(selected_df['time']).astype(int) // 10 ** 9
            selected_df['daytime_norm'] = selected_df['daytime'] / 60 / 60 / 24 * 2 - 1
            selected_df['norm_dt_from8'] = (selected_df['daytime'] // 60 - 8 * 60) / 360 * 2 - 1  # 从8点到14点，不超过6小时，值域为[0, 360]min, 规整到[-1, 1]，一分钟对应1/180
            images, ODTs, arrive_times = [], [], []
            for trip_id, group in tqdm(selected_df.groupby('traj_id'),
                                       desc="Gathering images",
                                       total=selected_df['traj_id'].drop_duplicates().shape[0]):
                if ((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60 > 35) or ((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60 < 5): # 小于5分钟，大于40分钟，过滤
                    continue
                group['offset'] = (group['time'] - group['time'].iloc[0]) / ( group['time'].iloc[-1] - group['time'].iloc[0]) * 2 - 1
                ##各网格与出发位置时间差
                group['time_diff'] = ((group['daytime'] - group.iloc[0]['daytime']) / 60) / 60 * 2 - 1 #与出发时间的秒级时间差转成分钟差(0, 60), 再规整到(-1, 1)，
                group['time_diff_min'] = deepcopy(group['time_diff'])
                group['time_diff_max'] = deepcopy(group['time_diff'])
                o_cell = int(group['cell_index'].iloc[0])
                d_cell = int(group['cell_index'].iloc[-1])
                ###
                if (o_cell, d_cell) in self.od_dict.keys():
                    od_length_limit = self.od_dict[(o_cell, d_cell)]['length_limit']
                    if len(group['cell_index'].unique()) >= od_length_limit:
                        continue
                # 超过长度考虑范围，暂时不用，之后再构建一次样本时需要
                start_t_5 = int((group['daytime'].iloc[0] / 60) // 5) # 出发5分钟级时间片
                start_t_10 = int((group['daytime'].iloc[0] / 60) // 30)  # 出发10分钟级时间片
                start_t_30 = int((group['daytime'].iloc[0] / 60) // 30)  # 出发30分钟级时间片，暂定30分钟，可以考虑换成5分钟，10分钟
                date = int(group['date'].iloc[-1])
                cell_group = group.groupby('cell_index').agg({'time_diff': np.mean, 'time_diff_min': np.min, 'time_diff_max':np.max}, as_index = False) # 网格最早访问时间，网格最晚访问时间
                cell_group['cell_index'] = cell_group.index
                #make odt dict, key:odt, value: {'total_travel_time': [], 'date': [], 'grid_seq': [[], [],..., []], 'grid_avg_t': [[], [],...,[]], 'grid_max_t:[[], [],...,[]]'}
                if (o_cell, d_cell, date, start_t_30) not in self.odt_dict.keys():
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)] = {}
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['total_travel_time'] = []
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['total_travel_time'].append(np.round((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60, 2))
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['date'] = []
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['date'].append(group.iloc[0]['date'])
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['grid_seq'] = []
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['grid_seq'].append(np.array(group['cell_index'].unique()).astype(int).tolist())
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['grid_min_t'] = []
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['grid_max_t'] = []
                    grid_min_t = np.round(np.sort(((np.array(cell_group['time_diff_min']) + 1) / 2 * 60)), 2).tolist()
                    grid_max_t = np.round(np.sort((np.array(cell_group['time_diff_max']) + 1) / 2 * 60), 2).tolist()
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['grid_min_t'].append(grid_min_t)
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['grid_max_t'].append(grid_max_t)
                else:
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['total_travel_time'].append(np.round((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60, 2))
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['date'].append(group.iloc[0]['date'])
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['grid_seq'].append(np.array(group['cell_index'].unique()).astype(int).tolist())
                    grid_min_t = np.round(np.sort((np.array(cell_group['time_diff_min']) + 1) / 2 * 60), 2).tolist()
                    grid_max_t = np.round(np.sort((np.array(cell_group['time_diff_max']) + 1) / 2 * 60), 2).tolist()
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['grid_min_t'].append(grid_min_t)
                    self.odt_dict[(o_cell, d_cell, date, start_t_30)]['grid_max_t'].append(grid_max_t)
                #make traffic state, 每个网格中轨迹点算距离，再统计时间。统计各段轨迹在网格的平均停留时间，一小段轨迹的停留时间包括出网格时间-进入网格时间，距离本身有误差，看每个网格中每一小段轨迹的停留时间
                traffic_group = deepcopy(group)
                traffic_group['ts'] = traffic_group['daytime'] / 60 // self.time_interval # self.time_interval分钟级时间片, 交通状况由10分钟级时间片构造得到
                traffic_group['daytime_diff'] = traffic_group['daytime'].diff() # 每个轨迹点与上一轨迹点的时间差
                traffic_group['daytime_diff'].iloc[0] = 0  # 首位nan换成0
                traffic_group['sub_traj_id'] = (traffic_group['cell_index'].diff() != 0).cumsum()
                cell_daytimenorm_diff_ts = traffic_group.groupby('sub_traj_id').agg({'daytime_diff': np.sum, 'ts': np.max}) # 该小段轨迹计入的时间片，小段轨迹通行时间间隔

                cell_index_diff = group['cell_index'].shift(1) - group['cell_index']
                traffic_cell_group = traffic_group.loc[cell_index_diff != 0]
                traffic_cell_group['daytime_diff_min'] =  np.array(cell_daytimenorm_diff_ts['daytime_diff']) / 60# 从上个格子到这个格子，加上子轨迹段在这个格子中的总停留时间, 秒转换为分钟
                traffic_cell_group['ts'] = np.array(cell_daytimenorm_diff_ts['ts']) # 每个cell的通行时间片
                ###
                traffic_cell_group = traffic_cell_group.iloc[1: -1] # 不考虑出发网格和到达网格的通行时间对交通状况的影响
                ###
                traffic_cell_group = traffic_cell_group.iloc[::-1]  # 来回绕网格如何处理的，确认下->现在对于来回绕的网格是当作两段子轨迹
                traffic_cell_travel_time = traffic_cell_group[['ts', 'date', 'cell_index', 'daytime_diff_min']] # 可能有来回绕的情况，分成两个cell和对应的通行时间
                #合并traffic state, 只能用for
                for i in range(len(traffic_cell_travel_time)):
                    ts = int(traffic_cell_group['ts'].iloc[i])
                    day = int(traffic_cell_group['date'].iloc[i])
                    traffic_cell_index = int(traffic_cell_group['cell_index'].iloc[i])
                    traffic_daytime_diff = traffic_cell_group['daytime_diff_min'].iloc[i]
                    if traffic_daytime_diff < 0:
                        continue
                    if (day, ts) not in self.traffic_dict.keys():
                        self.traffic_dict[(day, ts)] = {}
                        self.traffic_dict[(day, ts)][traffic_cell_index] = []
                        self.traffic_dict[(day, ts)][traffic_cell_index].append(np.round(traffic_daytime_diff, 2))
                    else:
                        if traffic_cell_index not in self.traffic_dict[(day, ts)].keys():  # ts在，cell_index不在
                            self.traffic_dict[(day, ts)][traffic_cell_index] = []
                            self.traffic_dict[(day, ts)][traffic_cell_index].append(np.round(traffic_daytime_diff, 2))
                        else:
                            self.traffic_dict[(day, ts)][traffic_cell_index].append(np.round(traffic_daytime_diff, 2))

                array_index = np.array(cell_group.index).astype(int).tolist()
                img = np.ones((3, self.split * self.split)) * -1.0  # (C, W*H)
                img[0, array_index] = 1  # Mask
                img[1, array_index] = cell_group['time_diff_min'] # 网格最早访问时间
                img[2, array_index] = cell_group['time_diff_max'] # 网格最晚访问时间
                img = img.reshape(img.shape[0], self.split, self.split)  # (C, W, H)
                #从起点到终点的分钟级时间差
                if (group.iloc[0]['daytime'] < 29401): # 出发时间需要超过8:10，否则不计入训练验证测试集，只用于算交通状况
                    continue
                images.append(img)
                ODTs.append((*group.iloc[0][['lng_norm', 'lat_norm']].to_list(),
                             *group.iloc[-1][['lng_norm', 'lat_norm']].to_list(),
                             group.iloc[0]['norm_dt_from8'],
                             int(group.iloc[0]['date']),
                             int(group.iloc[0]['daytime'] / 60 // self.time_interval))) # 增加出发天，出发时间片（10分钟级）
                arrive_times.append((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60)

            images, ODTs, arrive_times = np.stack(images, 0), np.array(ODTs), np.array(arrive_times) # 增加一个出发时间片和日期，可以直接取出对应的交通状态
            self.images[df_index] = (images, ODTs, arrive_times)
            self.num_channel = images.shape[1]
            np.save(self.traffic_state_path, self.traffic_dict)
            np.save(self.odt_path, self.odt_dict)
            self.traffic_dict = np.load(self.traffic_state_path, allow_pickle=True).item()
            print('traffic_state generated...')
            print('odt generated...')

        return (images, ODTs, arrive_times)

    def dump_images_state(self):
        #先获取images
        for i in range(3):
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

