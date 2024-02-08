import numpy as np
traffic_state = np.load('/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/1101_1116_15s_0117_3channel_S20_S20_FFalse_trafficstate.npy', allow_pickle=True).item()
traffic_state_save_path = '/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/1101_1116_15s_0116_3channel_S20_S20_FFalse_trafficstate_array_0117_test.npy'
normalized_stae_save_path = '/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/normalized_traffic_state_array_1101_1116_15s_0117.npy'
#[Day, Ts, cell_index] -> [b, cell_num]
Day = 17 # 1-16
Ts = 86 # 10分钟级时间片数量, 48 - 84
Cell_num = 400 # 网格数量
#先用有数据的时间，网格构建交通状态
traffic_array = np.zeros([Day, Ts, Cell_num])
for key in traffic_state.keys():
    day = key[0]
    ts = key[1]
    cell_states = traffic_state[key]
    for cell in cell_states.keys():
        state_list = cell_states[cell]
        #去掉离群值后取平均值
        if len(state_list) > 3:
            state_list.remove(np.array(state_list).min())
            state_list.remove(np.array(state_list).max())
        traffic_array[day, ts, cell] = np.mean(state_list)

saved_array = traffic_array
# saved_array = np.load(traffic_state_save_path, allow_pickle=True)
#对各个网格在不同时间片的初始化值，用1，2，5，6，7，8，9，12，13号相应的平均通行时间
#[Ts, Cell_num] 构建初始化交通状态表

mean_array = saved_array[[1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16]].mean(0) # 选中天，不同时间片，不同网格的通行时间，在天维度求平均 -> [Ts, Cell_num]
mean_array = np.repeat(mean_array[np.newaxis, :, :], Day, axis=0)

#用初始化的表构建traffic_array
for key in traffic_state.keys():
    day = key[0]
    ts = key[1]
    cell_states = traffic_state[key]
    for cell in cell_states.keys():
        state_list = cell_states[cell]
        if len(state_list) > 3:
            state_list.remove(np.array(state_list).min())
            state_list.remove(np.array(state_list).max())
        mean_array[day, ts, cell] = np.mean(state_list) # 平均通行时间

#看下通行分钟的最大值，取值范围为[0, max] -> (array / max ) * 2 - 1
normalized_array = (mean_array / mean_array.max()  ) * 2 - 1
np.save(normalized_stae_save_path, normalized_array)