import numpy as np
traffic_state = np.load('/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/1101_1116_15s_0116_3channel_S20_S20_FFalse_trafficstate_test.npy', allow_pickle=True).item()
traffic_state_save_path = '/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/1101_1116_15s_0116_3channel_S20_S20_FFalse_saved_traffic_array.npy'
#[Day, Ts, cell_index] -> [b, cell_num]
c = traffic_state
Day = 17 # 1-16
Ts = 86 # 10分钟级时间片数量, 到14点共
Cell_num = 400 # 网格数量
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
np.save(traffic_state_save_path, traffic_array)

