import numpy as np
traffic_dict = np.load('/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/201811_d1256789_h8_10_15s_0115_3channel_S20_test_S20_FFalse_trafficstate.npy', allow_pickle=True).item()
odt_dict = np.load('/data/XinZhi/ODUQ/DOT/data/cache_odt/201811_d1256789_h8_10_15s_0115_3channel_S20_test_S20_FFalse_odt.npy', allow_pickle=True).item()
c = traffic_dict