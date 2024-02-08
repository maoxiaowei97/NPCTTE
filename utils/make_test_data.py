import numpy as np
image_meta_path = '/data/XinZhi/ODUQ/DOT/data/20181101_20181110_all_15s_S20_FFalse_image.npz'
test_image_meta_path = '/data/XinZhi/ODUQ/DOT/data/20181101_20181110_all_15s_S20_FFalse_image_test.npz'
images = {}
update_images = {}
loaded_images = np.load(image_meta_path)
lng_min, lng_max, lat_min, lat_max = 104.03966, 104.12654, 30.65523, 30.72892
for i, label in enumerate(['train', 'val', 'test']):

    images[i] = (loaded_images[f'{label}_image'][:10000], loaded_images[f'{label}_odt'][:10000],
                      loaded_images[f'{label}_arr'][:10000])
    odt = images[i][1]
    start_x = np.around(((( ((odt[:, 0] + 1)/2) * (lng_max - lng_min) + lng_min) - lng_min) / ((lng_max - lng_min) / (20 - 1))))
    start_y = np.around((((((odt[:, 1] + 1)/2) * (lat_max - lat_min) + lat_min) - lat_min) / ((lat_max - lat_min) / (20 - 1))))
    end_x = np.around((((((odt[:, 2] + 1)/2) * (lng_max - lng_min) + lng_min) - lng_min) / ((lng_max - lng_min) / (20 - 1))))
    end_y = np.around((((((odt[:, 3] + 1)/2) * (lat_max - lat_min) + lat_min) - lat_min) / ((lat_max - lat_min) / (20 - 1))))
    start_cell_index = start_y * 20 + start_x
    end_cell_index = end_y * 20 + end_x
    t_minute = (((odt[:, 4] + 1) / 2 * 24 * 60 * 60) // 60).astype(int) # 一天中的多少分
    ts_10 = (t_minute // 10).astype(int) # 10分钟级时间片
    updated_odt = np.concatenate([images[i][1], start_cell_index.reshape(-1, 1), end_cell_index.reshape(-1, 1), t_minute.reshape(-1, 1), ts_10.reshape(-1, 1)], 1)
    update_images.update({f'{label}_image': images[i][0],
                   f'{label}_odt': updated_odt,
                   f'{label}_arr': images[i][2]})

gen_set_path = '/data/XinZhi/ODUQ/DOT/data/nppc_generation/20181101_20181110_all_15s_nppc_images_T1000_S20.npz'
test_gen_set_path = '/data/XinZhi/ODUQ/DOT/data/nppc_generation/20181101_20181110_all_15s_nppc_images_T1000_S20_test.npz'
gen_images = np.load(gen_set_path)
gen_images = [gen_images[label] for label in ['train', 'val', 'test']]
select_sets = [0, 1, 2]
for s in select_sets:
    gen_images[s] = gen_images[s][:10000]
np.savez(test_image_meta_path, **update_images)
np.savez(test_gen_set_path, train=gen_images[0], val=gen_images[1], test=gen_images[2])  # 生成数值型的image

gen_set_path_denoiser = '/data/XinZhi/ODUQ/DOT/data/20181101_20181110_all_15s_images_unet_T1000_S20.npz'
test_gen_set_path_denoiser = '/data/XinZhi/ODUQ/DOT/data/20181101_20181110_all_15s_images_unet_T1000_S20_test.npz'
gen_images = np.load(gen_set_path_denoiser)
gen_images = [gen_images[label] for label in ['train', 'val', 'test']]
select_sets = [0, 1, 2]
for s in select_sets:
    gen_images[s] = gen_images[s][:10000]
np.savez(test_gen_set_path_denoiser, train=gen_images[0], val=gen_images[1], test=gen_images[2])  # 生成数值型的image

