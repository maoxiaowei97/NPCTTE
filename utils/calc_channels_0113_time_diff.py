"""
1.13
ddpm生成 mask, time diff minute, offset
不shuffle dataset.get_images
问题：生成时间差也不准，训练集上的mape大概为52%，有可能是网格划分的原因，丢失了太多信息
"""
import numpy as np
images = {}
image_meta_path = '/data/XinZhi/ODUQ/DOT/data/20181101_20181110_all_15s_td_0112_S20_FFalse_image.npz'
loaded_images = np.load(image_meta_path)
for i, label in enumerate(['train', 'val', 'test']):
    images[i] = (loaded_images[f'{label}_image'], loaded_images[f'{label}_odt'],
                      loaded_images[f'{label}_arr'])

gen_set_path = '/data/XinZhi/ODUQ/DOT/data/20181101_20181110_all_15s_td_0110_images_unet_T1000_DenEpo-1_S20_test5k.npz'
gen_images = np.load(gen_set_path)
gen_images = [gen_images[label] for label in ['train', 'val', 'test']]
#   x[:, 1] = (x[:, 1] + 1) / 2 * 60
train_arrival = images[0][2]
train_tod = images[0][0][:, 1]
##取出第一个维度大于-0.90的部分，找出这部分的最小值和最大值
train_gen_tod = (gen_images[0][:,1].reshape(-1, 400) + 1 ) / 2 * 60 # train time diff minute
# train_true_tod = (train_tod.reshape(-1, 400) + 1 ) / 2 * 60
# train_od = train_true_tod
train_od = train_gen_tod
train_mask_gen = gen_images[0][:, 0].reshape(-1, 400) # mask of the sequence
train_arrival = train_arrival[:len(train_mask_gen)]
tod_min = []
tod_max = []
tod_dif_2biggesr = []
tod_dif_3biggesr = []
# for i in range(len(train_gen_tod)):
for i in range(len(train_mask_gen)):
    seq = train_od[i]
    mask = train_mask_gen[i] < 0
    seq[mask] = -1 # = 0
    valid_seq = seq

    # valid_seq = seq[mask]
    if (len(valid_seq) != 0 ) and (np.sort(valid_seq)[::-1][1] > 0) and ((np.sort(valid_seq)[::-1][2] > 0)):
        tod_min.append(valid_seq.min())
        tod_max.append(valid_seq.max())
        # tod_dif.append(((valid_seq.max() - valid_seq.min()) * 86400 / 2) // 60)
        tod_dif_2biggesr.append(np.sort(valid_seq)[::-1][1])
        tod_dif_3biggesr.append(np.sort(valid_seq)[::-1][2])
    else:
        tod_dif_2biggesr.append(15)
        tod_dif_3biggesr.append(15)

mae_2 = np.mean(np.abs(np.array(train_arrival) - np.array(tod_dif_2biggesr)))
mape_2 = np.mean(np.abs(np.array(train_arrival) - np.array(tod_dif_2biggesr)) / np.array(train_arrival))
mae_3 = np.mean(np.abs(np.array(train_arrival) - np.array(tod_dif_3biggesr)))
mape_3 = np.mean(np.abs(np.array(train_arrival) - np.array(tod_dif_3biggesr)) / np.array(train_arrival))
print('mae_2: ', mae_2, 'mape_2: ', mape_2)
print('mae_3: ', mae_3, 'mape_2: ', mape_3)

