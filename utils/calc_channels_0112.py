import numpy as np

images = {}
image_meta_path = '/data/XinZhi/ODUQ/DOT/data/20181101_20181110_all_15s_td_0110_S20_FFalse_image.npz'
loaded_images = np.load(image_meta_path)
for i, label in enumerate(['train', 'val', 'test']):
    images[i] = (loaded_images[f'{label}_image'], loaded_images[f'{label}_odt'],
                      loaded_images[f'{label}_arr'])

gen_set_path = '/data/XinZhi/ODUQ/DOT/data/20181101_20181110_all_15s_td_0110_images_unet_T1000_ddimTrue_DenEpo-1_S20.npz'
gen_images = np.load(gen_set_path)
gen_images = [gen_images[label] for label in ['train', 'val', 'test']]
#   x[:, 1] = (x[:, 1] + 1) / 2 * 60
train_arrival = images[0][2]
##取出第一个维度大于-0.90的部分，找出这部分的最小值和最大值
train_gen_tod = (gen_images[0][:,1].reshape(-1, 400) + 1 ) / 2 * 60 # train
train_mask_gen = gen_images[0][:, 0].reshape(-1, 400)
tod_min = []
tod_max = []
tod_dif = []
for i in range(len(train_gen_tod)):
    seq = train_gen_tod[i]
    mask = train_mask_gen[i] > 0
    seq[mask] = 0
    valid_seq = seq

    # valid_seq = seq[mask]
    if len(valid_seq) != 0:
        tod_min.append(valid_seq.min())
        tod_max.append(valid_seq.max())
        # tod_dif.append(((valid_seq.max() - valid_seq.min()) * 86400 / 2) // 60)
        tod_dif.append(valid_seq.max())
    else:
        tod_dif.append(15)
mae = np.mean(np.abs(np.array(train_arrival) - np.array(tod_dif)))
print('mae: ', mae)
