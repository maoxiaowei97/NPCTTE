#!/data/XinZhi/miniconda3/envs/py38/bin/python
"""
2023.1.24. 23:04
选用4天7小时数据，跑20， 30的数据
重点看生成特征质量，在split=20, 30时的情况
xian
201811_d1256789_h8_10.csv

测试给定ODT下的平均通行时间


"""
import os
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from model_0115.NppcNet import nppc_UNet, PCWrapper
from model_0115.NppcNet_Unet import Nppc_Unet as big_nppc_UNet
from dataet_statod_0124 import TrajectoryDataset
from model_0125_xian.trainer import DiffusionTrainer, UQTrainer, NppcTrainer
from model_0125_xian.diffusion import DiffusionProcess
from model_0125_xian.predictor import *
from model_0115.eta_trainer import ETATrainer
from model_0125_xian.denoiser import Unet
import plotly.express as px

parser = ArgumentParser()

parser.add_argument('--cuda', help='the index of the cuda device', type=int, default=1)
# Dataset arguments

parser.add_argument('-n', '--name', help='the name of the dataset', type=str, default='1101_1116_15s_0125_3channel_S20_xian') # 1.16. 20:31
parser.add_argument('--gen_image_name', help=' supp name of the generated images', type=str, default='01172300')
parser.add_argument('-s', '--split', help='the number of x and y split', type=int, default=20)
parser.add_argument('--flat', default=False)

# Denoiser arguments
parser.add_argument('--denoiser', help='name of the denoiser', type=str, default='unet')
parser.add_argument('--condition', type=str, default='odt')

# Training arguments
parser.add_argument('-e', '--epoch', help='number of training epoches', type=int, default=201)
parser.add_argument('-ea', '--early_stop', help='number of early stop epoches', type=int, default=15)
parser.add_argument('-b', '--batch', help='batch size', type=int, default=512)
parser.add_argument('-t', '--timestep', help='the total timesteps of diffusion process', type=int, default=1000)
parser.add_argument('-p', '--partial', default=1.0, type=float)
parser.add_argument('--n_dirs', help='num of nppc', default=5)
parser.add_argument('--nppc_step', help='npcc step', default=0)
parser.add_argument('--second_moment_loss_grace', help='second_moment_loss_grace', default=500)

# Diffusion training arguments
parser.add_argument('--traindiff', help='whether to train the diffusion model', default=False)
parser.add_argument('--loaddiff', help='whether to load the diffusion model from cached file',default=True)
parser.add_argument('--loadnppc', help='whether to load the nppc model from cached file',default=False)
parser.add_argument('--trainnppc', help='whether to train the nppc model', default=False)
parser.add_argument('--trainuq', help='whether to train the uq model', default=False)
# parser.add_argument('--loadepoch_denoiser_trained_epoch', help='the training epoch to load from, set to -1 for loading from the final model', default=101) # 表示用的denoiser是由多少轮训练得到的
# parser.add_argument('--loadepoch_denoiser', help='the training epoch to load from, set to -1 for loading from the final model', default=-1) # -1表示load最后一个epoch训练得到的denoiser
parser.add_argument('--loadepoch_denoiser_trained_epoch', help='the training epoch to load from, set to -1 for loading from the final model', default=201) # 表示用的denoiser是由多少轮训练得到的
parser.add_argument('--loadepoch_denoiser', help='load the denoiser best in the epoch of', default=-1)
parser.add_argument('--loadepoch_nppc', help='the training epoch to load from, set to -1 for loading from the final model', default=172)
parser.add_argument('--loadgen', help='whether to load generated images', default=False)
parser.add_argument('--loadnppcgen', help='whether to load generated nppc images', default=False)
parser.add_argument('--uq_estimate', help='whether estimate uq', default=False)
parser.add_argument('--uq_num', help='num of uncertainty quantification', default=3)
parser.add_argument('--ddim', help='whether ddim', default=False)
parser.add_argument('--draw', help='whether to draw generated images', default=False)
parser.add_argument('--draw_nppc', help='whether to draw generated nppc images', default=False)
parser.add_argument('--numimage', help='number of images to draw', type=int, default=100)
parser.add_argument('--test', help='whether test', default=False)
parser.add_argument('--test_evaluate', help='whether test by diffusion', default=False)
parser.add_argument('--temp', help='whether test by historical average', default=False)
parser.add_argument('--xgb', help='whether test by historical average', default=False)

# ETA training arguments
parser.add_argument('--traineta', help='whether to train the ETA prediction model', default=False)
parser.add_argument('--predictor', help='name of the predictor', type=str, default='trans')
parser.add_argument('--predict_type', help='only_mean, only_nppc, mean_nppc', type=str, default='only_mean')
parser.add_argument('--predict_task', help='only_mean, only_nppc, mean_nppc', type=str, default='mis')
parser.add_argument('--predictorL', help='number of layers in the predictor', type=int, default=2)
parser.add_argument('--trainorigin', help='whether to use original images to train ETA predictor', default=False)
parser.add_argument('--valorigin', help='whether to use original images to evaluate ETA predictor', default=False)
parser.add_argument('--stop', help='number of early stopping epoch for ETA training', type=int, default=10)
parser.add_argument('--eta_model_load_path', help='path of eta model', type=str, default='/data/maodawei/DOT/data/model/chengdu_predictor_trans_L2_D128_stTrue_gridFalse_T1000_S20.model_epoch7')

# Predictor arguments
parser.add_argument('-d', '--dmodel', help='dimension of predictor models', type=int, default=128)
parser.add_argument('--rmst',  help='whether to train the ETA prediction model with image information', default=False)
args = parser.parse_args()
if args.test:
    args.epoch = 5

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
if torch.cuda.is_available():
    device = 'cuda'
    small = True
else:
    device = 'cpu'
    small = True

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

# Loading dataset
dataset = TrajectoryDataset(args.name, split=args.split, partial=args.partial, small=False, flat=args.flat, traffic_states=False, is_test=args.test, get_dataframe = True)
# dataset.dump_images_state()
dataset.load_images()

# Create models.
denoiser = Unet(dim=args.split, channels=dataset.num_channel, init_dim=4, dim_mults=(1, 2, 4), condition=args.condition, traffic_state = False)
nppc_unet = PCWrapper(nppc_UNet(in_channels=3 + 3, out_channels=3 * args.n_dirs), n_dirs=args.n_dirs)
diffusion = DiffusionProcess(T=args.timestep, schedule_name='linear')

diffusion_trainer = DiffusionTrainer(diffusion=diffusion, denoiser=denoiser, dataset=dataset, lr=1e-3,
                                     batch_size=args.batch, loss_type='huber', device=device, traffic_state = False,
                                     num_epoch=args.epoch, early_stop = args.early_stop, is_test=args.test, denoiser_epoch=args.loadepoch_denoiser_trained_epoch, ddim = args.ddim)
#训练好denoiser, generate images


def scale_img(x):
    return x / torch.abs(x).flatten(-3).max(-1)[0][..., None, None, None] / 1.5 + 0.5

def tensor_img_to_numpy(x):
    return x.detach().permute(-2, -1, -3).cpu().numpy()
def imshow(img, scale=1, **kwargs):
    if isinstance(img, torch.Tensor):
        img = tensor_img_to_numpy(img)
    img = img.clip(0, 1)

    fig = px.imshow(img, **kwargs).update_layout(
        height=img.shape[0] * scale,
        width=img.shape[1] * scale,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
    )
    return fig

if args.loaddiff:
    denoiser = diffusion_trainer.load_model(None if args.loadepoch_denoiser == -1 else args.loadepoch_denoiser)
elif args.traindiff:
    denoiser = diffusion_trainer.train()

#xian route evaluate
if 1:
    test_images, test_ODTs, test_arrival = dataset.get_images(2) # 验证集所有图片
    denoiser = diffusion_trainer.load_model(None if args.loadepoch_denoiser == -1 else args.loadepoch_denoiser)
    diffusion_trainer.load_generation()
    test_gen = diffusion_trainer.gen_images[2]
    nppc_trainer = NppcTrainer(diffusion=diffusion, denoiser=denoiser, gen_images=diffusion_trainer.gen_images,
                               nppc_net=nppc_unet, dataset=dataset, lr=1e-3,
                               batch_size=args.batch, device=device, num_epoch=args.epoch, nppc_step=args.nppc_step,
                               second_moment_loss_grace=args.second_moment_loss_grace,
                               load_trained_nppc_epoch=args.loadepoch_nppc,
                               early_stopping=args.early_stop, is_test=args.test, ddim=args.ddim)
    nppc_trainer.load_generation()  # 更新self.images
    # test_gen = nppc_trainer.gen_images[2]
    total_num = len(test_images)
    pred_list = []
    label_list = []
    metrics = []
    for i in tqdm(range(total_num)):
        images = test_images[i]
        # gen_routes = test_gen[i][4][0].reshape(-1, args.split * args.split)
        gen_routes = test_gen[i][0].reshape(-1, args.split * args.split)
        gen_routes[gen_routes > 0] = 1
        gen_routes[gen_routes < 0] = 0
        true_routes = images[0].reshape(-1, args.split * args.split) #[b, N]
        true_routes[true_routes < 0] = 0
        prec =  precision_score(true_routes.reshape(-1), gen_routes.reshape(-1), zero_division=0)
        rec = recall_score(true_routes.reshape(-1), gen_routes.reshape(-1))
        f1 = f1_score(true_routes.reshape(-1), gen_routes.reshape(-1))
        metrics.append([prec, rec, f1])
    metrics = np.round(np.mean(metrics, axis=0) * 100, 3)
    print('Route metrics diffusion xian:', metrics)


indices_to_gen = [0, 1, 2]  # 按生成概率存多个图像, 通过diffusion_trainer.gen_images获得生成图像
# print('start generate denoiser images...')
# diffusion_trainer.save_generation(indices_to_gen)
# print(' denoiser images generated...')
if args.xgb:
    import xgboost as xgb
    od_info_dict = np.load('/data/XinZhi/ODUQ/DOT/data/cache_od_dict/1101_1116_15s_0125_3channel_S20_xian_od_info_avg_t_dict_without_val_test.npy', allow_pickle=True).item()
    train_images, train_ODTs, train_arrival = dataset.get_images(0)  # 验证集所有图片
    test_images, test_ODTs, test_arrival = dataset.get_images(2)  # 验证集所有图片
    ## test data
    lng_min, lng_max, lat_min, lat_max = dataset.lng_min, dataset.lng_max, dataset.lat_min, dataset.lat_max
    # test_sample_num = len(test_images) -  len(test_images)// 2
    test_sample_num = len(test_images)
    # test_sample_num = len(test_images)// 2
    test_start_x = np.around((((((test_ODTs[:, 0] + 1) / 2) * (lng_max - lng_min) + lng_min) - lng_min) / (
            (lng_max - lng_min) / (20 - 1))))
    test_start_y = np.around((((((test_ODTs[:, 1] + 1) / 2) * (lat_max - lat_min) + lat_min) - lat_min) / (
            (lat_max - lat_min) / (20 - 1))))
    test_end_x = np.around((((((test_ODTs[:, 2] + 1) / 2) * (lng_max - lng_min) + lng_min) - lng_min) / (
            (lng_max - lng_min) / (20 - 1))))
    test_end_y = np.around((((((test_ODTs[:, 3] + 1) / 2) * (lat_max - lat_min) + lat_min) - lat_min) / (
            (lat_max - lat_min) / (20 - 1))))
    test_start_cell_index = test_start_y * 20 + test_start_x
    test_end_cell_index = test_end_y * 20 + test_end_x
    test_norm_dt_from8 = test_ODTs[: ][:, 4]
    # test_norm_dt_from8 = test_ODTs[:len(test_images) // 2][:, 4]
    test_start_ts_30 = ((test_norm_dt_from8 + 1) / 2 * 360 + 480) // 30  # 第几个30min时间片
    test_start_ts_60 = ((test_norm_dt_from8 + 1) / 2 * 360 + 480) // 60  # 第几个30min时间片
    test_o_lng = test_ODTs[:][:, 0]
    test_o_lat = test_ODTs[:][:, 1]
    test_d_lng = test_ODTs[:][:, 2]
    test_d_lat = test_ODTs[:][:, 3]

    test_start_cell_index = test_start_cell_index[:]
    test_end_cell_index = test_end_cell_index[:]

    #train data
    dim_fea = 5
    train_start_x = np.around((((((train_ODTs[:, 0] + 1) / 2) * (lng_max - lng_min) + lng_min) - lng_min) / (
            (lng_max - lng_min) / (20 - 1))))
    train_start_y = np.around((((((train_ODTs[:, 1] + 1) / 2) * (lat_max - lat_min) + lat_min) - lat_min) / (
            (lat_max - lat_min) / (20 - 1))))
    train_end_x = np.around((((((train_ODTs[:, 2] + 1) / 2) * (lng_max - lng_min) + lng_min) - lng_min) / (
            (lng_max - lng_min) / (20 - 1))))
    train_end_y = np.around((((((train_ODTs[:, 3] + 1) / 2) * (lat_max - lat_min) + lat_min) - lat_min) / (
            (lat_max - lat_min) / (20 - 1))))

    train_start_cell_index = train_start_y * 20 + train_start_x
    train_end_cell_index = train_end_y * 20 + train_end_x
    train_norm_dt_from8 = train_ODTs[:, 4]
    train_start_ts_30 = ((train_norm_dt_from8 + 1) / 2 * 360 + 480) // 30  # 第几个30min时间片
    train_start_ts_60 = ((train_norm_dt_from8 + 1) / 2 * 360 + 480) // 60  # 第几个30min时间片
    train_o_lng = train_ODTs[:, 0]
    train_o_lat = train_ODTs[:, 1]
    train_d_lng = train_ODTs[:, 2]
    train_d_lat = train_ODTs[:, 3]

    x_train = np.zeros([len(train_images), dim_fea])
    y_train = np.array(train_arrival)
    #test data
    x_test = np.zeros([test_sample_num, dim_fea])
    y_test = np.array(test_arrival[:])

    #make train data
    for i in tqdm(range(len(train_images))):
        o = int(train_start_cell_index[i])
        d = int(train_end_cell_index[i])
        ts_30 = int(train_start_ts_30[i])
        ts_60 = int(train_start_ts_60[i])
        if (o, d) in od_info_dict.keys():
            if (ts_30 - 1) in od_info_dict[(o, d)]['odt_t'].keys():
                ha = np.mean(od_info_dict[(o, d)]['odt_t'][ts_30 - 1])
            else:  # o, d在，ts不在
                ha = od_info_dict[(o, d)]['od_avg_t']
        else:  # o, d不在，给全局平均
            ha = 11.72
        o_lng = train_o_lng[i]
        o_lat = train_o_lat[i]
        d_lng = train_d_lng[i]
        d_lat = train_d_lat[i]
        norm_dt_from8 = train_norm_dt_from8[i]
        # x_train[i, :] = np.array([o_lng, o_lat, d_lng, d_lat, norm_dt_from8, o, d, ts_30, ts_60, ha])  # b, 10
        x_train[i, :] = np.array([o_lng, o_lat, d_lng, d_lat, norm_dt_from8]) # b, 10

    #make test data
    for i in tqdm(range(len(test_start_cell_index))):
    # for i in tqdm(range (len(test_images) - len(test_images)// 2)):
        o = int(test_start_cell_index[i])
        d = int(test_end_cell_index[i])
        ts_30 = int(test_start_ts_30[i])
        ts_60 = int(test_start_ts_60[i])
        if (o, d) in od_info_dict.keys():
            if (ts_30 - 1) in od_info_dict[(o, d)]['odt_t'].keys():
                ha = np.mean(od_info_dict[(o, d)]['odt_t'][ts_30 - 1])
            else:  # o, d在，ts不在
                ha = od_info_dict[(o, d)]['od_avg_t']
        else:  # o, d不在，给全局平均
            ha = 10.72
        o_lng = test_o_lng[i]
        o_lat = test_o_lat[i]
        d_lng = test_d_lng[i]
        d_lat = test_d_lat[i]
        norm_dt_from8 = test_norm_dt_from8[i]
        # x_test[i, :] = np.array([o_lng, o_lat, d_lng, d_lat, norm_dt_from8, o, d, ts_30, ts_60, ha]) # b, 10
        x_test[i, :] = np.array([o_lng, o_lat, d_lng, d_lat, norm_dt_from8])  # b, 10

    dtrain = xgb.DMatrix(x_train[:60000], label=y_train[:60000])
    dtest = xgb.DMatrix(x_test, label=y_test)
    params = {'objective': 'reg:squarederror','max_depth': 5, 'alpha': 10 }
    print('start training xgb')
    model = xgb.train(params, dtrain)
    pred = model.predict(dtest)
    print('xgb well trained')

    print('final test sample num: ', len(test_images))
    print('final mae: ', np.mean(np.abs(np.array(pred) - np.array(y_test))))
    print('final mape: ', np.mean(np.abs(np.array(pred) - np.array(y_test)) / np.array(y_test)))
    print('final rmse: ', np.sqrt(np.mean( ((np.array(pred) - np.array(y_test)) ** 2) )))

if args.temp:
    od_info_dict = np.load('/data/XinZhi/ODUQ/DOT/data/cache_od_dict/1101_1116_15s_0125_3channel_S20_xian_od_info_avg_t_dict_without_val_test.npy', allow_pickle=True).item()
    test_images, test_ODTs, test_arrival = dataset.get_images(2)  # 测试集所有图片
    test_images, test_ODTs = (test_images, test_ODTs)
    lng_min, lng_max, lat_min, lat_max = dataset.lng_min, dataset.lng_max, dataset.lat_min, dataset.lat_max
    start_x = np.around((((((test_ODTs[:, 0] + 1) / 2) * (lng_max - lng_min) + lng_min) - lng_min) / (
            (lng_max - lng_min) / (20 - 1))))
    start_y = np.around((((((test_ODTs[:, 1] + 1) / 2) * (lat_max - lat_min) + lat_min) - lat_min) / (
            (lat_max - lat_min) / (20 - 1))))
    end_x = np.around((((((test_ODTs[:, 2] + 1) / 2) * (lng_max - lng_min) + lng_min) - lng_min) / (
            (lng_max - lng_min) / (20 - 1))))
    end_y = np.around((((((test_ODTs[:, 3] + 1) / 2) * (lat_max - lat_min) + lat_min) - lat_min) / (
            (lat_max - lat_min) / (20 - 1))))
    start_cell_index = start_y * 20 + start_x
    end_cell_index = end_y * 20 + end_x
    traffic_state = np.load(
        '/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/normalized_traffic_state_array_1101_1116_15s_0117.npy',
        allow_pickle=True)
    norm_dt_from8 = test_ODTs[:, 4]
    start_ts_30 = ((norm_dt_from8 + 1) / 2 * 360 + 480) // 30  # 第几个30min时间片
    start_ts_60 = ((norm_dt_from8 + 1) / 2 * 360 + 480) // 60  # 第几个30min时间片
    test_arrival = test_arrival[:]
    pred_list = []
    for i in tqdm(range (len(test_images))):
        o = int(start_cell_index[i])
        d = int(end_cell_index[i])
        ts_30 = int(start_ts_30[i])
        ts_60 = int(start_ts_60[i])
        if (o, d) in od_info_dict.keys():
            if (ts_30 - 1) in od_info_dict[(o, d)]['odt_t'].keys():
                pred_list.append(np.mean(od_info_dict[(o, d)]['odt_t'][ts_30 - 1])) # odt成功匹配
            else:# o, d在，ts不在
                pred_list.append(od_info_dict[(o, d)]['od_avg_t'])
            # if (ts_60 - 1) in od_info_dict[(o, d)]['odt_t_60'].keys():
            #     pred_list.append(np.mean(od_info_dict[(o, d)]['odt_t_60'][ts_60 - 1]))  # odt成功匹配
            # else:  # o, d在，ts不在
            #     pred_list.append(od_info_dict[(o, d)]['od_avg_t'])
        else: # o, d不在，给全局平均
            pred_list.append( 15.72)

    print('final test sample num: ', len(test_images))
    print('final mae: ', np.mean(np.abs(np.array(pred_list) - np.array(test_arrival))))
    print('final mape: ', np.mean(np.abs(np.array(pred_list) - np.array(test_arrival)) / np.array(test_arrival)))
    print('final rmse: ', np.sqrt(np.mean( ((np.array(pred_list) - np.array(test_arrival)) ** 2) )))


if args.test_evaluate:
    test_images, test_ODTs, test_arrival = dataset.get_images(2) # 验证集所有图片
    test_images, val_ODTs = (test_images, test_ODTs)
    traffic_state =  np.load('/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/normalized_traffic_state_array_1101_1116_15s_0117.npy', allow_pickle=True)
    bs = 512
    total_num = len(test_images) // 2
    batch_num = total_num // bs

    s_index = 0
    e_index = s_index + bs
    pred_list = []
    label_list = []
    for b in tqdm(range(batch_num)):
        print('total num: ',total_num, 'total batch num: ', batch_num)
        images = test_images[s_index:e_index]
        ODTs = test_ODTs[s_index:e_index]
        arrival_times = test_arrival[s_index:e_index]
        test_D = ODTs[:, 5].reshape(-1).astype(int)
        test_ts = ODTs[:, 6].reshape(-1).astype(int)
        val_traffic = traffic_state[test_D, test_ts].reshape(images.shape[0], 1, images.shape[2], images.shape[3])
        gen_steps = diffusion.p_sample_loop(denoiser, shape=(images.shape[0], *(images.shape[1:])),
                                            y=torch.from_numpy(ODTs).float().to(device), traffic = torch.from_numpy(val_traffic).float().to(device),
                                            display=True)
        gen_images = gen_steps[-1]
        #分享diffusion生成到达时间和label的差别
        t_diff_min = (gen_images[:, 1] + 1) / 2 * 60
        t_diff_max = (gen_images[:, 2] + 1) / 2 * 60
        pred = t_diff_max.reshape(images.shape[0], -1).max(1)
        label = arrival_times
        pred_list.extend(pred.tolist())
        label_list.extend(label.tolist())

        s_index += bs
        e_index += bs
        print('test sample num: ', e_index)
        print('mae: ', np.mean(np.abs(np.array(pred_list) - np.array(label_list))))
        print('mape: ', np.mean(np.abs(np.array(pred_list) - np.array(label_list)) / np.array(label_list)))
    print('final test sample num: ', e_index)
    print('final mae: ', np.mean(np.abs(np.array(pred_list) - np.array(label_list))))
    print('final mape: ', np.mean(np.abs(np.array(pred_list) - np.array(label_list)) / np.array(label_list)))

if args.draw:
    val_images, val_ODTs, val_arrival = dataset.get_images(1) # 验证集所有图片
    val_images, val_ODTs = (val_images, val_ODTs)
    traffic_state =  np.load('/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/normalized_traffic_state_array_1101_1116_15s_0117.npy', allow_pickle=True)
    val_num = 512
    val_images = val_images[:val_num]
    val_ODTs = val_ODTs[:val_num]
    val_D = val_ODTs[:, 5].reshape(-1).astype(int)
    val_ts = val_ODTs[:, 6].reshape(-1).astype(int)
    val_traffic = traffic_state[val_D, val_ts].reshape(val_images.shape[0], 1, val_images.shape[2], val_images.shape[3])
    gen_steps = diffusion.p_sample_loop(denoiser, shape=(val_num, *(val_images.shape[1:])),
                                        y=torch.from_numpy(val_ODTs[:val_num]).float().to(device), traffic = torch.from_numpy(val_traffic).float().to(device),
                                        display=True)
    gen_images = gen_steps[-1]
    #分享diffusion生成到达时间和label的差别
    t_diff_min = (gen_images[:, 1] + 1) / 2 * 60
    t_diff_max = (gen_images[:, 2] + 1) / 2 * 60
    origin_t_diff_min = (val_images[:, 1] + 1) / 2 * 60
    origin_t_diff_max = (val_images[:, 2] + 1) / 2 * 60
    num_channel = gen_images.shape[1]
    for i in tqdm(range(val_num), desc='Drawing generated images'):
        plt.figure(figsize=(num_channel / 2 * 5, 5))
        for c in range(num_channel):
            plt.subplot(2, num_channel, c + 1)
            plt.title(f'Generated channel {c + 1}')
            plt.imshow(gen_images[i][c]) #[20, 20]
            plt.subplot(2, num_channel, c + 1 + num_channel)
            plt.title(f'Real channel {c + 1}')
            plt.imshow(val_images[i][c])
        plt.savefig(os.path.join('data', 'images', f'E{args.loadepoch}_{dataset.name}_%03d_loadDenoiser_trafficstate.png' % i), bbox_inches='tight')
        plt.close()
# Load or train the diffusion model.

#训练好denoiser, generate images
# indices_to_gen = [0, 1, 2]  # 按生成概率存多个图像, 通过diffusion_trainer.gen_images获得生成图像
# print('start generate denoiser images...')
# diffusion_trainer.save_generation(indices_to_gen)
# print('denoiser images generated')
# diffusion_trainer.load_generation()
# print('denoiser images loaded')
#train nppc后，可以load nppc, denoiser, 然后输出不确定性度量结果，作为时间预测输入

#train nppc
# nppc_trainer = NppcTrainer(diffusion=diffusion, denoiser=denoiser, gen_images=diffusion_trainer.gen_images,
#                            nppc_net=nppc_unet, dataset=dataset, lr=1e-3,
#                            batch_size=args.batch, device=device, num_epoch=args.epoch,
#                            nppc_step=args.nppc_step, second_moment_loss_grace=args.second_moment_loss_grace,
#                            early_stopping=args.early_stop, is_test=args.test, ddim=args.ddim)
# print('start nppc training...')
# nppc_net = nppc_trainer.train()
# print('nppc training finished...')
# if args.draw_nppc:
#     # load denoiser and nppc
#     print('start drawing nppc images......')
#     val_images, val_ODTs, _ = dataset.get_images(1) # 验证集所有图片
#     val_images, val_ODTs = val_images, val_ODTs
#     val_gens = diffusion_trainer.gen_images[1]
#     val_num = 256
#     with torch.no_grad():
#         x_restored = val_gens[:val_num]
#         num_channel = x_restored.shape[1]
#         x_distorted = val_ODTs[:len(x_restored)]
#         w_mat = nppc_net(torch.tensor(x_restored).float().to(device), torch.tensor(x_distorted).float().to(device))
#         t_list = torch.linspace(-1, 1, 3).to(device)  # [-1, 1]范围内选3个
#         for i in tqdm(range(len(x_restored))):#遍历样本
#             imgs = t_list[:, None, None, None, None] * w_mat[i] + torch.from_numpy(x_restored[i][None][None]).float().to(device) # [sample_num, 5, 3, 20, 20] + [1, 1, 3, 20, 20] -> [sample_num, 5, 3, 20, 20] 每个样本的恢复图像
#             w = w_mat[i].unsqueeze(0) # [1, 5, 3, 20, 20]
#             imgs = torch.cat([w, imgs], axis = 0) # sample + 1
#             #sample 维度加1, t_list长度+1
#             imgs = imgs.transpose(0, 1).contiguous()# [n_dirs, sample_num + 1, channel, 20, 20]
#             #show imgs
#             # plt.figure(figsize=(num_channel * 5, (1 + len(t_list) * args.n_dirs) * 5))
#             plt.figure(figsize=(num_channel * 5, (1 + (len(t_list) + 1) * args.n_dirs) * 5))
#             for c in range(1, num_channel + 1): # channel
#                 plt.subplot(1 + args.n_dirs * (len(t_list) + 1), num_channel, c)  # 对于一个样本，长度上，对于一个主成分方向，有3行，一共5个主成分方向。nrows有16个，ncols有3个
#                 plt.title(f'Generated channel with nppc {c}')
#                 plt.imshow(val_images[i][c-1])
#             for dir in range(1, args.n_dirs + 1): # 从第 0 + 1个方向开始
#                 for dir_sample in range(1,  (len(t_list) + 1) + 1):
#                     for sample_c in range(1, num_channel + 1): # 先遍历channel
#                         image = imgs[dir-1][dir_sample-1][sample_c-1].detach().cpu().numpy()
#                         plt.subplot(1 + args.n_dirs * (len(t_list) + 1), num_channel, num_channel + sample_c + (dir - 1) * (len(t_list)) * num_channel + (dir_sample - 1) * num_channel)  # 对于一个样本，长度上，对于一个主成分方向，有3行，一共5个主成分方向。nrows有16个，ncols有3个
#                         plt.title(f'c{sample_c}-{dir}-th-dir-{dir_sample}-th-s')
#                         plt.imshow(image)
#             save_path = '/data/XinZhi/ODUQ/DOT/data/' + 'nppc_images/' + f'{args.gen_image_name}/'
#             dir_check(save_path)
#             plt.savefig(save_path + f'{dataset.name}_with_w_EpoDenoiser{args.loadepoch}_%03d.png' % i, bbox_inches='tight')
#             plt.close()

