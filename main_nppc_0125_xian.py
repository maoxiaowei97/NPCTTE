"""
2023.1.19
2018.11.1-11.10
nppc
"""
import os
from argparse import ArgumentParser

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from model_0125_xian.NppcNet import nppc_UNet, PCWrapper
from model_0115.NppcNet_Unet import Nppc_Unet as big_nppc_UNet
from dataet_statod_0124 import TrajectoryDataset
from model_0125_xian.trainer import DiffusionTrainer, UQTrainer, NppcTrainer
from model_0125_xian.diffusion_ddim import DiffusionProcess
from model_0125_xian.predictor import *
from model_0125_xian.eta_trainer import ETATrainer
from model_0125_xian.denoiser import Unet
import plotly.express as px

parser = ArgumentParser()
parser.add_argument('--cuda', help='the index of the cuda device', type=int, default=1)
# Dataset arguments
parser.add_argument('-n', '--name', help='the name of the dataset', type=str, default='1101_1116_15s_0125_3channel_S20_xian')
parser.add_argument('--gen_image_name', help=' supp name of the generated images', type=str, default='01261500')
parser.add_argument('-s', '--split', help='the number of x and y split', type=int, default=20)
parser.add_argument('--flat', default=False)

# Denoiser arguments
parser.add_argument('--denoiser', help='name of the denoiser', type=str, default='unet')
parser.add_argument('--condition', type=str, default='odt')

# Training arguments
parser.add_argument('-e', '--epoch', help='number of training epoches', type=int, default=201)
parser.add_argument('-ea', '--early_stop', help='number of early stop epoches', type=int, default=20)
parser.add_argument('-b', '--batch', help='batch size', type=int, default=512)
parser.add_argument('-t', '--timestep', help='the total timesteps of diffusion process', type=int, default=1000)
parser.add_argument('-p', '--partial', default=1.0, type=float)
parser.add_argument('--n_dirs', help='num of nppc', default=5)
parser.add_argument('--nppc_step', help='npcc step', default=0)
parser.add_argument('--second_moment_loss_grace', help='second_moment_loss_grace', default=500)

# Diffusion training arguments
parser.add_argument('--traindiff', help='whether to train the diffusion model', default=False)
parser.add_argument('--loaddiff', help='whether to load the diffusion model from cached file',default=False)
parser.add_argument('--loadnppc', help='whether to load the nppc model from cached file',default=True)
parser.add_argument('--trainnppc', help='whether to train the nppc model', default=False)
parser.add_argument('--trainuq', help='whether to train the uq model', default=False)
parser.add_argument('--trainepoch_denoiser', help='the training epoch to load from, set to -1 for loading from the final model', default=201)
parser.add_argument('--loadepoch_denoiser', help='load the denoiser best in the epoch of', default=-1)
parser.add_argument('--loadepoch_nppc', help='the training epoch to load from, set to -1 for loading from the final model', default=172)
# parser.add_argument('--loadepoch_nppc', help='the training epoch to load from, set to -1 for loading from the final model', default=91)
parser.add_argument('--loadgen', help='whether to load generated images', default=True)
parser.add_argument('--loadnppcgen', help='whether to load generated nppc images', default=False)
parser.add_argument('--uq_estimate', help='whether estimate uq', default=False)
parser.add_argument('--uq_num', help='num of uncertainty quantification', default=3)
parser.add_argument('--ddim', help='whether ddim', default=False)

parser.add_argument('--draw', help='whether to draw generated images', default=False)
parser.add_argument('--draw_nppc', help='whether to draw generated nppc images', default=False)
parser.add_argument('--numimage', help='number of images to draw', type=int, default=100)
parser.add_argument('--test', help='whether test', type=str, default=False)

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
dataset = TrajectoryDataset(args.name, split=args.split, partial=args.partial, small=False, flat=args.flat, traffic_states=False, is_test=args.test, get_dataframe=True)
dataset.load_images()

# Create models.
denoiser = Unet(dim=args.split, channels=dataset.num_channel, init_dim=4, dim_mults=(1, 2, 4), condition=args.condition, traffic_state = False)
nppc_unet = PCWrapper(nppc_UNet(in_channels=3 + 3, out_channels=3 * args.n_dirs), n_dirs=args.n_dirs)
diffusion = DiffusionProcess(T=args.timestep, schedule_name='linear')
diffusion_trainer = DiffusionTrainer(diffusion=diffusion, denoiser=denoiser, dataset=dataset, lr=1e-3,
                                     batch_size=args.batch, loss_type='huber', device=device, num_epoch=args.epoch, traffic_state=True,
                                     early_stop = args.early_stop, denoiser_epoch=args.trainepoch_denoiser, is_test=args.test, ddim=args.ddim)

if args.loaddiff:
    denoiser = diffusion_trainer.load_model(None if args.loadepoch_denoiser == -1 else args.loadepoch_denoiser)
    print('denoiser loaded...')
elif args.traindiff:
    denoiser = diffusion_trainer.train()


if args.loadnppc:
    #load训练好的nppc时不需要gen_images
    denoiser = diffusion_trainer.load_model(None if args.loadepoch_denoiser == -1 else args.loadepoch_denoiser)
    diffusion_trainer.load_generation()
    nppc_trainer = NppcTrainer(diffusion=diffusion, denoiser=denoiser, gen_images=diffusion_trainer.gen_images, nppc_net=nppc_unet, dataset=dataset, lr=1e-3,
                               batch_size=args.batch, device=device, num_epoch=args.epoch, nppc_step=args.nppc_step, load_trained_nppc_epoch = args.loadepoch_nppc,
                               second_moment_loss_grace=args.second_moment_loss_grace, early_stopping = args.early_stop, is_test=args.test, ddim=args.ddim)
    nppc_net = nppc_trainer.load_model(None if args.loadepoch_nppc == -1 else args.loadepoch_nppc)
    indices_to_gen = [0, 1, 2]
    nppc_trainer.save_generation(indices_to_gen)
    print('nppc gen saved')

if args.trainnppc:
    if args.loadgen:
        diffusion_trainer.load_generation()
    else:
        print('begin to generate images by denoiser...')
        indices_to_gen = [0, 1, 2]
        if len(indices_to_gen) > 0:
            diffusion_trainer.save_generation(indices_to_gen)
            print('denoiser generated images, conplete...')

    nppc_trainer = NppcTrainer(diffusion=diffusion, denoiser=denoiser, gen_images=diffusion_trainer.gen_images, nppc_net = nppc_unet, dataset=dataset, lr=1e-3,
                                     batch_size=args.batch, device=device, num_epoch=args.epoch, load_trained_nppc_epoch = args.loadepoch_nppc,
                               nppc_step=args.nppc_step, second_moment_loss_grace=args.second_moment_loss_grace, early_stopping= args.early_stop, is_test = args.test, ddim=args.ddim)
    print('start nppc training...')
    nppc_net = nppc_trainer.train()


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

#先把sigma加上
if args.draw_nppc:
    # load denoiser and nppc
    print('start drawing nppc images......')
    denoiser = diffusion_trainer.load_model(None if args.loadepoch_denoiser == -1 else args.loadepoch_denoiser)
    diffusion_trainer.load_generation()
    nppc_trainer = NppcTrainer(diffusion=diffusion, denoiser=denoiser, gen_images=diffusion_trainer.gen_images, nppc_net=nppc_unet, dataset=dataset, lr=1e-3,
                               batch_size=args.batch, device=device, num_epoch=args.epoch, nppc_step=args.nppc_step, load_trained_nppc_epoch = args.loadepoch_nppc,
                               second_moment_loss_grace=args.second_moment_loss_grace, early_stopping= args.early_stop, is_test=args.test, ddim=args.ddim)
    nppc_net = nppc_trainer.load_model(epoch=args.loadepoch_nppc)
    val_images, val_ODTs, _ = dataset.get_images(1) # 验证集所有图片
    val_images, val_ODTs = val_images, val_ODTs
    val_gens = diffusion_trainer.gen_images[1]
    val_num = 2048
    with torch.no_grad():
        x_restored = val_gens[:val_num]
        num_channel = x_restored.shape[1]
        x_distorted = val_ODTs[:len(x_restored)]
        w_mat = nppc_net(torch.tensor(x_restored).float().to(device), torch.tensor(x_distorted).float().to(device))
        w_mat_ = w_mat.flatten(2)
        w_norms = w_mat_.norm(dim=2) # [b, 5]
        w_hat_mat = w_mat_ / w_norms[:, :, None] # [b, 5, 3*20*20]
        sigma = torch.sqrt(w_norms.pow(2)) # 根据代码得到sigma = w_norm, [b, 5]

        t_list = torch.tensor([-2.5, -2, -1.5, -1, 0]).float().to(device)  # [-1, 1]范围内选3个
        for i in tqdm(range(len(x_restored))):  # 遍历样本
            # imgs = t_list[:, None, None, None, None] * w_mat[i] + torch.from_numpy(x_restored[i][None][None]).float().to(device)  # [sample_num, 5, 3, 20, 20] + [1, 1, 3, 20, 20] -> [sample_num, 5, 3, 20, 20] 每个样本的恢复图像
            # w = w_mat[i].unsqueeze(0)  # [1, 5, 3, 20, 20]
            imgs = t_list[:, None, None, None, None] * w_hat_mat.reshape(-1, args.n_dirs, 3, args.split, args.split)[i] * sigma[i][None, :, None, None, None] + torch.from_numpy( x_restored[i][None][None]).float().to(device)
            w = w_hat_mat[i].reshape(1, args.n_dirs, 3, args.split, args.split)

            imgs = torch.cat([w, imgs], axis=0)  # sample + 1
            # sample 维度加1, t_list长度+1
            imgs = imgs.transpose(0, 1).contiguous()  # [n_dirs, sample_num + 1, channel, 20, 20]
            # show imgs
            plt.figure(figsize=(num_channel * 5, (1 + (len(t_list) + 1) * args.n_dirs) * 5))
            for c in range(1, num_channel + 1):  # channel
                plt.subplot(1 + args.n_dirs * (len(t_list) + 1), num_channel, c)  # 对于一个样本，长度上，对于一个主成分方向，有3行，一共5个主成分方向。nrows有16个，ncols有3个
                plt.title(f'Real Images in channel {c}')
                plt.imshow(val_images[i][c - 1])
            for dir in range(1, args.n_dirs + 1):  # 从第 0 + 1个方向开始
                for dir_sample in range(1, (len(t_list) + 1) + 1):
                    for sample_c in range(1, num_channel + 1):  # 先遍历channel
                        image = imgs[dir - 1][dir_sample - 1][sample_c - 1].detach().cpu().numpy()
                        plt.subplot(1 + args.n_dirs * (len(t_list) + 1), num_channel,
                                    num_channel + sample_c + (dir - 1) * (len(t_list) + 1) * num_channel + (dir_sample - 1) * num_channel)  # 对于一个样本，长度上，对于一个主成分方向，有3行，一共5个主成分方向。nrows有16个，ncols有3个
                        plt.title(f'c{sample_c}-{dir}-th-dir-{dir_sample}-th-s')
                        plt.imshow(image)
            save_path = '/data/XinZhi/ODUQ/DOT/data/' + 'nppc_images/' + f'{args.gen_image_name}/'
            dir_check(save_path)
            plt.savefig(save_path + f'{dataset.name}_%03d.png' % i, bbox_inches='tight')
            plt.close()


