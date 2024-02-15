import os
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from model.NpcNet import npc_UNet, PCWrapper
from dataset import TrajectoryDataset
from model.trainer import DiffusionTrainer, UQTrainer, NpcTrainer
from model.diffusion import DiffusionProcess
from model.denoiser import Unet
from model.predictor import ETA_Unet
import plotly.express as px

parser = ArgumentParser()
parser.add_argument('--cuda', help='the index of the cuda device', type=int, default=1)
# Dataset arguments
parser.add_argument('-n', '--name', help='the name of the dataset', type=str, default='sample_0215')
parser.add_argument('--gen_image_name', help=' supp name of the generated images', type=str, default='01261500')
parser.add_argument('-s', '--split', help='the number of x and y split', type=int, default=20)
parser.add_argument('--flat', default=False)

# Denoiser arguments
parser.add_argument('--denoiser', help='name of the denoiser', type=str, default='unet')
parser.add_argument('--condition', type=str, default='odt')

# Training arguments
parser.add_argument('-e', '--epoch', help='number of training epoches', type=int, default=101)
parser.add_argument('-ea', '--early_stop', help='number of early stop epoches', type=int, default=20)
parser.add_argument('-b', '--batch', help='batch size', type=int, default=512)
parser.add_argument('-t', '--timestep', help='the total timesteps of diffusion process', type=int, default=1000)
parser.add_argument('-p', '--partial', default=1.0, type=float)
parser.add_argument('--n_dirs', help='num of npc', default=5)
parser.add_argument('--npc_step', help='npcc step', default=0)
parser.add_argument('--second_moment_loss_grace', help='second_moment_loss_grace', default=500)

# Diffusion training arguments
parser.add_argument('--traindiff', help='whether to train the diffusion model', default=True)
parser.add_argument('--loaddiff', help='whether to load the diffusion model from cached file',default=False)
parser.add_argument('--loadnpc', help='whether to load the npc model from cached file',default=False)
parser.add_argument('--trainnpc', help='whether to train the npc model', default=True)
parser.add_argument('--trainuq', help='whether to train the uq model', default=True)
parser.add_argument('--trainepoch_denoiser', help='the training epoch to load from, set to -1 for loading from the final model', default=101)
parser.add_argument('--loadepoch_denoiser', help='load the denoiser best in the epoch of', default=-1)
parser.add_argument('--loadepoch_npc', help='the training epoch to load from, set to -1 for loading from the final model', default=-1)
parser.add_argument('--loadgen', help='whether to load generated images', default=False)
parser.add_argument('--loadnpcgen', help='whether to load generated npc images', default=False)
parser.add_argument('--uq_estimate', help='whether estimate uq', default=False)
parser.add_argument('--uq_num', help='num of uncertainty quantification', default=3)
parser.add_argument('--ddim', help='whether ddim', default=False)

parser.add_argument('--draw', help='whether to draw generated images', default=False)
parser.add_argument('--draw_npc', help='whether to draw generated npc images', default=False)
parser.add_argument('--numimage', help='number of images to draw', type=int, default=100)
parser.add_argument('--test', help='whether test', type=str, default=False)

# ETA training arguments
parser.add_argument('--traineta', help='whether to train the ETA prediction model', default=False)
parser.add_argument('--predictor', help='name of the predictor', type=str, default='trans')
parser.add_argument('--predict_type', help='only_mean, only_npc, mean_npc', type=str, default='mean_npc')
parser.add_argument('--predict_task', help='mis, mae, mape', type=str, default='mae')
parser.add_argument('--predictorL', help='number of layers in the predictor', type=int, default=2)
parser.add_argument('--trainorigin', help='whether to use original images to train ETA predictor', default=False)
parser.add_argument('--valorigin', help='whether to use original images to evaluate ETA predictor', default=False)
parser.add_argument('--stop', help='number of early stopping epoch for ETA training', type=int, default=10)

# Predictor arguments
parser.add_argument('-d', '--dmodel', help='dimension of predictor models', type=int, default=128)
parser.add_argument('--rmst',  help='whether to train the ETA prediction model with image information', default=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

cur_path = os.path.abspath(__file__)
ws = os.path.dirname(cur_path)
print('ws: ', ws)
def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

# Loading dataset
dataset = TrajectoryDataset(args.name, split=args.split, partial=args.partial, small=False, flat=args.flat, is_test=args.test, get_dataframe=True)
dataset.dump_images_state()
dataset.load_images()

# Create models.
denoiser = Unet(dim=args.split, channels=dataset.num_channel, init_dim=4, dim_mults=(1, 2, 4), condition=args.condition)
npc_unet = PCWrapper(npc_UNet(in_channels=3 + 3, out_channels=3 * args.n_dirs), n_dirs=args.n_dirs)
diffusion = DiffusionProcess(T=args.timestep, schedule_name='linear')
diffusion_trainer = DiffusionTrainer(diffusion=diffusion, denoiser=denoiser, dataset=dataset, lr=1e-3,
                                     batch_size=args.batch, loss_type='huber', device=device, num_epoch=args.epoch,
                                     early_stop = args.early_stop, denoiser_epoch=args.trainepoch_denoiser, is_test=args.test, ddim=args.ddim)

if args.loaddiff:
    denoiser = diffusion_trainer.load_model(None if args.loadepoch_denoiser == -1 else args.loadepoch_denoiser)
    print('denoiser loaded...')
elif args.traindiff:
    denoiser = diffusion_trainer.train()

if args.loadnpc:
    denoiser = diffusion_trainer.load_model(None if args.loadepoch_denoiser == -1 else args.loadepoch_denoiser)
    diffusion_trainer.load_generation()
    npc_trainer = NpcTrainer(diffusion=diffusion, denoiser=denoiser, gen_images=diffusion_trainer.gen_images, npc_net=npc_unet, dataset=dataset, lr=1e-3,
                               batch_size=args.batch, device=device, num_epoch=args.epoch, npc_step=args.npc_step, load_trained_npc_epoch = args.loadepoch_npc,
                               second_moment_loss_grace=args.second_moment_loss_grace, early_stopping = args.early_stop, is_test=args.test, ddim=args.ddim)
    npc_net = npc_trainer.load_model(None if args.loadepoch_npc == -1 else args.loadepoch_npc)
    indices_to_gen = [0, 1, 2]
    npc_trainer.save_generation(indices_to_gen)
    print('npc gen saved')

if args.trainnpc:
    if args.loadgen:
        diffusion_trainer.load_generation()
    else:
        print('begin to generate images by denoiser...')
        indices_to_gen = [0, 1, 2]
        if len(indices_to_gen) > 0:
            diffusion_trainer.save_generation(indices_to_gen)
            print('denoiser generated images, conplete...')

    npc_trainer = NpcTrainer(diffusion=diffusion, denoiser=denoiser, gen_images=diffusion_trainer.gen_images, npc_net = npc_unet, dataset=dataset, lr=1e-3,
                                     batch_size=args.batch, device=device, num_epoch=args.epoch, load_trained_npc_epoch = args.loadepoch_npc,
                               npc_step=args.npc_step, second_moment_loss_grace=args.second_moment_loss_grace, early_stopping= args.early_stop, is_test = args.test, ddim=args.ddim)
    print('start npc training...')
    npc_net = npc_trainer.train()
    print('npc trained')
    indices_to_gen = [0, 1, 2]
    npc_trainer.save_generation(indices_to_gen)
    print('nppc gen saved')


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

# Train UQ predictor.
if args.trainuq:
    npc_trainer.load_generation()
    gen_images = npc_trainer.gen_images
    print('npc images generated')
    predictor = ETA_Unet(dim=args.split, channels=1, init_dim=4, dim_mults=(1, 2, 4), condition=args.condition,
                         split=args.split)

    uq_trainer = UQTrainer(diffusion=diffusion, predictor=predictor, dataset=dataset,
                            gen_images=gen_images, predict_type=args.predict_type,
                            predict_task=args.predict_task,
                            lr=1e-3, batch_size=args.batch, num_epoch=60, device=device,
                            early_stopping=args.stop,
                            train_origin=args.trainorigin, val_origin=args.valorigin, is_test = args.test, ddim=args.ddim)
    uq_trainer.train_val_test()
    print('done')

if args.draw_npc:
    # load denoiser and npc
    print('start drawing npc images......')
    denoiser = diffusion_trainer.load_model(None if args.loadepoch_denoiser == -1 else args.loadepoch_denoiser)
    diffusion_trainer.load_generation()
    npc_trainer = NpcTrainer(diffusion=diffusion, denoiser=denoiser, gen_images=diffusion_trainer.gen_images, npc_net=npc_unet, dataset=dataset, lr=1e-3,
                               batch_size=args.batch, device=device, num_epoch=args.epoch, npc_step=args.npc_step, load_trained_npc_epoch = args.loadepoch_npc,
                               second_moment_loss_grace=args.second_moment_loss_grace, early_stopping= args.early_stop, is_test=args.test, ddim=args.ddim)
    npc_net = npc_trainer.load_model(epoch=args.loadepoch_npc)
    val_images, val_ODTs, _ = dataset.get_images(1)
    val_images, val_ODTs = val_images, val_ODTs
    val_gens = diffusion_trainer.gen_images[1]
    val_num = 2048
    with torch.no_grad():
        x_restored = val_gens[:val_num]
        num_channel = x_restored.shape[1]
        x_distorted = val_ODTs[:len(x_restored)]
        w_mat = npc_net(torch.tensor(x_restored).float().to(device), torch.tensor(x_distorted).float().to(device))
        w_mat_ = w_mat.flatten(2)
        w_norms = w_mat_.norm(dim=2)
        w_hat_mat = w_mat_ / w_norms[:, :, None]
        sigma = torch.sqrt(w_norms.pow(2))

        t_list = torch.tensor([-2.5, -2, -1.5, -1, 0]).float().to(device)
        for i in tqdm(range(len(x_restored))):

            imgs = t_list[:, None, None, None, None] * w_hat_mat.reshape(-1, args.n_dirs, 3, args.split, args.split)[i] * sigma[i][None, :, None, None, None] + torch.from_numpy( x_restored[i][None][None]).float().to(device)
            w = w_hat_mat[i].reshape(1, args.n_dirs, 3, args.split, args.split)

            imgs = torch.cat([w, imgs], axis=0)  # sample + 1
            imgs = imgs.transpose(0, 1).contiguous()  # [n_dirs, sample_num + 1, channel, L, L]
            plt.figure(figsize=(num_channel * 5, (1 + (len(t_list) + 1) * args.n_dirs) * 5))
            for c in range(1, num_channel + 1):  # channel
                plt.subplot(1 + args.n_dirs * (len(t_list) + 1), num_channel, c)
                plt.title(f'Real Images in channel {c}')
                plt.imshow(val_images[i][c - 1])
            for dir in range(1, args.n_dirs + 1):
                for dir_sample in range(1, (len(t_list) + 1) + 1):
                    for sample_c in range(1, num_channel + 1):
                        image = imgs[dir - 1][dir_sample - 1][sample_c - 1].detach().cpu().numpy()
                        plt.subplot(1 + args.n_dirs * (len(t_list) + 1), num_channel,
                                    num_channel + sample_c + (dir - 1) * (len(t_list) + 1) * num_channel + (dir_sample - 1) * num_channel)
                        plt.title(f'c{sample_c}-{dir}-th-dir-{dir_sample}-th-s')
                        plt.imshow(image)
            save_path = ws + '/data/npc_images/' + f'{args.gen_image_name}/'
            dir_check(save_path)
            plt.savefig(save_path + f'{dataset.name}_%03d.png' % i, bbox_inches='tight')
            plt.close()


