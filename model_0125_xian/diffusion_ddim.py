import torch
from torch.nn import functional as F
from tqdm import tqdm
import torch.distributions as dist


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class DiffusionProcess:
    """
    Utilities used for the diffusion process.
    """

    def __init__(self, T, schedule_name):
        """
        :param T: the maximum timestamps of diffusion process.
        :param schedule_name: name of the diffusion schedule, either 'cosine', 'quadratic' or 'sigmoid'.
        """
        if schedule_name == 'cosine':
            self.schedule_func = cosine_beta_schedule
        elif schedule_name == 'quadratic':
            self.schedule_func = quadratic_beta_schedule
        elif schedule_name == 'sigmoid':
            self.schedule_func = sigmoid_beta_schedule
        elif schedule_name == 'linear':
            self.schedule_func = linear_beta_schedule
        else:
            raise NotImplementedError('Diffusion schedule ' + schedule_name + ' not implemented.')

        self.betas = self.schedule_func(T)
        self.T = T

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=T)

        # define alphas
        alphas = 1. - self.betas
        # α的累积乘积
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # 根号下1/α
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 根号下α的累积乘积
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        # 根号下1-α的累积乘积
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0) 方差
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def extract(self, a, t, x_shape):
        # 取出batch_size
        batch_size = t.shape[0]
        #
        out = a.gather(0, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def ddim_sample_sigmas(self, eta=0.0, sqrt_one_minus_alphas_cumprod_t_min_1=0.0,
                           sqrt_one_minus_alphas_cumprod_t=0.0, sqrt_alphas_cumprod_t_min_1=0.0):
        sigmas = eta * (sqrt_one_minus_alphas_cumprod_t_min_1 / sqrt_one_minus_alphas_cumprod_t) * (
                sqrt_one_minus_alphas_cumprod_t / sqrt_alphas_cumprod_t_min_1)
        return sigmas

    # 计算Xt diffuse过程
    def q_sample(self, x_start, t, noise=None):
        """
        Calculate the noisy data q(x_t|x_0).

        :param x_start: the noise free, original data from step 0, aka x_0.
        :param t: timestamps of the forward diffusion process.
        :param noise: given noise. If None, randomly samples from a standard Guassian.
        :return:
        """
        if noise is None:
            # 生成标准正态噪声，也就是ε
            noise = torch.randn_like(x_start)

        # 取出了所有t对应的根号下α的累积乘积
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        # 取出了所有t对应的根号下1-α的累积乘积
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # 这里return的是Xt，也就是timestamp=t时的实际图片
        # (batch_size,1,1,1)*(batch_size,3,20,20)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # diffuse过程，计算实际的噪声和Unet预测的噪声之间的loss
    def p_losses(self, denoise_model, x_start, t, y=None, noise=None, loss_type="l2"):
        """
        Calculate backward diffusion loss values for given diffusion timestamps.

        :param denoise_model: model for denoising noisy data, or as we know, predict the noise.
        :param x_start: the noise free, original data from step 0, aka x_0.
        :param t: timestamps of the backward diffusion process.
        :param noise: given noise. If None, randomly samples from a standard Guassian.
        :param loss_type: type of the loss function. Can be 'l1', 'l2' or 'huber'.
        :return:
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 形状是(batch_size,3,20,20)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, y)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    # denoise 估计mean过程
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, y=None):
        """
        Calculate backward diffusion process p(x_t-1|x_t).
        """
        # t的形状是(batch_size)
        # 取出betas_t
        # 形状是(batch_size,1,1,1)
        betas_t = self.extract(self.betas, t, x.shape)
        # 取出根号下1-α连乘
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        # 取出根号下1/α
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # p(x_t-1|x_t)的均值
        # model(x, t, y)输出的是ε，也就是预测的噪声
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t, y) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            normal_dist = dist.Normal(model_mean, posterior_variance_t)
            noise = torch.randn_like(x)
            sample_value = model_mean + torch.sqrt(posterior_variance_t) * noise
            log_prob = normal_dist.log_prob(sample_value).exp()
            if t_index == 1:
                print(log_prob)
            return sample_value

    @torch.no_grad()
    def p_sample_loop(self, model, shape, y=None, display=False):
        """
        Finish the full process of backward diffusion, from pure noise x_T, to the noise-free data x_0.
        """
        device = next(model.parameters()).device

        # b是batch_size
        b = shape[0]
        # start from pure noise (for each example in the batch)
        # img初始是高斯纯噪声，形状和shape一样
        img = torch.randn(shape, device=device)
        imgs = []

        iter = list(reversed(range(20, self.T, 50)))
        if display:
            iter = tqdm(iter, desc='Generating images through ddim p-sample')
        for i in iter:
            img = self.p_sample_ddim(model, img, torch.full((b,), i, device=device, dtype=torch.long),
                                     torch.full((b,), i - 20, device=device, dtype=torch.long), i, y=y)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, y=None, batch_size=16):
        return self.p_sample_loop(model, shape=(batch_size, image_size, image_size), y=y)

    @torch.no_grad()
    def p_sample_ddim(self, model, x, t, t_min1, t_index, y=None):
        # 取出根号下1-αt连乘
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        # 取出根号下αt的连乘
        sqrt_alphas_cumprod_t = self.extract(
            self.sqrt_alphas_cumprod, t, x.shape
        )

        # 取出根号下1-αt-1连乘
        sqrt_one_minus_alphas_cumprod_t_min_1 = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t_min1, x.shape
        )

        # 取出根号下αt-1的连乘
        sqrt_alphas_cumprod_t_min_1 = self.extract(
            self.sqrt_alphas_cumprod, t_min1, x.shape
        )

        model_mean = sqrt_alphas_cumprod_t_min_1 * ((x - sqrt_one_minus_alphas_cumprod_t * model(x, t,
                                                                                                 y)) / sqrt_alphas_cumprod_t) + sqrt_one_minus_alphas_cumprod_t_min_1 * model(
            x, t, y)
        sigma = self.ddim_sample_sigmas(0.0, sqrt_one_minus_alphas_cumprod_t_min_1,
                                        sqrt_one_minus_alphas_cumprod_t, sqrt_alphas_cumprod_t_min_1)

        noise = torch.randn_like(x)

        return model_mean + sigma * noise


'''
    @torch.no_grad()
    def p_sample_loop(self, model, shape, y=None, display=False):
        """
        Finish the full process of backward diffusion, from pure noise x_T, to the noise-free data x_0.
        """
        device = next(model.parameters()).device

        # b是batch_size
        b = shape[0]
        # start from pure noise (for each example in the batch)
        # img初始是高斯纯噪声，形状和shape一样
        img = torch.randn(shape, device=device)
        imgs = []

        iter = list(reversed(range(0, self.T)))
        if display:
            iter = tqdm(iter, desc='Generating images through p-sample')
        for i in iter:
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, y=y)
            imgs.append(img.cpu().numpy())
        return imgs
'''
