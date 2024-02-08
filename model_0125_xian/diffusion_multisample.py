import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
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
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


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
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """
        Calculate the noisy data q(x_t|x_0).

        :param x_start: the noise free, original data from step 0, aka x_0.
        :param t: timestamps of the forward diffusion process.
        :param noise: given noise. If None, randomly samples from a standard Guassian.
        :return:
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise #[B, C, H, W]

    def p_losses(self, denoise_model, x_start, t, y=None, noise=None, loss_type="l1"):
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

    @torch.no_grad()
    def p_sample(self, shape, model, x, t, t_index, sample_num, y=None):
        """
        Calculate backward diffusion process p(x_t-1|x_t).
        p(x_t-1|x_t) = N(x_t-1, u, sigma)
        """
        betas_t = self.extract(self.betas, t, shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, shape)

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t, y) / sqrt_one_minus_alphas_cumprod_t
        ) # sample_num * b, c, h, w

        # model_mean_for_distribution = model_mean #[b * s, c, h, w]
        # posterior_variance_t_for_distribution = torch.sqrt(self.extract(self.posterior_variance, t, x.shape).reshape(x.shape[0], 1)) * torch.ones([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]]).to(x.device)
        # covariance_matrix = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        # covariance_matrix.diagonal(dim1=-2, dim2=-1).copy_(posterior_variance_t_for_distribution)
        # mv_normal = MultivariateNormal(model_mean_for_distribution.reshape(x.shape[0], -1), covariance_matrix)
        # model_mean = model_mean.unsqueeze(0).repeat(sample_num, 1, 1, 1, 1)

        if t_index == 0:
            # b_lop_prob = mv_normal.log_prob((model_mean).reshape(x.shape[0], -1))
            # b_prob = torch.exp(b_lop_prob) #[s,b]
            # return model_mean, b_prob.reshape(shape.shape[0], sample_num)
            return model_mean.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape) #[b,1,1,1]
            # noise = torch.randn([sample_num, x.shape[0], x.shape[1],  x.shape[2],  x.shape[3]]).to(x.device) # noise可以有多个，多次采样得到去噪后的x_{t-1} [s,b,c*h*w]
            noise = torch.randn([x.shape[0], x.shape[1], x.shape[2], x.shape[3]]).to(x.device)  # noise可以有多个，多次采样得到去噪后的x_{t-1} [b*c,c*h*w]

            # b_lop_prob = mv_normal.log_prob((model_mean + torch.sqrt(posterior_variance_t).unsqueeze(0) * noise).reshape(sample_num, x.shape[0], -1))
            # b_lop_prob = mv_normal.log_prob((model_mean + torch.sqrt(posterior_variance_t) * noise).reshape(x.shape[0], -1))
            # b_prob = torch.exp(b_lop_prob)
            # probs = torch.exp(b_prob) #[sample_num * batch_size]
            # return model_mean + torch.sqrt(posterior_variance_t) * noise, torch.softmax(b_prob.reshape(shape[0], sample_num), dim=-1) #[b*s, c, h, w], [b, s], 每一步根据采样概率调整三条路线的权重
            return (model_mean + torch.sqrt(posterior_variance_t) * noise).reshape(shape[0] * sample_num, shape[1], shape[2], shape[3])
    @torch.no_grad()
    def p_sample_loop(self, model, sample_num, shape, y=None, display=False):
        """
        Finish the full process of backward diffusion, from pure noise x_T, to the noise-free data x_0.
        """
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        # img = torch.randn(shape, device=device).unsqueeze(1).repeat(1, sample_num, 1, 1, 1).reshape(shape[0] * sample_num, shape[1], shape[2], shape[3]) #[b*s, c, h, w]
        img = torch.randn([shape[0] * sample_num, shape[1], shape[2], shape[3]], device=device)
        imgs = []
        probs_iter = []

        iter = list(reversed(range(0, self.T)))
        if display:
            iter = tqdm(iter, desc='Generating images through p-sample')
        for i in tqdm(iter): # 逐步去噪，从第1000(T)步开始去噪。从一开始维护三条不同路线。每次从一个odt输入得到一个分布，每次从分布取出sample个值，下一次就得到三个分布。概率如何得到，从三个不同的分布采样得到不同的路线，再归一化
            # img, probs = self.p_sample(shape, model, img, torch.full((b*sample_num,), i, device=device, dtype=torch.long), i, sample_num, y=(y.unsqueeze(1).repeat(1, sample_num, 1).reshape(shape[0]*sample_num, -1))) # img: [s,b,c,h,w], probs: [s,b]
            # img, probs = img.permute(1, 0, 2, 3, 4).contiguous(), probs.permute(1, 0).contiguous()
            img = self.p_sample(shape, model, img,
                                       torch.full((b * sample_num,), i, device=device, dtype=torch.long), i, sample_num,
                                       y=(y.unsqueeze(1).repeat(1, sample_num, 1).reshape(shape[0] * sample_num, -1)))  # img: [s,b,c,h,w], probs: [s,b]
            imgs.append(img.reshape(b, sample_num, shape[1], shape[2], shape[3]).cpu().numpy())
            # probs_iter.append(probs)
        # return imgs, probs_iter
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, y=None, batch_size=16):
        return self.p_sample_loop(model, shape=(batch_size, image_size, image_size), y=y)