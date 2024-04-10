import os
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd

from model.diffusion import DiffusionProcess

cur_path = os.path.abspath(__file__)
path = os.path.dirname(cur_path)
ws = os.path.dirname(path)

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

def whether_stop(metric_lst=[], n=2, mode='maximize'):
    '''
    For fast parameter search, judge wether to stop the training process according to metric score
    n: Stop training for n consecutive times without rising
    mode: maximize / minimize
    '''
    if len(metric_lst) < 1: return False  # at least have 2 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx, v in enumerate(metric_lst):
        if v == max_v: max_idx = idx
    return max_idx < len(metric_lst) - n

class EarlyStop():
    """
    For training process, early stop strategy
    """

    def __init__(self, mode='maximize', patience=1):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1  # the best epoch
        self.is_best_change = False  # whether the best change compare to the last epoch

    def append(self, x):
        self.metric_lst.append(x)
        # update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        # update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize' else self.metric_lst.index(
            min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch  # update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:
            return -1
        else:
            return self.metric_lst[self.best_epoch]

def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]

def mean_absolute_percentage_error(y_true, y_pred):
    non_zero_indices = np.where(y_true != 0)
    actual_non_zero = y_true[non_zero_indices]
    predicted_non_zero = y_pred[non_zero_indices]
    mape = np.mean(np.abs((actual_non_zero - predicted_non_zero) / actual_non_zero))
    return mape

class DiffusionTrainer:
    def __init__(self, diffusion: DiffusionProcess, denoiser, dataset, lr, batch_size, loss_type,
                 num_epoch, early_stop, device,  denoiser_epoch):

        self.diffusion = diffusion
        self.denoiser = denoiser.to(device)

        self.optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=lr)
        self.dataset = dataset

        self.batch_size = batch_size
        self.loss_type = loss_type
        self.num_epoch = num_epoch
        self.device = device
        self.early_stopping = early_stop
        self.denoiser_epoch = denoiser_epoch

        self.ws =ws
        dir_check(self.ws + '/data/model')

        self.save_model_path = os.path.join(self.ws, 'data', 'model',
                                            f'{self.dataset.name}_denoiser_{self.denoiser.name}_'
                                            f'T{self.diffusion.T}_'
                                            f'DenEpo{self.denoiser_epoch}_'
                                            f'S{self.dataset.split}')

        self.save_loss_path = os.path.join(self.ws, 'data', 'model',
                                            f'{self.dataset.name}_denoiser_{self.denoiser.name}_'
                                            f'T{self.diffusion.T}_'
                                            f'E{self.num_epoch}_'
                                            f'DenEpo{self.denoiser_epoch}_'
                                            f'S{self.dataset.split}_loss.npy')

        self.gen_set_path = os.path.join(self.ws, 'data',
                                f'{self.dataset.name}_images_{self.denoiser.name}_'
                                f'T{self.diffusion.T}_'
                                f'sampleDate_0410_'
                                   f'DenEpo{self.denoiser_epoch}_'
                                f'S{self.dataset.split}.npz')

        self.traffic_condition_path = ws + '/data/' + self.dataset.name + f'_S{self.dataset.split}_F{self.dataset.flat}_trafficcondition.npy'
        self.traffic_condition = np.load(self.traffic_condition_path, allow_pickle = True)
        self.traffic_condition = torch.from_numpy(self.traffic_condition).float().to(self.device)

        self.gen_images = [np.array([]) for _ in range(3)]
        self.diff_loss = []

    def train_epoch(self, meta):
        self.denoiser.train()
        losses = []
        batch_iter = list(zip(*meta))
        desc_txt = 'Training diffusion, loss %05.6f'
        with tqdm(next_batch((batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size,
                  desc=desc_txt % 0.0) as pbar:

            for batch in pbar:
                self.optimizer.zero_grad()

                batch_img, batch_odt, _ = zip(*batch)

                # Create two batch tensors, with shape (N, C, X, Y) and (N, y_feat).
                batch_img, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                        for item in (batch_img, batch_odt))

                batch_D = batch_odt[:, 5].long().reshape(-1)
                batch_ts = batch_odt[:, 6].long().reshape(-1)
                batch_traffic_condition = self.traffic_condition[batch_D, batch_ts - 1].reshape(batch_img.size(0), 1, batch_img.size(2), batch_img.size(3))

                t = torch.randint(0, self.diffusion.T, (batch_img.size(0),)).long().to(self.device)

                loss = self.diffusion.p_losses(self.denoiser, batch_img, t, batch_odt, batch_traffic_condition, loss_type=self.loss_type)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                pbar.set_description(desc_txt % (loss.item()))
        return float(np.mean(losses))

    def train(self):
        train_meta = self.dataset.get_images(0)
        for epoch in range(self.num_epoch):
            print('denoising epoch: ', epoch)
            train_loss = self.train_epoch(train_meta)
            self.diff_loss.append(train_loss)
            if epoch % 10 == 0:
                self.save_model(epoch)

        self.save_model()
        return self.denoiser

    def save_model(self, epoch=None):
        save_path = self.save_model_path
        save_loss_path = self.save_loss_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        save_path += '.model'
        torch.save(self.denoiser.state_dict(), save_path)
        np.save(save_loss_path, np.array(self.diff_loss))

        print('[Saved denoiser] to ' + save_path)
        print('[Saved loss] to ' + save_loss_path)

    def load_model(self, epoch=None):
        save_path = self.save_model_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        save_path += '.model'
        self.denoiser.load_state_dict(torch.load(save_path, map_location=self.device), strict=True)
        print('[Loaded denoiser] from ' + save_path)
        return self.denoiser

    def generate(self, meta):
        self.denoiser.eval()
        batch_iter = list(zip(*meta))
        gens = []

        for batch in tqdm(next_batch(batch_iter, self.batch_size), total=len(batch_iter) // self.batch_size,
                          desc='Generating images'):
            batch_img, batch_odt, batch_arr = zip(*batch)
            batch_img, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                    for item in (batch_img, batch_odt))

            batch_D = batch_odt[:, 5].long().reshape(-1)
            batch_ts = batch_odt[:, 6].long().reshape(-1)
            batch_traffic_condition = self.traffic_condition[batch_D, batch_ts - 1].reshape(batch_img.size(0), 1,
                                                                                            batch_img.size(2),
                                                                                            batch_img.size(3))

            gen = self.diffusion.p_sample_loop(self.denoiser, shape=batch_img.shape, y=batch_odt, traffic_condition= batch_traffic_condition, display=False)[-1]
            gens.append(gen)

        return np.concatenate(gens, axis=0)

    def save_generation(self, select_sets=None):
        if select_sets is None:
            select_sets = range(3)

        for s in select_sets:
            self.gen_images[s] = self.generate(self.dataset.get_images(s))
        np.savez(self.gen_set_path, train=self.gen_images[0], val=self.gen_images[1], test=self.gen_images[2]) # 生成数值型的image
        print('[Saved generation] images to ' + self.gen_set_path)

    def save_generation_test(self, select_sets=None):
        if select_sets is None:
            select_sets = range(3)
        for s in select_sets:
            meta_images = self.dataset.get_images(s)
            if s == 0:
                meta_images = (meta_images[0][:1024], meta_images[1][:1024], meta_images[2][:1024])
            elif s == 1:
                meta_images = (meta_images[0][:512], meta_images[1][:512], meta_images[2][:512])
            elif s == 2:
                meta_images = (meta_images[0][:512], meta_images[1][:512], meta_images[2][:512])

            self.gen_images[s] = self.generate(meta_images)
        gen_path = os.path.join(self.ws, 'data',
                                f'{self.dataset.name}_images_{self.denoiser.name}_'
                                f'T{self.diffusion.T}_'
                                f'DenEpo{self.denoiser_epoch}_'
                                f'S{self.dataset.split}')
        self.gen_set_path = gen_path + '_test5k.npz'
        np.savez(self.gen_set_path, train=self.gen_images[0], val=self.gen_images[1], test=self.gen_images[2]) # 生成数值型的image
        print('[Saved generation] images to ' + self.gen_set_path)

    def load_generation(self):

        gen_images = np.load(self.gen_set_path)
        self.gen_images = [gen_images[label] for label in ['train', 'val', 'test']]
        print('[Loaded generation] from ' + self.gen_set_path)

    def load_generation_test(self):
        gen_path = os.path.join(self.ws, 'data',
                                f'{self.dataset.name}_images_{self.denoiser.name}_'
                                f'T{self.diffusion.T}_'
                                f'DenEpo{self.denoiser_epoch}_'
                                f'S{self.dataset.split}')
        self.gen_set_path = gen_path + '_test5k.npz'
        gen_images = np.load(self.gen_set_path)
        self.gen_images = [gen_images[label] for label in ['train', 'val', 'test']]
        print('[Loaded generation] from ' + self.gen_set_path)

def cal_uq_metric(label, mean, pred_upper, pred_lower, p=True, save=True, save_key='undefined'):
    rmse = math.sqrt(np.mean((label - mean) ** 2))
    mae = np.mean(np.abs(label - mean))
    mape = np.mean(np.abs(label - mean) / label)
    interval_width = np.mean(pred_upper - pred_lower)
    mis0 =np.mean(np.abs(mean - label))
    mis1 = torch.mean(torch.max(torch.tensor(pred_upper - pred_lower), torch.tensor([0.])))
    mis2 = torch.mean(torch.max(torch.tensor(pred_lower - label), torch.tensor([0.]))) * 20
    mis3 = torch.mean(torch.max(torch.tensor(label - pred_upper), torch.tensor([0.]))) * 20
    mis = mis0 + mis1.item() + mis2.item() + mis3.item()
    picp = np.sum((label > pred_lower) & (label < pred_upper) ) / len(label)
    print('rmse: ', rmse, 'mae: ', mae, 'mape: ', mape, 'interval_width: ', interval_width, 'mis: ', mis, 'picp: ', picp)
    if save == True:
        results = pd.Series([rmse, mae, mape, interval_width, mis, picp], index=['rmse', 'mae', 'mape', 'interval_width', 'mis', 'picp'])
        dir_check(ws + '/output' )
        save_path = ws + '/output/'  + save_key + '.csv'
        results.to_csv(save_path)
    return rmse, mae, mape, interval_width, mis, picp

class UQTrainer:
    """
    Trainer for the UQ predictor.
    """
    def __init__(self, diffusion, predictor, dataset, gen_images, lr, batch_size, num_epoch, device, predict_task, predict_type,
                 train_origin=False, val_origin=False, early_stopping=-1, confidence_level = None):
        self.diffusion = diffusion
        self.predictor = predictor.to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.dataset = dataset
        self.gen_images = gen_images
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.device = device

        self.train_origin = train_origin
        self.val_origin = val_origin
        self.early_stopping = early_stopping
        self.l1_loss = nn.L1Loss()
        self.predict_task = predict_task
        self.predict_type = predict_type
        self.ws = ws

        self.traffic_condition_path = ws + '/data/' + self.dataset.name + f'_S{self.dataset.split}_F{self.dataset.flat}_trafficcondition.npy'
        self.traffic_condition = np.load(self.traffic_condition_path, allow_pickle = True)
        self.traffic_condition = torch.from_numpy(self.traffic_condition).float().to(self.device)

        self.save_model_path = os.path.join('data', 'model',
                                            f'{self.dataset.name}_uq_predictor_{self.predictor.name}_'
                                            f'D{self.predictor.d_model}_'
                                            f'T{self.diffusion.T}_'
                                            f'P{self.predict_task}_'
                                            f'Type{self.predict_type}_'
                                            f'S{self.dataset.split}.model')
        self.split = self.dataset.split
        self.confidence_level = confidence_level

    def mae_mis_loss(self, pred_mean, pred_sigma, label):
        pred_upper = pred_mean + pred_sigma
        pred_lower = pred_mean - pred_sigma
        ###multi
        pred_upper = pred_upper.mean(1)
        pred_lower = pred_lower.mean(1)
        pred_mean = pred_mean.mean(1)

        loss0 = torch.mean(torch.abs(pred_mean - label))
        loss1 = torch.mean(torch.max(pred_upper - pred_lower, torch.tensor([0.]).to(pred_mean.device)))
        loss2 = torch.mean(torch.max(pred_lower - label.float(), torch.tensor([0.]).to(pred_mean.device))) * 2 / (1 - self.confidence_level)
        loss3 = torch.mean( torch.max(label.float() - pred_upper, torch.tensor([0.]).to(pred_mean.device))) * 2 / (1 - self.confidence_level)
        loss4 =  torch.mean(torch.abs(pred_mean - label) / label)

        return  loss4 + loss0 + loss1 + loss2 + loss3

    def train_epoch(self, eta_input, meta):
        self.predictor.train()
        losses = []
        origin_image, odt, label = meta
        batch_iter = list(zip(eta_input, odt, label))

        desc_txt = 'Training eta predictor, loss %05.6f'
        with tqdm(next_batch((batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size,desc=desc_txt % 0.0) as pbar:  # shuffle有问题
            for batch in pbar:
                batch_input, batch_odt, batch_label = zip(*batch)
                batch_input, batch_odt, batch_label = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                                       for item in (batch_input, batch_odt, batch_label))
                batch_D = batch_odt[:, 5].long().reshape(-1)
                batch_ts = batch_odt[:, 6].long().reshape(-1)
                batch_traffic_condition = self.traffic_condition[batch_D, batch_ts - 1].reshape(batch_input.size(0), 1, self.split, self.split)
                pre, sigma = self.predictor(x = batch_input, y = batch_odt, traffic_condition = batch_traffic_condition)
                loss = self.mae_mis_loss(pre, sigma, batch_label.reshape(-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                pbar.set_description(desc_txt % (loss.item()))
        return float(np.mean(losses))

    def eval_epoch(self, eta_input, meta, save_path, save=True):
        self.predictor.eval()
        origin_image, odt, label = meta

        batch_iter = list(zip(origin_image if self.val_origin else eta_input, odt))
        means = []
        pred_uppers = []
        pred_lowers = []

        for batch in next_batch(batch_iter, self.batch_size):
            batch_input, batch_odt = zip(*batch)
            batch_input, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                      for item in (batch_input, batch_odt))
            batch_D = batch_odt[:, 5].long().reshape(-1)
            batch_ts = batch_odt[:, 6].long().reshape(-1)
            batch_traffic_condition = self.traffic_condition[batch_D, batch_ts - 1].reshape(batch_input.size(0), 1,
                                                                                            self.split ,
                                                                                            self.split )
            mean, sigma = self.predictor(x = batch_input, y = batch_odt, traffic_condition = batch_traffic_condition)
            means.append(mean.detach().cpu().numpy().mean(1))
            pred_uppers.append((mean + sigma).detach().cpu().numpy().mean(1))
            pred_lowers.append((mean - sigma).detach().cpu().numpy().mean(1))

        means = np.concatenate(means)
        pred_uppers = np.concatenate(pred_uppers)
        pred_lowers = np.concatenate(pred_lowers)
        return cal_uq_metric(label.reshape(-1), means.reshape(-1), pred_uppers.reshape(-1), pred_lowers.reshape(-1), save=save,
                                     save_key=f'{self.predictor.name}_D{self.predictor.d_model}_'
                                              f'{self.dataset.name}_S{self.dataset.split}_P{self.dataset.partial}_task_{save_path}_T{self.predict_task}_Type{self.predict_type}')

    def train_val_test(self):
        train_gen = self.gen_images[0]
        train_meta = self.dataset.get_images(0)
        val_gen = self.gen_images[1]
        val_meta = self.dataset.get_images(1)
        test_gen = self.gen_images[2]
        test_meta = self.dataset.get_images(2)
        min_val_metric = 1e8
        epoch_before_stop = self.early_stopping

        for epoch in range(self.num_epoch):
            print('curren epoch: ', epoch)
            train_loss = self.train_epoch(train_gen, train_meta)
            rmse, val_mae, mape, interval_width, mis, picp = self.eval_epoch(val_gen, val_meta, save_path='uq', save=False)
            if self.predict_task == 'mae': # less is better
                if min_val_metric > val_mae:
                    min_val_metric = val_mae
                    epoch_before_stop = 0
                    self.save_model()
                else:
                    epoch_before_stop += 1

                if 0 < self.early_stopping <= epoch_before_stop:
                    print('\nEarly stopping, best epoch:', epoch - epoch_before_stop)
                    self.load_model(path=self.save_model_path)
                    break
            elif self.predict_task == 'mis': # less is better
                if min_val_metric > mis:
                    min_val_metric = mis
                    epoch_before_stop = 0
                    self.save_model()
                else:
                    epoch_before_stop += 1

                if 0 < self.early_stopping <= epoch_before_stop:
                    print('\nEarly stopping, best epoch:', epoch - epoch_before_stop)
                    self.load_model(path=self.save_model_path)
                    break
        rmse, val_mae, mape, interval_width, mis, picp = self.eval_epoch(test_gen, test_meta, save=True, save_path='uq')
        print('uq train val test finished...')
        return self.predictor

    def save_model(self, epoch=None):
        save_path = self.save_model_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        print('[Saved uq predictor] to ' + save_path)
        torch.save(self.predictor.state_dict(), save_path)

    def load_model(self, path, epoch=None):
        save_path = path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        print('[Loaded predictor] from ' + save_path)
        self.predictor.load_state_dict(torch.load(save_path, map_location=self.device), strict=True)
        self.predictor.eval()
        return self.predictor


class NpcTrainer:
    """
    Trainer for the npc model.
    """

    def __init__(self, diffusion: DiffusionProcess, denoiser, gen_images, npc_net, dataset, lr, batch_size,
                 num_epoch, npc_step, second_moment_loss_grace, early_stopping,  device, load_trained_npc_epoch):

        self.diffusion = diffusion
        self.denoiser = denoiser.to(device)
        self.gen_images = gen_images
        self.npc_net = npc_net.to(device)

        self.npc_optimizer = torch.optim.Adam(self.npc_net.parameters(), lr=lr)
        self.dataset = dataset

        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.device = device
        self.npc_step = npc_step
        self.second_moment_loss_grace = second_moment_loss_grace
        self.load_trained_npc_epoch = load_trained_npc_epoch
        self.sample_num = 13
        self.ws = ws
        dir_check(ws + '/data/npc_generation/')

        self.save_model_path = os.path.join('data', 'model',
                                            f'{self.dataset.name}_npcnet_'
                                            f'T{self.diffusion.T}_'
                                            f'npc_epoch{self.num_epoch}_'
                                            f'S{self.dataset.split}.model')

        self.gen_set_path = os.path.join(ws, 'data/npc_generation',
                                f'{self.dataset.name}_npc_images_'
                                f'T{self.diffusion.T}_'
                                f'npc_epoch{self.num_epoch}_'
                                f'load_trained_npc_epoch{self.load_trained_npc_epoch}_'
                                f'sample_num{self.sample_num}_'
                                f'S{self.dataset.split}.npz')

        self.early_stopping = early_stopping

        self.npc_loss = []
        self.save_loss_path = os.path.join('data', 'model',
                                            f'{self.dataset.name}_npcnet_{self.denoiser.name}_'
                                            f'T{self.diffusion.T}_'
                                            f'S{self.dataset.split}_loss.npy')

        self.traffic_condition_path = ws + '/data/' + self.dataset.name + f'_S{self.dataset.split}_F{self.dataset.flat}_trafficcondition.npy'
        self.traffic_condition = np.load(self.traffic_condition_path, allow_pickle = True)
        self.traffic_condition = torch.from_numpy(self.traffic_condition).float().to(self.device)

    def train_epoch(self, train_gen, meta):
        self.denoiser.eval()
        sample_num = len(train_gen)
        losses = []
        origin_image, odt, label = meta
        batch_iter = list(zip(train_gen, origin_image[:sample_num], odt[:sample_num], label[:sample_num]))

        desc_txt = 'Training npc, loss %05.6f'
        with tqdm(next_batch((batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size,
                  desc=desc_txt % 0.0) as pbar:
            for batch in pbar:

                batch_gen, batch_img, batch_odt, _ = zip(*batch)
                batch_gen, batch_img, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device) for item in (batch_gen, batch_img, batch_odt))
                batch_D = batch_odt[:, 5].long().reshape(-1)
                batch_ts = batch_odt[:, 6].long().reshape(-1)
                batch_traffic_condition = self.traffic_condition[batch_D, batch_ts - 1].reshape(batch_img.size(0), 1,
                                                                                                batch_img.size(2),
                                                                                                batch_img.size(3))
                w_mat = self.npc_net(batch_gen, batch_odt, batch_traffic_condition)
                w_mat_ = w_mat.flatten(2)
                w_norms = w_mat_.norm(dim=2)
                w_hat_mat = w_mat_ / w_norms[:, :, None]

                err = (batch_img - batch_gen).flatten(1)

                ## Normalizing by the error's norm
                ## -------------------------------
                err_norm = err.norm(dim=1)
                err = err / err_norm[:, None]
                w_norms = w_norms / err_norm[:, None]

                ## W hat loss
                ## ----------
                err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
                reconst_err = 1 - err_proj.pow(2).sum(dim=1)

                ## W norms loss
                ## ------------
                second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
                # sigma = w_norms
                # w = w_hat_mat

                second_moment_loss_lambda = -1 + 2 * self.npc_step / self.second_moment_loss_grace
                second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
                second_moment_loss_lambda *= second_moment_loss_lambda
                objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()

                self.npc_optimizer.zero_grad()
                objective.backward()
                self.npc_optimizer.step()
                self.npc_step += 1
                losses.append(objective.item())

                pbar.set_description(desc_txt % (objective.item()))
        return float(np.mean(losses))

    def train(self):
        self.train_meta = self.dataset.get_images(0)
        self.val_meta = self.dataset.get_images(1)
        train_gen = self.gen_images[0]

        print('start npc training')
        min_val_metric = 1e8
        epoch_before_stop = self.early_stopping

        for epoch in range(self.num_epoch):
            print('current npc epoch: ', epoch)
            train_loss = self.train_epoch(train_gen, self.train_meta)
            self.npc_loss.append(train_loss)
            val_metric = train_loss
            if min_val_metric > val_metric:
                min_val_metric = val_metric
                epoch_before_stop = 0
                self.save_model(epoch)
            else:
                epoch_before_stop += 1
            if epoch % 10 == 0:
                self.save_model(epoch)

            if 0 < self.early_stopping <= epoch_before_stop:
                print('\nEarly stopping, best epoch:', epoch - epoch_before_stop)
                self.load_model(epoch - epoch_before_stop)
                break

        print('final npc net saved')
        return self.npc_net

    def save_model(self, epoch=None):
        save_path = self.save_model_path
        save_loss_path = self.save_loss_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        torch.save(self.npc_net.state_dict(), save_path)
        np.save(save_loss_path, np.array(self.npc_loss))
        print('[Saved npc net] to ' + save_path)
        print('[Saved loss] to ' + save_loss_path)

    def load_model(self, epoch=None):
        save_path = self.save_model_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'

        self.npc_net.load_state_dict(torch.load(save_path, map_location=self.device), strict=True)
        print('[Loaded npc] from ' + save_path)
        return self.npc_net

    def save_generation(self, select_sets=None):
        if select_sets is None:
            select_sets = range(3)
        for s in select_sets:
            self.gen_images[s] = self.generate(self.gen_images[s], self.dataset.images[s]) # generated by denoiser, odt, origin images
        dir_check(self.gen_set_path)
        np.savez(self.gen_set_path, train=self.gen_images[0], val=self.gen_images[1], test=self.gen_images[2])
        print('[Saved npc generation] images to ' + self.gen_set_path)

    def load_generation(self):
        gen_images = np.load(self.gen_set_path)
        self.gen_images = [gen_images[label] for label in ['train', 'val', 'test']]
        print('[Loaded npc generation] from ' + self.gen_set_path)

    def generate(self, gen, meta):
        self.denoiser.eval()
        origin_image, odt, label = meta
        batch_iter = list(zip(gen, origin_image, odt, label))
        gens = []
        t_list = torch.FloatTensor([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]).to(self.device)
        self.sample_num = len(t_list)
        with tqdm(next_batch((batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size) as pbar:
            self.npc_net.eval()
            for batch in pbar:
                batch_gen, batch_img, batch_odt, batch_arrival = zip(*batch)
                batch_gen, batch_img, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device) for item in (batch_gen, batch_img, batch_odt))
                batch_D = batch_odt[:, 5].long().reshape(-1)
                batch_ts = batch_odt[:, 6].long().reshape(-1)
                batch_traffic_condition = self.traffic_condition[batch_D, batch_ts - 1].reshape(batch_img.size(0), 1,
                                                                                                batch_img.size(2),
                                                                                                batch_img.size(3))
                img_list = []
                with torch.no_grad():
                    w_mat = self.npc_net(batch_gen, batch_odt, batch_traffic_condition)
                    w_mat_ = w_mat.flatten(2)
                    w_norms = w_mat_.norm(dim=2)
                    w_hat_mat = w_mat_ / w_norms[:, :, None]
                    sigma = torch.sqrt(w_norms.pow(2))
                    for i in range(len(t_list)):
                        img_list.append(w_hat_mat.reshape(-1, 5, batch_img.size(1), batch_img.size(2), batch_img.size(3))[:, 0, :, :, :] * t_list[i] * sigma[:, 0][:, None, None, None]  + batch_gen)
                    img = torch.stack(img_list, dim=1)
                    gens.append(img.detach().cpu().numpy())

        return  np.concatenate(gens, axis=0)
