import os
import math
from datetime import datetime
from torch.nn import functional as F
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
from torch import nn
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from model.diffusion import DiffusionProcess

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

def cal_regression_metric(label, mean, p=True, save=True, save_key='undefined'):
    rmse = math.sqrt(mean_squared_error(label, mean))
    mae = mean_absolute_error(label, mean)
    mape = mean_absolute_percentage_error(label, mean)

    if p:
        print('rmse: %05.6f, mae: %05.6f, mape: %05.6f' % (rmse, mae, mape * 100), flush=True)

    if save:
        s = pd.Series([rmse, mae, mape], index=['rmse', 'mae', 'mape'])
        # s.to_hdf(os.path.join('/data/maodawei/DOT/output/', f'{save_key}.h5'),
        #          key=f't{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}', format='table')
        s.to_csv(f'/data/XinZhi/ODUQ/DOT/output/{save_key}.csv')
        np.savez(os.path.join('/data/XinZhi/ODUQ/DOT/output/', f'{save_key}.npz'),
                 pre=mean, label=label, rmse=rmse, mae=mae, mape=mape)
        print('[Saved prediction]')

    return rmse, mae, mape

class ETATrainer:
    """
    Trainer for the ETA predictor.
    """

    def __init__(self, diffusion, predictor, dataset, gen_images, lr, batch_size, num_epoch, device,
                 train_origin=False, val_origin=False, early_stopping=-1):
        self.diffusion = diffusion
        self.predictor = predictor.to(device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.dataset = dataset
        self.gen_images = gen_images
        self.loss_func = F.smooth_l1_loss
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.device = device

        self.train_origin = train_origin
        self.val_origin = val_origin
        self.early_stopping = early_stopping

        self.save_model_path = os.path.join('data', 'model',
                                            f'{self.dataset.name}_predictor_{self.predictor.name}_'
                                            f'D{self.predictor.d_model}_'
                                            f'T{self.diffusion.T}_'
                                            f'S{self.dataset.split}.model')

    def train_epoch(self, eta_input, meta):
        self.predictor.train()
        losses = []
        origin_image, odt, label = meta
        # eta_input[:, 3, :, :] = deepcopy(origin_image[:, 3, :, :]) 用真实值
        # eta_input[:, 0, :, :] = deepcopy(origin_image[:, 0, :, :]) # mask用真实值
        batch_iter = list(zip(origin_image if self.train_origin else eta_input, odt, label)) # origin_image最后一维是匹配的出发前一个时间片的交通状态

        desc_txt = 'Training eta predictor, loss %05.6f'
        with tqdm(next_batch(shuffle(batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size,
                  desc=desc_txt % 0.0) as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()

                batch_input, batch_odt, batch_label= zip(*batch)
                batch_input, batch_odt, batch_label = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                                       for item in (batch_input, batch_odt, batch_label))
                # batch_input = torch.concat([batch_input, traffic_states.unsqueeze(1)], dim=1)

                pre = self.predictor(batch_input, batch_odt)  # (batch_size)

                loss = self.loss_func(pre, batch_label)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                pbar.set_description(desc_txt % (loss.item()))
        return float(np.mean(losses))

    def eval_epoch(self, eta_input, meta, save=False):
        self.predictor.eval()
        origin_image, odt, label = meta
        # eta_input[:, 3, :, :] = deepcopy(origin_image[:, 3, :, :])
        # eta_input[:, 0, :, :] = deepcopy(origin_image[:, 0, :, :])
        batch_iter = list(zip(origin_image if self.val_origin else eta_input, odt))
        pres = []

        for batch in next_batch(batch_iter, self.batch_size):
            batch_input, batch_odt = zip(*batch)
            batch_input, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                      for item in (batch_input, batch_odt))
            # batch_input = torch.concat([batch_input, traffic_states.unsqueeze(1)], dim=1)
            pre = self.predictor(batch_input, batch_odt).detach().cpu().numpy()  # (batch_size)
            pres.append(pre)

        pres = np.concatenate(pres)
        return cal_regression_metric(label, pres, save=save,
                                     save_key=f'{self.predictor.name}_D{self.predictor.d_model}_'
                                              f'{self.dataset.name}_S{self.dataset.split}_P{self.dataset.partial}')

    def train(self):
        #先看eta是否可以过拟合
        train_gen = self.gen_images[0]
        train_meta = self.dataset.get_images(0)
        val_gen = self.gen_images[1]
        val_meta = self.dataset.get_images(1)
        test_gen = self.gen_images[2]
        test_meta = self.dataset.get_images(2)

        min_val_mae = 1e8
        epoch_before_stop = 0

        for epoch in range(self.num_epoch):
            loss_val = self.train_epoch(train_gen, train_meta)
            val_rmse, val_mae, val_mape = self.eval_epoch(val_gen, val_meta)

            if min_val_mae > val_mae:
                min_val_mae = val_mae
                epoch_before_stop = 0
                self.save_model()
            else:
                epoch_before_stop += 1

            if 0 < self.early_stopping <= epoch_before_stop:
                print('\nEarly stopping, best epoch:', epoch - epoch_before_stop)
                self.load_model()
                break

        test_rmse, test_mae, test_mape = self.eval_epoch(test_gen, test_meta, save=True)
        self.save_model()
        return self.predictor

    def save_model(self, epoch=None):
        save_path = self.save_model_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        print('[Saved predictor] to ' + save_path)
        torch.save(self.predictor.state_dict(), save_path)

    def load_model(self, epoch=None):
        save_path = self.save_model_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        print('[Loaded predictor] from ' + save_path)
        self.predictor.load_state_dict(torch.load(save_path, map_location=self.device), strict=True)
        self.predictor.eval()
        return self.predictor
