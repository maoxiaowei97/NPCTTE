# if args.draw_nppc:
#     # load denoiser and nppc
#     print('start drawing nppc images......')
#     denoiser = diffusion_trainer.load_model(None if args.loadepoch == -1 else args.loadepoch)
#     nppc_trainer = NppcTrainer(diffusion=diffusion, denoiser=denoiser, nppc_net=nppc_unet, dataset=dataset, lr=1e-3,
#                                batch_size=args.batch, device=device, num_epoch=args.epoch, nppc_step=args.nppc_step,
#                                second_moment_loss_grace=args.second_moment_loss_grace, early_stopping= args.early_stop)
#     nppc_net = nppc_trainer.load_model(None if args.loadepoch == -1 else args.loadepoch)
#     val_images, val_ODTs, _ = dataset.get_images(1) # 验证集所有图片
#     val_images, val_ODTs = shuffle(val_images, val_ODTs)
#     val_num = len(val_images)
#     with torch.no_grad():
#         gen_steps = diffusion.p_sample_loop(denoiser, shape=(val_num, *(val_images.shape[1:])),
#                                             y=torch.from_numpy(val_ODTs[:val_num]).float().to(device),
#                                             display=True)
#         x_restored = gen_steps[-1] # restored img, includes route
#         num_channel = x_restored.shape[1]
#         x_distorted = val_ODTs[:val_num]
#         # w_mat = nppc_net(torch.tensor(x_distorted).float().to(device), torch.tensor(x_restored).float().to(device)) #原nppc
#         w_mat = nppc_net(torch.tensor(x_restored).float().to(device), torch.tensor(x_distorted).float().to(device)) # big_nppc_unet
#         # t_list = torch.linspace(-1, 1, 3).to(device) # [-1, 1]范围内选3个
#         t_list = torch.linspace(-1, 1, 7).to(device)  # [-1, 1]范围内选3个
#         for i in tqdm(range(len(x_restored))):#遍历样本
#             imgs = t_list[:, None, None, None, None] * w_mat[i] + torch.from_numpy(x_restored[i][None][None]).float().to(device) # [21, 5, 3, 20, 20] + [1, 1, 3, 20, 20] -> [21, 5, 3, 20, 20] 每个样本的恢复图像
#             # imgs = torch.cat((scale_img(w_mat), imgs), dim=0)
#             imgs = imgs.transpose(0, 1).contiguous()# [n_dirs, sample_num, channel, 20, 20]
#             #show imgs
#             plt.figure(figsize=(num_channel * 5, (1 + len(t_list) * args.n_dirs) * 5))
#             for c in range(1, num_channel + 1): # channel
#                 plt.subplot(1 + args.n_dirs * len(t_list), num_channel, c)  # 对于一个样本，长度上，对于一个主成分方向，有3行，一共5个主成分方向。nrows有16个，ncols有3个
#                 plt.title(f'Generated channel with nppc {c}')
#                 plt.imshow(val_images[i][c-1])
#             for dir in range(1, args.n_dirs + 1): # 从第 0 + 1个方向开始
#                 for dir_sample in range(1,  len(t_list) + 1):
#                     for sample_c in range(1, num_channel + 1): # 先遍历channel
#                         image = imgs[dir-1][dir_sample-1][sample_c-1].detach().cpu().numpy()
#                         if sample_c == 1:
#                             mask = image[:, :] < 0 # 对第一个维度进行mask
#                             show = image[:, :] > 0
#                             image[mask] = -1
#                             image[show] = 1
#                         plt.subplot(1 + args.n_dirs * len(t_list), num_channel, num_channel + sample_c + (dir - 1) * (len(t_list)) * num_channel + (dir_sample - 1) * num_channel)  # 对于一个样本，长度上，对于一个主成分方向，有3行，一共5个主成分方向。nrows有16个，ncols有3个
#                         plt.title(f'c{sample_c}-{dir}-th-dir-{dir_sample}-th-s')
#                         plt.imshow(image)
#             # plt.savefig(os.path.join('data', 'images_nppc_20181101_100w_1228_2251', f'{dataset.name}_%03d.png' % i), bbox_inches='tight')
#             save_path = '/data/maodawei/DOT/data/' + 'nppc_images/' + f'{args.gen_image_name}/'
#             dir_check(save_path)
#             plt.savefig(save_path + f'{dataset.name}_%03d.png' % i, bbox_inches='tight')
#             plt.close()
#
#
#     def save_generation(self, select_sets=None, sample_num = 3):
#         if select_sets is None:
#             select_sets = range(3)
#         for s in select_sets:
#             self.gen_images[s] = self.generate(self.dataset.get_images(s), sample_num)
#         np.savez(self.gen_set_path, train=self.gen_images[0], val=self.gen_images[1], test=self.gen_images[2]) # 生成数值型的image
#         print('[Saved generation] images to ' + self.gen_set_path)
#
#     def load_generation(self):
#         gen_images = np.load(self.gen_set_path)
#         self.gen_images = [gen_images[label] for label in ['train', 'val', 'test']]
#         print('[Loaded generation] from ' + self.gen_set_path)
#
#     # self.train_meta = self.dataset.get_images(0)
#     # self.val_meta = self.dataset.get_images(1)
#     # self.test_meta = self.dataset.get_images(2)
#     # data_meta = []
#     # self.denoiser.eval()
#     # with torch.no_grad():
#     #     self.x_restored_train = self.diffusion.p_sample_loop(self.denoiser, shape=(self.train_meta[1].shape[0], *(self.train_meta[0].shape[1:])), y=torch.from_numpy(self.train_meta[1]).float().to(self.device), display=True)[-1]
#     #     self.x_restored_val = self.diffusion.p_sample_loop(self.denoiser, shape=(self.val_meta[1].shape[0], *(self.val_meta[0].shape[1:])), y=torch.from_numpy(self.val_meta[1]).float().to(self.device), display=True)[-1]
#     #     self.x_restored_test = self.diffusion.p_sample_loop(self.denoiser, shape=(self.test_meta[1].shape[0], *(self.test_meta[0].shape[1:])), y=torch.from_numpy(self.test_meta[1]).float().to(self.device), display=True)[-1]
#     # self.denoised_images = [self.x_restored_train, self.x_restored_val, self.x_restored_test]
#
#     def generate(self, meta):
#         self.denoiser.eval()
#
#         batch_iter = list(zip(*meta))
#         gens = []
#         # probs = []
#         for batch in tqdm(next_batch(batch_iter, self.batch_size), total=len(batch_iter) // self.batch_size,
#                           desc='Generating images'):
#             batch_img, batch_odt, batch_arr = zip(*batch)
#             batch_odt = torch.from_numpy(np.stack(batch_odt, 0)).float().to(self.device)
#             # gen_batch = np.zeros([sample_num, np.array(batch_img).shape[0], np.array(batch_img).shape[1], np.array(batch_img).shape[2], np.array(batch_img).shape[3]])
#             # prob_batch = np.zeros([sample_num, np.array(batch_img).shape[0], np.array(batch_img).shape[1]])
#
#
#             gen = self.diffusion.p_sample_loop(self.denoiser, shape=np.array(batch_img).shape,
#                                                y=batch_odt, display=False)[-1] # gen: [s, b, c, h, w], probs: [b, s, T]
#             gens.append(gen)
#             # probs.append(prob)
#         # return np.concatenate(gens, axis=0), np.concatenate(probs, axis=0)
#         return np.concatenate(gens, axis=0)
#
#
#   def train_epoch(self, meta):
#         self.denoiser.eval()
#         with torch.no_grad():
#             self.x_restored_train = self.diffusion.p_sample_loop(self.denoiser, shape=(meta[1].shape[0], *(meta[0].shape[1:])), y=torch.from_numpy(meta[1]).float().to(self.device), display=True)[-1]
#         losses = []
#         batch_iter = list(zip(*meta))
#
#         desc_txt = 'Training nppc, loss %05.6f'
#
#         s_index = 0
#         e_index = self.batch_size
#         with tqdm(next_batch(shuffle(batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size,
#                   desc=desc_txt % 0.0) as pbar:
#             for batch in pbar:
#
#                 batch_img, batch_odt, _ = zip(*batch)
#                 batch_img, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
#                                         for item in (batch_img, batch_odt))
#
#                 x_restored = self.x_restored_train[s_index: e_index]
#                 s_index = e_index
#                 e_index += self.batch_size
#                 w_mat = self.nppc_net(torch.tensor(x_restored).float().to(self.device), torch.tensor(batch_odt).float().to(self.device))
#
#                 w_mat_ = w_mat.flatten(2)
#                 w_norms = w_mat_.norm(dim=2)
#                 w_hat_mat = w_mat_ / w_norms[:, :, None]
#
#                 err = (batch_img - torch.from_numpy(x_restored).float().to(self.device)).flatten(1)  # label + 去噪后的结果 -> error [B, C*N*N]
#
#                 ## Normalizing by the error's norm
#                 ## -------------------------------
#                 err_norm = err.norm(dim=1)
#                 err = err / err_norm[:, None]
#                 w_norms = w_norms / err_norm[:, None]
#
#                 ## W hat loss
#                 ## ----------
#                 err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)  # 主成分 + error
#                 reconst_err = 1 - err_proj.pow(2).sum(dim=1)  # 1在这里是希望loss>0, 本质是L_w
#
#                 ## W norms loss
#                 ## ------------
#                 second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
#                 # variance: w_norms.pow(2) -> 各个主成分方向的权重 -> 生成路线的probability (torch.softmax(w_norms.pow(2), dim=1))
#
#                 second_moment_loss_lambda = -1 + 2 * self.nppc_step / self.second_moment_loss_grace
#                 second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
#                 second_moment_loss_lambda *= second_moment_loss_lambda
#                 objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()
#
#                 self.nppc_optimizer.zero_grad()
#                 objective.backward()
#                 self.nppc_optimizer.step()
#                 self.nppc_step += 1
#                 losses.append(objective.item())
#
#                 pbar.set_description(desc_txt % (objective.item()))
#         return float(np.mean(losses))
#
#
# def train_epoch(self, meta):
#     self.denoiser.train()
#     losses = []
#     batch_iter = list(zip(*meta))
#
#     desc_txt = 'Training diffusion, loss %05.6f'
#     with tqdm(next_batch(shuffle(batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size,
#               desc=desc_txt % 0.0) as pbar:
#         for batch in pbar:
#             self.optimizer.zero_grad()
#
#             batch_img, batch_odt, _ = zip(*batch)
#             # Create two batch tensors, with shape (N, C, X, Y) and (N, y_feat).
#             batch_img, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
#                                     for item in (batch_img, batch_odt))
#             t = torch.randint(0, self.diffusion.T, (batch_img.size(0),)).long().to(self.device)
#
#             loss = self.diffusion.p_losses(self.denoiser, batch_img, t, batch_odt, loss_type=self.loss_type)
#             loss.backward()
#             self.optimizer.step()
#
#             losses.append(loss.item())
#             pbar.set_description(desc_txt % (loss.item()))
#     return float(np.mean(losses))
def train_epoch(self, meta):
    self.denoiser.train()
    losses = []
    batch_iter = list(zip(*meta))
    self.traffic_state = np.load(
        '/data/XinZhi/ODUQ/DOT/data/cache_traffic_state/normalized_traffic_state_array_1101_1116_15s_0117.npy',
        allow_pickle=True)
    self.traffic_state = torch.from_numpy(self.traffic_state).float().to(self.device)
    # [D, Ts, C]

    desc_txt = 'Training diffusion, loss %05.6f'
    with tqdm(next_batch(shuffle(batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size,
              desc=desc_txt % 0.0) as pbar:
        for batch in pbar:
            self.optimizer.zero_grad()

            batch_img, batch_odt, _ = zip(*batch)

            # Create two batch tensors, with shape (N, C, X, Y) and (N, y_feat).
            batch_img, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                    for item in (batch_img, batch_odt))
            batch_D = batch_odt[:, 5].long().reshape(-1)
            batch_ts = batch_odt[:, 6].long().reshape(-1)
            batch_traffic_state = self.traffic_state[batch_D, batch_ts].reshape(batch_img.size(0), 1, batch_img.size(2),
                                                                                batch_img.size(3))  # (b, h, w)

            t = torch.randint(0, self.diffusion.T, (batch_img.size(0),)).long().to(self.device)

            loss = self.diffusion.p_losses(self.denoiser, batch_img, t, batch_odt, batch_traffic_state,
                                           loss_type=self.loss_type)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            pbar.set_description(desc_txt % (loss.item()))
    return float(np.mean(losses))