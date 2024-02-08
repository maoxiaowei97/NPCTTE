def train(self):
    train_meta = self.dataset.get_images(0)
    val_meta = self.dataset.get_images(1)
    print('start nppc training')
    min_val_metric = 1e8
    epoch_before_stop = self.early_stopping

    for epoch in range(self.num_epoch):
        print('current nppc epoch: ', epoch)
        train_loss = self.train_epoch(train_meta)
        self.nppc_loss.append(train_loss)
        # val_metric = self.eval_epoch(val_meta)
        val_metric = train_loss
        if min_val_metric > val_metric:
            min_val_metric = val_metric
            epoch_before_stop = 0
            # self.save_model(epoch)
            self.save_model()
        else:
            epoch_before_stop += 1

        if 0 < self.early_stopping <= epoch_before_stop:
            print('\nEarly stopping, best epoch:', epoch - epoch_before_stop)
            # self.load_model(epoch - epoch_before_stop)
            self.load_model()
            break

    # self.save_model()

    print('final nppc net saved')
    return self.nppc_net