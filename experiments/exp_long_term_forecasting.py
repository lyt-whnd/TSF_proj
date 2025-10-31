from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

from model.Statistics_prediction import Statistics_prediction


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):

        #添加SAN统计量估计
        if self.args.adaptive_norm:
            self.statistics_pred = Statistics_prediction(self.args).to(self.device)

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        #添加SAN统计量估计
        if self.args.adaptive_norm:
            self.station_optim = optim.Adam(self.statistics_pred.parameters(), lr=self.args.station_lr)

        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def station_loss(self, y, statistics_pred):
        bs, len, dim = y.shape
        y = y.reshape(bs, -1, self.args.period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_ture = torch.cat([mean, std], dim=-1)
        loss = self.criterion(statistics_pred, station_ture)
        return loss

    def vali(self, vali_data, vali_loader, criterion,epoch=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.adaptive_norm:
                    batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)
                    if epoch + 1 <= self.station_pretrain_epoch:
                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        if self.args.features == 'MS':
                            statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                        loss = self.station_loss(batch_y, statistics_pred)
                    else:
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_label = batch_x[:, -self.args.label_len:, :]
                        dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
                else:
                    # channel_decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if "Cycle" in self.args.model:
                    batch_cycle = batch_cycle.int().to(self.device)
                # fc1 - channel_decoder
                #添加CycleNet
                if any(substr in self.args.model for substr in {'Cycle'}):
                    outputs = self.model(batch_x, batch_cycle)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    loss = mae
                else:
                    loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.adaptive_norm:
            path_station = './station/' + '{}_s{}_p{}'.format(self.args.data, self.args.seq_len, self.args.pred_len)

            if not os.path.exists(path_station):
                os.makedirs(path_station)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        if self.args.adaptive_norm:
            early_stopping_station_model = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            if self.args.adaptive_norm:
                if epoch == self.station_pretrain_epoch and self.args.station_type == 'adaptive':
                    best_model_path = path_station + '/' + 'checkpoint.pth'
                    self.statistics_pred.load_state_dict(torch.load(best_model_path))
                    print('loading pretrained adaptive station model')
                    # self.statistics_pred.requires_grad_(False)
                self.statistics_pred.train()

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                if self.args.adaptive_norm:
                    batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)
                    if epoch + 1 <= self.station_pretrain_epoch:
                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        if self.args.features == 'MS':
                            statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                        loss = self.station_loss(batch_y, statistics_pred)
                        train_loss.append(loss.item())
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_label = batch_x[:, -self.args.label_len:, :]
                        dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
                else:

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)


                    # channel_decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if "Cycle" in self.args.model:
                    batch_cycle = batch_cycle.int().to(self.device)

                if any(substr in self.args.model for substr in {'Cycle'}):
                    outputs = self.model(batch_x, batch_cycle)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                if self.args.adaptive_norm:
                    if self.args.features == 'MS':
                        statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                    outputs = self.statistics_pred.de_normalize(outputs, statistics_pred)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self.criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                else:

                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.model == "TimeBridge":
                        loss = self.time_freq_mae(batch_y, outputs)
                    else:
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                if self.args.adaptive_norm:
                    # two-stage training schema
                    if epoch + 1 <= self.station_pretrain_epoch:
                        self.station_optim.step()
                    else:
                        model_optim.step()
                model_optim.step()
                if self.args.adaptive_norm:
                    self.station_optim.zero_grad()


                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion,epoch)
            # test_loss = 0
            test_loss = self.vali(test_data, test_loader, criterion,epoch)

            if self.args.adaptive_norm:
                if epoch + 1 <= self.station_pretrain_epoch:
                    print(
                        "Station Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    early_stopping_station_model(vali_loss, self.statistics_pred, path_station)
                    adjust_learning_rate(self.station_optim, epoch + 1, self.args, self.args.station_lr)
                else:
                    print(
                        "Backbone Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                            epoch + 1 - self.station_pretrain_epoch, train_steps, train_loss, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                    adjust_learning_rate(model_optim, epoch + 1 - self.station_pretrain_epoch, self.args,
                                         self.args.learning_rate)
                    adjust_learning_rate(self.station_optim, epoch + 1 - self.station_pretrain_epoch, self.args,
                                         self.args.station_lr)
            else:

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if self.args.lradj != 'TST':
                    adjust_learning_rate(model_optim, None, epoch + 1, self.args)
                else:
                    print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def time_freq_mae(self, batch_y, outputs):
        # time mae loss
        t_loss = (outputs - batch_y).abs().mean()

        # freq mae loss
        f_loss = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
        f_loss = f_loss.abs().mean()

        return (1 - self.args.alpha) * t_loss + self.args.alpha * f_loss

    def test(self, setting, test=0,epoch=None):
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if test:
            print(f'loading model on {device}')
            checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.args.adaptive_norm:
            self.statistics_pred.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.adaptive_norm:
                    input_x = batch_x

                    batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
                else:

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)


                    # channel_decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if "Cycle" in self.args.model:
                    batch_cycle = batch_cycle.int().to(self.device)

                # fc1 - channel_decoder
                if any(substr in self.args.model for substr in {'Cycle'}):
                    outputs = self.model(batch_x, batch_cycle)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                if self.args.adaptive_norm:
                    if self.args.features == 'MS':
                        statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                    outputs = self.statistics_pred.de_normalize(outputs, statistics_pred)

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        #print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if "Cycle" in self.args.model:
                    batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if any(substr in self.args.model for substr in {'Cycle'}):
                    outputs = self.model(batch_x, batch_cycle)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
