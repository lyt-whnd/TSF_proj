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
import pandas as pd
from collections import defaultdict

from model.DDN import DDN

# from model.Statistics_prediction import Statistics_prediction


warnings.filterwarnings('ignore')

def infer_future_timestamps_auto_batch(x: torch.Tensor, pred_len: int, freq: str = 'h') -> torch.Tensor:
    """
    从形状 [B, T] 的 Unix 秒时间戳中，根据最后一个时间点生成未来 pred_len 个时间点。

    Args:
        x (torch.Tensor): 输入张量，形状 [B, T]，单位为秒（Unix 时间）
        pred_len (int): 需要预测/扩展的时间步数
        freq (str): 时间频率，可选:
            's' - 秒
            'm' - 分钟
            'h' - 小时（默认）
            'd' - 天
            也可以指定自定义间隔，例如 '15m', '3h', '2d'

    Returns:
        torch.Tensor: 形状 [B, pred_len]，未来每步的 Unix 秒时间戳
    """
    B = x.size(0)
    device = x.device

    # 每个样本的最后一个时间戳
    last_t = x[:, -1].to(torch.long)  # [B]

    # 将 freq 转换为秒数
    def freq_to_seconds(freq_str: str) -> int:
        import re
        m = re.match(r"(\d*)([smhd])", freq_str)
        if not m:
            raise ValueError(f"Unsupported frequency format: {freq_str}")
        num = int(m.group(1)) if m.group(1) else 1
        unit = m.group(2)
        if unit == 's':
            return num
        elif unit == 'm':
            return num * 60
        elif unit == 'h':
            return num * 3600
        elif unit == 'd':
            return num * 86400
        else:
            raise ValueError(f"Unsupported unit: {unit}")

    step_seconds = freq_to_seconds(freq)

    # 生成偏移序列
    offsets = torch.arange(1, pred_len + 1, device=device, dtype=torch.long) * step_seconds  # [pred_len]

    # 广播加法得到未来时间戳
    future_times = last_t.unsqueeze(1) + offsets.unsqueeze(0)  # [B, pred_len]

    return future_times


#推测未来小时信息函数
def infer_time(batch_time, pred_len):
    # batch_time: [batch_size, seq_len]
    batch_size, seq_len = batch_time.shape
    pred_time = torch.zeros((batch_size, pred_len), dtype=batch_time.dtype).to(batch_time.device)
    for i in range(batch_size):
        last_hour = int(batch_time[i, -1].item())
        for j in range(pred_len):
            next_hour = (last_hour + j + 1) % 24
            pred_time[i, j] = next_hour
    return pred_time


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.station_pretrain_epoch = args.pre_epoch if self.args.station_type == 'adaptive' else 0
        self.station_type = args.station_type

    def _build_model(self):

        # #添加SAN统计量估计
        # if self.args.adaptive_norm:
        #     self.statistics_pred = Statistics_prediction(self.args).to(self.device)
        #采用效果更好的DDN
        self.statistics_pred = DDN(self.args).to(self.device)
        self.station_loss = self.sliding_loss

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
            self.criterion = nn.MSELoss()
        # return criterion

    def station_loss(self, y, statistics_pred):
        bs, len, dim = y.shape
        y = y.reshape(bs, -1, self.args.period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_ture = torch.cat([mean, std], dim=-1)
        loss = self.criterion(statistics_pred, station_ture)
        return loss

    def san_loss(self, y, statistics_pred):
        bs, len, dim = y.shape
        y = y.reshape(bs, -1, self.args.period_len, dim)
        mean = torch.mean(y, dim=2)
        std = torch.std(y, dim=2)
        station_ture = torch.cat([mean, std], dim=-1)
        loss = self.criterion(statistics_pred, station_ture)
        return loss

    def sliding_loss(self, y, statistics_pred):
        _, (mean, std) = self.statistics_pred.norm(y.transpose(-1, -2), False)
        station_ture = torch.cat([mean, std], dim=1).transpose(-1, -2)
        loss = self.criterion(statistics_pred, station_ture)
        return loss

    def _irm_penalty(self, y_pred, y_true, env_ids, criterion=None):
        """
        IRMv1 penalty: sum_e || d/dw L_e(w * y_pred, y_true) |_{w=1} ||^2
        y_pred, y_true: 形状需可被 criterion 接受（通常 [B, pred_len, C] 或被你展平后的）
        env_ids: [B] 的长整型域 id
        """
        device = y_pred.device
        envs = torch.unique(env_ids)
        penalty = 0.0
        used = 0
        for e in envs:
            mask = (env_ids == e)
            if mask.sum() < 2:  # 太少就跳过，避免数值不稳定
                continue
            w = torch.tensor(1.0, requires_grad=True, device=device)
            loss_e = criterion(w * y_pred[mask], y_true[mask])
            grad = torch.autograd.grad(loss_e, [w], create_graph=True)[0]
            penalty = penalty + grad.pow(2)
            used += 1
        if used == 0:
            return torch.tensor(0.0, device=device)
        return penalty / used

    def _vrex_penalty(self, losses_per_env):
        """
        V-REx penalty: variance of per-environment empirical risks
        Args:
            losses_per_env: List[Tensor] where each tensor is a scalar loss for one environment (already averaged over that env's batch samples)
        Returns:
            Tensor scalar = Var_e[R_e]
        注：与论文 V-REx 一致（REx = Equalize risks across domains），当 β>0 时鼓励各域风险相等。
        """
        if losses_per_env is None or len(losses_per_env) <= 1:
            # 少于两个域时无法计算方差，返回 0
            return torch.tensor(0.0, device=self.device)
        L = torch.stack(losses_per_env)  # [E]
        # 无偏或有偏都可，这里按论文实现常用的有偏方差（与代码更稳定）
        return L.var(unbiased=False)

    def vali(self, vali_data, vali_loader, criterion,epoch=None):
        total_loss = []
        vali_domain_loss = defaultdict(list)
        self.model.eval()
        self.statistics_pred.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, *rest = batch
                env_id = rest[0] if rest else None
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'solar' in self.args.data.lower():
                    batch_x = batch_x[:, :, 1:]

                if 'Wind' in self.args.data or 'multi_wind' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.adaptive_norm:
                    if epoch + 1 <= self.station_pretrain_epoch and self.args.use_norm == 'sliding':
                        batch_x, statistics_pred, statistics_seq = self.statistics_pred.normalize(batch_x,
                                                                                                  p_value=False)
                    elif self.args.use_norm == 'sliding':
                        batch_x, statistics_pred, statistics_seq = self.statistics_pred.normalize(batch_x)
                    else:
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

                        if "Cycle" in self.args.model:
                            batch_cycle = batch_cycle.int().to(self.device)
                        # fc1 - channel_decoder
                        # 添加CycleNet
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        if self.args.features == 'MS':
                            statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        outputs = self.statistics_pred.de_normalize(outputs, statistics_pred)
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        pred = outputs.detach().cpu()
                        true = batch_y.detach().cpu()

                        loss = criterion(pred, true)
                else:
                    # channel_decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()

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
                            if self.args.use_norm == 'sliding':
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)
                total_loss.append(loss.cpu().item())

                # --- per-batch per-domain loss logging ---
                with torch.no_grad():
                    doms, counts = torch.unique(env_id, return_counts=True)
                    msg = []
                    for d in doms:
                        mask = (env_id == d)
                        if mask.any():
                            did = d.item()
                            dom_loss = criterion(outputs[mask], batch_y[mask]).item()
                            msg.append(f"domain {int(d.item())}: {dom_loss:.6f} (n={int(mask.sum().item())})")
                            vali_domain_loss[did].append(dom_loss)

        total_loss = np.average(total_loss)
        self.model.train()
        if self.args.adaptive_norm:
            self.statistics_pred.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.pred_len == 1:
            self.args.adaptive_norm = 0
            if self.args.model == 'iTransformer':
                self.args.use_norm = 'revin'
                print("使用iTransformer revin")
        print("是否使用归一化：",self.args.adaptive_norm)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.adaptive_norm:
            path_station = './station/' + '{}_s{}_p{}_{}_m'.format(self.args.data, self.args.seq_len, self.args.pred_len,self.args.model)

            if not os.path.exists(path_station):
                os.makedirs(path_station)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        if self.args.adaptive_norm:
            early_stopping_station_model = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        self._select_criterion()

        use_IRM_flag = False  # 是否使用IRM正则化
        use_VREX_flag = True  # 是否使用V-REx风险外推正则（Risk Extrapolation, variance of per-domain risks）
        # 测试batch size大小
        print(">>> train_loader.batch_size =", getattr(train_loader, "batch_size", None))
        print(">>> test_loader.batch_size =", getattr(test_loader, "batch_size", None))

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs + self.station_pretrain_epoch):
            iter_count = 0
            train_loss = []
            train_domain_loss = defaultdict(list)

            if self.args.adaptive_norm:
                if epoch == self.station_pretrain_epoch and self.args.station_type == 'adaptive':
                    best_model_path = path_station + '/' + 'checkpoint.pth'
                    self.statistics_pred.load_state_dict(torch.load(best_model_path))
                    print('loading pretrained adaptive station model')
                    if self.args.use_norm == 'sliding' and self.args.twice_epoch >= 0:
                        print('reset station model optim for finetune')

                if self.args.use_norm == 'sliding' and 0 <= self.args.twice_epoch == epoch - self.station_pretrain_epoch:
                    lr = model_optim.param_groups[0]['lr']
                    model_optim.add_param_group({'params': self.statistics_pred.parameters(), 'lr': lr})
                    # self.statistics_pred.requires_grad_(False)
                self.statistics_pred.train()

            self.model.train()
            epoch_time = time.time()

            # 添加IRM参数
            if use_IRM_flag:
                print("使用IRM")
                irm_lambda = getattr(self.args, 'irm_lambda', 10.0)  # e.g. 1.0
                irm_anneal = getattr(self.args, 'irm_anneal_iters', 10000)  # e.g. 500
                global_step = 0
            elif use_VREX_flag:
                print("使用VREX")
                # --- V-REx 超参（均使用 getattr 以避免未在 args 中定义时报错）---
                beta_vrex = getattr(self.args, 'beta_vrex', 50)  # e.g. 5.0
                vrex_anneal = getattr(self.args, 'vrex_anneal_iters', 10000)  # e.g. 500
                global_step = 0
                print("beta_vrex的值为：", beta_vrex)

            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, *rest = batch
                env_id = rest[0] if rest else None

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                if 'solar' in self.args.data.lower():
                    batch_x = batch_x[:, :, 1:]

                batch_y = batch_y.float().to(self.device)

                if 'Wind' in self.args.data or 'mutli_wind' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.adaptive_norm:
                    if epoch + 1 <= self.station_pretrain_epoch and self.args.use_norm == 'sliding':
                        batch_x, statistics_pred, statistics_seq = self.statistics_pred.normalize(batch_x,
                                                                                                  p_value=False)
                    elif self.args.use_norm == 'sliding':
                        batch_x, statistics_pred, statistics_seq = self.statistics_pred.normalize(batch_x)
                    else:
                        batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)
                    if epoch + 1 <= self.station_pretrain_epoch:
                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        if self.args.features == 'MS':
                            statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                        loss = self.station_loss(batch_y, statistics_pred)
                        train_loss.append(loss.item())
                    else:
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_label = batch_x[:, -self.args.label_len:, :]
                        dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)

                        if "Cycle" in self.args.model:
                            batch_cycle = batch_cycle.int().to(self.device)

                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                if self.args.use_norm == 'sliding':
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        if self.args.features == 'MS':
                            statistics_pred = statistics_pred[:, :, [self.args.enc_in - 1, -1]]
                        outputs = self.statistics_pred.de_normalize(outputs, statistics_pred)
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # --- 选择目标：V-REx > IRM > ERM ---
                        if use_VREX_flag:
                            # 1) 逐域损失
                            losses_e = []
                            envs = torch.unique(env_id)
                            for e in envs:
                                m = (env_id == e)
                                if m.any():
                                    losses_e.append(self.criterion(outputs[m], batch_y[m]))
                            # 2) ERM 基础项（域均值）
                            if len(losses_e) > 0:
                                erm_loss = torch.stack(losses_e).mean()
                            else:
                                erm_loss = self.criterion(outputs, batch_y)
                            # 3) V-REx 方差项（退火）
                            vrex_w = 0.0 if global_step < vrex_anneal else beta_vrex
                            vrex_pen = self._vrex_penalty(losses_per_env=losses_e) if vrex_w > 0.0 else torch.tensor(
                                0.0, device=self.device)
                            loss = erm_loss + vrex_w * vrex_pen
                            global_step += 1
                        # 是否使用IRM
                        elif use_IRM_flag:
                            pred = outputs
                            # 添加IRM参数
                            erm_loss = self.criterion(pred, batch_y[:, -pred.shape[1]:, :])  # 让 y 的时间维与 pred 对齐
                            irm_w = 0.0 if global_step < irm_anneal else irm_lambda
                            if irm_w > 0.0:
                                irm_pen = self._irm_penalty(pred, batch_y[:, -pred.shape[1]:, :], env_id,
                                                            criterion=self.criterion)
                                loss = erm_loss + irm_w * irm_pen
                            else:
                                irm_pen = torch.tensor(0.0, device=self.device)
                                loss = erm_loss

                            global_step += 1
                        # else:
                        #     loss = self.criterion(outputs, batch_y)

                        elif self.args.model == "TimeBridge":
                            loss = self.time_freq_mae(batch_y, outputs)
                        else:
                            loss = self.criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:

                    # channel_decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
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
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # --- 选择目标：V-REx > IRM > ERM ---
                    if use_VREX_flag:
                        # 1) 逐域损失
                        losses_e = []
                        envs = torch.unique(env_id)
                        for e in envs:
                            m = (env_id == e)
                            if m.any():
                                losses_e.append(self.criterion(outputs[m], batch_y[m]))
                        # 2) ERM 基础项（域均值）
                        if len(losses_e) > 0:
                            erm_loss = torch.stack(losses_e).mean()
                        else:
                            erm_loss = self.criterion(outputs, batch_y)
                        # 3) V-REx 方差项（退火）
                        vrex_w = 0.0 if global_step < vrex_anneal else beta_vrex
                        vrex_pen = self._vrex_penalty(losses_per_env=losses_e) if vrex_w > 0.0 else torch.tensor(
                            0.0, device=self.device)
                        loss = erm_loss + vrex_w * vrex_pen
                        global_step += 1
                    # 是否使用IRM
                    elif use_IRM_flag:
                        pred = outputs
                        # 添加IRM参数
                        erm_loss = self.criterion(pred, batch_y[:, -pred.shape[1]:, :])  # 让 y 的时间维与 pred 对齐
                        irm_w = 0.0 if global_step < irm_anneal else irm_lambda
                        if irm_w > 0.0:
                            irm_pen = self._irm_penalty(pred, batch_y[:, -pred.shape[1]:, :], env_id,
                                                        criterion=self.criterion)
                            loss = erm_loss + irm_w * irm_pen
                        else:
                            irm_pen = torch.tensor(0.0, device=self.device)
                            loss = erm_loss

                        global_step += 1
                    # else:
                    #     loss = self.criterion(outputs, batch_y)

                    elif self.args.model == "TimeBridge":
                        loss = self.time_freq_mae(batch_y, outputs)
                    else:
                        loss = self.criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    # if self.args.model == "TimeBridge":
                    #     loss = self.time_freq_mae(batch_y, outputs)
                    # else:
                    #     loss = self.criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                    # --- per-batch per-domain loss logging ---
                    with torch.no_grad():
                        doms, counts = torch.unique(env_id, return_counts=True)
                        msg = []
                        for d in doms:
                            mask = (env_id == d)
                            if mask.any():
                                did = d.item()
                                dom_loss = self.criterion(outputs[mask], batch_y[mask]).item()
                                msg.append(f"domain {int(d.item())}: {dom_loss:.6f} (n={int(mask.sum().item())})")
                                train_domain_loss[did].append(dom_loss)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    # 添加IRM损失
                    if use_IRM_flag:
                        print(f'| loss: {erm_loss.item():.4f} | irm: {irm_pen.item():.4f} | total: {loss.item():.4f}')
                    if use_VREX_flag:
                        # 若启用 V-REx，打印方差正则与权重
                        try:
                            print(
                                f"| erm: {erm_loss.item():.4f} | vrex: {float(vrex_pen.item()) if 'vrex_pen' in locals() else 0.0:.4f} | beta: {vrex_w:.4f} | total: {loss.item():.4f}")
                        except Exception:
                            pass
                    time_now = time.time()

                loss.backward()
                #梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
            vali_loss = self.vali(vali_data, vali_loader, self.criterion,epoch)
            # test_loss = 0
            test_loss = self.vali(test_data, test_loader, self.criterion,epoch)

            if self.args.adaptive_norm:
                if epoch + 1 <= self.station_pretrain_epoch:
                    print(
                        "Station Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    early_stopping_station_model(vali_loss, self.statistics_pred, path_station)
                    adjust_learning_rate(self.station_optim, None, epoch + 1, self.args, self.args.station_lr)
                else:
                    print(
                        "Backbone Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                            epoch + 1 - self.station_pretrain_epoch, train_steps, train_loss, vali_loss, test_loss))
                    # 若有更新之后stop,即保存
                    if self.args.use_norm == 'sliding' and 0 <= self.args.twice_epoch <= epoch - self.station_pretrain_epoch:
                        early_stopping(vali_loss, self.model, path, self.statistics_pred, path_station)
                    else:
                        early_stopping(vali_loss, self.model, path)

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                    adjust_learning_rate(model_optim, None, epoch + 1 - self.station_pretrain_epoch, self.args,
                                         self.args.learning_rate)
                    adjust_learning_rate(self.station_optim, None, epoch + 1 - self.station_pretrain_epoch, self.args,
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

        if self.args.pred_len == 1:
            self.args.adaptive_norm = 0
            if self.args.model == 'iTransformer':
                self.args.use_norm = 'revin'
                print("使用iTransformer revin")

        print("Adaptive norm:", self.args.adaptive_norm)

        self.device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if test:
            print(f'loading model on {self.device}')
            checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model", ckpt)
            msg = self.model.load_state_dict(state_dict, strict=False)
            print("missing:", msg.missing_keys)
            print("unexpected:", msg.unexpected_keys)
            # self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            if self.args.adaptive_norm:
                path_station = './station/' + '{}_s{}_p{}_{}_m'.format(self.args.data, self.args.seq_len,
                                                                       self.args.pred_len, self.args.model)
                print("加载归一模型：" + path_station + 'checkpoint.pth')
                ckpt_s = torch.load(os.path.join(path_station, 'checkpoint.pth'),
                                    map_location=self.device)
                state_s = ckpt_s.get('model', ckpt_s)  # 兼容有/无“model”键
                self.statistics_pred.load_state_dict(state_s, strict=False)
                self.statistics_pred.to(self.device)  # 保证模型也在CPU
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
                if 'solar' in self.args.data.lower():
                    batch_time = batch_x[:, :, 0]
                    batch_x = batch_x[:, :, 1:]
                    pred_time = infer_time(batch_time, self.args.pred_len)

                if 'Wind' in self.args.data or 'mutli_wind' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                input_x = batch_x

                if self.args.adaptive_norm:
                    if self.args.use_norm == 'sliding':
                        batch_x, statistics_pred, statistics_seq = self.statistics_pred.normalize(batch_x)
                    else:
                        batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)

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

                        # 获取缩放后的0
                        if not self.args.inverse:
                            zero_scaler = - test_data.scaler.mean_[-1] / test_data.scaler.scale_[-1]
                        else:
                            zero_scaler = 0

                        # 将时间戳信息重新cat到output上
                        outputs = torch.cat((pred_time.unsqueeze(-1), outputs), dim=-1)
                        # 更改 tensor 中的目标通道：若小时在 0-6 或 18-23，则将对应位置改为 "原始值 0 经 scaler 后的数值"
                        zero_val = float(zero_scaler)  # 转成 Python float，避免 numpy 与 torch 混用
                        mask = (outputs[:, :, 0] <= 6) | (outputs[:, :, 0] >= 18)  # [B, L] 布尔掩码
                        outputs[:, :, -1] = outputs[:, :, -1].masked_fill(mask, zero_val)

                        # 去掉tensor中时间的列
                        outputs = outputs[:, :, 1:]


                else:
                    # channel_decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()

                    if "Cycle" in self.args.model:
                        batch_cycle = batch_cycle.int().to(self.device)

                    # fc1 - channel_decoder
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            if self.args.use_norm == 'sliding':
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # 获取缩放后的0
                    if not self.args.inverse:
                        zero_scaler = - test_data.scaler.mean_[-1] / test_data.scaler.scale_[-1]
                    else:
                        zero_scaler = 0

                    # 将时间戳信息重新cat到output上
                    outputs = torch.cat((pred_time.unsqueeze(-1), outputs), dim=-1)
                    # 更改 tensor 中的目标通道：若小时在 0-6 或 18-23，则将对应位置改为 "原始值 0 经 scaler 后的数值"
                    zero_val = float(zero_scaler)  # 转成 Python float，避免 numpy 与 torch 混用
                    mask = (outputs[:, :, 0] <= 6) | (outputs[:, :, 0] >= 18)  # [B, L] 布尔掩码
                    outputs[:, :, -1] = outputs[:, :, -1].masked_fill(mask, zero_val)

                    # 去掉tensor中时间的列
                    outputs = outputs[:, :, 1:]


                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    if self.args.features == 'MS':
                        target_idx = -1
                        # 取出目标列的均值和尺度
                        mean_t = float(test_data.scaler.mean_[target_idx])
                        scale_t = float(test_data.scaler.scale_[target_idx])
                        outputs_norm =  outputs[..., 0]
                        outputs = (outputs_norm * scale_t + mean_t)[..., None]
                        batch_y_norm = batch_y[..., 0]
                        batch_y = (batch_y_norm * scale_t + mean_t)[..., None]
                    else:
                        outputs = test_data.inverse_transform(outputs)
                        batch_y = test_data.inverse_transform(batch_y)
                    # outputs = test_data.inverse_transform(outputs)
                    # batch_y = test_data.inverse_transform(batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    # x = input_x.detach().cpu().numpy()
                    inputs = input_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        if self.args.features == 'MS':
                            target_idx = -1
                            mean_t = float(test_data.scaler.mean_[target_idx])
                            scale_t = float(test_data.scaler.scale_[target_idx])
                            input_norm = inputs[..., target_idx]
                            inputs = (input_norm * scale_t + mean_t)[..., None]
                        else:
                            shape = input.shape
                            input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)

                    gt = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
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
        self.args.inverse = 1
        print("inverse:", self.args.inverse)

        if self.args.pred_len == 1:
            self.args.adaptive_norm = 0
            if self.args.model == 'iTransformer':
                self.args.use_norm = 'revin'
                print("使用iTransformer revin")

        self.device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
            if self.args.adaptive_norm:
                path_station = './station/' + '{}_s{}_p{}_{}_m'.format(self.args.data, self.args.seq_len,
                                                                       self.args.pred_len, self.args.model)
                print("加载归一模型：" + path_station + 'checkpoint.pth')
                ckpt_s = torch.load(os.path.join(path_station, 'checkpoint.pth'),
                                    map_location=self.device)
                state_s = ckpt_s.get('model', ckpt_s)  # 兼容有/无“model”键
                self.statistics_pred.load_state_dict(state_s, strict=False)
                self.statistics_pred.to(self.device)  # 保证模型加载在cpu，cuda都能运行

        preds = []

        self.model.eval()
        if self.args.adaptive_norm:
            self.statistics_pred.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'solar' in self.args.data.lower():
                    batch_time = batch_x[:, :, 0]
                    batch_x = batch_x[:, :, 1:]
                    batch_y = batch_y[:, :, 1:]
                    pred_time = infer_time(batch_time, self.args.pred_len)

                if 'Wind' in self.args.data or 'multi_wind' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)


                if self.args.adaptive_norm:
                    if self.args.use_norm == 'sliding':
                        batch_x, statistics_pred, statistics_seq = self.statistics_pred.normalize(batch_x)
                    else:
                        batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)

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
                        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        if pred_data.scale and self.args.inverse:
                            if self.args.features == 'MS':
                                target_idx = -1
                                # 取出目标列的均值和尺度
                                mean_t = float(pred_data.scaler.mean_[target_idx])
                                scale_t = float(pred_data.scaler.scale_[target_idx])
                                outputs_norm = outputs[..., 0]
                                outputs = (outputs_norm * scale_t + mean_t)[..., None]
                            else:
                                outputs = pred_data.inverse_transform(outputs)

                        # 获取缩放后的0
                        if not self.args.inverse:
                            zero_scaler = - pred_data.scaler.mean_[-1] / pred_data.scaler.scale_[-1]
                        else:
                            zero_scaler = 0

                        # 将时间戳信息重新cat到output上
                        outputs = torch.cat((pred_time.unsqueeze(-1), outputs), dim=-1)
                        # 更改 tensor 中的目标通道：若小时在 0-6 或 18-23，则将对应位置改为 "原始值 0 经 scaler 后的数值"
                        zero_val = float(zero_scaler)  # 转成 Python float，避免 numpy 与 torch 混用
                        mask = (outputs[:, :, 0] <= 6) | (outputs[:, :, 0] >= 18)  # [B, L] 布尔掩码
                        outputs[:, :, -1] = outputs[:, :, -1].masked_fill(mask, zero_val)

                        # 去掉tensor中时间的列
                        outputs = outputs[:, :, 1:]

                else:
                    if "Cycle" in self.args.model:
                        batch_cycle = batch_cycle.int().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_label = batch_x[:, -self.args.label_len:, :]
                    dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
                    # encoder - decoder
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                    if pred_data.scale and self.args.inverse:
                        if self.args.features == 'MS':
                            target_idx = -1
                            # 取出目标列的均值和尺度
                            mean_t = float(pred_data.scaler.mean_[target_idx])
                            scale_t = float(pred_data.scaler.scale_[target_idx])
                            outputs_norm = outputs[..., 0]
                            outputs = (outputs_norm * scale_t + mean_t)[..., None]
                        else:
                            outputs = pred_data.inverse_transform(outputs)

                    # 获取缩放后的0
                    # zero_scaler = - pred_data.scaler.mean_ / pred_data.scaler.scale_
                    if not self.args.inverse:
                        zero_scaler = - pred_data.scaler.mean_[-1] / pred_data.scaler.scale_[-1]
                    else:
                        zero_scaler = 0


                    # 将时间戳信息重新cat到output上
                    outputs = torch.cat((pred_time.unsqueeze(-1), outputs), dim=-1)
                    # 更改 tensor 中的目标通道：若小时在 0-6 或 18-23，则将对应位置改为 "原始值 0 经 scaler 后的数值"
                    zero_val = float(zero_scaler)  # 转成 Python float，避免 numpy 与 torch 混用
                    mask = (outputs[:, :, 0] <= 6) | (outputs[:, :, 0] >= 18)  # [B, L] 布尔掩码
                    outputs[:, :, -1] = outputs[:, :, -1].masked_fill(mask, zero_val)

                    # 去掉tensor中时间的列
                    outputs = outputs[:, :, 1:]

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
        # print("preds.shape:", preds.shape)
        preds = np.array(preds)
        print("preds.shape:", preds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        print("preds.shape:", preds.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        preds = preds.reshape(-1, preds.shape[-1])
        df = pd.DataFrame(preds, columns=[f"var_{i + 1}" for i in range(preds.shape[1])])
        excel_path = os.path.join(folder_path, 'real_prediction.xlsx')
        df.to_excel(excel_path, index=False)

        print(f"✅ 保存成功: {excel_path}")

        return
