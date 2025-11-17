from data_provider.data_loader import (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS,
                                       Dataset_Pred,solar_data,Dataset_wind_multi_domain)
from torch.utils.data import DataLoader
import torch
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'solar_data': solar_data,
    'Wind_multi_domain': Dataset_wind_multi_domain,
}

#是否要进行多域随机采样
sampler_flag = True
print("是否进行多域随机采样：",sampler_flag)


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        cycle=args.cycle
    )
    print(flag, len(data_set))
    print(batch_size)
    if sampler_flag and 'wind' in args.data.lower() and flag != 'pred':
        domains = np.unique(data_set.start_domains)
        d2i = {int(d): np.where(data_set.start_domains == d)[0].tolist() for d in domains}
        sampler = BalancedBatchSampler(d2i, batch_size=batch_size, domains_per_batch=8,drop_last=drop_last,shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_sampler=sampler,
            num_workers=args.num_workers,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader

    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)
    # return data_set, data_loader



class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    让每个 batch 同时包含多个域的起点索引（基于 Dataset_wind_multi_domain.start_domains）
    使用方式：
        ds = Dataset_wind_multi_domain(..., flag='train', ...)
        # 构造 domain -> 起点索引 的映射
        domains = np.unique(ds.start_domains)
        d2i = {d: np.where(ds.start_domains == d)[0].tolist() for d in domains}
        sampler = BalancedBatchSampler(d2i, batch_size=64, domains_per_batch=2)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, drop_last=True)
    """
    def __init__(self, domain_to_indices: dict, batch_size: int, domains_per_batch: int = 2,shuffle=True, drop_last=True):
        self.domain_to_indices = {int(d): list(v) for d, v in domain_to_indices.items()}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.domains_per_batch = max(1, min(domains_per_batch, len(self.domain_to_indices)))

    def __iter__(self):
        pools = {d: v[:] for d, v in self.domain_to_indices.items()}
        rng = np.random.default_rng()
        for d in pools: rng.shuffle(pools[d])
        while True:
            avail = [d for d, v in pools.items() if len(v)]
            if len(avail) < self.domains_per_batch:
                break
            chosen = rng.choice(avail, size=self.domains_per_batch, replace=False).tolist()
            per = max(1, self.batch_size // self.domains_per_batch)
            batch = []
            for d in chosen:
                take = min(per, len(pools[d]))
                batch += pools[d][:take]
                pools[d] = pools[d][take:]
            if len(batch) < self.batch_size:
                for d in list(pools.keys()):
                    need = self.batch_size - len(batch)
                    if need <= 0: break
                    take = min(need, len(pools[d]))
                    batch += pools[d][:take]
                    pools[d] = pools[d][take:]
            if not batch:
                break
            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.domain_to_indices.values())
        return (total + self.batch_size - 1) // self.batch_size