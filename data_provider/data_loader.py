import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class solar_data(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Solar_Power.xlsx',
                 target='data', scale=True, timeenc=0, freq='h',cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_excel(os.path.join(self.root_path,
                                          self.data_path))
        #将数据发电量放在最后一列
        col = df_raw.pop("data")
        df_raw['data'] = col

        # 代表左右边界
        # num_train = int(len(df_raw) * 0.7)
        # num_test = int(len(df_raw) * 0.2)
        # num_valid = int(len(df_raw) * 0.1)
        border1s = [0, 9 * 30 * 24 - self.seq_len, 9 * 30 * 24 + 2 * 30 * 24 - self.seq_len]
        border2s = [len(df_raw), 9 * 30 * 24 + 2 * 30 * 24, 9 * 30 * 24 + 3 * 30 * 24]
        # border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 代表读取不包含第一列的所有列，第一列是时间
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            #测试问题
            print("target is:", self.target)
            df_data = df_raw[[self.target]]  # df_raw[[self.target]]返回一个DataFrame，即只包含一列的Pandas数据，
            # 而df_raw[self.target]返回一个Pandas Series。
        print("self.scale:", self.scale)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values  # .value返回一个二维的numpy数组

        df_stamp = df_raw[['date']][border1:border2]
        #重新处理太阳能中date中的格式
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # 计算置0
        # df_timestamp = df_raw[['date']][border1:border2]
        #运气bug，😁
        df_timestamp = df_raw[['date']][border1:border2]
        df_timestamp['date'] = pd.to_datetime(df_timestamp.date)
        df_timestamp['hour'] = df_timestamp.date.apply(lambda row: row.hour, 1)
        df_timestamp = df_timestamp.drop(['date'], axis=1)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            '''
                     在进入该分支时，我们选择 arg.embed 为 timeF，这意味着我们要对时间信息进行 
                     编码时间信息。freq "应该是最小的时间步长，有以下选项 
                      选项：[s:秒，t:分钟，h:小时，d:日，b:工作日，w:周，m:月]，也可以使用更详细的 freq，如 15 分钟或 3 小时')
                     因此，你应该检查数据的时间步长，并设置 “freq ”参数。
                     在对 time_features 进行编码后，每种日期信息格式将被编码成 
                     一个列表，每个元素表示该时间点的相对位置
                     (例如，周日、月日、日小时），并且每个元素都在范围[-0.5, 0.5]内进行归一化。  
                     '''
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # self.data_x = data[border1:border2]
        # 将时间戳信息重新贴到data里面
        self.data_x = np.concatenate(
            (df_timestamp['hour'].values.reshape(border2 - border1, 1), data[border1:border2]), axis=1)
        self.data_y = data[border1:border2]


        self.data_stamp = data_stamp

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark,cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


## TODO add cycle
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        # self.inverse = 1
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.cycle = cycle
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        df_raw = pd.read_excel(os.path.join(self.root_path,
                                            self.data_path))
        # 将数据发电量放在最后一列
        col = df_raw.pop("data")
        df_raw['data'] = col
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])

        # 计算置0
        df_timestamp = df_raw[['date']][border1:border2]
        df_timestamp['date'] = pd.to_datetime(df_timestamp.date)
        df_timestamp['hour'] = df_timestamp.date.apply(lambda row: row.hour, 1)
        df_timestamp = df_timestamp.drop(['date'], axis=1)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # self.data_x = data[border1:border2]
        if 'solar' in self.data_path.lower():
            self.data_x = np.concatenate(
                (df_timestamp['hour'].values.reshape(border2 - border1, 1), data[border1:border2]), axis=1)
        else:
            self.data_x = data[border1:border2]

        if self.inverse:
            # self.data_y = df_data.values[border1:border2]
            if 'solar' in self.data_path.lower():
                self.data_y = np.concatenate(
                    (df_timestamp['hour'].values.reshape(border2 - border1, 1), df_data[border1:border2]), axis=1)
            else:
                self.data_y = df_data.values[border1:border2]
        else:
            # self.data_y = data[border1:border2]
            if 'solar' in self.data_path.lower():
                self.data_y = np.concatenate(
                    (df_timestamp['hour'].values.reshape(border2 - border1, 1), data[border1:border2]), axis=1)
            else:
                self.data_y = data.values[border1:border2]
        self.data_stamp = data_stamp

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        print("s_end:", s_end)
        cycle_index = torch.tensor(self.cycle_index[s_end % len(self.cycle_index)])
        # cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark,cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_wind_multi_domain(Dataset):
    """
    多域风电数据集（不跨域、不跨断档滑窗）
    - 读取单个 CSV，其中必须包含：
        - 时间列：'date'（若存在'统计时间'会自动重命名为'date'）
        - 域ID列：默认 'domain_id'（可通过 domain_col 指定）
        - 其余为数值特征/目标列
    - 与本仓库其它 Dataset 保持接口一致：__getitem__ 返回 (seq_x, seq_y, seq_x_mark, seq_y_mark)
    - 只在“训练段”拟合 scaler；val/test 仅 transform
    - 每个 batch 的样本起点来自预先计算的 valid_starts，确保窗口不跨越域或断档
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='wind.csv',
                 target='OT', scale=True, timeenc=0, freq='h',cycle=None,
                 ):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.domain_col = 'domain_id'
        self.step_minutes = 10
        self.gap_mult = 2
        self.start_domains = None

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    @staticmethod
    def _segments_by_gap(time_series, expected_minutes, gap_mult=1.5):
        """
        给定按时间排序的 Series，返回连续片段 [a,b]（闭区间）列表。
        相邻时间差 > expected_minutes * gap_mult 认为有断档。
        """
        t = pd.to_datetime(time_series).reset_index(drop=True)
        diff = t.diff().dt.total_seconds().fillna(0) / 60.0
        cut_idx = np.where(diff.values > expected_minutes * gap_mult)[0]
        segs = []
        start = 0
        for c in cut_idx:
            segs.append((start, c - 1))
            start = c
        segs.append((start, len(t) - 1))
        return segs

    def _check_valid_starts(self):
        ok = True
        for i, sb in enumerate(self.valid_starts):
            s_begin = int(sb)
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            Lx = s_end - s_begin
            Ly = r_end - r_begin
            # 越界 or 长度不等就报
            cond = (
                    s_begin >= 0 and r_begin >= 0 and
                    s_end <= len(self.data_x) and
                    r_end <= len(self.data_y) and
                    Lx == self.seq_len and
                    Ly == (self.label_len + self.pred_len) and
                    self.data_stamp.shape[0] >= max(s_end, r_end)
            )
            if not cond:
                print("[BAD START]", dict(
                    idx=i, s_begin=int(s_begin), s_end=int(s_end),
                    r_begin=int(r_begin), r_end=int(r_end),
                    len_x=len(self.data_x), len_y=len(self.data_y),
                    seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len
                ))
                ok = False
                break
        if ok:
            print("[CHECK] all valid_starts OK")

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # 统一时间列名
        if 'date' not in df_raw.columns and '统计时间' in df_raw.columns:
            df_raw = df_raw.rename(columns={"统计时间": "date"})
        if 'date' not in df_raw.columns:
            raise ValueError("数据中必须包含时间列 'date'（或原名 '统计时间'）。")
        if self.domain_col not in df_raw.columns:
            raise ValueError(f"数据中必须包含域ID列 '{self.domain_col}'。")

        # 排序，便于按域、按时间切片
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.sort_values([self.domain_col, 'date']).reset_index(drop=True)

        # 选择特征列
        if self.features in ['M', 'MS']:
            cols_data = [c for c in df_raw.columns if c not in ['date', self.domain_col]]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"不支持的 features 类型: {self.features}")

        # --- 仅用“各域的训练段”拟合 scaler ---
        train_mask_all = np.zeros(len(df_raw), dtype=bool)
        for d, g in df_raw.groupby(self.domain_col, sort=False):
            n = len(g)
        #将训练与所有数据全部进行归一化
            # n_tr = int(n)
            n_tr = int(n * 0.8)
            # # 训练段是该域的前 70%（按时间）
            idx = g.index.values
            train_mask_all[idx[:n_tr]] = True
        if self.scale:
            self.scaler.fit(df_data.values[train_mask_all])
            data_all = self.scaler.transform(df_data.values)
        else:
            data_all = df_data.values

        # --- 根据 flag 取出各域当前 split 的行，并记录连续片段，构造 valid_starts ---
        data_rows = []
        stamp_rows = []
        domain_rows = []          # 与 data_rows 对齐的域ID（行级）
        valid_starts = []         # 窗口起点（在 concat 后的索引）
        base = 0

        for d, g in df_raw.groupby(self.domain_col, sort=False):
            # print("d:",d)  #X03
            n = len(g)
            n_tr = int(n * 0.8)
            n_te = int(n * 0.1)
            n_va = n - n_tr - n_te
            # # 三段边界（域内）
            b1s = [0, n_tr - self.seq_len, n - n_te - self.seq_len]
            b2s = [n_tr, n_tr + n_va, n]
            # b2s = [n, n_tr + n_va, n]
            # if d == 'X03':
            #     b1s = [0, n_tr - self.seq_len, n - n_te - self.seq_len]
            #     b2s = [n_tr, n_tr + n_va, n]
            # else:
            #     b1s = [0, n_tr - self.seq_len, n - n_te - self.seq_len]
            #     b2s = [n, n_tr + n_va, n]
            b1 = b1s[self.set_type]
            b2 = b2s[self.set_type]
            b1 = max(b1, 0)
            b2 = max(b2, 0)
            # if d == 'X03' and self.set_type == 2:
            #     b1 = b1s[self.set_type]
            #     b2 = b2s[self.set_type]
            #     gi = g.iloc[b1:b2].copy()  # 当前 flag 对应的域内片段
            #     print("b1:",b1)
            #     print("b2:",b2)
            #     print("gi:",len(gi))
            # else:
            #     gi = None
            # if d == 'X03':
            #     b1 = b1s[self.set_type]
            #     b2 = b2s[self.set_type]
            #     gi = g.iloc[b1:b2].copy()  # 当前 flag 对应的域内片段
            #     print("b1:",b1)
            #     print("b2:",b2)
            #     print("gi:",len(gi))
            # else:
            #     gi = None
            # if gi is None or gi.empty:
            #     continue
            gi = g.iloc[b1:b2].copy()
            if gi.empty:
                continue

            # 取数值（已整体变换过的 data_all），用原始索引映射
            Xi = data_all[gi.index.values]
            # 时间编码（不喂绝对时间也可以，这里保持与仓库风格一致）
            if self.timeenc == 0:
                tmp = pd.DataFrame({
                    'month': gi['date'].dt.month.values,
                    'day': gi['date'].dt.day.values,
                    'weekday': gi['date'].dt.weekday.values,
                    'hour': gi['date'].dt.hour.values
                })
                data_stamp_i = tmp.values
            elif self.timeenc == 1:
                dates_i = pd.DatetimeIndex(pd.to_datetime(gi['date'], errors='coerce'))
                # data_stamp_i = time_features(gi['date'].values, freq=self.freq).transpose(1, 0)
                data_stamp_i = time_features(dates_i, freq=self.freq).transpose(1, 0)


            # 断档切段，并在段内产“合法起点”（不跨断档、不跨域）
            segs = self._segments_by_gap(gi['date'], expected_minutes=self.step_minutes, gap_mult=self.gap_mult)
            for a, b in segs:
                L = b - a + 1
                # 起点定义与其它 Dataset 一致：窗口结束于 t，预测段从 t+1 开始
                # for t in range(a + self.seq_len - 1, a + L - self.pred_len):
                #     valid_starts.append(base + t)
                left = a + max(self.seq_len, self.label_len) - 1
                right = a + L - self.pred_len
                if right > left:
                    for t in range(left, right):
                        valid_starts.append(base + t)

            # 累积到全局拼接数组
            # if d == 'X03' and self.set_type == 2:
            #     data_rows.append(Xi)
            # else:
            #     data_rows.append(Xi)
            data_rows.append(Xi)
            stamp_rows.append(data_stamp_i)
            domain_rows.append(np.full(len(gi), d))
            base += len(gi)

        if not data_rows:
            raise ValueError("当前 split 下没有可用样本，请检查数据或 split 设置。")

        self.data_x = np.concatenate(data_rows, axis=0)
        self.data_y = self.data_x
        self.data_stamp = np.concatenate(stamp_rows, axis=0)
        self.domain_rows = np.concatenate(domain_rows, axis=0)  # 行级域ID
        self.valid_starts = np.array(valid_starts, dtype=np.int64)
        #防御出现坏点
        vs = []
        T = len(self.data_x)
        for sb in self.valid_starts:
            s_begin = int(sb)
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            if 0 <= s_begin and 0 <= r_begin and s_end <= T and r_end <= T:
                vs.append(sb)
        self.valid_starts = np.array(vs, dtype=np.int64)
        # 方便外部做“按域均衡采样”的索引：每个起点对应的域ID
        self.start_domains = self.domain_rows[self.valid_starts]
        print("start_domains", self.start_domains)
        # 将domain映射为数字
        unique_ids = np.unique(self.start_domains)  # 自动排序去重
        mapping = {v: i for i, v in enumerate(unique_ids)}
        self.start_domains = np.vectorize(mapping.get)(self.start_domains)
        print("start_domains", self.start_domains)
        self._check_valid_starts()



    def __getitem__(self, index):
        s_begin = int(self.valid_starts[index])
        # s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        env_id = torch.tensor(int(self.start_domains[index]), dtype=torch.long)
        cycle_index = torch.tensor(1)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, env_id

    def __len__(self):
        return len(self.valid_starts)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)