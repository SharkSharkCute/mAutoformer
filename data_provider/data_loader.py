import os
import pandas as pd
import traceback
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import numpy as np
import calendar
from datetime import datetime
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
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
            data_stamp = df_stamp.drop(columns=['date']).values

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
                 target='OT', scale=True, timeenc=0, freq='t'):
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
            data_stamp = df_stamp.drop(columns=['date']).values

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

"""
python -u run.py `
--is_training 1 `
--root_path ./data/ `
--data_path test.csv `
--model_id FD_5_25 `
--model Autoformer `
--data custom `
--features MS `
--target RUL `
--seq_len 10 `
--label_len 0 `
--pred_len 1 `
--e_layers 4 `
--d_layers 2 `
--factor 3 `
--enc_in 25 `
--dec_in 25 `
--c_out 1 `
--embed none `
--des quick_test `
--itr 1 `
--train_epochs 100 `
--batch_size 128 `
--d_model 768 `
--e_layers 4 `
--d_layers 2 `
--num_workers 4
"""
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
        # print(cols)
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
            data_stamp = df_stamp.drop(['date'],axis=1).values
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
        print("---------------")
        traceback.print_stack()
        print("---------------")
        print("")
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

"""
python -u run.py `
--is_training 1 `
--root_path ./data/ `
--data_path NULL `
--seq_len 30 `
--label_len 0 `
--pred_len 1 `
--features NULL `
--target RUL `
--freq NULL `
--model_id FD_5_25 `
--model Autoformer `
--data CMAPSS `
--e_layers 4 `
--d_layers 2 `
--factor 3 `
--enc_in 24 `
--dec_in 24 `
--c_out 1 `
--embed none `
--des quick_test `
--itr 1 `
--train_epochs 100 `
--d_model 768 `
--num_workers 5 `
--batch_size 128
"""
class Dataset_CMAPSS(Dataset):
    def __init__(self, root_path, data_path,size, features, target, timeenc, freq,  flag, scaler=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.target = target
        print(flag)
        assert flag in ['train', 'test', 'val', 'real_test']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'real_test':3}
        self.set_type = type_map[flag]
        
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler
        self.__read_data__()

    def build_timestamp(self, stamp_seq, base_year=1970):
        n = len(stamp_seq)
        ts_list = []
        for idx in range(n):
            row = stamp_seq.iloc[idx]
            year  = base_year + idx // 360  
            month = int(row.month)
            day   = int(row.day)
            max_day = calendar.monthrange(year, month)[1]
            if day > max_day:
                day = max_day
            hour   = int(row.hour)
            minute = int(row.minute) * 15   
            ts_list.append(datetime(year, month, day, hour, minute, 0).strftime('%Y-%m-%d %H:%M:%S'))
        return pd.Series(ts_list, name="date")
    
    def __add_remaining_useful_life__(self, df):
        max_cycle = df["time_cycles"].max()
        df["RUL"] = max_cycle - df["time_cycles"]
        return df
    
    def __save_autoformer_csv__(self, file_name, df_list):
        save_path = os.path.join(self.root_path, file_name)
        all_df = pd.concat(df_list, ignore_index=True)
        all_df.to_csv(save_path, index=False)
        print(f'[Info] Saved preprocessed data to {save_path}')
    
    def _load_train_data_(self):
        names = ['unit_nr', 'time_cycles']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
        col_names = names + setting_names + sensor_names 

        pd_list = []
        self.data_stamp = []
        debug_num = 0
        for file_index in range(1,5):
            index_train_file = os.path.join(self.root_path, f'train_FD00{file_index}.txt')
            df = pd.read_csv(index_train_file, sep=r'\s+', header=None, names=col_names)

            grouped = df.groupby('unit_nr')
            for index, unit_df in grouped:
                unit_df = self.__add_remaining_useful_life__(unit_df)
                cols = list(unit_df.columns)
                cols.remove('time_cycles')
                cols.remove('unit_nr')
                #cols.remove('RUL')
                unit_df = unit_df[cols]

                time_step = unit_df.shape[0]
                
                stamp_seq = pd.DataFrame({
                    "month": [(i // 30) % 12 + 1 for i in range(time_step)],
                    "day": [(i % 30) + 1 for i in range(time_step)],
                    "weekday": [i % 7 for i in range(time_step)],
                    "hour": [i % 24 for i in range(time_step)],
                    "minute": [(i % 60) // 15 for i in range(time_step)], 
                })
                stamp_seq = stamp_seq.values.astype(np.float32) 
                unit_df = unit_df.reset_index(drop=True)
                #unit_df = pd.concat([self.build_timestamp(stamp_seq), unit_df], axis=1)
                
                length = self.seq_len + self.pred_len
                if unit_df.shape[0] < length:
                    continue
                
                pd_list.append(unit_df)
                self.data_stamp.append(stamp_seq)

        self.targetCol = [df.iloc[:, -1].tolist() for df in pd_list]
        for df in pd_list:
            df.iloc[:, -1] = 0

        return pd_list
    
    def __load_test_data_(self):
        names = ['unit_nr', 'time_cycles']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
        col_names = names + setting_names + sensor_names 

        pre_list = []
        self.data_stamp = []
        self.targetCol = []
        for file_index in range(1,5):
            index_train_file = os.path.join(self.root_path, f'test_FD00{file_index}.txt')
            df = pd.read_csv(index_train_file, sep=r'\s+', header=None, names=col_names)

            org_rul_list = []
            with open(os.path.join(self.root_path, f'RUL_FD00{file_index}.txt'), "r") as f:
                org_rul_list = [int(line.strip()) for line in f if line.strip()]
            #print(org_rul_list)
            grouped = df.groupby('unit_nr')
            if(len(org_rul_list) != grouped.ngroups):
                print("len(org_rul_list) != grouped.ngroups")
                quit()

            for index, (unit_id, unit_df) in enumerate(grouped):
                cols = list(unit_df.columns)
                cols.remove('time_cycles')
                cols.remove('unit_nr')

                unit_df = unit_df[cols]

                time_step = unit_df.shape[0]
                stamp_seq = pd.DataFrame({
                    "month": [(i // 30) % 12 + 1 for i in range(time_step)],
                    "day": [(i % 30) + 1 for i in range(time_step)],
                    "weekday": [i % 7 for i in range(time_step)],
                    "hour": [i % 24 for i in range(time_step)],
                    "minute": [(i % 60) // 15 for i in range(time_step)], 
                })
                stamp_seq = stamp_seq.values.astype(np.float32) 
                unit_df = unit_df.reset_index(drop=True)
                unit_df["RUL"] = 0 
                
                length = self.seq_len + self.pred_len
                if unit_df.shape[0] < length:
                    #print(f"Drop {index}, {org_rul_list[index]}")
                    continue
                
                unit_df = unit_df.iloc[-length:]
                pre_list.append(unit_df)
                self.data_stamp.append(stamp_seq)
                #print(org_rul_list[index])
                self.targetCol.append(org_rul_list[index])
        return pre_list
    
    def __read_data__(self):
        pd_list = self._load_train_data_()
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        total_len = len(pd_list)
        num_train = int(total_len * 0.7)
        num_vali = int(total_len * 0.2)
        num_test = int(total_len - num_train - num_vali)

        concat_df = pd.concat(pd_list[:num_train], ignore_index=True)
        self.scaler_x.fit(concat_df.values)

        concat_targetCol_df = np.concatenate(self.targetCol[:num_train]).reshape(-1, 1)
        self.scaler_y.fit(concat_targetCol_df)
        
        if self.set_type != 3:
            self.data_x = [self.scaler_x.transform(df.values) for df in pd_list]
            if self.set_type == 0:
                self.targetCol = self.targetCol[:num_train]
                self.data_x = self.data_x[:num_train]
                self.data_stamp = self.data_stamp[:num_train]
            elif self.set_type == 1:
                self.targetCol = self.targetCol[num_train: num_train + num_vali]
                self.data_x = self.data_x[num_train: num_train + num_vali]
                self.data_stamp = self.data_stamp[num_train:num_train + num_vali] 
            elif self.set_type == 2:
                self.targetCol = self.targetCol[num_train + num_vali:]
                self.data_x = self.data_x[num_train + num_vali:]
                self.data_stamp = self.data_stamp[num_train + num_vali:]
            
            self.data_y = [x.copy() for x in self.data_x]
            for i in range(len(self.data_y)):
                if len(self.data_y[i]) != len(self.targetCol[i]):
                    print(f"{i}: {len(self.data_y[i])} != {len(self.targetCol[i])}")
                    quit()
                self.data_y[i][:, -1] = self.scaler_y.transform(
                    np.array(self.targetCol[i]).reshape(-1, 1)
                ).flatten()

            #self.data_y = [self.scaler.transform(y) for y in self.data_y]

        else:
            pred_list = self.__load_test_data_()
            self.data_x = [self.scaler_x.transform(df.values) for df in pred_list]
            self.data_y = [pred.copy() for pred in pred_list]
            for i in range(len(self.data_y)):
                self.data_y[i].iloc[-1, -1] = self.targetCol[i]
            self.data_y = [self.scaler_x.transform(y) for y in self.data_y]

        self.index_list = []
        for i, seq in enumerate(self.data_x):
            max_len = max(0, len(seq) - self.seq_len - self.pred_len + 1)
            for j in range(max_len):
                self.index_list.append((i, j))
        
        #print(self.targetCol)
        
    def __getitem__(self, index):
        i, j = self.index_list[index]

        s_begin = j
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[i][s_begin:s_end]
        seq_y = self.data_y[i][r_begin:r_end]
        seq_x_mark = self.data_stamp[i][s_begin:s_end]
        seq_y_mark = self.data_stamp[i][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.index_list)

    def inverse_transform(self, data):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return self.scaler_x.inverse_transform(data)
    
"""
python -u run.py `
--is_training 0 `
--root_path ./data/ `
--data_path NULL `
--seq_len 30 `
--label_len 0 `
--pred_len 1 `
--features NULL `
--target RUL `
--freq NULL `
--model_id FD_5_25 `
--model Autoformer `
--data CMAPSS `
--checkpoints ./checkpoints/ `
--e_layers 4 `
--d_layers 2 `
--factor 3 `
--enc_in 24 `
--dec_in 24 `
--c_out 1 `
--embed none `
--des quick_test `
--itr 1 `
--d_model 768 `
--batch_size 1
"""

