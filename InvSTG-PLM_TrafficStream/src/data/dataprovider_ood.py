import torch
import numpy as np
import torch.utils.data
from utils.utils import get_adjacency_matrix_2direction, get_adjacency_matrix, get_randmask, linear_interpolate, get_block_mask
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
from data.scaler import StandardScaler, MinMaxScaler
import os
import re

def generate_sample_by_sliding_window(data, sample_len, step=1):

    sample = []

    for i in range(0, data.shape[0] - sample_len, step):

        sample.append(torch.unsqueeze(data[i:i+sample_len] , 0))
    
    if (data.shape[0] - sample_len) % step !=0 :
        sample.append(torch.unsqueeze(data[-sample_len:] , 0))

    sample = torch.concat(sample,dim=0)

    return sample


def _has_structure_shift(max_increase_ratio):
    return max_increase_ratio is not None and float(max_increase_ratio) > 0


def _safe_std(value):
    if isinstance(value, torch.Tensor):
        return value if value.abs().item() > 0 else torch.tensor(1.0, device=value.device)
    return value if abs(float(value)) > 0 else 1.0

class BasicDataset(torch.utils.data.Dataset):

    history  : torch.Tensor #(B,sample_len,node_num,features)
    target   : torch.Tensor #(B,(sample_len)+predict_len,node_num,features)
    timestamp: torch.Tensor #(B,sample_len*2, 4 ) 12(months) + 31(day) + 7(week) + 24(day)
    cond_mask: torch.Tensor #(B,sample_len,node_num)
    ob_mask  : torch.Tensor #(B,sample_len,node_num)
    env_ids  : torch.Tensor #(B,) 
    spurious_features: torch.Tensor #(B,) 

    def __init__(self, history, target, timestamp, cond_mask, ob_mask, 
                 env_ids=None, spurious_features=None, training=False) -> None:
        
        self.history = history
        self.target = target
        self.timestamp = timestamp
        self.cond_mask = cond_mask
        self.ob_mask = ob_mask
        self.env_ids = env_ids
        self.spurious_features = spurious_features
        self.training = training

    def __len__(self):
        return self.history.shape[0]
    
    def __getitem__(self, index):
        if self.env_ids is not None and self.spurious_features is not None:
            return (
                self.history[index], 
                self.target[index], 
                self.timestamp[index], 
                self.cond_mask[index], 
                self.ob_mask[index],
                self.env_ids[index],
                self.spurious_features[index]
            )
        else:
            return (
                self.history[index], 
                self.target[index], 
                self.timestamp[index], 
                self.cond_mask[index], 
                self.ob_mask[index]
            )


class DataProvider():

    node_num : int
    features : int
    data  : torch.Tensor #(T, node_num,features)
    timestamp: torch.Tensor #(T,sample_len+predict_len, 5 ) 12(months) + 31(day) + 7(week) + 24(hours) +60(minutes)
    mask:torch.Tensor #(T, node_num,features) 
    mask_eval:torch.Tensor #(T, node_num,features) position for evaluate, mask_eval&mask=mask

    def __init__(self, data_path, adj_path ,dataset, node_shuffle_seed=None) -> None:

        self.dataset = dataset

        self.data, self.node_num, self.features, \
        self.adj_mx, self.distance_mx, \
        self.timestamp,self.mask,self.mask_eval =  self.read_data(data_path, adj_path)

        if node_shuffle_seed is not None:
            rdm = np.random.RandomState(node_shuffle_seed)
            idx = np.arange(self.node_num)
            rdm.shuffle(idx)
            idx = torch.from_numpy(idx)
            self.data = self.data[:,idx,:]
            self.adj_mx = self.adj_mx[idx,:][:,idx]
            node_order = np.arange(self.node_num)
            rdm.shuffle(node_order)
            self.node_order = torch.from_numpy(node_order)
        else:
            node_order = np.arange(self.node_num)
            self.node_order = torch.from_numpy(node_order)

    def getdataset(self, sample_len, output_len, window_size, \
                input_dim, output_dim, \
               train_ratio, val_ratio, target_strategy, few_shot=1, 
               device=None, num_envs=4, max_increase_ratio=0, test_increase_ratio=0, test_decrease_ratio=0):

        if device is None:
            raise ValueError("device parameter is required. Please pass args.device explicitly.")

        self.data = self.data.float().to(device)
        self.timestamp = self.timestamp.to(device)
        self.mask = self.mask.float().to(device)
        self.mask_eval = self.mask_eval.float().to(device)
        # self.node_order = self.node_order.long().to(device)

        self.adj_mx_train = self.adj_mx
        self.distance_mx_train = self.distance_mx
        self.adj_mx_test = self.adj_mx
        self.distance_mx_test = self.distance_mx

        if _has_structure_shift(max_increase_ratio): #with structure shifts
            node_num_training = int(self.node_num / (1.+ max_increase_ratio))
            node_num_test_increase = int(node_num_training * test_increase_ratio)
            node_num_test_decrease = int(node_num_training * test_decrease_ratio)
            node_training = self.node_order[:node_num_training]
            node_test = np.concatenate([self.node_order[:node_num_training-node_num_test_decrease], 
                                        self.node_order[node_num_training:node_num_training+node_num_test_increase]])
            self.adj_mx_train = self.adj_mx[node_training,:][:,node_training]
            self.distance_mx_train = self.distance_mx[node_training,:][:,node_training]
            self.adj_mx_test = self.adj_mx[node_test,:][:,node_test]
            self.distance_mx_test = self.distance_mx[node_test,:][:,node_test]

        all_len = self.data.shape[0]
        train_len = int(all_len * train_ratio)
        val_len = int(all_len * val_ratio)

        train_range = [0,int(train_len * few_shot)]
        val_range = [train_len, train_len+val_len]
        test_range = [train_len+val_len, all_len]

        if _has_structure_shift(max_increase_ratio):
            train_data = self.data[train_range[0]:train_range[1], node_training, :]
            train_mask = self.mask[train_range[0]:train_range[1], node_training, :]
            train_mask_eval = self.mask_eval[train_range[0]:train_range[1], node_training, :]
        else:
            train_data = self.data[train_range[0]:train_range[1]]
            train_mask = self.mask[train_range[0]:train_range[1]]
            train_mask_eval = self.mask_eval[train_range[0]:train_range[1]]
        train_te = self.timestamp[train_range[0]:train_range[1]]
        train_sample = generate_sample_by_sliding_window(train_data, sample_len=window_size)
        train_x, train_y = train_sample[:,:sample_len,...][...,:input_dim],  train_sample[:,-output_len:,...][...,:output_dim]

        if _has_structure_shift(max_increase_ratio):
            scaler_mask = self.mask[train_range[0]:train_range[1], node_training, :]!=0
            scaler_data = self.data[train_range[0]:train_range[1], node_training, :]
        else:
            scaler_mask = self.mask[train_range[0]:train_range[1]]!=0
            scaler_data = self.data[train_range[0]:train_range[1]]
        dim = scaler_data.shape[-1]
        mean = [scaler_data[...,i:i+1][scaler_mask[...,i:i+1]].mean() for i in range(dim)]
        std = [_safe_std(scaler_data[...,i:i+1][scaler_mask[...,i:i+1]].std()) for i in range(dim)]
        self.scaler = self.getscalerclass()(mean,std)

        train_x = self.scaler.transform(train_x)
        train_te = generate_sample_by_sliding_window(train_te, sample_len=window_size)
        train_ob_mask = generate_sample_by_sliding_window(train_mask, sample_len=window_size)[...,:input_dim]
        if target_strategy=='random':
            train_cond_mask = get_randmask(train_ob_mask[:,:sample_len],0,1).to(device)[...,:input_dim]
        else :
            train_len = train_ob_mask.shape[0]
            train_cond_mask = torch.concat([get_block_mask(train_ob_mask[i,:sample_len].cpu(), target_strategy='hybrid',min_seq=3, max_seq=12).to(device).unsqueeze(0) for i in range(train_len)])[...,:input_dim]
        
        if num_envs > 0:
            
            num_train_samples = train_x.shape[0]
            samples_per_env = num_train_samples // num_envs
            
            train_g = torch.arange(num_train_samples, device=device) // samples_per_env
            train_g = torch.clamp(train_g, 0, num_envs - 1)  
            
            train_c = torch.rand(num_train_samples, device=device)
            
            train_dataset = BasicDataset(
                history=train_x, 
                target=train_y, 
                timestamp=train_te,
                cond_mask=train_cond_mask,
                ob_mask=train_ob_mask,
                env_ids=train_g,
                spurious_features=train_c,
                training=True
            )
        else:
            train_dataset = BasicDataset(
                history=train_x, 
                target=train_y, 
                timestamp=train_te,
                cond_mask=train_cond_mask,
                ob_mask=train_ob_mask,
                training=True
            )

        if _has_structure_shift(max_increase_ratio):
            val_data = self.data[val_range[0]:val_range[1], node_training, :]
            val_mask = self.mask[val_range[0]:val_range[1], node_training, :]
            val_mask_eval = self.mask_eval[val_range[0]:val_range[1], node_training, :]
        else:
            val_data = self.data[val_range[0]:val_range[1]]
            val_mask, val_mask_eval = self.mask[val_range[0]:val_range[1]], self.mask_eval[val_range[0]:val_range[1]]
        val_te = self.timestamp[val_range[0]:val_range[1]]
        val_sample = generate_sample_by_sliding_window(val_data, sample_len=window_size)
        val_x, val_y = val_sample[:,:sample_len,...][...,:input_dim],  val_sample[:,-output_len:,...][...,:output_dim]
        val_x = self.scaler.transform(val_x)
        val_te = generate_sample_by_sliding_window(val_te, sample_len=window_size)

        val_ob_mask = generate_sample_by_sliding_window(val_mask_eval, sample_len=window_size)[...,:input_dim]
        val_cond_mask = generate_sample_by_sliding_window(val_mask, sample_len=window_size)[:,:sample_len][...,:input_dim]
    
        val_dataset = BasicDataset(history=val_x, target=val_y, timestamp=val_te, cond_mask=val_cond_mask,ob_mask=val_ob_mask)

        if _has_structure_shift(max_increase_ratio):
            test_data = self.data[test_range[0]:test_range[1], node_test, :]
            test_mask = self.mask[test_range[0]:test_range[1], node_test, :]
            test_mask_eval = self.mask_eval[test_range[0]:test_range[1], node_test, :]
        else:
            test_data = self.data[test_range[0]:test_range[1]]
            test_mask, test_mask_eval = self.mask[test_range[0]:test_range[1]], self.mask_eval[test_range[0]:test_range[1]]
        test_te = self.timestamp[test_range[0]:test_range[1]]
        test_sample = generate_sample_by_sliding_window(test_data, sample_len=window_size)
        test_x, test_y = test_sample[:,:sample_len,...][...,:input_dim],  test_sample[:,-output_len:,...][...,:output_dim]
        test_x = self.scaler.transform(test_x)
        test_te = generate_sample_by_sliding_window(test_te, sample_len=window_size)

        test_ob_mask = generate_sample_by_sliding_window(test_mask_eval, sample_len=window_size)[...,:input_dim]
        test_cond_mask = generate_sample_by_sliding_window(test_mask, sample_len=window_size)[:,:sample_len][...,:input_dim]
        test_dataset = BasicDataset(history=test_x, target=test_y, timestamp=test_te, cond_mask=test_cond_mask,ob_mask=test_ob_mask)

        return train_dataset, val_dataset, test_dataset
                

    def getadj(self):

        return self.adj_mx_train, self.distance_mx_train, self.adj_mx_test, self.distance_mx_test
    
    def getscalerclass(self):
        
        return StandardScaler


def generatetimestamp(start, periods, freq):

    time = pd.date_range(start=start,periods=periods,freq=freq)

    month = np.reshape(time.month, (-1, 1))
    dayofmonth = np.reshape(time.day, (-1, 1))
    dayofweek = np.reshape(time.weekday, (-1, 1))
    hour = np.reshape(time.hour, (-1, 1))
    minute = np.reshape(time.minute, (-1, 1))

    timestamp = np.concatenate((month, dayofmonth, dayofweek, hour, minute), -1)

    timestamp = torch.tensor(timestamp)

    return timestamp

timestampfun = {
    'PEMS08': lambda T : generatetimestamp(start='20160701 00:00:00',periods=T,freq='5min'),
    'PEMS07': lambda T : generatetimestamp(start='20170501 00:00:00',periods=T,freq='5min'),
    'PEMS04': lambda T : generatetimestamp(start='20180101 00:00:00',periods=T,freq='5min'),
    'PEMS03': lambda T : generatetimestamp(start='20180901 00:00:00',periods=T,freq='5min'),
    'NYCTAXI': lambda T : generatetimestamp(start='20160401 00:00:00',periods=T,freq='30min'),
    'CHIBIKE': lambda T : generatetimestamp(start='20160401 00:00:00',periods=T,freq='30min'),
}

class PEMSFLOWProvider(DataProvider):

    def read_data(self, data_path,  adj_path = None ) -> None:


        data = torch.from_numpy(np.load(data_path)['data'][...,:])
        
        T, node_num, features = data.shape
        if 'PEMS03' in self.dataset:
            id_filename = adj_path.replace('csv','txt')
        else :
            id_filename = None
        adj_mx, distance_mx = get_adjacency_matrix(adj_path, node_num, id_filename)
        adj_mx = np.where(np.eye(node_num).astype('bool'),1,adj_mx)

        timestamp = timestampfun[self.dataset[:6]](T)

        return data, node_num, features, \
               adj_mx, distance_mx, \
               timestamp,torch.ones_like(data),torch.ones_like(data)

class PEMSMISSINGProvider(DataProvider):

    def read_data(self, data_path,  adj_path = None ) -> None:

        dir_name = os.path.dirname(data_path)
        fileName = os.path.basename(data_path)

        true_datapath = os.path.join(dir_name,fileName.replace('miss','true')) 
        miss_datapath = os.path.join(dir_name,fileName.replace('true','miss')) 

        miss_data = np.load(miss_datapath)
        mask = torch.from_numpy(miss_data['mask'][:, :, :].astype('long'))
        data = np.load(true_datapath)['data'].astype(np.float32)[:, :, :]
        data[np.isnan(data)] = 0
        data = torch.from_numpy(data)

        T, node_num, features = data.shape
 
        adj_mx, distance_mx = get_adjacency_matrix(adj_path, node_num)
        adj_mx = np.where(np.eye(node_num).astype('bool'),1,adj_mx)

        timestamp = timestampfun[self.dataset[:6]](T)

        return data, node_num, features, \
               adj_mx, distance_mx, \
               timestamp,mask,torch.ones_like(data)
    

class NYCTAXIProvider(DataProvider):

    def read_data(self, data_path,  adj_path = None ) -> None:


        data = torch.from_numpy(np.load(data_path)['data'][...,:])
        data = np.transpose(data,(1,0,2))
        
        T, node_num, features = data.shape

        adj_mx, distance_mx = np.ones((node_num,node_num)).astype(np.float32),np.ones((node_num,node_num)).astype(np.float32)
        timestamp = timestampfun[self.dataset](T)

        return data, node_num, features, \
               adj_mx, distance_mx, \
               timestamp,torch.ones_like(data),torch.ones_like(data)


def _resolve_yearly_path(path_value, year, default_name):
    if path_value is None:
        return None

    normalized = os.path.normpath(path_value)
    if os.path.isdir(normalized):
        return os.path.join(normalized, default_name.format(year=year))

    if '{year}' in normalized:
        return normalized.format(year=year)

    if '%YEAR%' in normalized:
        return normalized.replace('%YEAR%', str(year))

    dirname = os.path.dirname(normalized)
    basename = os.path.basename(normalized)
    if re.search(r'20\d{2}', basename):
        return os.path.join(dirname, re.sub(r'20\d{2}', str(year), basename, count=1))

    candidate = os.path.join(dirname, default_name.format(year=year))
    if dirname and os.path.isdir(dirname):
        return candidate
    return normalized


def _extract_npz_array(npz_file):
    for key in ('data', 'x', 'flow', 'traffic', 'arr_0'):
        if key in npz_file.files:
            return npz_file[key]
    raise KeyError(f'Unsupported npz keys: {npz_file.files}')


def _orient_trafficstream_array(data, adj_node_num=None):
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3:
        raise ValueError(f'TrafficStream yearly data must be 2D/3D, got shape {data.shape}')

    if adj_node_num is not None:
        if data.shape[1] == adj_node_num:
            return data
        if data.shape[0] == adj_node_num:
            return np.transpose(data, (1, 0, 2))

    if data.shape[0] < data.shape[1]:
        return np.transpose(data, (1, 0, 2))
    return data


def _build_trafficstream_features(flow_data, input_dim):
    flow = flow_data[..., :1].astype(np.float32)
    num_samples, num_nodes, _ = flow.shape
    feature_list = [flow]

    if input_dim >= 2:
        time_in_day = ((np.arange(num_samples) % 288) / 288.0).astype(np.float32)
        time_in_day = np.tile(time_in_day[:, None, None], (1, num_nodes, 1))
        feature_list.append(time_in_day)

    if input_dim >= 3:
        day_in_week = ((np.arange(num_samples) // 288) % 7).astype(np.float32)
        day_in_week = np.tile(day_in_week[:, None, None], (1, num_nodes, 1))
        feature_list.append(day_in_week)

    if flow_data.shape[-1] > 1 and len(feature_list) < input_dim:
        remaining = input_dim - len(feature_list)
        feature_list.append(flow_data[..., 1:1 + remaining].astype(np.float32))

    return np.concatenate(feature_list, axis=-1)[..., :input_dim]


def _resolve_split_ratio(train_ratio, val_ratio):
    train_ratio = float(train_ratio) if train_ratio is not None else 0.6
    val_ratio = float(val_ratio) if val_ratio is not None else 0.2
    if train_ratio <= 0 or train_ratio >= 1 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        return 0.6, 0.2
    return train_ratio, val_ratio


def _parse_year_spec(year_spec):
    if year_spec is None:
        return []

    tokens = []
    for chunk in re.split(r'[, _]+', str(year_spec).strip()):
        if not chunk:
            continue
        if '-' in chunk:
            start_str, end_str = chunk.split('-', 1)
            start_year = int(start_str)
            end_year = int(end_str)
            step = 1 if end_year >= start_year else -1
            tokens.extend([str(year) for year in range(start_year, end_year + step, step)])
        else:
            tokens.append(str(int(chunk)))

    seen = set()
    ordered_years = []
    for year in tokens:
        if year not in seen:
            ordered_years.append(year)
            seen.add(year)
    return ordered_years


class TrafficStreamProvider:
    def __init__(self, data_path, adj_path, dataset, years, check_years=None,
                 node_shuffle_seed=None, tood=1, sood=1) -> None:
        del node_shuffle_seed  # STOP-style TrafficStream keeps the original node order.
        if years is None:
            raise ValueError('TrafficStream requires --years to locate STOP-style yearly artifacts.')

        self.dataset = dataset
        self.train_year = str(years)
        self.eval_years = _parse_year_spec(check_years if check_years is not None else years)
        if not self.eval_years:
            self.eval_years = [self.train_year]
        self.tood = bool(tood)
        self.sood = bool(sood)
        self._data_path = data_path
        self._adj_path = adj_path

        self.train_bundle = self._load_year_bundle(self.train_year)
        self.eval_bundles = [self._load_year_bundle(year) for year in self.eval_years] if self.tood else [self.train_bundle]

        self.scaler = None
        self.features = 0
        self.adj_mx_train = self.train_bundle['adj_mx']
        self.distance_mx_train = self.train_bundle['distance_mx']
        self.adj_mx_test = self.train_bundle['adj_mx']
        self.distance_mx_test = self.train_bundle['distance_mx']

    def _load_year_bundle(self, year):
        data_file = _resolve_yearly_path(self._data_path, year, os.path.join('{year}', 'his.npz'))
        adj_file = _resolve_yearly_path(self._adj_path if self._adj_path else self._data_path, year, os.path.join('{year}', '{year}_adj.npy'))

        with np.load(data_file) as npz_file:
            raw_payload = _extract_npz_array(npz_file)
            raw_array = np.asarray(raw_payload, dtype=np.float32)

        raw_array[np.isnan(raw_array)] = 0.0
        adj_node_num = None
        if adj_file is not None and os.path.exists(adj_file):
            adj_probe = np.load(adj_file)
            if adj_probe.ndim == 2:
                adj_node_num = adj_probe.shape[0]
        raw_array = _orient_trafficstream_array(raw_array, adj_node_num=adj_node_num)
        node_num = raw_array.shape[1]
        if os.path.exists(adj_file):
            adj_mx, distance_mx = get_adjacency_matrix(adj_file, node_num)
        else:
            adj_mx = np.eye(node_num, dtype=np.float32)
            distance_mx = np.eye(node_num, dtype=np.float32)

        timestamp = generatetimestamp(start=f'{year}0101 00:00:00', periods=raw_array.shape[0], freq='5min')
        return {
            'year': year,
            'data': raw_array,
            'adj_mx': adj_mx.astype(np.float32),
            'distance_mx': distance_mx.astype(np.float32),
            'timestamp': timestamp,
        }

    def _prepare_year_tensors(self, bundle, input_dim, device, node_limit=None):
        data = bundle['data'][..., :input_dim]
        if node_limit is not None:
            data = data[:, :node_limit, :]
        data = torch.from_numpy(data).float().to(device)
        timestamp = bundle['timestamp'].to(device)
        mask = torch.ones_like(data)
        return data, timestamp, mask

    def _build_scaler(self, train_data):
        mean = [train_data[..., i:i+1].mean() for i in range(train_data.shape[-1])]
        std = [_safe_std(train_data[..., i:i+1].std()) for i in range(train_data.shape[-1])]
        return StandardScaler(mean, std)

    def _build_dataset(self, data, timestamp, sample_len, output_len,
                       input_dim, output_dim, target_strategy, device, num_envs, training):
        window_size = sample_len + output_len
        sample = generate_sample_by_sliding_window(data, sample_len=window_size)
        history = sample[:, :sample_len, ..., :input_dim]
        target = sample[:, -output_len:, ..., :output_dim]
        timestamp_window = generate_sample_by_sliding_window(timestamp, sample_len=window_size)

        ob_mask = torch.ones((history.shape[0], window_size, data.shape[1], input_dim), device=device)
        if training:
            if target_strategy == 'random':
                cond_mask = get_randmask(ob_mask[:, :sample_len], 0, 1).to(device)[..., :input_dim]
            else:
                cond_mask = torch.concat([
                    get_block_mask(ob_mask[i, :sample_len].cpu(), target_strategy='hybrid', min_seq=3, max_seq=12)
                    .to(device)
                    .unsqueeze(0)
                    for i in range(history.shape[0])
                ])[..., :input_dim]
        else:
            cond_mask = ob_mask[:, :sample_len]

        history = self.scaler.transform(history)

        if training and num_envs > 0:
            num_train_samples = history.shape[0]
            samples_per_env = max(1, num_train_samples // num_envs)
            env_ids = torch.arange(num_train_samples, device=device) // samples_per_env
            env_ids = torch.clamp(env_ids, 0, num_envs - 1)
            spurious_features = torch.rand(num_train_samples, device=device)
            return BasicDataset(history, target, timestamp_window, cond_mask, ob_mask, env_ids, spurious_features, training=True)

        return BasicDataset(history, target, timestamp_window, cond_mask, ob_mask, training=training)

    def _concat_eval_years(self, bundles, input_dim, device, node_limit):
        data_parts = []
        timestamp_parts = []
        for bundle in bundles:
            year_data, year_timestamp, _ = self._prepare_year_tensors(bundle, input_dim, device, node_limit=node_limit)
            data_parts.append(year_data)
            timestamp_parts.append(year_timestamp)
        return torch.concat(data_parts, dim=0), torch.concat(timestamp_parts, dim=0)

    def getdataset(self, sample_len, output_len, window_size, input_dim, output_dim,
                   train_ratio, val_ratio, target_strategy, few_shot=1,
                   device=None, num_envs=4, max_increase_ratio=0, test_increase_ratio=0, test_decrease_ratio=0):
        del max_increase_ratio, test_increase_ratio, test_decrease_ratio, window_size
        if device is None:
            raise ValueError("device parameter is required. Please pass args.device explicitly.")

        train_ratio, val_ratio = _resolve_split_ratio(train_ratio, val_ratio)
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio <= 0:
            raise ValueError('TrafficStream requires train_ratio + val_ratio < 1 for temporal split.')

        node_limit = min(
            [self.train_bundle['data'].shape[1]] + [bundle['data'].shape[1] for bundle in self.eval_bundles]
        )

        train_year_data, train_year_timestamp, _ = self._prepare_year_tensors(
            self.train_bundle, input_dim, device, node_limit=node_limit
        )
        test_source_data, test_source_timestamp = self._concat_eval_years(
            self.eval_bundles, input_dim, device, node_limit=node_limit
        )

        train_year_len = train_year_data.shape[0]
        train_end = int(train_year_len * train_ratio)
        val_end = int(train_year_len * (train_ratio + val_ratio))
        if train_end <= 0 or val_end <= train_end:
            raise ValueError('TrafficStream split generated an empty training or validation set.')

        train_full_data = train_year_data[:train_end]
        train_full_timestamp = train_year_timestamp[:train_end]
        train_data = train_full_data[:max(1, int(train_full_data.shape[0] * few_shot))]
        train_timestamp = train_full_timestamp[:train_data.shape[0]]

        val_data = train_year_data[train_end:val_end]
        val_timestamp = train_year_timestamp[train_end:val_end]

        test_source_len = test_source_data.shape[0]
        test_len = int(test_source_len * test_ratio)
        if test_len <= 0:
            raise ValueError('TrafficStream split generated an empty test set.')
        test_data = test_source_data[-test_len:]
        test_timestamp = test_source_timestamp[-test_len:]

        self.adj_mx_train = self.train_bundle['adj_mx'][:node_limit, :node_limit]
        self.distance_mx_train = self.train_bundle['distance_mx'][:node_limit, :node_limit]
        # Multi-year test concatenation requires a shared node space, so test uses the same common-node graph.
        self.adj_mx_test = self.adj_mx_train
        self.distance_mx_test = self.distance_mx_train
        self.features = input_dim
        self.scaler = self._build_scaler(train_data)

        train_dataset = self._build_dataset(
            train_data, train_timestamp, sample_len, output_len,
            input_dim, output_dim, target_strategy, device, num_envs, training=True
        )
        val_dataset = self._build_dataset(
            val_data, val_timestamp, sample_len, output_len,
            input_dim, output_dim, target_strategy, device, num_envs, training=False
        )
        test_dataset = self._build_dataset(
            test_data, test_timestamp, sample_len, output_len,
            input_dim, output_dim, target_strategy, device, num_envs, training=False
        )
        return train_dataset, val_dataset, test_dataset

    def getadj(self):
        return self.adj_mx_train, self.distance_mx_train, self.adj_mx_test, self.distance_mx_test

