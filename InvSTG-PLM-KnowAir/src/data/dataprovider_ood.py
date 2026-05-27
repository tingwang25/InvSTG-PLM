import torch
import numpy as np
import torch.utils.data
from utils.utils import get_adjacency_matrix_2direction, get_adjacency_matrix, get_randmask, linear_interpolate, get_block_mask
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
from data.scaler import StandardScaler, MinMaxScaler
import os


def _load_numpy_payload(data_path):
    loaded = np.load(data_path, allow_pickle=True)

    if isinstance(loaded, np.lib.npyio.NpzFile):
        if 'data' in loaded.files:
            return loaded['data']
        return loaded[loaded.files[0]]

    if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
        item = loaded.item()
        if isinstance(item, dict):
            for key in ('data', 'x', 'arr_0'):
                if key in item:
                    return item[key]
        return item

    return loaded


def _normalize_knowair_data(raw_data):
    data = np.asarray(raw_data, dtype=np.float32)
    known_node_counts = {184, 196}

    if data.ndim == 2:
        if data.shape[1] in known_node_counts and data.shape[0] not in known_node_counts:
            pass
        elif data.shape[0] in known_node_counts and data.shape[1] not in known_node_counts:
            data = data.transpose(1, 0)
        elif data.shape[0] <= 512 and data.shape[1] > data.shape[0]:
            data = data.transpose(1, 0)
        data = data[..., None]
    elif data.ndim == 3:
        if data.shape[1] in known_node_counts and data.shape[0] not in known_node_counts:
            pass
        elif data.shape[0] in known_node_counts and data.shape[1] not in known_node_counts:
            data = data.transpose(1, 0, 2)
        elif data.shape[2] in known_node_counts and data.shape[1] not in known_node_counts:
            data = data.transpose(0, 2, 1)
        elif data.shape[2] <= 512 and data.shape[0] > data.shape[2]:
            data = data.transpose(0, 2, 1)
    else:
        raise ValueError(f"Unsupported KnowAir data shape: {data.shape}")

    return data


def _build_mask_from_data(data):
    finite_mask = np.isfinite(data)
    cleaned = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    mask = finite_mask.astype(np.float32)
    return cleaned, mask


def _build_knownair_composed_features(raw_data, timestamp, target_col):
    feature_count = raw_data.shape[-1]
    if target_col < 0 or target_col >= feature_count:
        raise ValueError(
            f"KnowAir target column {target_col} is out of range for feature count {feature_count}"
        )

    target = raw_data[..., target_col:target_col + 1]
    raw_features = raw_data[..., :target_col]

    time_in_day = (
        (timestamp[:, 3].cpu().numpy().astype(np.float32) * 60.0 + timestamp[:, 4].cpu().numpy().astype(np.float32))
        / (24.0 * 60.0)
    )
    day_in_week = timestamp[:, 2].cpu().numpy().astype(np.float32) / 6.0

    node_num = raw_data.shape[1]
    time_in_day = np.repeat(time_in_day[:, None, None], node_num, axis=1)
    day_in_week = np.repeat(day_in_week[:, None, None], node_num, axis=1)

    return np.concatenate([target, raw_features, time_in_day, day_in_week], axis=-1).astype(np.float32)


def _build_knownair_composed_mask(raw_mask, target_col):
    feature_count = raw_mask.shape[-1]
    if target_col < 0 or target_col >= feature_count:
        raise ValueError(
            f"KnowAir target column {target_col} is out of range for feature count {feature_count}"
        )

    target_mask = raw_mask[..., target_col:target_col + 1]
    feature_mask = raw_mask[..., :target_col]
    temporal_mask = np.ones((*raw_mask.shape[:2], 2), dtype=np.float32)
    return np.concatenate([target_mask, feature_mask, temporal_mask], axis=-1)


def _read_city_coordinates(city_path, node_num):
    coords = []
    with open(city_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            coords.append((float(parts[2]), float(parts[3])))

    if len(coords) < node_num:
        raise ValueError(f"City file {city_path} only contains {len(coords)} nodes, expected {node_num}")

    return np.asarray(coords[:node_num], dtype=np.float32)


def _read_city_metadata(city_path, node_num):
    nodes = []
    with open(city_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            nodes.append({
                'idx': int(parts[0]),
                'city': parts[1],
                'lon': float(parts[2]),
                'lat': float(parts[3]),
            })

    if len(nodes) < node_num:
        raise ValueError(f"City file {city_path} only contains {len(nodes)} nodes, expected {node_num}")

    return nodes[:node_num]


def _haversine_distance_matrix(coords):
    lon = np.radians(coords[:, 0])[:, None]
    lat = np.radians(coords[:, 1])[:, None]
    dlon = lon - lon.T
    dlat = lat - lat.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def _thresholded_gaussian_kernel(distance_mx, epsilon=0.01):
    adj = distance_mx.copy().astype(np.float32)
    nonzero_mask = adj != 0
    if np.any(nonzero_mask):
        std = np.std(adj[nonzero_mask])
        if std > 0:
            adj[nonzero_mask] = np.exp(-1.0 * (adj[nonzero_mask] / std) ** 2)
        else:
            adj[nonzero_mask] = 1.0
    adj[adj < epsilon] = 0.0
    np.fill_diagonal(adj, 1.0)
    return adj


def _candidate_altitude_paths(base_path):
    if not base_path:
        return []

    if os.path.isdir(base_path):
        return [os.path.join(base_path, 'altitude.npy')]

    if base_path.lower().endswith('altitude.npy'):
        return [base_path]

    return []


def _knownair_repo_data_dirs():
    current_dir = os.path.dirname(__file__)
    return [
        os.path.normpath(os.path.join(current_dir, '..', '..', 'data', 'KnowAir', 'data')),
        os.path.normpath(os.path.join(current_dir, '..', '..', '..', 'KnowAir', 'data')),
    ]


def _resolve_knownair_altitude_path(data_path, adj_path):
    data_dir = os.path.dirname(os.path.abspath(data_path))

    candidates = []
    candidates.extend(_candidate_altitude_paths(adj_path))
    candidates.extend(_candidate_altitude_paths(data_dir))
    for repo_knownair_dir in _knownair_repo_data_dirs():
        candidates.extend(_candidate_altitude_paths(repo_knownair_dir))

    for altitude_path in candidates:
        if altitude_path and os.path.isfile(altitude_path):
            return altitude_path

    return None


def _lonlat_to_altitude_xy(lon, lat):
    lon_l = 100.0
    lat_u = 48.0
    res = 0.05
    x = int(np.round((lon - lon_l - res / 2.0) / res))
    y = int(np.round((lat_u + res / 2.0 - lat) / res))
    return x, y


def _clamp_xy(x, y, altitude_shape):
    height, width = altitude_shape
    return (
        int(np.clip(x, 0, width - 1)),
        int(np.clip(y, 0, height - 1)),
    )


def _line_points(x0, y0, x1, y1):
    steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int64)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int64)
    return xs, ys


def _apply_altitude_filter(distance_mx, nodes, altitude, alti_thres=1200.0):
    filtered = distance_mx.copy()
    altitude_shape = altitude.shape

    for i in range(len(nodes)):
        src_x, src_y = _lonlat_to_altitude_xy(nodes[i]['lon'], nodes[i]['lat'])
        src_x, src_y = _clamp_xy(src_x, src_y, altitude_shape)
        altitude_src = altitude[src_y, src_x]

        for j in range(i + 1, len(nodes)):
            if filtered[i, j] == 0:
                continue

            dest_x, dest_y = _lonlat_to_altitude_xy(nodes[j]['lon'], nodes[j]['lat'])
            dest_x, dest_y = _clamp_xy(dest_x, dest_y, altitude_shape)
            altitude_dest = altitude[dest_y, dest_x]

            xs, ys = _line_points(src_x, src_y, dest_x, dest_y)
            xs = np.clip(xs, 0, altitude_shape[1] - 1)
            ys = np.clip(ys, 0, altitude_shape[0] - 1)
            altitude_points = altitude[ys, xs]

            blocked_from_src = np.sum(altitude_points - altitude_src > alti_thres)
            blocked_from_dest = np.sum(altitude_points - altitude_dest > alti_thres)
            if blocked_from_src >= 3 or blocked_from_dest >= 3:
                filtered[i, j] = 0.0
                filtered[j, i] = 0.0

    return filtered


def _build_stop_style_graph(city_path, altitude_path, node_num, dist_thres=3.0):
    nodes = _read_city_metadata(city_path, node_num)
    coords = np.asarray([[node['lon'], node['lat']] for node in nodes], dtype=np.float32)

    coord_diff = coords[:, None, :] - coords[None, :, :]
    coord_dist = np.sqrt(np.sum(coord_diff ** 2, axis=-1))
    edge_mask = (coord_dist <= dist_thres).astype(np.float32)
    np.fill_diagonal(edge_mask, 0.0)

    geo_distance_mx = _haversine_distance_matrix(coords).astype(np.float32)
    distance_mx = geo_distance_mx * edge_mask

    if altitude_path is not None:
        altitude = np.load(altitude_path).astype(np.float32)
        distance_mx = _apply_altitude_filter(distance_mx, nodes, altitude)

    adj_mx = _thresholded_gaussian_kernel(distance_mx)
    return adj_mx.astype(np.float32), distance_mx.astype(np.float32)


def _candidate_city_paths(base_path):
    if not base_path:
        return []

    if os.path.isdir(base_path):
        return [
            os.path.join(base_path, 'city.txt'),
            os.path.join(base_path, 'city_196.txt'),
        ]

    if base_path.lower().endswith('.txt'):
        return [base_path]

    return []


def _resolve_knownair_city_path(data_path, adj_path, node_num):
    data_dir = os.path.dirname(os.path.abspath(data_path))

    candidates = []
    candidates.extend(_candidate_city_paths(adj_path))
    candidates.extend(_candidate_city_paths(data_dir))
    for repo_knownair_dir in _knownair_repo_data_dirs():
        candidates.extend(_candidate_city_paths(repo_knownair_dir))

    for city_path in candidates:
        if not city_path or not os.path.isfile(city_path):
            continue
        try:
            coords = _read_city_coordinates(city_path, node_num)
        except ValueError:
            continue
        if coords.shape[0] == node_num:
            return city_path

    raise FileNotFoundError(
        f"Could not resolve a city coordinate file for {node_num} nodes. "
        f"Tried paths derived from data_path={data_path} and adj_path={adj_path}."
    )

def generate_sample_by_sliding_window(data, sample_len, step=1):

    sample = []

    for i in range(0, data.shape[0] - sample_len, step):

        sample.append(torch.unsqueeze(data[i:i+sample_len] , 0))
    
    if (data.shape[0] - sample_len) % step !=0 :
        sample.append(torch.unsqueeze(data[-sample_len:] , 0))

    sample = torch.concat(sample,dim=0)

    return sample


def _build_knownair_absolute_timestamps(periods):
    return pd.date_range(start='2015-01-01 00:00:00', periods=periods, freq='3h')


def _get_contiguous_range(indices, split_name):
    if indices.size == 0:
        raise ValueError(f"KnowAir split '{split_name}' is empty.")
    return int(indices[0]), int(indices[-1]) + 1


def _get_knownair_yearly_ratio_ranges(periods, train_ratio, val_ratio, few_shot):
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"KnowAir requires train_ratio + val_ratio < 1, got {train_ratio + val_ratio:.4f}."
        )

    absolute_time = _build_knownair_absolute_timestamps(periods)
    years = absolute_time.year

    train_val_indices = np.flatnonzero(years == 2015)
    test_source_indices = np.flatnonzero((years >= 2016) & (years <= 2018))

    train_val_start, train_val_end = _get_contiguous_range(train_val_indices, 'train_val_2015')
    test_source_start, test_source_end = _get_contiguous_range(test_source_indices, 'test_source_2016_2018')

    train_val_len = train_val_end - train_val_start
    train_end = train_val_start + int(train_val_len * train_ratio)
    val_end = train_val_start + int(train_val_len * (train_ratio + val_ratio))

    if train_end <= train_val_start or val_end <= train_end:
        raise ValueError(
            "KnowAir train/val split is empty. Please check train_ratio and val_ratio."
        )

    train_full_end = train_end
    train_end = train_val_start + max(1, int((train_full_end - train_val_start) * few_shot))

    test_source_len = test_source_end - test_source_start
    test_len = int(test_source_len * test_ratio)
    if test_len <= 0:
        raise ValueError("KnowAir test split is empty. Please check train_ratio and val_ratio.")

    test_start = test_source_end - test_len

    return {
        'train': [train_val_start, train_end],
        'val': [train_full_end, val_end],
        'test': [test_start, test_source_end],
    }

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

    def __init__(self, data_path, adj_path ,dataset, node_shuffle_seed=None, **read_data_kwargs) -> None:

        self.dataset = dataset

        self.data, self.node_num, self.features, \
        self.adj_mx, self.distance_mx, \
        self.timestamp,self.mask,self.mask_eval =  self.read_data(data_path, adj_path, **read_data_kwargs)

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
        has_structure_shift = max_increase_ratio is not None and max_increase_ratio != 0

        if has_structure_shift: # with structure shifts
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
        if self.dataset == 'KnowAir':
            split_ranges = _get_knownair_yearly_ratio_ranges(all_len, train_ratio, val_ratio, few_shot)
            train_range = split_ranges['train']
            val_range = split_ranges['val']
            test_range = split_ranges['test']
        else:
            train_len = int(all_len * train_ratio)
            val_len = int(all_len * val_ratio)

            train_range = [0, int(train_len * few_shot)]
            val_range = [train_len, train_len+val_len]
            test_range = [train_len+val_len, all_len]

        if has_structure_shift:
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

        if has_structure_shift:
            scaler_mask = self.mask[train_range[0]:train_range[1], node_training, :]!=0
            scaler_data = self.data[train_range[0]:train_range[1], node_training, :]
        else:
            scaler_mask = self.mask[train_range[0]:train_range[1]]!=0
            scaler_data = self.data[train_range[0]:train_range[1]]
        dim = scaler_data.shape[-1]
        mean = [scaler_data[...,i:i+1][scaler_mask[...,i:i+1]].mean() for i in range(dim)]
        std = [scaler_data[...,i:i+1][scaler_mask[...,i:i+1]].std() for i in range(dim)]
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

        if has_structure_shift:
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

        if has_structure_shift:
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
    'KnowAir': lambda T : generatetimestamp(start='20150101 00:00:00',periods=T,freq='3h'),
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


class KnowAirProvider(DataProvider):

    def read_data(self, data_path, adj_path=None, knowair_target_col=17) -> None:
        raw_data = _load_numpy_payload(data_path)
        raw_data = _normalize_knowair_data(raw_data)
        raw_data, raw_mask = _build_mask_from_data(raw_data)

        T, node_num, _ = raw_data.shape

        if adj_path and os.path.isfile(adj_path) and adj_path.lower().endswith(('.csv', '.npy')):
            adj_mx, distance_mx = get_adjacency_matrix(adj_path, node_num)
        else:
            city_path = _resolve_knownair_city_path(data_path, adj_path, node_num)
            altitude_path = _resolve_knownair_altitude_path(data_path, adj_path)
            adj_mx, distance_mx = _build_stop_style_graph(city_path, altitude_path, node_num)

        adj_mx = np.where(np.eye(node_num).astype(bool), 1.0, adj_mx).astype(np.float32)
        timestamp = timestampfun['KnowAir'](T)
        data = _build_knownair_composed_features(raw_data, timestamp, knowair_target_col)
        mask = _build_knownair_composed_mask(raw_mask, knowair_target_col)

        data = torch.from_numpy(data)
        mask = torch.from_numpy(mask)
        features = data.shape[-1]

        return data, node_num, features, \
               adj_mx, distance_mx, \
               timestamp, mask, mask.clone()

