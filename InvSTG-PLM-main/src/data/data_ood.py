import torch
import numpy as np
import torch.utils.data
from typing import Dict
from data.dataprovider_ood import PEMSFLOWProvider,PEMSMISSINGProvider,NYCTAXIProvider

data_dict = {
    'PEMS08FLOW': PEMSFLOWProvider,
    'PEMS04FLOW': PEMSFLOWProvider,
    'PEMS03FLOW': PEMSFLOWProvider,
    'PEMS07FLOW': PEMSFLOWProvider,
    'PEMS08MISSING': PEMSMISSINGProvider,
    'PEMS04MISSING': PEMSMISSINGProvider,
    'PEMS03MISSING': PEMSMISSINGProvider,
    'PEMS07MISSING': PEMSMISSINGProvider,
    'NYCTAXI':NYCTAXIProvider,
    'CHITAXI':NYCTAXIProvider,
}

def data_loader(dataset, batch_size, shuffle=True, drop_last=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def load_data_ood(dataset, batch_size, sample_len, output_len, window_size, 
                  input_dim, output_dim,
                  train_ratio, val_ratio, data_path, adj_path, target_strategy, 
                  few_shot=1, node_shuffle_seed=None, device=None,
                  num_envs=4, max_increase_ratio=0, test_increase_ratio=0, test_decrease_ratio=0):
    
    if device is None:
        raise ValueError("device parameter is required. Please pass args.device explicitly.")
    
    dataprovider = data_dict[dataset](data_path, adj_path, dataset, node_shuffle_seed)

    train_set, val_set, test_set = dataprovider.getdataset(
        sample_len=sample_len,
        output_len=output_len,
        window_size=window_size,
        input_dim=input_dim,
        output_dim=output_dim,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        target_strategy=target_strategy,
        few_shot=few_shot,
        device=device,
        num_envs=num_envs, 
        max_increase_ratio=max_increase_ratio,
        test_increase_ratio=test_increase_ratio,
        test_decrease_ratio=test_decrease_ratio
    )

    scaler = dataprovider.scaler
    features = dataprovider.features
    adj_mx_train, distance_mx_train, adj_mx_test, distance_mx_test = dataprovider.getadj()

    train_loader = data_loader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = data_loader(val_set,   batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader  = data_loader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, \
           scaler, features, \
           adj_mx_train, distance_mx_train, adj_mx_test, distance_mx_test

