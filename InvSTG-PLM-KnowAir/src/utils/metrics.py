import torch

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def MAPE_torch(pred, true, mask_value=1e-6):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def MAPE_torch_node(pred, true, mask_value=1e-6):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = pred*mask
        true = true*mask + (1-mask.float())
        count = mask.sum(dim=-1)
    return torch.sum(torch.abs(torch.div((true - pred)*mask, true)),dim=-1)/count


def weighted_MAE_torch(pred, true, weight_mask):
    weight_mask = weight_mask.float()
    weight_sum = weight_mask.sum()
    if weight_sum <= 0:
        return torch.tensor(float('nan'), device=pred.device)
    return torch.sum(torch.abs(true - pred) * weight_mask) / weight_sum


def weighted_RMSE_torch(pred, true, weight_mask):
    weight_mask = weight_mask.float()
    weight_sum = weight_mask.sum()
    if weight_sum <= 0:
        return torch.tensor(float('nan'), device=pred.device)
    return torch.sqrt(torch.sum(((pred - true) ** 2) * weight_mask) / weight_sum)


def build_sudden_change_mask(targets, eval_mask=None, threshold_start=75.0, threshold_change=20.0):
    sudden_mask = targets >= threshold_start
    if targets.shape[1] > 1:
        change_mask = torch.zeros_like(sudden_mask, dtype=torch.bool)
        change_mask[:, 1:] = torch.abs(targets[:, 1:] - targets[:, :-1]) >= threshold_change
        sudden_mask = sudden_mask | change_mask
    if eval_mask is not None:
        sudden_mask = sudden_mask & eval_mask.bool()
    return sudden_mask


def cal_weighted_metrics(predicts, targets, weight_mask, reference_mask=None):
    F = targets.shape[-1]

    mae = []
    for f in range(F):
        mask = weight_mask[..., f]
        mae.append(weighted_MAE_torch(pred=predicts[..., f], true=targets[..., f], weight_mask=mask).item())

    rmse = []
    for f in range(F):
        mask = weight_mask[..., f]
        rmse.append(weighted_RMSE_torch(pred=predicts[..., f], true=targets[..., f], weight_mask=mask).item())

    coverage = []
    for f in range(F):
        mask = weight_mask[..., f].bool()
        if reference_mask is None:
            coverage.append(mask.float().mean().item())
        else:
            ref = reference_mask[..., f].bool()
            ref_count = ref.sum().item()
            coverage.append(mask.sum().item() / ref_count if ref_count else float('nan'))

    return mae, rmse, coverage


def cal_metrics(predicts,targets,eval_mask):
    F = targets.shape[-1]

    mae = []
    for f in range(F):
        mask = eval_mask[...,f]
        mae.append(MAE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())

    rmse = []
    for f in range(F):
        mask = eval_mask[...,f]
        rmse.append(RMSE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())

    mape = []
    for f in range(F):
        mask = eval_mask[...,f]
        mape.append(MAPE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())

    mape_10 = []
    for f in range(F):
        mask = eval_mask[...,f]
        mask = mask & (targets[...,0] >= 10)
        mape_10.append(MAPE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())

    mape_20 = []
    for f in range(F):
        mask = eval_mask[...,f]
        mask = mask & (targets[...,0] >= 20)
        mape_20.append(MAPE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())  

    return mae,rmse,mape,mape_10,mape_20