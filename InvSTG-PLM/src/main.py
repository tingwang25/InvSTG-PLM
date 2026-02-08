import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
import os
from utils.utils import get_time_str,check_dir,draw_loss_line,draw_mape_node,get_randmask,get_block_mask, cal_shortest_path_length
from logger import getlogger
from model.model import STALLM
from model.llm import Phi2,T5,LLAMA3,Transformer,GPT2,Qwen3
from data.data_ood import load_data_ood
from utils.metrics import MAE_torch,RMSE_torch,MAPE_torch,MAPE_torch_node,cal_metrics
from utils.argsinit_ood import InitArgs
import copy
from torch.optim.lr_scheduler import ExponentialLR
import nni
import random
import string

random_str = lambda : ''.join(random.sample(string.ascii_letters + string.digits, 6))

class EBD(nn.Module):
    def __init__(self, num_envs=None, device=None):
      super(EBD, self).__init__()
      self.num_envs = num_envs
      if device is None:
          device = 'cuda' if torch.cuda.is_available() else 'cpu'
          import warnings
          warnings.warn(f"EBD device parameter not provided, using default: {device}. Please pass args.device explicitly.")
      self.device = device
      self.embedings = torch.nn.Embedding(num_envs, 1)
      self.re_init()

    def re_init(self):
      self.embedings.weight.data.fill_(1.)

    def re_init_with_noise(self, noise_sd):
      rd = torch.normal(
         torch.Tensor([1.0] * self.num_envs),
         torch.Tensor([noise_sd] * self.num_envs))
      self.embedings.weight.data = rd.view(-1, 1).to(self.device)

    def forward(self, e):
      return self.embedings(e.long())

def TrainEpoch(loader, model, optim, conf_opt, loss_fn,  prompt_prefix,scaler, epoch, alpha, need_step : bool, mylogger=None):
    if need_step:
        model.train()
        var_ratio = 1
        init_epoch = 5
        if epoch < init_epoch: 
            alpha_prime = 0
            if mylogger is not None:
                mylogger.info(f"epoch {epoch} with alpha_prime 0 for better learnable param")
        else:
            alpha_prime = alpha * ((epoch - init_epoch) ** var_ratio)
    else :
        model.eval()

    loss_item = 0
    conf_loss_item = 0
    count = 0   
    
    for batch in loader:  
        if need_step:
            input, target, timestamp,cond_mask,ob_mask, g, c = batch
        else:
            input, target, timestamp,cond_mask,ob_mask = batch
        
        B,T,N,F = input.shape
        if args.task == 'prediction':
            cond_mask = ob_mask[:,:T]
        if args.trainset_dynamic_missing and need_step:
            cond_mask = get_randmask(cond_mask,0,0.1)   
        input = torch.where(cond_mask==0,0,input)
        input = input.permute(0,2,1,3).contiguous().view(B,N,-1)
        cond_mask_n = cond_mask.permute(0,2,1,3).contiguous()
        # Forward pass
        predict, predict_conf, other_loss, s_state_conf = model(input,timestamp,prompt_prefix,cond_mask_n)
        predict = predict.view(B,N,-1,args.output_dim).permute(0,2,1,3).contiguous()
        predict_conf = predict_conf.view(B,N,-1,args.output_dim).permute(0,2,1,3).contiguous()
        predict = scaler.inverse_transform(predict)
        predict_conf = scaler.inverse_transform(predict_conf)
        if args.task != 'prediction':
            cond_mask = torch.concat(
        (cond_mask, torch.zeros(B, ob_mask.shape[1]-cond_mask.shape[1], N, F, device=args.device)),
        dim=1
    )
            eval_mask = (ob_mask - cond_mask).bool()[...,:args.output_dim]
        else:
            eval_mask = ob_mask[:,-args.predict_len:].bool()[...,:args.output_dim]
        loss = MAE_torch(pred=predict[:,:args.sample_len][eval_mask], true=target[:,:args.sample_len][eval_mask]) 
        conf_loss = MAE_torch(pred=predict_conf[:,:args.sample_len][eval_mask], true=target[:,:args.sample_len][eval_mask])
        irm_type = args.irm_type
        loss_list = []
        train_nll = 0
        if need_step:
            if irm_type == "rex_causal":    
                env_losses = []
                for conf in s_state_conf:
                    pred_combine = model.combine_with_ze(predict, conf)    # (B,L,N,D)
                    pred_combine = pred_combine.view(B,N,-1,args.output_dim).permute(0,2,1,3).contiguous()
                    pred_combine = scaler.inverse_transform(pred_combine)
                    enll = MAE_torch(pred=pred_combine[:,:args.sample_len][eval_mask], true=target[:,:args.sample_len][eval_mask])
                    env_losses.append(enll)
                env_loss = torch.stack(env_losses)  # [K]
                train_penalty = alpha_prime * torch.var(env_loss) + min(alpha_prime, 1) * env_loss.mean()

            elif irm_type == "irmv1":
                num_envs = int(g.max().item()) + 1
                ebd = EBD(num_envs, device=args.device).to(args.device)
                predict_flat = predict[:,:args.sample_len][eval_mask]
                target_flat = target[:,:args.sample_len][eval_mask]
                # g_expanded = g.view(B, 1, 1, 1).expand(B, T, N, args.output_dim)
                # g_flat = g_expanded[eval_mask].view(-1)
                train_logits = ebd(g).view(-1, 1) * predict_flat
                train_nll = MAE_torch(pred=train_logits, true=target_flat)
                grad = torch.autograd.grad(
                    train_nll * num_envs, ebd.parameters(),
                    create_graph=True)[0]
                train_penalty = alpha_prime * torch.mean(grad ** 2)

            elif irm_type == "rex":    
                #  Calculate loss for each environment
                num_envs = int(g.max().item()) + 1
                for i in range(num_envs):
                    env_mask = (g == i)
                    if env_mask.sum() == 0:
                        continue
                    pred_env_i = predict[env_mask]
                    target_env_i = target[env_mask]
                    eval_mask_env_i = eval_mask[env_mask]
                    enll = MAE_torch(pred=pred_env_i[:,:args.sample_len][eval_mask_env_i], true=target_env_i[:,:args.sample_len][eval_mask_env_i])
                    train_nll += enll / num_envs
                    loss_list.append(enll)

                # Calculate REx penalty
                loss_t = torch.stack(loss_list)
                train_penalty = alpha_prime * ((loss_t - loss_t.mean()) ** 2).mean()

            weight_norm = torch.tensor(0.).to(args.device)
            for n, m in model.named_modules():
                if n.startswith("out_mlp_conf"):
                    continue
                if hasattr(m, "weight") and m.weight is not None:
                    if isinstance(m.weight, torch.Tensor) and m.weight.dtype.is_floating_point:
                        weight_norm += m.weight.norm().pow(2).to(args.device)
                if hasattr(m, "bias") and m.bias is not None:
                    if isinstance(m.bias, torch.Tensor) and m.bias.dtype.is_floating_point:
                        weight_norm += m.bias.norm().pow(2).to(args.device)
            loss = loss + train_penalty + alpha_prime * weight_norm
            # if alpha_prime > 1.0:
            #     loss /= (1. + alpha_prime)

        loss_item += loss.item()
        conf_loss_item += conf_loss.item()
        count += 1

        if need_step:
            conf_opt.zero_grad()
            conf_loss.backward()
            conf_opt.step()

            optim.zero_grad()
            L = loss
            for l in other_loss:
                L += l
            L.backward()
            optim.step()

    if count:
        loss_item /= count
        conf_loss_item /= count
    return loss_item, conf_loss_item

def TestEpoch(loader, model,  prompt_prefix, scaler, save=False):

    with torch.no_grad():
        model.eval()
        targets = []
        predicts = []
        predict_confs = []
        eval_masks = []

        # Initialize fixed node selection if needed (when not random per batch)
        fixed_noise_node_indices = None

        for batch_idx, (input, target, timestamp,cond_mask,ob_mask) in enumerate(loader):
            B,T,N,F = input.shape

            input = torch.where(cond_mask==0,0,input)
            input = input.permute(0,2,1,3).contiguous().view(B,N,-1)

            if args.gaussian_noise_ratio > 0 or args.gaussian_noise_mean_shift != 0:
                if isinstance(scaler.std, (list, tuple)):
                    std_values = []
                    for std_val in scaler.std:
                        if isinstance(std_val, torch.Tensor):
                            std_values.append(std_val.item() if std_val.numel() == 1 else float(std_val))
                        else:
                            std_values.append(float(std_val))
                    original_std = np.mean(std_values) if std_values else 1.0
                else:
                    if isinstance(scaler.std, torch.Tensor):
                        original_std = scaler.std.item() if scaler.std.numel() == 1 else float(scaler.std)
                    else:
                        original_std = float(scaler.std)
                
                noise_node_indices = None   
                if args.gaussian_noise_node_ratio < 1.0:
                    num_noise_nodes = max(1, int(N * args.gaussian_noise_node_ratio))
                    
                    if args.gaussian_noise_node_random_per_batch:
                        noise_node_indices = np.random.choice(N, size=num_noise_nodes, replace=False)
                    else:
                        if fixed_noise_node_indices is None:
                            fixed_noise_node_indices = np.random.choice(N, size=num_noise_nodes, replace=False)
                        noise_node_indices = fixed_noise_node_indices
                
                if noise_node_indices is not None:
                    noise_mask = torch.zeros(N, dtype=torch.bool, device=input.device)
                    noise_mask[noise_node_indices] = True
                    noise_mask_expanded = noise_mask.unsqueeze(0).unsqueeze(-1)
                    
                    if args.gaussian_noise_mean_shift != 0:
                        input = input + args.gaussian_noise_mean_shift * noise_mask_expanded
                    
                    if args.gaussian_noise_ratio > 0:
                        noise = torch.randn_like(input) * args.gaussian_noise_ratio
                        input = input + noise * noise_mask_expanded
                else:
                    if args.gaussian_noise_mean_shift != 0:
                        input = input + args.gaussian_noise_mean_shift
                    
                    if args.gaussian_noise_ratio > 0:
                        noise = torch.randn_like(input) * args.gaussian_noise_ratio
                        input = input + noise

            predict, predict_conf, _, _ = model(input,timestamp,prompt_prefix,cond_mask)

            predict_conf = predict_conf.view(B,N,-1,args.output_dim).permute(0,2,1,3).contiguous()
            predict = predict.view(B,N,-1,args.output_dim).permute(0,2,1,3).contiguous()

            if args.task != 'prediction':
                cond_mask = torch.concat((cond_mask,torch.zeros(B,ob_mask.shape[1]-cond_mask.shape[1],N,F).to(args.device)),dim=1)
                eval_mask = (ob_mask - cond_mask).bool()[...,:args.output_dim]
            else:
                eval_mask = ob_mask[:,-args.predict_len:].bool()[...,:args.output_dim]

            targets.append(target.detach())
            predicts.append(predict.detach())
            predict_confs.append(predict_conf.detach())
            eval_masks.append(eval_mask.detach())

        targets = torch.concat(targets,dim = 0)
        predicts = torch.concat(predicts,dim = 0)
        predict_confs = torch.concat(predict_confs,dim = 0)
        eval_masks = torch.concat(eval_masks,dim = 0)

        predicts = scaler.inverse_transform(predicts)
        predict_confs = scaler.inverse_transform(predict_confs)

        mae_recon, mae_pred = None, None
        rmse_recon, rmse_pred = None, None
        mape_recon, mape_pred = None, None

        if args.task in ['all','imputation']:
            eval_mask = eval_masks[:,:args.sample_len]
            mae_recon, rmse_recon, mape_recon, _,_ = cal_metrics(predicts=predicts[:,:args.sample_len],targets=targets[:,:args.sample_len],eval_mask=eval_mask)
        
        if args.task in ['all','prediction']:
            eval_mask = eval_masks[:,-args.predict_len:]
            mae_pred, rmse_pred, mape_pred, _,_ = cal_metrics(predicts=predicts[:,-args.predict_len:],targets=targets[:,-args.predict_len:],eval_mask=eval_mask)
            mae_conf, rmse_conf, mape_conf, _,_ = cal_metrics(predicts=predict_confs[:,-args.predict_len:],targets=targets[:,-args.predict_len:],eval_mask=eval_mask)

    if save:
        np.savez(os.path.join(LOG_DIR,'test.npz'),targets=targets.cpu().numpy(),predicts=predicts.cpu().numpy(),mask=eval_masks.cpu().numpy())

    return mae_recon, rmse_recon, mape_recon, mae_pred, rmse_pred, mape_pred, mae_conf, rmse_conf, mape_conf

def Train(args, mylogger, model, prompt_prefix, scaler, train_loader, val_loader, test_loader, LOG_DIR):
    patience_count = 0

    max_epoch = args.epoch

    if args.zero_shot:
        max_epoch = 0

    lr = args.lr
    val_epoch = args.val_epoch
    test_epoch = args.test_epoch

    optim = torch.optim.AdamW([
        {'params': (p for name, p in model.named_parameters() if ('bias' not in name) and p.requires_grad), 'weight_decay': args.weight_decay},
        {'params': (p for name, p in model.named_parameters() if ('bias' in name) and p.requires_grad)}
    ],lr=lr)
    conf_opt = torch.optim.Adam(
        list(model.out_mlp_conf.parameters()) + list(model.env_gate.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=10,min_lr=1e-6) 

    loss_fn = torch.nn.L1Loss()

    best_loss = 1e9
    best_model = copy.deepcopy(model.grad_state_dict())

    train_loss_line = {'x':[],'y':[]}
    val_loss_line = {'x':[],'y':[]}

    for epoch in range(max_epoch):

        train_loss, conf_loss = TrainEpoch(train_loader,model,optim,conf_opt,loss_fn,prompt_prefix,scaler, epoch, args.alpha, need_step=True, mylogger=mylogger)

        train_loss_line['x'].append(epoch)
        train_loss_line['y'].append(train_loss)

        mylogger.info(f"epoch {epoch} train_loss:{train_loss}, train_conf_loss:{conf_loss}")

        if epoch % val_epoch == 0:

            val_loss, val_conf_loss = TrainEpoch(val_loader,model,optim,conf_opt, loss_fn,prompt_prefix,scaler, epoch, args.alpha,need_step=False, mylogger=mylogger)
            val_loss_line['x'].append(epoch)
            val_loss_line['y'].append(val_loss)

            if val_loss < best_loss :
                patience_count = 0
                best_loss = val_loss
                best_model = copy.deepcopy(model.grad_state_dict())
            else :
                patience_count += 1
            
            if args.nni:
                nni.report_intermediate_result(val_loss)
            mylogger.info(f"[Validation] epoch {epoch} val_loss:{val_loss}, val_conf_loss:{val_conf_loss}")
            scheduler.step(val_loss)

        if epoch % test_epoch == 0:

            mae_recon, rmse_recon, mape_recon, mae_pred, rmse_pred, mape_pred, mae_conf, rmse_conf, mape_conf = TestEpoch(test_loader,model,prompt_prefix,scaler=scaler)

            if args.task in ['all','imputation']:
            
                mylogger.info(f"[Test][imputation] epoch {epoch} mae:{mae_recon} rmse:{rmse_recon} mape:{mape_recon}")
            
            if args.task in ['all','prediction']:
            
                mylogger.info(f"[Test][prediction] epoch {epoch} mae:{mae_pred} rmse:{rmse_pred} mape:{mape_pred} mae_conf:{mae_conf} rmse_conf:{rmse_conf} mape_conf:{mape_conf}")
        
        #scheduler.step()
        mylogger.info(f"[Scheduler] epoch {epoch} lr:{optim.param_groups[0]['lr']}")
        

        if patience_count >= args.patience:
                mylogger.info('early stop')
                break
        
    if args.nni:
        nni.report_final_result(best_loss)

    
    model.load_state_dict(best_model,strict=False)

    mae_recon, rmse_recon, mape_recon, mae_pred, rmse_pred, mape_pred, mae_conf, rmse_conf, mape_conf = TestEpoch(test_loader,model,prompt_prefix,scaler,save=args.save_result)

    if args.task in ['all','imputation']:
    
        mylogger.info(f"[Test][imputation] best model mae:{mae_recon} rmse:{rmse_recon} mape:{mape_recon}")
    
    if args.task in ['all','prediction']:
    
        mylogger.info(f"[Test][prediction] best model mae:{mae_pred} rmse:{rmse_pred} mape:{mape_pred} mae_conf:{mae_conf} rmse_conf:{rmse_conf} mape_conf:{mape_conf}")   

    draw_loss_line(train_loss_line,val_loss_line,os.path.join(LOG_DIR,'loss.png'))


def getllm(args):
    device = args.device
    
    if args.model == 't5':
        basemodel = T5(args.causal, args.lora, args.ln_grad, args.llm_layers, device=device)
    elif args.model == 'gpt2':
        basemodel = GPT2(args.causal, args.lora, args.ln_grad, args.llm_layers, device=device)
    elif args.model == 'llama3':
        basemodel = LLAMA3(args.causal, args.lora, args.ln_grad, args.llm_layers, device=device)
    elif args.model == 'qwen3':
        basemodel = Qwen3(args.causal, args.lora, args.ln_grad, args.llm_layers, device=device)
    return basemodel

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def setup_device(args):
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            args.device = 'cpu'
            print("Warning: CUDA not available, using CPU instead")
        else:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                args.device = 'cpu'
                print("Warning: No CUDA devices found, using CPU instead")
            elif num_gpus == 1:
                args.device = 'cuda:0'
                print("Using GPU 0 (only one GPU available)")
            else:
                best_gpu = 0
                max_free_memory = 0
                gpu_info = []
                
                for i in range(num_gpus):
                    try:
                        torch.cuda.set_device(i)
                        
                        free_memory, total_memory = torch.cuda.mem_get_info(i)
                        reserved = torch.cuda.memory_reserved(i)
                        allocated = torch.cuda.memory_allocated(i)
                        
                        total_gb = total_memory / (1024**3)
                        free_gb = free_memory / (1024**3)
                        reserved_gb = reserved / (1024**3)
                        allocated_gb = allocated / (1024**3)
                        
                        gpu_info.append({
                            'id': i,
                            'total': total_gb,
                            'free': free_gb,
                            'reserved': reserved_gb,
                            'allocated': allocated_gb
                        })
                        
                        if free_memory > max_free_memory:
                            max_free_memory = free_memory
                            best_gpu = i
                    except Exception as e:
                        print(f"Warning: Cannot access GPU {i}: {e}")
                        continue
                
                if not gpu_info:
                    args.device = 'cpu'
                    print("Warning: All GPUs are unavailable, using CPU instead")
                else:
                    args.device = f'cuda:{best_gpu}'
                    best_info = gpu_info[best_gpu]
                    print(f"Auto-selected GPU {best_gpu}: {best_info['free']:.2f}GB free / {best_info['total']:.2f}GB total")
                    
                    if len(gpu_info) > 1:
                        print("Available GPUs:")
                        for info in gpu_info:
                            marker = " <-- selected" if info['id'] == best_gpu else ""
                            print(f"  GPU {info['id']}: {info['free']:.2f}GB free / {info['total']:.2f}GB total{marker}")
    
    if args.device.startswith('cuda:'):
        try:
            gpu_id = int(args.device.split(':')[1])
            torch.cuda.set_device(gpu_id)
            print(f"Set default CUDA device to {args.device}")
        except Exception as e:
            print(f"Warning: Failed to set CUDA device: {e}")
            args.device = 'cpu'
    
    return args.device

def main(args):

    output_len = args.predict_len
    window_size = args.sample_len + args.predict_len
    if args.task == 'all':
        output_len += args.sample_len
    elif args.task == 'imputation':
        output_len = args.sample_len
        window_size -= args.predict_len

    if args.nni:
        params = nni.get_next_parameter()
        args.time_token_dim = params['time_token_dim']
        args.node_emb_dim = params['node_emb_dim']
        args.trunc_k = params['trunc_k']

    num_envs = 4

    basemodel = getllm(args)

    train_loader, val_loader, test_loader, scaler, features, adj_mx_train, distance_mx_train, adj_mx_test, distance_mx_test = load_data_ood(
        dataset=args.dataset,
        batch_size=args.batch_size,
        sample_len=args.sample_len,
        output_len=output_len,
        window_size=window_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        data_path=args.data_path,
        adj_path=args.adj_filename,
        target_strategy=args.target_strategy,
        few_shot = args.few_shot, node_shuffle_seed = args.node_shuffle_seed, device = args.device,
        num_envs=num_envs,
        max_increase_ratio = args.max_increase_ratio,
        test_increase_ratio = args.test_increase_ratio,
        test_decrease_ratio = args.test_decrease_ratio,
    )
    LOG_DIR = os.path.join(args.log_root,f'{get_time_str()}_{args.desc}_test_{args.dataset}_{random_str()}')
    check_dir(LOG_DIR,mkdir=True)
    logpath = os.path.join(LOG_DIR,f'experiments.log')
    mylogger = getlogger(logpath)
    mylogger.info(f'Structural shift: {args.max_increase_ratio}, test increase: {args.test_increase_ratio}, test decrease: {args.test_decrease_ratio}')
    
    prompt_prefix = None
    if not args.prompt_prefix is None:
        prompt_prefix = args.prompt_prefix

        tokenizer = basemodel.gettokenizer()

        prompt_prefix = tokenizer(prompt_prefix, 
                        return_tensors="pt", return_attention_mask=False)
        prompt_prefix = prompt_prefix['input_ids'].to(args.device).view(-1,1)#[:-1,:]


    modelpath = os.path.join(LOG_DIR,f'{get_time_str()}_{args.desc}.pth')

    mylogger.info(args)

    model = STALLM(basemodel=basemodel, sample_len= args.sample_len, output_len = output_len, \
                    input_dim = args.input_dim , output_dim = args.output_dim , \
                     node_emb_dim=args.node_emb_dim , \
                    sag_dim = args.sag_dim, sag_tokens = args.sag_tokens, \
                     adj_mx = adj_mx_train, dis_mx = distance_mx_train, \
                    use_node_embedding = args.node_embedding ,use_timetoken= args.time_token, \
                    use_sandglassAttn = args.sandglassAttn, dropout = args.dropout, trunc_k = args.trunc_k, t_dim = args.t_dim,wo_conloss=args.wo_conloss, args=args).to(args.device)
    
    if not args.from_pretrained_model is None:
        model.load(args.from_pretrained_model)
    
    if args.zero_shot and args.from_pretrained_model is None :
        mylogger.info(f'Please specify pretrained model when test zero-shot')
        exit()
    
    mylogger.info(model)
    total_params, total_trainable_params = model.params_num()
    mylogger.info(f'total_trainable_params:{total_trainable_params}')

    mylogger.info(model.grad_state_dict().keys())

    Train(args, mylogger, model, prompt_prefix, scaler, train_loader, val_loader, test_loader, LOG_DIR)

    model.save(modelpath)

if __name__ == '__main__':
    args = InitArgs()

    setup_device(args)

    main(args)


    



    
    