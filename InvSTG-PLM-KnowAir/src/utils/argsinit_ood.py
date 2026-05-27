import argparse

def AddModelArgs(parser):

    parser.add_argument("--lora",action="store_true", help="whether use lora fine-tunning")

    parser.add_argument("--prompt_pool",action="store_true")

    parser.add_argument("--ln_grad",action="store_true", help="whether to calculate gradient of LayerNorm ")

    parser.add_argument("--causal", default=0, type=int,
                            help="LLM causal attention")
    
    parser.add_argument("--prompt_prefix", default=None ,type=str, help="whether use prompt or not")


    parser.add_argument("--node_embedding", action="store_true")

    parser.add_argument("--time_token", action="store_true")


    parser.add_argument("--model", default="gpt2" ,type=str)

    parser.add_argument("--llm_layers", default=None, type=int)

    parser.add_argument("--dropout", default=0, type=float)

    parser.add_argument("--trunc_k", default=16, type=int)

    parser.add_argument("--t_dim", default=64, type=int)

    parser.add_argument("--node_emb_dim", default=128, type=int)

    parser.add_argument("--sandglassAttn", action="store_true")
    parser.add_argument("--wo_conloss" , action="store_true")
    parser.add_argument("--sag_dim", default=128, type=int)
    parser.add_argument("--sag_tokens", default=128, type=int)


def AddDataArgs(parser):

    parser.add_argument("--dataset" ,type=str)

    parser.add_argument("--data_path" ,type=str)

    parser.add_argument("--adj_filename" ,default=None , type=str)

    parser.add_argument("--sample_len", default=12, type=int)

    parser.add_argument("--predict_len", default=12, type=int)



    parser.add_argument("--train_ratio", default=0.6, type=float)

    parser.add_argument("--val_ratio", default=0.6, type=float)

    parser.add_argument("--input_dim", default=1, type=int)

    parser.add_argument("--output_dim", default=1, type=int)

    parser.add_argument("--knowair_target_col", default=17, type=int)

def AddTrainArgs(parser):

    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--lr_decay", default=0.99, type=float)

    parser.add_argument("--weight_decay", default=0.05, type=float)

    parser.add_argument("--batch_size", default=4, type=int)

    parser.add_argument("--epoch", default=100, type=int)

    parser.add_argument("--val_epoch", default=5, type=int)

    parser.add_argument("--test_epoch", default=5, type=int)

    parser.add_argument("--patience", default=100, type=int)

def AddCausalArgs(parser):
    # dataset
    parser.add_argument('--alpha', default=1e-6, type=float, help='invariant loss')

    parser.add_argument('--causal_ratio', default=0.9, type=float, help='causal_ratio r')    
    parser.add_argument('--normalized_k', default=0.01, type=float, 
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')    # k
    parser.add_argument('--quantile_k', default=0.1, type=float, help='quantile_k')
    # hyperparam
    
    parser.add_argument('--intervention_mechanism',type=int,default=1, help='0 none ; 1 DIR; 2 DIR2; 3 DIR3')    
    parser.add_argument('--var_ratio', default=1, type=float, help='alpha increase ratio for each epoch') 
    parser.add_argument('--edge_score', type=int, default=1, help='0 mlp_cat; 1 mlp_hadamard;') #


    parser.add_argument('--clip', default=5, type=int, help='gradient clip.')  # None, 2

    parser.add_argument('--heads', type=int, default=1, help='attention heads.')

    parser.add_argument('--num_nodes', type=int, default=358, help='num of nodes')
    parser.add_argument('--input_length', default=12, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--irm_type', default='rex_causal', type=str)


def InitArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--desc", default='phi2_s_token', type=str,
                            help="description")
    
    parser.add_argument("--log_root", default='/root/data1/My-STD-PLM/code/STD-PLM/logs', type=str,
                            help="Log root directory")
    
    parser.add_argument("--from_pretrained_model" , default=None ,type=str)

    parser.add_argument("--zero_shot" , action="store_true")

    parser.add_argument("--nni" , action="store_true")

    parser.add_argument("--save_result" , action="store_true")

    parser.add_argument("--few_shot" , default=1, type=float)

    parser.add_argument("--node_shuffle_seed" , default=None, type=int)

    parser.add_argument("--trainset_dynamic_missing" , action="store_true")

    parser.add_argument("--task" , default='prediction' ,choices=['prediction','imputation','all'],type=str)
    
    parser.add_argument("--target_strategy" , default='random' ,choices=['random','hybrid'],type=str)

    # OOD related arguments
    parser.add_argument("--max_increase_ratio", default=0, type=float, help='dataset-driven hyperparameter for OOD node partitioning')
    parser.add_argument("--test_increase_ratio", default=0, type=float, help='ratio of unseen nodes in test set (must less than max_increase_ratio)')
    parser.add_argument("--test_decrease_ratio", default=0, type=float, help='ratio of overlapping nodes to remove from test set (must less than 1)')
    parser.add_argument("--gaussian_noise_ratio", default=0, type=float, help='ratio of Gaussian noise std to data std for test input transformation (0.0 means no noise)')
    parser.add_argument("--gaussian_noise_mean_shift", default=0, type=float, help='mean shift of Gaussian noise for test input transformation (0.0 means no mean shift)')
    parser.add_argument("--gaussian_noise_node_ratio", default=1.0, type=float, help='ratio of nodes to add Gaussian noise to (0.0-1.0, 1.0 means all nodes)')
    parser.add_argument("--gaussian_noise_node_random_per_batch", default=False, help='if set, randomly select nodes for each batch; otherwise, use fixed node selection for all batches')
    parser.add_argument("--eval_sudden_change", action="store_true", help='evaluate sudden-change/high-value points with weighted MAE/RMSE')
    parser.add_argument("--sudden_threshold_start", default=75.0, type=float, help='label threshold used to define high-value points')
    parser.add_argument("--sudden_threshold_change", default=20.0, type=float, help='label difference threshold used to define sudden-change points')

    AddDataArgs(parser)

    AddModelArgs(parser)

    AddTrainArgs(parser)

    AddCausalArgs(parser)

    args = parser.parse_args()

    return args