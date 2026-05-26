from typing import Iterator, Mapping
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from utils.utils_GNN import norm_Adj, lap_eig, topological_sort
from model.sandglassAttn import SAG
import numpy as np
from model.position import PositionalEncoding
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops
# import torchsort


class DecodingLayer(nn.Module):

    def __init__(self, input_dim ,emb_dim, output_dim):
        super().__init__()

        hidden_size = (emb_dim+output_dim)*2//3
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
        #nn.Linear(in_features=input_dim,out_features=output_dim)
        
    def forward(self, llm_hidden):

        out = self.fc(llm_hidden)

        return out

class TimeEmbedding(nn.Module):

    def __init__(self,t_dim):
        super().__init__()

        #self.hour_embedding = nn.Embedding(num_embeddings=24,embedding_dim=t_dim)
        self.day_embedding = nn.Embedding(num_embeddings=288,embedding_dim=t_dim)
        self.week_embedding = nn.Embedding(num_embeddings=7,embedding_dim=t_dim)

    def forward(self,TE):

        # TE (B,T,5)

        B,T,_ = TE.shape

        week = (TE[...,2].to(torch.long) % 7).view(B*T,-1)
        hour = (TE[...,3].to(torch.long) % 24).view(B*T,-1)
        minute = (TE[...,4].to(torch.long) % 60).view(B*T,-1)

        DE = self.day_embedding((hour*60+minute)//5)
        #HE = self.hour_embedding(hour)
        WE = self.week_embedding(week)

        te = torch.concat((DE,WE),dim=-1).view(B,T,-1)

        return te

class NodeEmbedding(nn.Module):
    def __init__(self, adj_mx, node_emb_dim, k = 16, dropout = 0 ):
        super().__init__()
        N,_ = adj_mx.shape
        self.k = k
        self.max_num_nodes = N 

        self.setadj(adj_mx=adj_mx)

        self.fc = nn.Linear(in_features=k,out_features=node_emb_dim)

    def forward(self, adj_mx=None):
        if adj_mx is None:
            lap_eigvec = self.lap_eigvec
        else:
            eigvec, eigval = lap_eig(adj_mx)
            k = self.k
            N_actual = adj_mx.shape[0]
            if k > N_actual:
                eigvec = np.concatenate((eigvec, np.zeros((N_actual, k-N_actual))), axis=-1)
                eigval = np.concatenate((eigval, np.zeros(k-N_actual)), axis=-1)
            ind = np.abs(eigval).argsort(axis=0)[::-1][:k]
            eigvec = eigvec[:, ind]
            lap_eigvec = torch.as_tensor(eigvec, dtype=self.lap_eigvec.dtype, device=self.lap_eigvec.device)
        return self.fc(lap_eigvec)
    
    def setadj(self,adj_mx):
        N,_ = adj_mx.shape

        self.adj_mx = adj_mx

        eigvec, eigval = lap_eig(self.adj_mx)
        k = self.k
        if k>N:
            eigvec = np.concatenate((eigvec, np.zeros((N, k-N))), axis=-1)
            eigval = np.concatenate((eigval, np.zeros(k-N)), axis=-1)
        
        ind = np.abs(eigval).argsort(axis=0)[::-1][:k]

        eigvec = eigvec[:, ind]        

        if hasattr(self,'lap_eigvec'):
            self.lap_eigvec = torch.tensor(eigvec).float()
        else :
            self.register_buffer('lap_eigvec', torch.tensor(eigvec).float())
    
class Time2Token(nn.Module):
    def __init__(self,sample_len, features, emb_dim, tim_dim, dropout):
        super().__init__()
        
        self.sample_len = sample_len

        in_features =  sample_len*features*2 + tim_dim
        hidden_size = (in_features + emb_dim)*2//3
        self.fc_state = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        input_dim = tim_dim + (sample_len-1)*features*2
        hidden_size = (input_dim+emb_dim)*2//3
        self.fc_grad = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )        

        self.ln = nn.LayerNorm(emb_dim)

    def forward(self,x,te,mask):
        # te(B,T,tim_dim)

        B,N,TF = x.shape

        x = x.view(B,N,self.sample_len,-1) #B,N,T,F
        x = torch.concat((x,mask.view(B,N,self.sample_len,-1)),dim=-1)
        x = x.mean(dim=1) #B,T,F

        state = x.view(B,1,-1)
        state = torch.concat((state,te[:,-1:,:]),dim=-1)#(B,1,TF+tim_dim)
        state = self.fc_state(state)

        grad = (x[:,1:,:] - x[:,:-1,:]).view(B,1,-1)#(B,1,(T-1)F)
        grad = torch.concat((grad,te[:,-1:,:]),dim=-1)#(B,1,(T-1)F+tim_dim)
        grad = self.fc_grad(grad)

        out = torch.concat((state,grad),dim=1)

        out = self.ln(out)

        return out


class Node2Token(nn.Module):
    def __init__(self,sample_len, features, node_emb_dim, emb_dim, tim_dim, dropout, use_node_embedding):
        super().__init__()

        in_features = sample_len*features*2
        
        self.use_node_embedding = use_node_embedding

        state_features =  tim_dim
        if use_node_embedding:
            state_features += node_emb_dim

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, emb_dim),
        )
        
        hidden_size = node_emb_dim
        self.state_fc = nn.Sequential(
            nn.Linear(state_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,emb_dim),
        )

        self.mask_token = nn.Linear(in_features=sample_len*features,out_features=emb_dim)

        self.ln = nn.LayerNorm(emb_dim)

    def forward(self,x,te,ne_causal, ne_conf, mask):

        B,N,TF = x.shape

        mask = mask.permute(0,2,1,3).contiguous().view(B,N,-1) #(B,N,TF)
        x = torch.concat((x,mask),dim=-1)
        x = self.fc1(x) #B,N,D
        x += self.mask_token(mask) #test

        state= te[:,-1:,:].repeat(1,N,1)

        # state_causal
        if self.use_node_embedding:
            ne_causal = torch.unsqueeze(ne_causal,dim=0).repeat(B,1,1)
            state_causal = torch.concat((state,ne_causal),dim=-1)
        state_causal = self.state_fc(state_causal)
        out_causal = state_causal + x
        out_causal = self.ln(out_causal)

        # state_conf
        if self.use_node_embedding:
            ne_conf = torch.unsqueeze(ne_conf,dim=0).repeat(B,1,1)
            state_conf = torch.concat((state,ne_conf),dim=-1)
        state_conf = self.state_fc(state_conf)
        out_conf = state_conf + x
        out_conf = self.ln(out_conf)

        return out_causal, out_conf

class EstimationGate(nn.Module):
    """The estimation gate module."""

    def __init__(self, hid_dim):
        """
        hid_dim: hidden dimension
        """
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(hid_dim*2, hid_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hid_dim, 1)

    def forward(self, node_embedding, time_feat, history_data):
        """Generate gate value in (0, 1) based on current node and time step embeddings to roughly estimating the proportion of the two hidden time series."""

        batch_size, seq_length, _ ,_ = time_feat.shape
        estimation_gate_feat = torch.cat([time_feat, node_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1)], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)
        # activation
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        return history_data

class Causal_Token(nn.Module):
    """
    This is a method for calculating zi and zv
    We define edge_score by node similarity
    """
    def __init__(self, args, tim_dim, node_emb_dim, sample_len, input_dim):
        super(Causal_Token, self).__init__()
        self.top_k_ratio = args.causal_ratio  # hyperparam
        self.num_nodes = args.num_nodes
        self.hid_dim = node_emb_dim
        self.device = args.device
        self.batch_size = args.batch_size
        # self.use_distance = args.use_distance
        self.normalized_k = args.normalized_k
        self.quantile_k = args.quantile_k
        self.input_length = sample_len

        self.edge_score = args.edge_score

        self.num_feat = 1

        self.tim_dim = nn.Linear(tim_dim, self.hid_dim)
        # self.node_emb_dim = nn.Linear(node_emb_dim, self.hid_dim)
        self.ori_data_emb = nn.Linear(input_dim, self.hid_dim)
        self.input_dim = input_dim

        self.estimation_gate= EstimationGate(self.hid_dim)
        
        # node embeddings （trainable）
        self.node_embedding = nn.Parameter(torch.empty(self.num_nodes, self.hid_dim))

        self.mlp_cat = nn.Sequential(
            nn.Linear(self.hid_dim*2, self.hid_dim*4),
            nn.ReLU(),
            nn.Linear(self.hid_dim*4, 1)
        )

        self.mlp_had = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim*2),
            nn.ReLU(),
            nn.Linear(self.hid_dim*2, 1)
        )
        
        # start embedding layer for node feature
        self.embedding  = nn.Linear(self.num_feat, self.hid_dim)

        in_channel = self.input_length*self.hid_dim
        self.node_emb = nn.Linear(in_channel, self.hid_dim)
        self.node_emb_1 = nn.Linear(in_channel, self.hid_dim)   #source
        self.node_emb_2 = nn.Linear(in_channel, self.hid_dim)   #target
        # self.node_emb_12 = nn.Linear(2*self.hid_dim, self.hid_dim)   
        
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_embedding)

    def edge_index_to_adj(self, edge_index, num_nodes):
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        
        if edge_index.shape[1] > 0:
            row, col = edge_index[0], edge_index[1]
            adj[row, col] = 1.0
            
            adj = adj + adj.t()
            
            adj = torch.clamp(adj, 0, 1)
        
        adj.fill_diagonal_(1)
        
        return adj

    def topk_split_from_score(self, edge_score, epoch: int, training_stage: bool):
        score_g = edge_score.mean(dim=0)  # [E]
        E = score_g.numel()
        k = max(1, int(self.top_k_ratio * E))

        topk_idx = torch.topk(score_g, k, largest=True).indices
        hard = torch.zeros_like(score_g)
        hard[topk_idx] = 1.0

        if not training_stage:
            causal = hard
        else:
            t = min(1.0, max(0.0, (epoch - 80) / 50.0)) 
            tau = 0.5 * (1 - t) + 0.05 * t
            thr = torch.quantile(score_g.detach(), 1 - self.top_k_ratio)
            soft = torch.sigmoid((score_g - thr) / tau)
            causal = hard + (soft - soft.detach())

        conf = 1.0 - causal
        return causal, conf


    def forward(self, ori_data , ori_adj, time_embedding, epoch=None, training_stage=False):

        B, N, TF = ori_data.shape
        # prepare data
        time_emb = self.tim_dim(time_embedding)  # [B, T, hid_dim]
        time_feat = time_emb.unsqueeze(2).expand(B, self.input_length, N, self.hid_dim)  # [B, T, N, hid_dim]
        history_data = self.ori_data_emb(ori_data.view(B,N,self.input_length,-1)).permute(0,2,1,3).contiguous()  # [B, T, N, hid_dim]
        if N <= self.node_embedding.shape[0]:
            node_emb = self.node_embedding[:N, :]  # [N, hid_dim]
        else:
            additional_nodes = N - self.node_embedding.shape[0]
            additional_emb = torch.zeros(additional_nodes, self.hid_dim, device=self.node_embedding.device)
            node_emb = torch.cat([self.node_embedding, additional_emb], dim=0)  # [N, hid_dim]

        # cal graph structure
        adaptive_adj = F.softmax(F.relu(torch.mm(node_emb, node_emb.T)), dim=1)  # [1, N, N] with direction
        adaptive_adj = adaptive_adj * (adaptive_adj >= self.normalized_k)
        adaptive_adj = 0.5 * (adaptive_adj + adaptive_adj.T)  
        adaptive_adj = adaptive_adj + ori_adj

        edge_index, edge_attr = dense_to_sparse(adaptive_adj)
        row, col = edge_index

        gated_history_data  = self.estimation_gate(node_emb, time_feat, history_data)
        local_st = gated_history_data.transpose(1,2)
        _, _, all_feature1, all_feature2 = local_st.shape
        local_st = local_st.contiguous().view(-1, all_feature1*all_feature2)    # for batch node-level x [B*N L*d]
        
        x = self.node_emb(local_st)
        x = x.view(B, N, -1)
        x_row = x[:, row, :]
        x_col = x[:, col, :]

        if(self.edge_score==0):
            edge_rep = torch.cat([x_row, x_col], dim=-1)
            edge_score = self.mlp_cat(edge_rep).squeeze(-1)
        elif(self.edge_score==1):
            edge_rep = x_row * x_col
            edge_score = self.mlp_had(edge_rep).squeeze(-1)

        edge_mask_causal_g, edge_mask_conf_g = self.topk_split_from_score(edge_score, epoch, training_stage)
            
        edge_mask_adj_causal = torch.zeros_like(adaptive_adj)
        edge_mask_adj_conf   = torch.zeros_like(adaptive_adj)
        edge_mask_adj_causal[row, col] = edge_mask_causal_g
        edge_mask_adj_conf[row, col]   = edge_mask_conf_g
        
        causal_adj = adaptive_adj * edge_mask_adj_causal
        
        conf_adj = adaptive_adj * edge_mask_adj_conf

        return causal_adj, conf_adj


class STALLM(nn.Module):
    def __init__(self,basemodel,sample_len, output_len,\
                 input_dim , output_dim , 
                  node_emb_dim , sag_dim, sag_tokens, \
                 adj_mx = None, dis_mx = None , use_node_embedding = True,\
                 use_timetoken = True, use_sandglassAttn = True, \
                 dropout = 0, trunc_k = 16, t_dim = 64,wo_conloss=False,args=None):
        super().__init__()

        self.topological_sort_node = True

        tim_dim = t_dim *2 #hour,week    

        self.device = args.device
        self.register_buffer("ori_adj", torch.as_tensor(adj_mx, dtype=torch.float32))
        self.register_buffer("dis_mx", torch.as_tensor(dis_mx, dtype=torch.float32))
        self.setadj(self.ori_adj, self.dis_mx)

        self.output_dim = output_dim
        self.input_dim = input_dim

        self.emb_dim = basemodel.emb_dim
        self.basemodel = basemodel
        

        self.sample_len = sample_len
        self.output_len = output_len
        self.sag_tokens = sag_tokens


        self.use_sandglassAttn = use_sandglassAttn
        if use_sandglassAttn:
            self.wo_conloss = wo_conloss
            self.sandglassAttn = SAG(sag_dim=sag_dim, sag_tokens=sag_tokens, emb_dim=self.emb_dim, sample_len=sample_len, features=input_dim ,dropout=dropout)

        self.causal_token = Causal_Token(args, tim_dim, node_emb_dim, sample_len, input_dim)

        self.spatialTokenizer =  Node2Token(sample_len=sample_len,features=input_dim,node_emb_dim=node_emb_dim,\
                                            emb_dim=self.emb_dim, \
                                            tim_dim=tim_dim,dropout=dropout,use_node_embedding=use_node_embedding)

        self.out_mlp = DecodingLayer(input_dim=output_dim*sample_len, \
                                     emb_dim=self.emb_dim, \
                                     output_dim=output_dim*output_len)

        self.timeembedding = TimeEmbedding(t_dim=t_dim)

        self.use_node_embedding = use_node_embedding
        if use_node_embedding:
            self.node_embd_layer = NodeEmbedding(adj_mx=adj_mx, node_emb_dim=node_emb_dim, dropout=dropout)

        self.use_timetoken = use_timetoken
        if use_timetoken:
            self.timeTokenizer = Time2Token(sample_len=sample_len,features=input_dim,\
                                            emb_dim=self.emb_dim, \
                                            tim_dim=tim_dim,dropout=dropout)

        self.layer_norm = nn.LayerNorm(self.emb_dim)

        self.out_mlp_conf = DecodingLayer(input_dim=output_dim*sample_len, \
                                     emb_dim=self.emb_dim, \
                                     output_dim=output_dim*output_len)
        self.env_gate = nn.Linear(output_dim*output_len, 1)  # z_e -> scalar gate

        self.epoch = 0
        self.training_stage = False
        self.test_causal_zero = False  # Flag for testing causal_adj sensitivity

        node_order,node_order_rev = topological_sort(adj_mx)
        self.register_buffer("node_order", torch.as_tensor(node_order, dtype=torch.long))
        self.register_buffer("node_order_rev", torch.as_tensor(node_order_rev, dtype=torch.long))

    def forward(self,x:torch.FloatTensor,timestamp:torch.Tensor,prompt_prefix:Optional[torch.LongTensor],mask:torch.LongTensor):
        other_loss = []

        # timestamp (B,T,5)
        timestamp = timestamp[:,:self.sample_len,:]

        B,N,TF = x.shape #(Batch,N,T*features)

        te = self.timeembedding(timestamp)

        causal_edge_adj, conf_edge_adj = self.causal_token(x, self.ori_adj, te, self.epoch, self.training_stage)
        
        if self.test_causal_zero:
            causal_edge_adj = torch.zeros_like(causal_edge_adj)
        
        if self.use_node_embedding:
            ne_causal = self.node_embd_layer(causal_edge_adj)
            ne_conf = self.node_embd_layer(conf_edge_adj)
        else:
            ne_causal = None
            ne_conf = None
        self.setadj(causal_edge_adj, self.dis_mx) # 1

        # spatial token
        spatial_token, spatial_token_conf = self.spatialTokenizer(x,te,ne_causal,ne_conf,mask)
        if self.topological_sort_node:
            spatial_token = spatial_token[:,self.node_order,:]

            # node_order_conf,node_order_rev_conf = topological_sort(conf_edge_adj.detach().cpu().numpy())
            spatial_token_conf = spatial_token_conf[:,self.node_order,:]

        # spatial -> sandglassAttn
        st_embedding = spatial_token
        st_embedding_conf = spatial_token_conf
        s_num = N
        if self.use_sandglassAttn:
            s_num = self.sag_tokens
            st_embedding,attn_weights = self.sandglassAttn.encode(st_embedding) #(B,N',D) #attn_weights(B,N',N)
            st_embedding_conf,attn_weights_conf = self.sandglassAttn.encode(st_embedding_conf) #(B,N',D) #attn_weights(B,N',N)
            if not self.wo_conloss:
                scale = attn_weights.sum(dim=1)#(B,N)

                order = self.node_order
                A = self.adj_mx
                A_used = A[order][:, order]
                sag_score = torch.einsum('mn,bhn->bhm', A_used, attn_weights)
                other_loss_item = -((sag_score*attn_weights-attn_weights*attn_weights)).sum(dim=2).mean()*10
                other_loss.append(other_loss_item)

                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())

        if self.use_timetoken:
            time_tokens = self.timeTokenizer(x,te,mask)
            time_tokens_idx = st_embedding.shape[1]
            st_embedding = torch.concat([time_tokens,st_embedding],dim=1)
            st_embedding_conf = torch.concat([time_tokens,st_embedding_conf],dim=1)

        if prompt_prefix is not None:
            prompt_len,_ = prompt_prefix.shape
            prompt_embedding = self.basemodel.getembedding(prompt_prefix).view(1,prompt_len,-1)
            prompt_embedding = prompt_embedding.repeat(B,1,1)
            st_embedding = torch.concat([prompt_embedding,st_embedding],dim=1)
            st_embedding_conf = torch.concat([prompt_embedding,st_embedding_conf],dim=1)

        hidden_state = st_embedding
        hidden_state_conf = st_embedding_conf

        # a new method for inv st_embedding
        hidden_state = self.basemodel(hidden_state)
        s_state = hidden_state[:,-s_num:,:]
        if self.use_sandglassAttn:
            s_state = self.sandglassAttn.decode(s_state,spatial_token) 
        s_state += spatial_token
        if self.topological_sort_node: ######
            s_state = s_state[:,self.node_order_rev,:]
        if self.use_timetoken:
            t_state = hidden_state[:,-time_tokens_idx-1:-time_tokens_idx,:]
            t_state += time_tokens[:,-1:,:]
            s_state += t_state
        s_state = self.layer_norm(s_state) # z_s [B, N, D]

        out = self.out_mlp(s_state)

        hidden_state_conf = self.basemodel(hidden_state_conf).detach()
        s_state_conf = hidden_state_conf[:,-s_num:,:]
        if self.use_sandglassAttn:
            s_state_conf = self.sandglassAttn.decode(s_state_conf,spatial_token_conf)
        s_state_conf += spatial_token_conf
        if self.topological_sort_node: ######
            s_state_conf = s_state_conf[:,self.node_order_rev,:]
        if self.use_timetoken:
            t_state_conf = hidden_state_conf[:,-time_tokens_idx-1:-time_tokens_idx,:]
            t_state_conf += time_tokens[:,-1:,:]
            s_state_conf += t_state_conf
        s_state_conf = self.layer_norm(s_state_conf) # z_e [B, N, D]

        out_conf = self.out_mlp_conf(s_state_conf.detach())

        return out, out_conf, other_loss, s_state_conf
        # return out, other_loss

    def combine_with_ze(self, out, s_state_conf):
        conf_nd = self.out_mlp_conf(s_state_conf).detach()
        gate_n = torch.sigmoid(self.env_gate(conf_nd))
        gate = gate_n.unsqueeze(0)
        return gate * out  # [B，N, output_dim*output_len]
        

    def grad_state_dict(self):
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return  {name: param.detach() for name, param in self.state_dict().items() if name in save_list}
        
    
    def save(self, path:str):
        
        selected_state_dict = self.grad_state_dict()
        torch.save(selected_state_dict, path)
    
    def load(self, path:str):

        loaded_params = torch.load(path)
        self.load_state_dict(loaded_params,strict=False)
    
    def params_num(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params += sum(p.numel() for p in self.buffers())
        
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, total_trainable_params

    def setadj(self,adj_mx,dis_mx):

        if torch.is_tensor(adj_mx):
            self.adj_mx = adj_mx
        else:
            self.adj_mx = torch.as_tensor(adj_mx, dtype=torch.float32, device=dis_mx.device)

        self.dis_mx = dis_mx if torch.is_tensor(dis_mx) else torch.as_tensor(dis_mx, dtype=torch.float32, device=self.adj_mx.device)

        self.d_mx = self.adj_mx.sum(dim=1).float()

        base = self.d_mx.new_full((self.d_mx.size(0),), 1.05)
        self.alpha = base + torch.softmax(self.d_mx, dim=0) * 5


