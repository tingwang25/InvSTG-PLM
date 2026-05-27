from typing import Iterator, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models import Model
from torch.nn.parameter import Parameter
from typing import Any, Dict, Optional, Tuple, Union
from swift import Swift, LoRAConfig
from modelscope import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM


class BaseModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
    
    def forward(self,x):
        raise NotImplementedError('error')
    
    def getembedding(self,x):
        raise NotImplementedError('error')
    
    def gettokenizer(self):
        raise NotImplementedError('error')
    
    def getmonthembedding(self):
        months = ['January','February','March','April','May','June','July','August','September','October','November','December']
        inputs = self.tokenizer.convert_tokens_to_ids(months)
        month_ids = torch.tensor(inputs).to(self.device).view(-1,1)
        month_embedding = self.getembedding(month_ids).view(-1,self.emb_dim)
        return month_embedding
    
    def getweekembedding(self):
        weeks = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
        inputs = self.tokenizer.convert_tokens_to_ids(weeks)
        week_ids = torch.tensor(inputs).to(self.device).view(-1,1)
        week_embedding = self.getembedding(week_ids).view(-1,self.emb_dim)
        return week_embedding

class Phi2(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None,device='cuda'):
        super().__init__(device=device)

        causal = bool(causal)

        self.emb_dim = 2560

        llm = Model.from_pretrained('AI-ModelScope/phi-2',trust_remote_code=True)

        if not layers is None:

            llm.transformer.h = llm.transformer.h[:layers]

        for pblock in llm.transformer.h:
            mixer = pblock.mixer
            mixer.inner_attn.causal = causal
            mixer.inner_attn.causal = causal
        
        for name, param in llm.named_parameters():
            param.requires_grad_(False)

        if lora:

            lora_config = LoRAConfig(
                    r=16,
                    target_modules=['Wqkv'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            llm = Swift.prepare_model(llm, lora_config,trust_remote_code=True)

        self.llm_embd = llm.transformer.embd # wte:51200->2560  (B,len,1) -> (B,len,emb_dim)

        self.llm_h = llm.transformer.h # ModuleList (B,len,emb_dim) ->  (B,len,emb_dim)
        
        if ln_grad:
            for i, (name, param) in enumerate(self.llm_h.named_parameters()):
                if 'ln' in name: # or 'mlp' in name:
                    param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/phi-2", trust_remote_code=True)

    def forward(self,x:torch.FloatTensor):

        hidden_state = x

        for layer in self.llm_h:
            hidden_state = layer(hidden_state)

        out = hidden_state

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm_embd(x)
    
    def gettokenizer(self):

        return self.tokenizer 
    
    def getmonthembedding(self):
        inputs = self.tokenizer('January,February,March,April,May,June,July,August,September,October,November,December', 
                    return_tensors="pt", return_attention_mask=False)
        month_ids= inputs['input_ids'].to(self.device).view(-1,1)[::2]
        month_embedding = self.getembedding(month_ids).view(-1,self.emb_dim)
        return month_embedding


class GPT2(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None,device='cuda'):
        super().__init__(device=device)

        causal = bool(causal)

        self.emb_dim = 768

        self.llm = Model.from_pretrained('AI-ModelScope/gpt2',trust_remote_code=True)

        if not layers is None:

            self.llm.transformer.h = self.llm.transformer.h[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:

            lora_config = LoRAConfig(
                    r=16,
                    target_modules=['q_attn','c_attn'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True).model

        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'ln' in name  or 'wpe' in name:
                    param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/gpt2", trust_remote_code=True)

    def forward(self,x:torch.FloatTensor,attention_mask=None):

        out = self.llm(inputs_embeds=x,attention_mask=attention_mask,output_hidden_states=True).hidden_states[-1]

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm.transformer.wte(x)
    
    def gettokenizer(self):

        return self.tokenizer 


class Transformer(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None,device='cuda'):
        super().__init__(device=device)


        self.emb_dim = 768

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=12)
        self.llm = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=3)


    def forward(self,x:torch.FloatTensor,attention_mask=None):

        out = self.llm(x)

        return out

   

class LLAMA3(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None,device='cuda'):
        super().__init__(device=device)

        causal = bool(causal)

        self.emb_dim = 4096

        self.llm = Model.from_pretrained('LLM-Research/Meta-Llama-3-8B-Instruct',trust_remote_code=True)

        print(self.llm)

        if not layers is None:

            self.llm.model.layers = self.llm.model.layers[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:

            lora_config = LoRAConfig(
                    r=16,
                    target_modules=['q_proj','k_proj','v_proj','o_proj'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True).model

        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'norm' in name  or 'wpe' in name:
                    param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained("LLM-Research/Meta-Llama-3-8B-Instruct", trust_remote_code=True)

    def forward(self,x:torch.FloatTensor,attention_mask=None):

        out = self.llm(inputs_embeds=x,attention_mask=attention_mask,output_hidden_states=True).hidden_states[-1]

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm.model.embed_tokens(x)
    
    def gettokenizer(self):

        return self.tokenizer 

class T5(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None,device='cuda'):
        super().__init__(device=device)

        causal = bool(causal)

        self.emb_dim = 768

        self.llm = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base',trust_remote_code=True)

        if not layers is None:
            self.llm.encoder.block = self.llm.encoder.block[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:
            lora_config = LoRAConfig(
                    r=16,
                    target_modules=['q','k','v','o'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True).model

        
        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'layer_norm' in name or 'final_layer_norm' in name:
                    param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", trust_remote_code=True)

    def forward(self,x:torch.FloatTensor,attention_mask=None):

        encoder_outputs = self.llm.encoder(inputs_embeds=x, attention_mask=attention_mask, output_hidden_states=True)
        out = encoder_outputs.last_hidden_state

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm.shared(x)
    
    def gettokenizer(self):

        return self.tokenizer 


class Qwen3(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None,device='cuda'):
        super().__init__(device=device)

        causal = bool(causal)

        # Qwen3-0.6B has embedding dimension of 512
        self.emb_dim = 1024

        model_path = None
        try:
            self.llm = Model.from_pretrained('Qwen/Qwen3-0.6B',trust_remote_code=True)
            model_path = 'Qwen/Qwen3-0.6B'
        except Exception as e:
            print(f"Failed to load from ModelScope: {e}")
            try:
                self.llm = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',trust_remote_code=True)
                model_path = 'Qwen/Qwen3-0.6B'
            except Exception as e2:
                print(f"Failed to load from HuggingFace Qwen/Qwen3-0.6B: {e2}")
                try:
                    self.llm = AutoModelForCausalLM.from_pretrained('qwen/Qwen3-0.6B',trust_remote_code=True)
                    model_path = 'qwen/Qwen3-0.6B'
                except Exception as e3:
                    print(f"Failed to load from qwen/Qwen3-0.6B: {e3}")
                    raise RuntimeError("Failed to load Qwen3-0.6B model from all attempted paths")

        if hasattr(self.llm, 'to'):
            self.llm = self.llm.to(self.device)
        elif hasattr(self.llm, 'model') and hasattr(self.llm.model, 'to'):
            self.llm.model = self.llm.model.to(self.device)

        print(f"Loaded Qwen3-0.6B model from {model_path}")
        print(self.llm)

        if not layers is None:
            if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'layers'):
                self.llm.model.layers = self.llm.model.layers[:layers]
            elif hasattr(self.llm, 'transformer') and hasattr(self.llm.transformer, 'h'):
                self.llm.transformer.h = self.llm.transformer.h[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:
            lora_config = LoRAConfig(
                    r=16,
                    target_modules=['q_proj','k_proj','v_proj','o_proj'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True)
            if hasattr(self.llm, 'model'):
                self.llm = self.llm.model

        
        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'norm' in name or 'ln' in name or 'layer_norm' in name:
                    param.requires_grad = True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen3-0.6B', trust_remote_code=True)

    def forward(self,x:torch.FloatTensor,attention_mask=None):
        if x.device != self.device:
            x = x.to(self.device)
        if attention_mask is not None and attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)
        
        outputs = self.llm(inputs_embeds=x,attention_mask=attention_mask,output_hidden_states=True)
        
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            out = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            out = outputs.last_hidden_state
        else:
            if hasattr(outputs, 'logits'):
                raise RuntimeError("Cannot extract hidden states from Qwen3 model output")
            out = outputs

        return out

    def getembedding(self, x:torch.FloatTensor):
        if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'embed_tokens'):
            return self.llm.model.embed_tokens(x)
        elif hasattr(self.llm, 'transformer') and hasattr(self.llm.transformer, 'wte'):
            return self.llm.transformer.wte(x)
        elif hasattr(self.llm, 'embed_tokens'):
            return self.llm.embed_tokens(x)
        else:
            raise AttributeError("Cannot find embedding layer in Qwen3 model")
    
    def gettokenizer(self):

        return self.tokenizer 


class Qwen3_1_7B(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None,device='cuda'):
        super().__init__(device=device)

        causal = bool(causal)

        # Qwen3-1.7B has embedding dimension of 1536
        self.emb_dim = 1536

        model_path = None
        try:
            self.llm = Model.from_pretrained('Qwen/Qwen3-1.7B',trust_remote_code=True)
            model_path = 'Qwen/Qwen3-1.7B'
        except Exception as e:
            print(f"Failed to load from ModelScope: {e}")
            try:
                self.llm = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B',trust_remote_code=True)
                model_path = 'Qwen/Qwen3-1.7B'
            except Exception as e2:
                print(f"Failed to load from HuggingFace Qwen/Qwen3-1.7B: {e2}")
                try:
                    self.llm = AutoModelForCausalLM.from_pretrained('qwen/Qwen3-1.7B',trust_remote_code=True)
                    model_path = 'qwen/Qwen3-1.7B'
                except Exception as e3:
                    print(f"Failed to load from qwen/Qwen3-1.7B: {e3}")
                    raise RuntimeError("Failed to load Qwen3-1.7B model from all attempted paths")

        if hasattr(self.llm, 'to'):
            self.llm = self.llm.to(self.device)
        elif hasattr(self.llm, 'model') and hasattr(self.llm.model, 'to'):
            self.llm.model = self.llm.model.to(self.device)

        print(f"Loaded Qwen3-1.7B model from {model_path}")

        if not layers is None:
            if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'layers'):
                self.llm.model.layers = self.llm.model.layers[:layers]
            elif hasattr(self.llm, 'transformer') and hasattr(self.llm.transformer, 'h'):
                self.llm.transformer.h = self.llm.transformer.h[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:
            lora_config = LoRAConfig(
                    r=16,
                    target_modules=['q_proj','k_proj','v_proj','o_proj'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True)
            if hasattr(self.llm, 'model'):
                self.llm = self.llm.model

        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'norm' in name or 'ln' in name or 'layer_norm' in name:
                    param.requires_grad = True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', trust_remote_code=True)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen3-1.7B', trust_remote_code=True)

    def forward(self,x:torch.FloatTensor,attention_mask=None):
        if x.device != self.device:
            x = x.to(self.device)
        if attention_mask is not None and attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)
        
        outputs = self.llm(inputs_embeds=x,attention_mask=attention_mask,output_hidden_states=True)
        
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            out = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            out = outputs.last_hidden_state
        else:
            raise RuntimeError("Cannot extract hidden states from Qwen3-1.7B model output")

        return out

    def getembedding(self, x:torch.FloatTensor):
        if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'embed_tokens'):
            return self.llm.model.embed_tokens(x)
        elif hasattr(self.llm, 'transformer') and hasattr(self.llm.transformer, 'wte'):
            return self.llm.transformer.wte(x)
        elif hasattr(self.llm, 'embed_tokens'):
            return self.llm.embed_tokens(x)
        else:
            raise AttributeError("Cannot find embedding layer in Qwen3-1.7B model")
    
    def gettokenizer(self):
        return self.tokenizer 