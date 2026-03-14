

    
```python


from essentials import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class STthreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        binary_output = (x > threshold).float()
        ctx.save_for_backward(x)
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_threshold = None
        return grad_x, grad_threshold

apply_ste = STthreshold.apply

class MicroFilterGate(nn.Module):
    def __init__(self, dims, mem=64, thresh=0.5):
        super().__init__()
        self.mkey = nn.Parameter(torch.randn(mem, dims))
        self.mval = nn.Parameter(torch.randn(mem, 1))
        self.mlp = nn.Sequential(nn.Linear(dims, dims // 2), nn.SiLU(), nn.Linear(dims // 2, 1))
        
        self.threshold = nn.Parameter(torch.tensor(thresh, dtype=dtype), requires_grad=False)
        self.concat = nn.Linear(2, 1, device=device, dtype=dtype)

    def forward(self, x):
        key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(self.mkey, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        x_val = self.concat(torch.cat((torch.matmul(key, self.mval), self.mlp(x)), dim=-1))
        
        survival_mask = apply_ste(x_val, self.threshold)
        return survival_mask, x_val

    def update_threshold(self, loss, current_loss_ema, lr=0.01):
        """Dynamic regulation. Sparsify if learning is stable, Densify if loss spikes."""
        if loss > current_loss_ema:
            self.threshold.sub_(lr)
        else:
            self.threshold.add_(lr)
        self.threshold.data = torch.clamp(self.threshold.data, 0.05, 0.95)

class MiniConnection(nn.Module):
    def __init__(self, dims, expand=2):
        super().__init__()
        self.dims = dims
        self.expand = expand
        self.parallel = nn.ModuleList([nn.Linear(dims, dims) for _ in range(expand)])
        self.network = nn.Linear(dims, expand)
        self.relu = nn.ReLU()

    def forward(self, input_features):
        features = [pathway(input_features) for pathway in self.parallel]
        weights = torch.softmax(self.network(input_features), dim=-1)
        weighted_combined = sum(w * f for w, f in zip(weights.unbind(dim=-1), features))
        return self.relu(weighted_combined)

class MacroPolicyNet(nn.Module):
    def __init__(self, dims, max_jump=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims, 128),
            nn.SiLU(),
            nn.Linear(128, max_jump + 1)
        )
        
    def forward(self, pooled_features):
        return F.softmax(self.net(pooled_features), dim=-1)

class HybridMyelinatedBlock(nn.Module):
    def __init__(self, dims, num_layers, mini_hc=True, hc_expansion_rate=2):
        super().__init__()
        self.num_layers = num_layers
        self.dims = dims
        
        self.work_mem = nn.Parameter(torch.zeros(1, 1, dims), requires_grad=True)
        self.mem_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]), requires_grad=True)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_dict = {
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'micro_filter': MicroFilterGate(dims, mem=64, thresh=0.3),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None,
            }
            if mini_hc:
                layer_dict['mini_moe'] = MiniConnection(dims, expand=hc_expansion_rate)
            else:
                layer_dict['mini_moe'] = None

            self.layers.append(nn.ModuleDict(layer_dict))

        self.policy_net = MacroPolicyNet(dims, max_jump=2)

        self.mlp_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(
            nn.Linear(dims, dims * 4), 
            nn.SiLU(), 
            nn.Linear(dims * 4, dims)
        )
        self.mlp_ln = nn.LayerNorm(dims)

    def forward(self, x): 
        batch, ctx = x.shape[:2]
        original_x = x
        
        work_mem = self.work_mem.expand(batch, -1, -1)
        
        pooled_representation = x.mean(dim=1)
        policy = self.policy_net(pooled_representation)
        
        history = []
        i = 0
        while i < self.num_layers:
            layer = self.layers[i]
            
            survival_mask, survival_logits = layer['micro_filter'](x)
            mask_layer = survival_mask.expand(-1, ctx, self.dims)
            
            px = layer['ln'](x)  

            if layer['adapter'] is not None:
                adapted_px = layer['adapter'](px)
            else:
                adapted_px = px

            if layer['mini_moe'] is not None:
                layer_out = layer['mini_moe'](adapted_px)
            else:
                layer_out = adapted_px
                
            gate_val = layer['gate'](px)
            x = x + gate_val * (layer_out * mask_layer)

            mem = x.mean(dim=1, keepdim=True)
            mem_val = self.mem_gate(mem)
            work_mem = mem_val * work_mem + (1 - mem_val) * mem
            
            survival_rate = survival_mask.mean()
            
            if survival_rate < 0.1 and i < self.num_layers - 1:
                action = 1
            elif i < self.num_layers - 1:
                action = torch.multinomial(policy, 1).squeeze(-1).item()
            else:
                action = 0
                
            if action > 0:
                jump_distance = action
                i_next = min(i + jump_distance + 1, self.num_layers)
                jump_weight = self.jump_weights[min(jump_distance-1, 2)]               
                
                x = x + jump_weight * original_x + (1-jump_weight) * work_mem.expand(-1, ctx, -1)
                
                i = i_next
                history.append({'layer': i, 'status': 'jumped_to'})
            else:
                i += 1
                history.append({'layer': i, 'status': 'processed'})
                
        mlp_gate = self.mlp_gate(x)
        mlp_output = self.mlp(self.mlp_ln(x))
        x = x + mlp_gate * mlp_output
        
        return x, {'jump_history': history}


```
