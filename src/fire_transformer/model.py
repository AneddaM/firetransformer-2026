import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=512):
        super().__init__(); pe=torch.zeros(max_len,d_model); pos=torch.arange(max_len,dtype=torch.float32).unsqueeze(1); div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); self.register_buffer("pe",pe.unsqueeze(0),persistent=False)
    def forward(self,x): return x+self.pe[:,:x.size(1)]
class AttentionPool(nn.Module):
    def __init__(self,d_model): super().__init__(); self.score=nn.Linear(d_model,1,bias=True)
    def forward(self,h):
        a=torch.softmax(self.score(h).squeeze(-1),dim=1); return torch.sum(h*a.unsqueeze(-1),dim=1),a
class FireTransformer(nn.Module):
    def __init__(self,n_features=20,d_model=64,n_heads=4,n_layers=3,d_ff=128,dropout=0.1,gas_feature_index=3):
        super().__init__(); self.input_proj=nn.Linear(n_features,d_model)
        if gas_feature_index is not None and 0<=gas_feature_index<n_features:
            with torch.no_grad(): self.input_proj.weight[:,gas_feature_index].mul_(2.0)
        self.cls=nn.Parameter(torch.zeros(1,1,d_model)); nn.init.normal_(self.cls,std=0.02); self.pos=SinusoidalPositionalEncoding(d_model)
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,dim_feedforward=d_ff,dropout=dropout,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=n_layers,norm=nn.LayerNorm(d_model)); self.pool=AttentionPool(d_model)
        self.classifier=nn.Sequential(nn.Linear(2*d_model,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,1))
    def forward(self,x,return_attention=False):
        h=self.input_proj(x); h=torch.cat([self.cls.expand(x.size(0),-1,-1),h],dim=1); h=self.encoder(self.pos(h)); hc=h[:,0]; pooled,a=self.pool(h[:,1:]); logits=self.classifier(torch.cat([hc,pooled],dim=-1)).squeeze(-1); return (logits,a) if return_attention else logits
class BiLSTMClassifier(nn.Module):
    def __init__(self,n_features=20,hidden_size=96,num_layers=2,dropout=0.1):
        super().__init__(); self.lstm=nn.LSTM(n_features,hidden_size,num_layers=num_layers,dropout=dropout if num_layers>1 else 0,bidirectional=True,batch_first=True); self.classifier=nn.Linear(hidden_size*2,1)
    def forward(self,x): h,_=self.lstm(x); return self.classifier(h[:,-1]).squeeze(-1)
def build_model(model_cfg,n_features=20,gas_feature_index=3):
    if model_cfg["type"]=="transformer": return FireTransformer(n_features,model_cfg["d_model"],model_cfg["n_heads"],model_cfg["n_layers"],model_cfg["d_ff"],model_cfg.get("dropout",0.1),gas_feature_index)
    if model_cfg["type"]=="bilstm": return BiLSTMClassifier(n_features,model_cfg.get("hidden_size",96),model_cfg.get("num_layers",2),model_cfg.get("dropout",0.1))
    raise ValueError(model_cfg["type"])
