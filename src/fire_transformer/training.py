from __future__ import annotations
import copy, math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .augmentation import augment_batch
from .evaluation import collect_probabilities
from .losses import BinaryFocalLoss

def build_loss(model_cfg,loss_cfg,positive_weight=None,device="cpu"):
    if model_cfg.get("loss","focal")=="bce":
        pw=None if positive_weight is None else torch.tensor([positive_weight],device=device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    return BinaryFocalLoss(loss_cfg.get("focal_gamma",2.0),loss_cfg.get("focal_alpha",0.75),loss_cfg.get("label_smoothing",0.05))
def _lr_lambda(epoch,warmup,max_epochs):
    if epoch<warmup: return max(1e-8,(epoch+1)/max(1,warmup))
    progress=(epoch-warmup)/max(1,max_epochs-warmup); return 0.5*(1+math.cos(math.pi*min(1.0,progress)))
def train_model(model,train_loader,val_loader,model_cfg,training_cfg,loss_cfg,device="cpu",positive_weight=None,use_augmentation=True):
    model=model.to(device); lr=float(model_cfg.get("lr",1e-3)); max_epochs=int(training_cfg.get("max_epochs",150)); patience=int(training_cfg.get("patience",15)); warmup=int(training_cfg.get("warmup_epochs",10))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=float(training_cfg.get("weight_decay",0.01))); scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:_lr_lambda(e,warmup,max_epochs)); loss_fn=build_loss(model_cfg,loss_cfg,positive_weight,device)
    best_state=None; best_auc=-np.inf; stale=0; history=[]
    for epoch in range(max_epochs):
        model.train(); losses=[]
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.float().to(device)
            if use_augmentation and model_cfg["type"]=="transformer": xb=augment_batch(xb,float(training_cfg.get("gaussian_noise_std",0.01)),float(training_cfg.get("time_warp_prob",0.15)))
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        scheduler.step(); yv,pv=collect_probabilities(model,val_loader,device); val_auc=roc_auc_score(yv,pv) if len(np.unique(yv))>1 else 0.5
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"val_auc":float(val_auc),"lr":opt.param_groups[0]["lr"]})
        if val_auc>best_auc+1e-8: best_auc=val_auc; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=patience: break
    if best_state is None: raise RuntimeError("Training failed to produce a checkpoint")
    model.load_state_dict(best_state); return model,history,float(best_auc)
