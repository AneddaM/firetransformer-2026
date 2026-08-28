from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def choose_threshold(y,p,objective="f1"):
    best_t,best_s=0.5,-np.inf
    for t in np.linspace(0.01,0.99,99):
        pred=(p>=t).astype(int)
        score=f1_score(y,pred,zero_division=0) if objective=="f1" else recall_score(y,pred,zero_division=0)
        if score>best_s: best_s,best_t=score,float(t)
    return best_t,float(best_s)

@torch.inference_mode()
def collect_probabilities(model,loader,device="cpu"):
    model.eval(); ys=[]; ps=[]
    for xb,yb in loader:
        ps.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(ys),np.concatenate(ps)

def _normalize_event_key(event):
    f,o=event; return (str(f),round(float(o),6))
def _normalize_events(events):
    return None if events is None else tuple(dict.fromkeys(_normalize_event_key(e) for e in events))
def _event_groups(onset_ts,file_ids=None):
    onset_ts=np.asarray(onset_ts,dtype=float); n=len(onset_ts)
    file_ids=np.full(n,"__single_acquisition__",dtype=object) if file_ids is None else np.asarray(file_ids,dtype=object)
    groups={}
    for i in np.flatnonzero(np.isfinite(onset_ts)):
        key=(str(file_ids[i]),round(float(onset_ts[i]),6)); groups.setdefault(key,[]).append(int(i))
    return groups

def event_coverage(pred,onset_ts,file_ids=None,evaluable_events=None):
    groups=_event_groups(onset_ts,file_ids); expected=_normalize_events(evaluable_events) or tuple(groups.keys())
    if not expected: return np.nan
    pred=np.asarray(pred); covered=0
    for event in expected:
        idx=np.asarray(groups.get(event,[]),dtype=int)
        if idx.size and np.any(pred[idx]==1): covered+=1
    return covered/len(expected)

def lead_time_stats(pred,y,window_end_ts,onset_ts,file_ids=None):
    pred=np.asarray(pred); y=np.asarray(y); end=np.asarray(window_end_ts,dtype=float); onset=np.asarray(onset_ts,dtype=float)
    leads=[]
    for idxs in _event_groups(onset,file_ids).values():
        idx=np.asarray(idxs,dtype=int); correct=idx[(pred[idx]==1)&(y[idx]==1)]
        if correct.size:
            leads.append(float(onset[correct[0]])-float(np.min(end[correct])))
    if not leads: return {"lead_mean_s":np.nan,"lead_median_s":np.nan,"lead_p25_s":np.nan,"lead_p75_s":np.nan}
    a=np.asarray(leads,float); return {"lead_mean_s":float(a.mean()),"lead_median_s":float(np.median(a)),"lead_p25_s":float(np.percentile(a,25)),"lead_p75_s":float(np.percentile(a,75))}

def compute_metrics(y,p,threshold,window_end_ts=None,onset_ts=None,file_ids=None,physical_events=None,evaluable_events=None):
    y=np.asarray(y,dtype=int); p=np.asarray(p,float); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    out={
        "threshold":float(threshold),"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"auc_roc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,"auc_pr":float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan,"specificity":float(tn/(tn+fp)) if tn+fp else np.nan,"false_alarm_rate":float(fp/(tn+fp)) if tn+fp else np.nan,"missed_detection_rate":float(fn/(fn+tp)) if fn+tp else np.nan,"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)
    }
    phys=_normalize_events(physical_events) or (); evals=_normalize_events(evaluable_events) or ()
    out["events_physical_total"]=len(phys); out["events_evaluable_total"]=len(evals); out["events_total"]=len(evals)
    if onset_ts is not None:
        groups=_event_groups(onset_ts,file_ids); covered=0
        for e in evals:
            idx=np.asarray(groups.get(e,[]),dtype=int)
            if idx.size and np.any(pred[idx]==1): covered+=1
        out["events_covered"]=covered; out["coverage"]=covered/len(evals) if evals else np.nan
        if window_end_ts is not None: out.update(lead_time_stats(pred,y,window_end_ts,onset_ts,file_ids))
    return out
