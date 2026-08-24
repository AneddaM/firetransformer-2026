#!/usr/bin/env python3
from pathlib import Path
import tempfile
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fire_transformer.config import load_config
from fire_transformer.data import DatasetCatalog, build_fold
from fire_transformer.model import build_model


def main():
    with tempfile.TemporaryDirectory() as td:
        subprocess.check_call([sys.executable,str(ROOT/"scripts"/"make_synthetic_dataset.py"),"--output",td,"--samples","180"])
        cfg=load_config(ROOT/"configs"/"default.yaml")
        cat=DatasetCatalog(td,cfg["schema"])
        assert cat.nodes == [f"NODO{i}" for i in range(1,6)]
        train,val,test,_,features,train_a,val_a,test_a=build_fold(cat,"NODO5",cfg["rolling_window"],cfg["window"],cfg["horizon"],cfg["val_fraction_files"],1337)
        assert train.X.shape[-1] == 20
        assert set(a.node for a in test_a)=={"NODO5"}
        assert all(a.node!="NODO5" for a in train_a+val_a)
        model=build_model(cfg["models"]["ft64"],n_features=20,gas_feature_index=3)
        with torch.inference_mode():
            z=model(torch.from_numpy(train.X[:2]))
        assert z.shape==(2,)
        assert sum(p.numel() for p in model.parameters())==110338
        print("SMOKE TEST PASSED")

if __name__=="__main__": main()
