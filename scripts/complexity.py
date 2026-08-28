#!/usr/bin/env python3
MODELS = {
    "FT-32":  dict(S=61,d=32,dff=64,L=2,params=20002),
    "FT-64":  dict(S=61,d=64,dff=128,L=3,params=110338),
    "FT-128": dict(S=61,d=128,dff=256,L=3,params=433666),
}
def macs(c):
    S,d,dff,L = c["S"],c["d"],c["dff"],c["L"]
    layer = 4*S*d*d + 2*S*d*dff + 2*S*S*d
    overhead = 20*d*S + S*d + 2*d*d + 3*d  # projection + rough pooling/classifier
    return L*layer + overhead
for name,c in MODELS.items():
    m=macs(c)
    print(f"{name:6s} params={c['params']:7d} FP32={c['params']*4/1024:.1f} KiB MAC={m/1e6:.2f} M")
