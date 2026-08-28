def transformer_macs(S,d,dff,layers,n_features=20):
    layer=4*S*d*d+2*S*d*dff+2*S*S*d
    overhead=n_features*d*(S-1)+S*d+2*d*d+3*d
    return layers*layer+overhead
