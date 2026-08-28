import torch


def augment_batch(x, gaussian_noise_std=0.01, time_warp_prob=0.15):
    if gaussian_noise_std > 0:
        x = x + gaussian_noise_std * torch.randn_like(x)
    if time_warp_prob > 0 and x.size(1) > 2:
        mask = torch.rand(x.size(0), device=x.device) < time_warp_prob
        if mask.any():
            shift = torch.where(torch.rand(mask.sum(), device=x.device) < 0.5, -1, 1)
            warped = x[mask].clone()
            for i, s in enumerate(shift.tolist()):
                warped[i] = torch.roll(warped[i], shifts=s, dims=0)
            x = x.clone(); x[mask] = warped
    return x
