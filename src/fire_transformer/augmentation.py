import torch


def augment_batch(x, gaussian_noise_std=0.01, time_warp_prob=0.15):
    """Mild batch-time augmentation used only during training.

    Time warping is implemented as a conservative one-step local temporal roll on a
    random subset of batches; this preserves shape and avoids interpolation artifacts.
    """
    if gaussian_noise_std > 0:
        x = x + torch.randn_like(x) * gaussian_noise_std
    if time_warp_prob > 0 and torch.rand(()) < time_warp_prob:
        if x.shape[1] > 4:
            split = int(torch.randint(2, x.shape[1] - 1, (1,)).item())
            direction = 1 if torch.rand(()) > 0.5 else -1
            x = x.clone()
            x[:, split:] = torch.roll(x[:, split:], shifts=direction, dims=1)
    return x
