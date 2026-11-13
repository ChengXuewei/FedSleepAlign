"""The seven augmentations of the training windows. Every epoch of an augmented copy has one randomly chosen channel
transformed by two augmentations drawn without replacement; the other channels stay unchanged."""
import torch
import torch.nn.functional as F

AUG_TYPES = ["noise", "shift", "scale", "filter", "warp", "freq", "crop"]


def _augment_single_channel(x):
    """Two random augmentations of one channel, x: (1, 1, L)."""
    device, orig_len = x.device, x.shape[-1]
    augmented = x.clone()
    for aug_idx in torch.randperm(len(AUG_TYPES))[:2]:
        aug = AUG_TYPES[aug_idx]
        if aug == "noise":                               # Gaussian noise, SD 0.01 of the channel
            augmented = augmented + torch.randn_like(augmented) * 0.01 * augmented.std()
        elif aug == "shift":                             # up to 0.5 s, padded with the boundary value
            shift = torch.randint(-50, 51, (1,)).item()
            augmented = torch.roll(augmented, shifts=shift, dims=-1)
            if shift > 0:
                augmented[..., :shift] = augmented[..., shift:shift + 1].expand_as(augmented[..., :shift])
            elif shift < 0 and -shift < orig_len:
                s = -shift
                augmented[..., -s:] = augmented[..., -s - 1:-s].expand_as(augmented[..., -s:])
        elif aug == "scale":                             # amplitude factor in [0.9, 1.1]
            augmented = augmented * torch.empty(1, device=device).uniform_(0.9, 1.1)
        elif aug == "filter":                            # random 5-tap kernel with unit l1 norm
            weights = torch.rand(5, device=device) - 0.5
            weights = weights / weights.abs().sum()
            augmented = F.conv1d(augmented, weights.view(1, 1, -1), padding="same")
        elif aug == "warp":                              # resampled to a length factor in [0.9, 1.1] and back
            new_len = max(2, int(orig_len * torch.empty(1).uniform_(0.9, 1.1).item()))
            augmented = F.interpolate(augmented, size=new_len, mode="linear", align_corners=True)
            augmented = F.interpolate(augmented, size=orig_len, mode="linear", align_corners=True)
        elif aug == "freq":                              # phase of every Fourier coefficient rotated by <= 0.1 rad
            fft_signal = torch.fft.fft(augmented)
            phase_shift = torch.empty_like(fft_signal.real).uniform_(-0.1, 0.1)
            augmented = torch.fft.ifft(fft_signal * torch.exp(1j * phase_shift)).real
        elif aug == "crop":                              # 90-100% of the epoch, resampled to the full length
            crop_size = max(2, int(orig_len * torch.empty(1).uniform_(0.9, 1.0).item()))
            max_start = orig_len - crop_size
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            augmented = F.interpolate(augmented[..., start:start + crop_size], size=orig_len, mode="linear",
                                      align_corners=True)
    return augmented


def augment(signal):
    """signal: (B, 3, L) epochs; each epoch gets one randomly chosen channel augmented."""
    out = signal.clone()
    for b in range(out.shape[0]):
        ch = torch.randint(0, out.shape[1], (1,)).item()
        out[b, ch] = _augment_single_channel(out[b, ch].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out
