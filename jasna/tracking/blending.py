from __future__ import annotations

import torch
import torch.nn.functional as F

_KERNEL_CACHE: dict[tuple[str, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]] = {}

BLEND_DILATION_RATIO = 0.028
BLEND_FALLOFF_RATIO = 0.028


def _box_kernels(device: torch.device, dtype: torch.dtype, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached separable 1D box-blur kernels (horizontal, vertical).

    A K×K uniform kernel (value 1/K²) is the outer product of two 1D uniform
    kernels of value 1/K, so the 2D box blur factors into a horizontal pass
    (1×K) followed by a vertical pass (K×1).
    """
    cache_key = (str(device), dtype, kernel_size)
    kernels = _KERNEL_CACHE.get(cache_key)
    if kernels is None:
        kh = torch.ones((1, 1, 1, kernel_size), device=device, dtype=dtype) / kernel_size
        kv = torch.ones((1, 1, kernel_size, 1), device=device, dtype=dtype) / kernel_size
        kernels = (kh, kv)
        _KERNEL_CACHE[cache_key] = kernels
    return kernels


def _box_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # Separable box blur: O(2K) per pixel instead of O(K²) for the dense conv2d.
    # Pad once in 2D (reflect) then run two "valid" 1D convolutions, which is
    # numerically identical to padding + a single dense K×K conv2d.
    kh, kv = _box_kernels(x.device, x.dtype, kernel_size)
    pad = kernel_size // 2
    x4d = F.pad(x.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(F.conv2d(x4d, kh), kv)
    return blurred.squeeze(0).squeeze(0)


def _make_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def create_blend_mask(crop_mask: torch.Tensor, frame_height: int) -> torch.Tensor:
    """Create blend mask from detection mask with dilation and falloff.

    Dilation ensures blend weight=1.0 extends past the mask edge to cover
    any adjacent mosaic blocks the detector missed.  Falloff creates a
    smooth transition entirely outside the mosaic area.
    Both are proportional to frame height (~30px each at 1080p).
    """
    mask = crop_mask.squeeze()
    blend_dtype = mask.dtype if mask.is_floating_point() else torch.get_default_dtype()

    dilation_px = max(3, round(frame_height * BLEND_DILATION_RATIO))
    falloff_px = max(3, round(frame_height * BLEND_FALLOFF_RATIO))

    dilate_k = _make_odd(dilation_px * 2 + 1)
    falloff_k = _make_odd(falloff_px * 2 + 1)

    blend = (mask > 0).to(dtype=blend_dtype)
    blend = _box_blur(blend, dilate_k)
    blend = (blend > 0.01).to(dtype=blend_dtype)
    blend = _box_blur(blend, falloff_k)

    return blend.clamp_(0.0, 1.0)

