import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from PIL import Image, ImageFilter, ImageEnhance, ImageChops

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def leaf_mask_auto(rgb_u8: np.ndarray) -> np.ndarray:
    corners = np.stack([rgb_u8[0, 0], rgb_u8[0, -1], rgb_u8[-1, 0], rgb_u8[-1, -1]], axis=0).astype(np.float32)
    bg = corners.mean(axis=0)
    bg_luma = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    r, g, b = rgb_u8[..., 0], rgb_u8[..., 1], rgb_u8[..., 2]
    if bg_luma > 180:
        return (r < 240) | (g < 240) | (b < 240)
    return (r > 15) | (g > 15) | (b > 15)

def deltae2000_map(real_u8: np.ndarray, fake_u8: np.ndarray) -> np.ndarray:
    real = real_u8.astype(np.float32) / 255.0
    fake = fake_u8.astype(np.float32) / 255.0
    return deltaE_ciede2000(rgb2lab(real), rgb2lab(fake))

def topk_mean(de: np.ndarray, mask: np.ndarray, pct: float) -> float:
    v = de[mask].reshape(-1)
    if v.size == 0:
        return 0.0
    k = max(1, int(np.ceil(v.size * pct)))
    idx = np.argpartition(v, -k)[-k:]
    return float(v[idx].mean())

def topk_energy_fraction(de: np.ndarray, mask: np.ndarray, pct: float) -> float:
    v = de[mask].reshape(-1)
    if v.size == 0:
        return 0.0
    total = float(v.sum() + 1e-12)
    k = max(1, int(np.ceil(v.size * pct)))
    idx = np.argpartition(v, -k)[-k:]
    top = float(v[idx].sum())
    return float(top / total)

def hist_intersection_similarity(real_u8: np.ndarray, fake_u8: np.ndarray, bins: int = 256) -> float:
    sims = []
    for ch in range(3):
        hr, _ = np.histogram(real_u8[..., ch], bins=bins, range=(0, 256), density=False)
        hf, _ = np.histogram(fake_u8[..., ch], bins=bins, range=(0, 256), density=False)
        hr = hr.astype(np.float32)
        hf = hf.astype(np.float32)
        hr = hr / (hr.sum() + 1e-12)
        hf = hf / (hf.sum() + 1e-12)
        sims.append(np.minimum(hr, hf).sum())
    return float(np.mean(sims) * 100.0)


def heatmap_deltae_glow(de: np.ndarray,
                        cmap_name: str = "viridis",
                        p_low: float = 1.0,
                        p_high: float = 99.0,
                        gamma: float = 0.60,
                        glow_radius: float = 6.0,
                        glow_strength: float = 1.25) -> Image.Image:
    
    d = de.astype(np.float32)

    lo = np.percentile(d, p_low)
    hi = np.percentile(d, p_high)
    d = (d - lo) / (hi - lo + 1e-12)
    d = np.clip(d, 0.0, 1.0)

    d = np.power(d, gamma)

    cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(d)[..., :3] * 255.0).astype(np.uint8)
    base = Image.fromarray(rgb, mode="RGB")

    glow = base.filter(ImageFilter.GaussianBlur(radius=glow_radius))
    glow = ImageEnhance.Brightness(glow).enhance(glow_strength)
    out = ImageChops.screen(base, glow)

    out = ImageEnhance.Contrast(out).enhance(1.10)
    return out

_pix2pix = None


class Pix2PixTorchScript:
    def __init__(self, ts_path: str, image_size: int = 256):
        self.device = torch.device("cpu")
        self.image_size = image_size
        self.netG = torch.jit.load(ts_path, map_location="cpu")
        self.netG.eval()

    @torch.no_grad()
    def infer(self, pil_img: Image.Image) -> dict:
        B = pil_img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        A = B.convert("L").convert("RGB")  

        a = np.asarray(A).astype(np.float32) / 255.0
        a = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
        a = (a * 2.0) - 1.0

        fake = self.netG(a)
        fake = (fake + 1.0) / 2.0
        fake = torch.clamp(fake, 0.0, 1.0)[0].cpu().permute(1, 2, 0).numpy()
        fake_u8 = (fake * 255).astype(np.uint8)

        real_u8 = np.asarray(B).astype(np.uint8)
        mask = leaf_mask_auto(real_u8)
        de = deltae2000_map(real_u8, fake_u8)

        return {
            "input_gray": A,  
            "reconstructed": Image.fromarray(fake_u8),
            "deltae_heatmap": heatmap_deltae_glow(de),
            "metrics": {
                "Ciede2000_sum_mask": float(de[mask].sum()) if mask.any() else 0.0,
                "Top2pct_mean_deltaE2000": float(topk_mean(de, mask, pct=0.02)),
                "Top1pct_energy_fraction": float(topk_energy_fraction(de, mask, pct=0.01)),
                "Hist_similarity_rgb": float(hist_intersection_similarity(real_u8, fake_u8)),
            }
        }


def infer_pix2pix(pil_img: Image.Image, ts_path: str) -> dict:
    global _pix2pix
    if _pix2pix is None:
        _pix2pix = Pix2PixTorchScript(ts_path=ts_path, image_size=256)

    out = _pix2pix.infer(pil_img)

    
    return {
        "input_gray": out["input_gray"],
        "reconstructed": out["reconstructed"],
        "deltae_heatmap": out["deltae_heatmap"],
        "metrics": out["metrics"],
    }