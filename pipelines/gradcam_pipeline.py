import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms

def _resize_to_match(rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    pil = Image.fromarray(rgb)
    pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(pil)


def _overlay(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:

    H, W = base_rgb.shape[:2]
    if heat_rgb.shape[:2] != (H, W):
        heat_rgb = np.array(
            Image.fromarray(heat_rgb).resize((W, H), resample=Image.BILINEAR)
        )

    out = base_rgb.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


_predictor = None

class GradCamPredictor:
    def __init__(self, weights_path: str, image_size: int = 224):
        self.device = torch.device("cpu")
        self.image_size = image_size

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        ckpt = torch.load(weights_path, map_location="cpu")

        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.target_layer = self.model.layer2[-1]
        self._acts = None
        self._grads = None

        def fwd_hook(_, __, output):
            self._acts = output

        def bwd_hook(_, grad_in, grad_out):
            self._grads = grad_out[0]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

        self.tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        x = self.tf(pil_img.convert("RGB"))
        return x.unsqueeze(0)

    def infer(self, pil_img: Image.Image) -> dict:
        x = self._preprocess(pil_img)
        x.requires_grad_(True)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)

        cls = int(torch.argmax(probs, dim=1).item())
        score = logits[0, cls]

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        acts = self._acts[0]              
        grads = self._grads[0]            
        w = grads.mean(dim=(1, 2))        
        cam = (w[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-12)

        cam_np = cam.detach().cpu().numpy()
        cam_u8 = (cam_np * 255).astype(np.uint8)
        heat_gray = Image.fromarray(cam_u8).resize((self.image_size, self.image_size), Image.BILINEAR)
        heat_rgb = Image.merge("RGB", (heat_gray, heat_gray, heat_gray))

        base = pil_img.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        base_np = np.array(base).astype(np.float32)
        heat_np = np.array(heat_rgb).astype(np.float32)
        overlay = (0.55 * base_np + 0.45 * heat_np)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return {
            "img1": Image.fromarray(overlay),
            "img2": heat_rgb,
            "metrics": {
                "pred_class": cls,
                "prob_class0": float(probs[0, 0].item()),
                "prob_class1": float(probs[0, 1].item()),
            }
        }

def infer_gradcam(pil_img: Image.Image, weights_path: str) -> dict:
    global _predictor
    if _predictor is None:
        _predictor = GradCamPredictor(weights_path=weights_path, image_size=224)
    return _predictor.infer(pil_img)




