import base64
import io
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
from torchvision.models import densenet169

MODEL_PATH = os.getenv("MODEL_PATH", "best_densenet169_legfracture.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["NORMAL", "FRACTURE"]
THRESHOLD = 0.5
CAM_LAYER_ALIASES = {
    "denseblock4_last_conv": "features.denseblock4.denselayer32.conv2",
    "denseblock4": "features.denseblock4",
    "norm5": "features.norm5",
}
DEFAULT_CAM_METHOD = os.getenv("CAM_METHOD", "gradcampp").lower()
DEFAULT_CAM_TARGET_LAYER = os.getenv("CAM_TARGET_LAYER", "denseblock4_last_conv").lower()

infer_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_model():
    model = densenet169(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def load_model():
    model = build_model().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def predict_pil(model, img: Image.Image):
    img = img.convert("RGB")
    x = infer_tfms(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    prob_pos = torch.sigmoid(logits)[0, 0].item()
    pred = 1 if prob_pos >= THRESHOLD else 0

    return {
        "label": CLASSES[pred],
        "probability": prob_pos,
    }


def _to_data_url_png(img: Image.Image):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{image_b64}"


def _resolve_module_by_path(root: nn.Module, module_path: str) -> nn.Module:
    module = root
    for token in module_path.split("."):
        if token.isdigit():
            module = module[int(token)]
        else:
            if not hasattr(module, token):
                raise ValueError(f"Unknown target layer: {module_path}")
            module = getattr(module, token)
    if not isinstance(module, nn.Module):
        raise ValueError(f"Target layer is not a torch module: {module_path}")
    return module


def _get_target_layer(model: nn.Module, target_layer: str) -> Tuple[nn.Module, str]:
    layer_key = (target_layer or DEFAULT_CAM_TARGET_LAYER).strip().lower()
    resolved_path = CAM_LAYER_ALIASES.get(layer_key, target_layer)
    if resolved_path is None:
        resolved_path = CAM_LAYER_ALIASES[DEFAULT_CAM_TARGET_LAYER]
    return _resolve_module_by_path(model, resolved_path), resolved_path


def _jet_colormap(cam: np.ndarray):
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def _gradcampp_weights(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    grads2 = grads.pow(2)
    grads3 = grads.pow(3)
    denom = 2.0 * grads2 + (acts * grads3).sum(dim=(2, 3), keepdim=True)
    denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
    alpha = grads2 / (denom + 1e-8)
    positive_grads = torch.relu(grads)
    return (alpha * positive_grads).sum(dim=(2, 3), keepdim=True)


def _compute_cam_map(
    model: nn.Module,
    x: torch.Tensor,
    target_layer: nn.Module,
    cam_method: str = "gradcampp",
) -> np.ndarray:
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def _forward_hook(_, __, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        activations["value"] = out

        def _capture_grad(grad):
            gradients["value"] = grad

        out.register_hook(_capture_grad)

    handle = target_layer.register_forward_hook(_forward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[:, 0].sum().backward()

        acts = activations.get("value")
        grads = gradients.get("value")
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM failed to capture intermediate tensors.")

        method = (cam_method or DEFAULT_CAM_METHOD).lower()
        if method == "gradcampp":
            weights = _gradcampp_weights(acts, grads)
        elif method == "gradcam":
            weights = grads.mean(dim=(2, 3), keepdim=True)
        else:
            raise ValueError("cam_method must be 'gradcam' or 'gradcampp'")

        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam[0].detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def _clean_mask(mask: np.ndarray, morph_kernel: int, tighten_iter: int) -> np.ndarray:
    if morph_kernel < 1:
        morph_kernel = 1
    if morph_kernel % 2 == 0:
        morph_kernel += 1

    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if morph_kernel >= 3:
        # Opening: remove small noisy spots before hotspot selection.
        img = img.filter(ImageFilter.MinFilter(morph_kernel))
        img = img.filter(ImageFilter.MaxFilter(morph_kernel))

    for _ in range(max(0, int(tighten_iter))):
        # Tighten mask with one-pixel erosion per iteration.
        img = img.filter(ImageFilter.MinFilter(3))

    return np.array(img, dtype=np.uint8) > 0


def _extract_components(mask: np.ndarray, cam_np: np.ndarray) -> List[Dict[str, object]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[Dict[str, object]] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            q = deque([(y, x)])
            visited[y, x] = True
            ys: List[int] = []
            xs: List[int] = []

            while q:
                cy, cx = q.pop()
                ys.append(cy)
                xs.append(cx)

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        q.append((ny, nx))

            ys_np = np.asarray(ys, dtype=np.int32)
            xs_np = np.asarray(xs, dtype=np.int32)
            values = cam_np[ys_np, xs_np]
            components.append(
                {
                    "ys": ys_np,
                    "xs": xs_np,
                    "area": int(values.size),
                    "peak": float(values.max() if values.size else 0.0),
                    "mean": float(values.mean() if values.size else 0.0),
                    "x0": int(xs_np.min() if xs_np.size else 0),
                    "y0": int(ys_np.min() if ys_np.size else 0),
                    "x1": int(xs_np.max() if xs_np.size else 0),
                    "y1": int(ys_np.max() if ys_np.size else 0),
                }
            )

    return components


def _select_hotspot(
    cam_np: np.ndarray,
    cam_thr: float,
    hotspot_percentile: float,
    min_area: int,
    morph_kernel: int,
    tighten_iter: int,
) -> Tuple[np.ndarray, Optional[List[int]], float, float]:
    percentile_thr = float(np.percentile(cam_np, np.clip(hotspot_percentile, 0.0, 100.0)))
    thr_used = float(max(cam_thr, percentile_thr))
    raw_mask = cam_np >= thr_used
    cleaned = _clean_mask(raw_mask, morph_kernel=morph_kernel, tighten_iter=tighten_iter)

    components = _extract_components(cleaned, cam_np)
    if not components:
        return np.zeros_like(cleaned, dtype=bool), None, thr_used, percentile_thr

    valid = [c for c in components if c["area"] >= int(min_area)]
    candidates = valid if valid else components
    best = max(candidates, key=lambda c: (c["peak"], c["mean"], -c["area"]))

    hotspot_mask = np.zeros_like(cleaned, dtype=bool)
    hotspot_mask[best["ys"], best["xs"]] = True
    box = [
        int(best["x0"]),
        int(best["y0"]),
        int(best["x1"] - best["x0"] + 1),
        int(best["y1"] - best["y0"] + 1),
    ]
    return hotspot_mask, box, thr_used, percentile_thr


def _overlay_with_focus(
    input_np: np.ndarray,
    heatmap_np: np.ndarray,
    cam_np: np.ndarray,
    thr_used: float,
    hotspot_mask: np.ndarray,
) -> np.ndarray:
    input_f = input_np.astype(np.float32)
    heatmap_f = heatmap_np.astype(np.float32)
    alpha = np.clip((cam_np - thr_used) / (1.0 - thr_used + 1e-8), 0.0, 1.0)
    alpha = 0.15 + 0.65 * alpha
    overlay = np.clip(
        input_f * (1.0 - alpha[..., None]) + heatmap_f * alpha[..., None], 0, 255
    ).astype(np.uint8)

    if hotspot_mask.any():
        dimmed = overlay.astype(np.float32)
        dimmed[~hotspot_mask] *= 0.65
        overlay = np.clip(dimmed, 0, 255).astype(np.uint8)
    return overlay


def _mask_to_data_url(mask: np.ndarray) -> str:
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return _to_data_url_png(mask_img)


def gradcam_pil(
    model,
    img: Image.Image,
    cam_thr: float = 0.8,
    min_area: int = 35,
    hotspot_percentile: float = 92.0,
    cam_method: str = DEFAULT_CAM_METHOD,
    target_layer: str = DEFAULT_CAM_TARGET_LAYER,
    morph_kernel: int = 3,
    tighten_iter: int = 1,
):
    layer_module, resolved_layer = _get_target_layer(model, target_layer)

    img_rgb = img.convert("RGB")
    input_224 = img_rgb.resize((224, 224))
    x = infer_tfms(img_rgb).unsqueeze(0).to(DEVICE)
    cam_np = _compute_cam_map(
        model=model, x=x, target_layer=layer_module, cam_method=cam_method
    )

    hotspot_mask, box, thr_used, percentile_thr = _select_hotspot(
        cam_np=cam_np,
        cam_thr=float(cam_thr),
        hotspot_percentile=float(hotspot_percentile),
        min_area=int(min_area),
        morph_kernel=int(morph_kernel),
        tighten_iter=int(tighten_iter),
    )

    input_np = np.array(input_224, dtype=np.uint8)
    heatmap_np = (_jet_colormap(cam_np) * 255).astype(np.uint8)
    overlay_np = _overlay_with_focus(input_np, heatmap_np, cam_np, thr_used, hotspot_mask)

    overlay_box_img = Image.fromarray(overlay_np)
    if box is not None:
        x0, y0, w, h = box
        draw = ImageDraw.Draw(overlay_box_img)
        draw.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], outline=(255, 240, 0), width=2)

    mask_area_pct = float(hotspot_mask.mean() * 100.0)

    return {
        "cam_method": (cam_method or DEFAULT_CAM_METHOD).lower(),
        "target_layer": resolved_layer,
        "cam_thr": float(cam_thr),
        "thr_used": float(thr_used),
        "percentile_thr": float(percentile_thr),
        "hotspot_percentile": float(hotspot_percentile),
        "min_area": int(min_area),
        "morph_kernel": int(morph_kernel),
        "tighten_iter": int(tighten_iter),
        "mask_area_pct": mask_area_pct,
        "box_224": box,
        "input_224_data_url": _to_data_url_png(input_224),
        "cam_heatmap_data_url": _to_data_url_png(Image.fromarray(heatmap_np)),
        "overlay_data_url": _to_data_url_png(Image.fromarray(overlay_np)),
        "overlay_box_data_url": _to_data_url_png(overlay_box_img),
        "mask_data_url": _mask_to_data_url(hotspot_mask),
    }
