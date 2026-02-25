import base64
import io
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import densenet169

MODEL_PATH = os.getenv("MODEL_PATH", "best_densenet169_legfracture.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["NORMAL", "FRACTURE"]
THRESHOLD = 0.5

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


def _jet_colormap(cam: np.ndarray):
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def gradcam_pil(model, img: Image.Image, cam_thr: float = 0.55):
    target_layer = model.features.norm5
    activations = {}
    gradients = {}

    def _forward_hook(_, __, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    forward_handle = target_layer.register_forward_hook(_forward_hook)

    try:
        img_rgb = img.convert("RGB")
        input_224 = img_rgb.resize((224, 224))
        x = infer_tfms(img_rgb).unsqueeze(0).to(DEVICE)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, 0].backward()

        acts = activations.get("value")
        grads = gradients.get("value")
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM failed to capture intermediate tensors.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.detach().cpu().numpy()
    finally:
        forward_handle.remove()
        model.zero_grad(set_to_none=True)

    input_np = np.array(input_224, dtype=np.uint8)
    heatmap_np = (_jet_colormap(cam_np) * 255).astype(np.uint8)
    overlay_np = np.clip(0.6 * input_np + 0.4 * heatmap_np, 0, 255).astype(np.uint8)

    mask = cam_np >= cam_thr
    mask_area_pct = float(mask.mean() * 100.0)

    box = None
    overlay_box_img = Image.fromarray(overlay_np)
    if mask.any():
        ys, xs = np.where(mask)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        box = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        draw = ImageDraw.Draw(overlay_box_img)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)

    return {
        "cam_thr": cam_thr,
        "mask_area_pct": mask_area_pct,
        "box_224": box,
        "input_224_data_url": _to_data_url_png(input_224),
        "cam_heatmap_data_url": _to_data_url_png(Image.fromarray(heatmap_np)),
        "overlay_data_url": _to_data_url_png(Image.fromarray(overlay_np)),
        "overlay_box_data_url": _to_data_url_png(overlay_box_img),
    }
