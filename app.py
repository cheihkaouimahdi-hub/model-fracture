import base64
import io
from typing import Literal

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from frontend import UI_HTML
from model_service import DEVICE, gradcam_pil, load_model, predict_pil

app = FastAPI(title="X-ray Classifier API")


@app.on_event("startup")
def startup_load_model():
    app.state.model = load_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/", response_class=HTMLResponse)
def interface():
    return HTMLResponse(content=UI_HTML)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    show_image: bool = False,
    show_cam: bool = False,
    cam_thr: float = Query(0.8, ge=0.0, le=1.0),
    cam_method: Literal["gradcam", "gradcampp"] = Query("gradcampp"),
    target_layer: Literal["denseblock4_last_conv", "denseblock4", "norm5"] = Query(
        "denseblock4_last_conv"
    ),
    hotspot_percentile: float = Query(92.0, ge=0.0, le=100.0),
    min_area: int = Query(35, ge=1, le=224 * 224),
    morph_kernel: int = Query(3, ge=1, le=9),
    tighten_iter: int = Query(1, ge=0, le=3),
):
    if file.content_type and not (
        file.content_type.startswith("image/")
        or file.content_type == "application/octet-stream"
    ):
        return JSONResponse(status_code=400, content={"error": "Upload an image file."})

    data = await file.read()
    if not data:
        return JSONResponse(status_code=400, content={"error": "Empty upload."})

    try:
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError):
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    result = predict_pil(app.state.model, img)

    if show_image:
        format_to_mime = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "BMP": "image/bmp",
            "GIF": "image/gif",
        }
        mime_type = (
            file.content_type
            if file.content_type and file.content_type.startswith("image/")
            else format_to_mime.get((img.format or "").upper(), "image/png")
        )
        image_b64 = base64.b64encode(data).decode("ascii")
        result["image_data_url"] = f"data:{mime_type};base64,{image_b64}"

    if show_cam and result.get("label") != "NORMAL":
        result.update(
            gradcam_pil(
                app.state.model,
                img,
                cam_thr=cam_thr,
                min_area=min_area,
                hotspot_percentile=hotspot_percentile,
                cam_method=cam_method,
                target_layer=target_layer,
                morph_kernel=morph_kernel,
                tighten_iter=tighten_iter,
            )
        )

    return result
