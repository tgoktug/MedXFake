# ddpm_api.py
# FastAPI tabanlı DDPM arayüzü

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import os
import tempfile
import uuid
from typing import Tuple

from ddpm_interface import (
    generate_images,
    manipulate_image,
    detect_manipulated,
    detect_fake_synthesis,
)

# --------------------------------------------------------------------
# UYGULAMA
# --------------------------------------------------------------------
app = FastAPI(title="DDPM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static klasörü mevcut değilse oluştur
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")


# --------------------------------------------------------------------
# YARDIMCI FONKSİYON — UploadFile → rastgele geçici dosya yolu
# --------------------------------------------------------------------
def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".png"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(upload.file.read())
    return tmp_path


# --------------------------------------------------------------------
# 1) DDPM SENTEZ (IMAGE GENERATION)
# --------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root_page():
    """
    /templates/index.html dosyasını döndürür.
    """
    index_path = "templates/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html)
    else:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
@app.post("/synthesis")
async def api_synthesis(
    model_path: str = Form(...),
    num_images: int = Form(...),
):
    """
    DDPM ile sahte medikal görüntü üretir ve ZIP döndürür.
    ZIP dosyası random bir isimle static/ içine kaydedilir.
    """
    try:
        # rastgele zip adı oluştur
        zip_filename = f"gen_{uuid.uuid4().hex}.zip"
        zip_path = os.path.join("static", zip_filename)

        # üretim
        generate_images(
            model_path=model_path,
            num_images=num_images,
            output_zip=zip_path
        )

        return JSONResponse({
            "status": "ok",
            "zip_url": f"/static/{zip_filename}"
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)}, status_code=500
        )


# --------------------------------------------------------------------
# 2) MANIPULATION (inpainting)
# --------------------------------------------------------------------
@app.post("/manipulate")
async def api_manipulate(
    model_path: str = Form(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    image: UploadFile = File(...)
):
    """
    BBox + DDPM inpainting ile manipüle görüntü üretir.
    Çıktı random isimle static/ içine kaydedilir.
    """
    try:
        # uploaded image → random temp file
        orig_path = save_upload_to_temp(image)
        bbox: Tuple[int, int, int, int] = (x1, y1, x2, y2)

        # rastgele çıktı adı
        output_filename = f"manip_{uuid.uuid4().hex}.png"
        output_path = os.path.join("static", output_filename)

        # işlem
        final_path = manipulate_image(
            model_path=model_path,
            original_path=orig_path,
            bbox=bbox,
            output_path=output_path
        )

        return JSONResponse({
            "status": "ok",
            "output_path": f"/static/{output_filename}"
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)}, status_code=500
        )


# --------------------------------------------------------------------
# 3) MANIPULATION DETECTION (binary classifier)
# --------------------------------------------------------------------
@app.post("/detect/manipulated")
async def api_detect_manipulated(
    model_path: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Contrastive + CNN tabanlı manipülasyon tespiti.
    """
    try:
        img_path = save_upload_to_temp(image)

        result = detect_manipulated(
            model_path=model_path,
            image_path=img_path
        )

        return JSONResponse({
            "status": "ok",
            "result": result
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)}, status_code=500
        )


# --------------------------------------------------------------------
# 4) 8-Class FAKE SYNTHESIS DETECTION (ResNet-18)
# --------------------------------------------------------------------
@app.post("/detect/synthesis8")
async def api_detect_synthesis8(
    model_path: str = Form(...),
    image: UploadFile = File(...)
):
    """
    8 sınıflı: [brain_real, brain_fake, chest_real, chest_fake, kidney_real, ...]
    """
    try:
        img_path = save_upload_to_temp(image)

        result = detect_fake_synthesis(
            model_path=model_path,
            image_path=img_path
        )

        return JSONResponse({
            "status": "ok",
            "result": result
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)}, status_code=500
        )
