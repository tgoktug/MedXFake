# ddpm_interface.py

import torch
from zipfile import ZipFile
import os
from PIL import Image
import numpy as np

from ddpm_synthesis import UNetSynthesis, ddpm_sample, DEVICE, CHANNELS, IMG_SIZE
from ddpm_manipulation import UNetManip, ddpm_inpaint


# ---------------------------------------------------------
# 1) SYNTHESIS API —> RESİM ÜRET + ZIP OLARAK KAYDET
# ---------------------------------------------------------
def generate_images(model_path, num_images, output_zip="generated.zip"):

    model = UNetSynthesis(img_channels=CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    os.makedirs("tmp_gen", exist_ok=True)

    images = ddpm_sample(model, n_samples=num_images)

    file_paths = []
    for i in range(num_images):
        arr = images[i].numpy()[0] * 255
        img = Image.fromarray(arr.astype(np.uint8))
        fname = f"tmp_gen/img_{i}.png"
        img.save(fname)
        file_paths.append(fname)

    with ZipFile(output_zip, "w") as z:
        for f in file_paths:
            z.write(f)

    return os.path.abspath(output_zip)



# ---------------------------------------------------------
# 2) MANIPULATION API —> ORİJİNAL + MASKE → MANİPÜLE EDİLMİŞ GÖRSEL
# ---------------------------------------------------------
def manipulate_image(model_path, original_path, bbox, output_path="manipulated.png"):
    """
    bbox: (x1, y1, x2, y2)
    Orijinal görüntü 128×128'e resize edilir.
    Bu bounding box'a göre maske otomatik üretilir.
    """

    # 1. Modeli yükle
    model = UNetManip().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. Orijinal görüntüyü al
    orig = Image.open(original_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    orig_np = np.array(orig) / 255.0
    orig_t = torch.tensor(orig_np * 2 - 1).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 3. Bounding box → mask (1 = manipulate, 0 = koru)
    x1, y1, x2, y2 = bbox
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0

    mask_t = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 4. Inpainting
    out = ddpm_inpaint(model, orig_t, mask_t)[0,0]

    # 5. Kaydet
    out = Image.fromarray((out.numpy() * 255).astype(np.uint8))
    out.save(output_path)

    return os.path.abspath(output_path)

# ddpm_interface.py içine ekle

def detect_manipulated(model_path="./models/deepfake_classifier.pth", image_path="original-2.jpg"):
    from ddpm_manip_detection import detect_manipulated
    result = detect_manipulated(
        model_path=model_path,
        image_path=image_path
    )
    return result
def detect_fake_synthesis(model_path="./models/resnet_8class_best.pth", image_path="original-2.jpg"):
    from medical8_detect import detect_medical8

    result = detect_medical8(
        model_path=model_path,
        image_path=image_path
    )
    return result