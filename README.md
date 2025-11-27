# 🧬 MedXFake  
### **A New, Comprehensive, Realistic and Applicable Deep-Fake Synthesis, Manipulation and Detection System for Medical Images**

**Authors:**  
- **Mehmet Karakose**, Firat University, Elazig, Türkiye — *mkarakose@firat.edu.tr*  
- **T. Göktuğ Altundoğan**, Celal Bayar University, Manisa, Türkiye — *turan.altundogan@cbu.edu.tr*  
- **Mert Çeçen**, Firat University, Elazig, Türkiye — *mert.cecen23@gmail.com*

---

## 📄 Abstract

Accessing the outputs of medical imaging applications is costly and challenging due to patient rights and confidentiality. In this study, realistic medical deepfake images were generated using DDPM and GAN models with medical images collected from online platforms. A realistic medical image manipulation approach was implemented using a DDPM-based method. A large medical deepfake dataset was created using these models, and multiple detection strategies were developed.

An API integration capable of generating fake images was implemented using four different UNet-DDPM models (brain MRI, kidney CT, lung CT, breast ultrasound). Deep-fakes synthesized from scratch were detected using a ResNet-based classifier that achieved **99.78% F1-score**.

Manipulation detection is more challenging; therefore, the CNN classifier was fine-tuned using a contrastive learning approach, improving the F1-score from **89.90% → 99.74%**.

All synthesis, manipulation, and detection modules are integrated with an easy-to-use web interface.

---

## 🔑 Keywords

`Medical Imaging`, `Deep Fake Detection`, `Diffusion Models`, `DDPM`, `UNet`,  
`Inpainting`, `Contrastive Learning`

---

# 🚀 Overview

MedXFake provides a complete deep-fake framework for medical images:

✔ **DDPM-based synthesis** (brain, chest, kidney, lung)  
✔ **DDPM-based manipulation (inpainting)**  
✔ **8-class ResNet synthetic deepfake detection**  
✔ **Contrastive-learning manipulation detector**  
✔ **REST API**  
✔ **Local HTML interface** (index.html **direct file access**)  

---

# 📁 Repository Structure

```
project_root/
│
├── ddpm_api.py
├── ddpm_interface.py
├── ddpm_synthesis.py
├── ddpm_manipulation.py
├── contrastive_manip_detect.py
├── medical8_detect.py
│
├── templates/
│    └── index.html
│
├── static/
├── models/
└── README.md
```

---

# 🧬 1. DDPM Synthesis System

Custom UNet architecture with sinusoidal time embeddings (T=1000).

### API Endpoint:
```
POST /synthesis
```

---

# 🎭 2. DDPM Manipulation (Inpainting)

Semantic region editing using DDPM inpainting.

### API Endpoint:
```
POST /manipulate
```

---

# 🛡 3. Manipulation Detection (Contrastive Learning)

Binary classifier using contrastive encoder.

### API Endpoint:
```
POST /detect/manipulated
```

---

# 🧪 4. 8-Class Synthetic DeepFake Detection

ResNet18 classifier for:
```
brain_real, brain_fake, chest_real, chest_fake,
kidney_real, kidney_fake, lung_real, lung_fake
```

### API Endpoint:
```
POST /detect/synthesis8
```

---

# 🌐 5. Web Interface (IMPORTANT)

⚠ index.html **FastAPI tarafından serve edilmez.**  
**Dosyadan direkt açılmalıdır.**

---

# 📦 Installation

```
pip install -r requirements.txt
```

---

# 📥 Model Download Links

Place all models into:
```
models/
```

[From this link:](https://drive.google.com/drive/folders/1n6WZlOAS-KHumren2st5QnTzjLUut2nf?usp=sharing)

---

# 📝 Citation

```
@article{MedXFake2025,
  title={MedXFake: A New, Comprehensive, Realistic and Applicable Deep-Fake Synthesis, Manipulation and Detection System for Medical Images},
  author={Karakose, Mehmet and Altundoğan, T. Göktuğ and Çeçen, Mert},
  journal={SoftwareX},
  year={2025}
}
```

---

MIT License
