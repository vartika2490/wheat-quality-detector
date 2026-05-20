# 🌾 Wheat Grain Quality Detection System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://wheat-quality-detector-s5nc2pwh6pdek6dpmssdpp.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Model](https://img.shields.io/badge/Model-ResNet50-green?style=for-the-badge)](https://pytorch.org/vision/stable/models/resnet.html)

> **Live demo:** https://wheat-quality-detector-s5nc2pwh6pdek6dpmssdpp.streamlit.app/

An AI-powered web application that detects and classifies wheat grain quality from images in real time. Built as part of a data science internship at an agricultural trading company, this project solves a real business problem — screening wheat quality before purchase to reduce financial risk.

---

## Why I Built This

The company I work for acts as a middleman between wheat buyers and sellers. We offer buyers a one-month credit facility and take a 2% margin on each transaction. The problem was simple but costly: we had no reliable way to quickly assess the quality of wheat before approving credit. Manual inspection was slow, inconsistent, and human error was common.

So I built this system. Upload a grain image, get an instant quality grade, a physical description of what the model sees, and a clear business recommendation — accept, conditional, or reject. The whole thing runs in under two seconds.

---

## What It Does

- Classifies wheat grain into **6 quality categories** from a single image
- Returns a **confidence score** so you know how certain the model is
- Gives a **business recommendation** (Accept / Conditional / Reject)
- Generates a **downloadable PDF quality report** for record keeping
- Works on any device through a clean web interface

**The 6 categories:**

| Category | Grade | Decision |
|---|---|---|
| Healthy | Grade A — Premium Quality | ✅ Accept |
| Broken | Grade C — Broken Grain | ⚠️ Conditional |
| Black Tip | Grade C — Black Tip | ⚠️ Conditional |
| Shriveled | Grade C — Shriveled Grain | ⚠️ Conditional |
| Fungus Damage | Grade D — Fungus Damaged | ❌ Reject |
| Insect Damage | Grade D — Insect Damaged | ❌ Reject |

---

## How It Works — The Pipeline

### Stage 1 — Data Collection
The dataset is GrainSet, a professional scanner dataset of 175,000 wheat grain images captured under controlled lighting with a black background. Each image contains 4 panels — two real grain photos and two silhouette masks. The scanner setup ensures consistent image quality across the entire dataset, which is one reason the model performs so well.

### Stage 2 — Preprocessing
Before training, every image is cropped to keep only the left half — the actual grain photos — and discard the silhouette masks on the right. Feeding mask images to the model adds noise without any texture information, so removing them improves accuracy. This was done using PIL (Pillow) and resulted in 169,999 clean cropped images.

### Stage 3 — Data Labeling
I built a custom labeling tool using Python's tkinter library. The tool displays each grain image on screen and lets you press a number key to assign a label. It saves progress automatically every 20 images so you can stop and resume at any time. I labeled 200 images per class (1,400 total) by hand to create the ground truth dataset.

One thing I learned the hard way: the folder numbers in the original dataset (0, 1, 2, 3, 4, 5, 6) did not map to the class names I expected. I had to visually inspect sample images from each folder to figure out the correct mapping before retraining. Getting labels right is more important than having more data.

### Stage 4 — Model Architecture
I used **ResNet50 with transfer learning** rather than building a CNN from scratch. ResNet50 is a 50-layer deep neural network pretrained by Facebook on ImageNet — 1.2 million images across 1,000 categories. It already knows how to detect edges, textures, and shapes. I froze the base layers to preserve that knowledge, unfroze the final convolutional block (layer4) for fine-tuning, and replaced the classification head with a custom layer for our 6 wheat categories:

```
Dropout(0.4) → Linear(2048 → 512) → ReLU → Dropout(0.3) → Linear(512 → 6)
```

ResNet50 specifically was chosen because its residual connections solve the vanishing gradient problem, making it reliably trainable even at 50 layers deep.

### Stage 5 — Training
- **Framework:** PyTorch
- **Loss function:** CrossEntropyLoss (standard for multi-class classification)
- **Optimizer:** Adam with learning rate 0.0001
- **Scheduler:** StepLR — reduces learning rate by 10x every 5 epochs
- **Augmentation:** Random horizontal flip, vertical flip, rotation up to 15°
- **Split:** 70% train / 15% validation / 15% test
- **Epochs:** 20
- **Hardware:** CPU only

The model hit **100% validation accuracy by epoch 6** and maintained it through epoch 20. Test set accuracy came in at **99%** across all six classes.

### Stage 6 — Full Dataset Labeling
After training a working model on 1,400 images, I used it to generate labels for the remaining 169,999 images with confidence scores. Predictions above 80% confidence were trusted automatically. This gave us a fully labeled large dataset for potential future retraining without months of manual work.

### Stage 7 — Web Application
Built with **Streamlit** for the frontend and **ReportLab** for PDF generation. The app accepts any wheat grain image, runs it through the model, and returns the result within two seconds. The PDF report includes the grain classification, confidence score, physical description, and business recommendation.

---

## Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Healthy | 1.00 | 1.00 | 1.00 |
| Broken | 1.00 | 1.00 | 1.00 |
| Black Tip | 1.00 | 1.00 | 1.00 |
| Fungus Damage | 0.97 | 0.97 | 0.97 |
| Insect Damage | 1.00 | 1.00 | 1.00 |
| Shriveled | 1.00 | 0.97 | 0.98 |
| **Overall** | **0.99** | **0.99** | **0.99** |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| PyTorch | Deep learning framework |
| torchvision | Pretrained models + image transforms |
| ResNet50 | CNN architecture (50 layers, 23M parameters) |
| PIL / Pillow | Image reading, cropping, resizing |
| pandas | Label management and CSV handling |
| scikit-learn | Train/test split, evaluation metrics |
| matplotlib + seaborn | Training curves, confusion matrix |
| Streamlit | Web application frontend |
| ReportLab | PDF report generation |
| tkinter | Custom data labeling tool |

---

## Project Structure

```
wheat-detection/
├── scripts/
│   ├── 01_data_labeling.py       # Custom tkinter labeling tool
│   ├── 02_data_preprocessing.py  # Image cropping pipeline
│   ├── 03_model_training.py      # ResNet50 training script
│   ├── 04_web_application.py     # Flask app (local)
│   └── 05_streamlit_app.py       # Streamlit app (deployed)
├── models/
│   └── final_wheat_model.pth     # Trained model weights
├── reports/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── correct_labels.csv
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/vartika2490/wheat-detection-model
cd wheat-detection-model

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run scripts/05_streamlit_app.py
```

---

## Challenges I Ran Into

**Wrong label mapping.** The original dataset used folder numbers (0–6) instead of class names. I initially mapped them incorrectly, which caused the model to confuse broken grain with insect damage. I fixed this by writing a script that displayed sample images from each folder so I could visually verify the correct mapping before retraining. Getting labels right matters more than having more data.

**18GB dataset extraction.** Windows has a file path length limit of 260 characters. Many image filenames in the dataset were extremely long, causing extraction to fail silently. I wrote a custom extraction script that renames files with short names during extraction to bypass this limit.

**Training time on CPU.** Training on 6,000 images took significantly longer than expected on CPU. I managed this by reducing batch size and optimizing the data loading pipeline, bringing full training down to under 90 minutes.

**Library version conflicts.** NumPy 2.x broke PyTorch compatibility. Fixed by pinning NumPy to version 1.26.4.

---

## Business Impact

- **Faster decisions:** Quality grading that used to take hours now takes seconds
- **Consistent results:** No variation between inspectors or shift changes
- **Reduced credit risk:** Reject poor quality batches before approving buyer credit
- **Audit trail:** Every quality check generates a PDF report for documentation
- **Scalable:** The same model can process any volume without additional cost

---

## About

Built during a data science internship at an agricultural trading company. This was my first end-to-end machine learning project — from raw dataset to deployed web application.

**Dataset:** GrainSet — published on Figshare (175,000 professional scanner images)
**GitHub:** [vartika2490](https://github.com/vartika2490)
**Live Demo:** [wheat-quality-detector.streamlit.app](https://wheat-quality-detector-s5nc2pwh6pdek6dpmssdpp.streamlit.app/)
