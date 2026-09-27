
# ORAL-SENSE EDGE 🦷

### Privacy-First, Explainable & Offline Oral Lesion Screening using MobileNetV3

ORAL-SENSE EDGE is a lightweight Edge AI prototype designed to assist with
oral lesion screening using smartphone/webcam images.

The system uses **MobileNetV3** for image classification and **Grad-CAM**
for visual explainability. The prototype is designed around local/offline
processing to improve privacy.

> ⚠️ ORAL-SENSE EDGE is a screening/research prototype and does NOT provide
> a medical diagnosis.

---

## 🚀 Key Features

- 📷 Webcam image capture
- 🖼️ Upload existing oral images
- 🔍 Image quality checking
- 🤖 MobileNetV3-based classification
- 🧠 AI model confidence/probability
- 🔥 Grad-CAM explainability heatmap
- 📊 Screening result dashboard
- 🔄 Visual change monitoring between images
- 🩺 Referral guidance
- 📜 Local screening history
- 📈 Screening analytics
- 📄 Professional PDF report generation
- 🔐 Privacy Center
- ☁️ No automatic cloud upload
- 🌐 Local/offline processing
- 🎨 Futuristic dark-mode dashboard

---

## 🧠 System Architecture

```text
Camera / Image Upload
        ↓
Image Quality Check
        ↓
MobileNetV3
        ↓
Classification
        ↓
Model Probability
        ↓
Grad-CAM Explainability
        ↓
Screening Guidance
        ↓
History / Analytics / PDF Report
````

---

## 🤖 AI Model

The project uses:

**Model:** MobileNetV3 Small
**Framework:** PyTorch
**Input Size:** 224 × 224 pixels

### Classes

* `benign_lesions`
* `malignant_lesions`

The model is intended for prototype-level screening research.

---

## 📊 Prototype Evaluation

The current prototype was evaluated on a held-out test set.

| Metric                   | Result |
| ------------------------ | -----: |
| Test Images              |     51 |
| Correct Predictions      |     39 |
| Test Accuracy            | 76.47% |
| Best Validation Accuracy | 78.72% |

### Confusion Matrix

```text
[[21, 5],
 [ 7, 18]]
```

These results are from the current prototype dataset and should not be
interpreted as clinical performance.

---

## 🔥 Explainable AI

ORAL-SENSE EDGE uses **Grad-CAM** to visualize image regions that influenced
the model's prediction.

This helps make the AI output more interpretable.

> Grad-CAM highlights model-influencing regions. It does not prove the
> presence, absence, or exact location of a lesion.

---

## 🔐 Privacy

ORAL-SENSE EDGE follows a privacy-first prototype design.

* Processing is performed locally.
* No automatic cloud upload is performed.
* Screening history is stored locally.
* Generated screening data can be cleared by the user.

The privacy design is intended for the current prototype and does not
guarantee security against all operating-system or device-level threats.

---

## 🛠️ Technology Stack

* Python
* PyTorch
* Torchvision
* MobileNetV3
* OpenCV
* CustomTkinter
* Pillow
* NumPy
* ReportLab
* Grad-CAM
* JSON-based local history

---

## 📁 Project Structure

```text
ORAL-SENSE-EDGE/
│
├── app/
│   ├── camera.py
│   ├── quality_check.py
│   ├── input_image.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── test_model.py
│   ├── gradcam.py
│   ├── screening_pipeline.py
│   ├── dashboard.py
│   ├── multi_screening.py
│   ├── change_monitor.py
│   ├── referral_guidance.py
│   ├── screening_history.py
│   ├── history_viewer.py
│   ├── analytics.py
│   ├── privacy_manager.py
│   └── report_generator.py
│
├── model/
│   └── oral_sense_mobilenetv3.pth
│
├── dataset/
│
├── results/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/khushiaroura-del/ORAL-SENSE-EDGE.git
```

### 2. Open the project

```bash
cd ORAL-SENSE-EDGE
```

### 3. Create virtual environment

```bash
python -m venv .venv
```

### 4. Activate environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the dashboard

```bash
python app\dashboard.py
```

---

## 🔬 Dataset

The project uses oral lesion image data organized into two classes:

```text
benign_lesions
malignant_lesions
```

The dataset is divided into:

* Training
* Validation
* Testing

Only the original images were used for the dataset split to reduce the risk
of data leakage.

---

## ⚠️ Limitations

This project is a student/research prototype.

The current model:

* Has a relatively small evaluation set.
* Has not undergone clinical validation.
* Does not replace a dentist or medical professional.
* Should not be used for definitive cancer diagnosis.
* Requires larger and independently collected patient-level datasets for
  meaningful clinical evaluation.

---

## 🔮 Future Scope

Possible future improvements include:

* Larger patient-level datasets
* External validation
* Sensitivity and specificity analysis
* Better lesion localization
* Improved image quality assessment
* Mobile/Android deployment
* On-device optimized inference
* Multilingual interface
* Voice-based guidance
* Secure encrypted local storage
* Long-term image comparison
* Clinical collaboration and validation

---

## 💡 USP

ORAL-SENSE EDGE combines multiple components into a single screening workflow:

**Quality-Aware + MobileNetV3 + Explainable AI + Offline Processing +
Privacy + History + Analytics + Reporting**

The goal is not simply to classify an image, but to provide a more
interpretable and privacy-focused screening workflow.

---

## 🩺 Responsible Use

ORAL-SENSE EDGE is intended for educational, research, and prototype
screening purposes.

It should not be used as a substitute for professional medical examination,
diagnosis, or treatment.

If an oral lesion or abnormality is concerning or persistent, professional
evaluation should be sought.

---

## 👩‍💻 Project

**ORAL-SENSE EDGE**

**Technology:** Edge AI + MobileNetV3 + Explainable AI

**Area:** AI-Based Oral Health Screening

**Project Period:** 2025–26

````
