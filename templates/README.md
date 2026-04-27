# DeepShield — Explainable Deepfake Detection

Cloud Computing End Semester Project  
Amrita Vishwa Vidyapeetham, Amritapuri

## What it does
Detects deepfake face images using MobileNetV2 + Grad-CAM explainability.
Shows which facial regions triggered the fake detection.

## GCP Components
- Vertex AI — model training
- AutoML Vision — image classification
- Cloud Storage — dataset and model storage
- Cloud Run — app deployment

## Setup for teammates

### 1. Clone the repo
git clone https://github.com/YOURUSERNAME/deepshield.git
cd deepshield

### 2. Create virtual environment (needs Python 3.11)
py -3.11 -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add GCP credentials
Get credentials JSON from [your name] on WhatsApp
Place it at: C:\Users\YOURNAME\Downloads\deepshield-493817-ca210739a648.json
OR update the path in setup_credentials() in app.py

### 5. Run
python app.py
Open http://localhost:5000

## Model info
- Architecture: MobileNetV2 transfer learning
- Dataset: 140k Real and Fake Faces (Kaggle)
- Accuracy: 91% train, 87% val
- Class mapping: fake=0, real=1
- Stored in GCS: deepfake-dataset-26/deepshield_final.keras

## Project structure
deepshield/
    app.py
    requirements.txt
    Dockerfile
    README.md
    templates/
        index.html