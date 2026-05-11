# DeepShield — Explainable Deepfake Detection

DeepShield is a cloud-native deepfake detection system built using TensorFlow, Flask, and Google Cloud Platform. The system detects manipulated facial images and generates Grad-CAM visual explanations highlighting the regions responsible for the prediction.

In addition to inference, DeepShield integrates a complete cloud analytics pipeline using Google Cloud Storage, BigQuery, Looker Studio, and Cloud Run.

---

## Features

- Deepfake image classification (REAL vs FAKE)
- Grad-CAM explainability heatmaps
- Cloud-native deployment using Google Cloud Run
- Model storage and loading from Google Cloud Storage
- Prediction logging with BigQuery
- Interactive analytics dashboard using Looker Studio
- Lightweight MobileNetV2-based inference pipeline
- Real-time browser-based interface

---

## System Architecture

```text
User Upload
    ↓
Flask Backend (Cloud Run)
    ↓
Model Download (Cloud Storage)
    ↓
MobileNetV2 Inference
    ↓
Grad-CAM Generation
    ↓
Prediction Logging (BigQuery)
    ↓
Analytics Dashboard (Looker Studio)
```

---

## Tech Stack

| Component          | Technology           |
| ------------------ | -------------------- |
| Backend            | Flask                |
| Deep Learning      | TensorFlow / Keras   |
| CNN Architecture   | MobileNetV2          |
| Explainability     | Grad-CAM             |
| Cloud Deployment   | Google Cloud Run     |
| Model Storage      | Google Cloud Storage |
| Analytics Database | BigQuery             |
| Dashboard          | Looker Studio        |
| Image Processing   | OpenCV, Pillow       |

---

## Project Structure

```text
deepshield/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── templates/
│   └── index.html
├── static/
├── demo_samples/
└── README.md
```

---

## How It Works

### 1. Image Upload

Users upload a facial image through the browser interface.

### 2. Image Preprocessing

The image is resized to 224×224 and preprocessed using MobileNetV2 preprocessing.

### 3. Model Inference

The trained MobileNetV2 model predicts whether the image is REAL or FAKE.

### 4. Grad-CAM Explainability

A Grad-CAM heatmap is generated to visualize the regions that influenced the prediction.

### 5. Prediction Logging

Prediction metadata is stored in BigQuery:

- Timestamp
- Prediction label
- Confidence score
- Image hash
- Grad-CAM generation status

### 6. Analytics Visualization

BigQuery data is visualized using Looker Studio dashboards.

---

## Google Cloud Services Used

### Google Cloud Run

Hosts the Flask backend as a serverless containerized application.

### Google Cloud Storage

Stores the trained deep learning model and enables runtime model loading.

### BigQuery

Stores prediction logs for analytics and monitoring.

### Looker Studio

Creates interactive dashboards from BigQuery data.

---

## Grad-CAM Explainability

DeepShield uses Gradient-weighted Class Activation Mapping (Grad-CAM) to improve transparency.

Instead of only predicting whether an image is fake, the system highlights the facial regions responsible for the prediction.

This makes the model more interpretable and trustworthy.

---

## Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/meenakshi04cy/deepshield.git
cd deepshield
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Google Cloud Credentials

Set:

```text
GOOGLE_APPLICATION_CREDENTIALS
```

or place local credentials JSON file.

### 5. Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Cloud Deployment

DeepShield is deployed using Google Cloud Run.

### Deployment Command

```bash
gcloud run deploy deepshield --source .
```

---

## BigQuery Analytics

Example analytics queries:

### REAL vs FAKE Distribution

```sql
SELECT prediction, COUNT(*) AS total
FROM `deepshield-493817.deepshield_data.prediction_logs`
GROUP BY prediction;
```

### Average Confidence

```sql
SELECT AVG(confidence_score) AS avg_confidence
FROM `deepshield-493817.deepshield_data.prediction_logs`;
```

### Predictions Over Time

```sql
SELECT DATE(timestamp) AS day, COUNT(*) AS predictions
FROM `deepshield-493817.deepshield_data.prediction_logs`
GROUP BY day
ORDER BY day;
```

---

## Dashboard Features

The Looker Studio dashboard includes:

- Total predictions scorecard
- REAL vs FAKE pie chart
- Prediction timeline
- Confidence score analysis
- Recent prediction logs
- Interactive filters

---

## Model Information

| Property        | Detail               |
| --------------- | -------------------- |
| Architecture    | MobileNetV2          |
| Framework       | TensorFlow / Keras   |
| Input Size      | 224×224 RGB          |
| Classification  | Binary (REAL / FAKE) |
| Explainability  | Grad-CAM             |

---

## Limitations

- Model performance depends on dataset distribution.
- Generalization to unseen generators or diffusion-based images may vary.
- Current system focuses on image-based deepfake detection only.

---

## Future Scope

- Video deepfake detection
- Multi-face analysis
- Diffusion-model detection
- Improved dataset diversity
- Real-time webcam inference
- Advanced explainability methods

---

## Team

Amrita Vishwa Vidyapeetham — B.Tech CSE

- Krishnapriya
- Meenakshi
- Shubhangi

Cloud Computing End Semester Project

---

## License

This project was developed for academic and educational purposes.

---

## Acknowledgements

- TensorFlow
- Google Cloud Platform
- OpenCV
- Flask
- Keras
- Looker Studio
- BigQuery