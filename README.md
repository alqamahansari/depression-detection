# 🧠 Depression Detection from Social Media Text  
### Deep Learning-Based Mental Health Classification System

Live Demo:  
🔗 https://depression-detection-mp9i.onrender.com/


## 📌 Project Overview

This project presents a Deep Learning-based system for detecting depressive tendencies in Reddit-style social media posts.

The objective is to compare traditional machine learning approaches with sequential deep learning models and evaluate their effectiveness in binary classification (Depressed vs Not Depressed).

The final system is deployed as a Dockerized web application with a Flask backend and modern HTML/CSS/JS frontend.


## 🎯 Problem Statement

Early identification of depressive signals in text can assist in mental health research and intervention systems. This project builds and evaluates models that classify user-generated posts into:

- 0 → Not Depressed  
- 1 → Depressed  

⚠️ This system is for academic research purposes only and is NOT a clinical diagnostic tool.


## 🗂 Dataset

- Source: Reddit Depression Dataset  
- Balanced subset: 60,000 samples  
- Input: Combined `title + body`  
- Output: Binary label  

### Preprocessing Steps:
- Lowercasing
- URL removal
- Special character removal
- Tokenization
- Vocabulary size: 10,000
- Sequence padding length: 200


## 🤖 Models Implemented

### 1️⃣ Baseline: TF-IDF + Logistic Regression
- Traditional ML approach
- Feature-based vectorization
- Strong linear baseline

### 2️⃣ LSTM
- Embedding Layer
- Unidirectional LSTM
- Fully Connected + Sigmoid

### 3️⃣ BiLSTM (Final Model)
- Embedding Layer
- Bidirectional LSTM
- Dropout Regularization
- Early Stopping
- Best validation model checkpoint


## 📊 Results Comparison

| Model | Accuracy | F1 Score |
|--------|----------|----------|
| Logistic Regression | 0.9180 | 0.9164 |
| LSTM | 0.9155 | 0.9149 |
| BiLSTM (Final) | 0.9114 | 0.9115 |

### Additional Metrics:
- ROC AUC: ~0.97
- Confusion Matrix evaluated
- Train/Validation/Test split used
- Early stopping applied


## 📉 Model Evaluation

- Proper Train / Validation / Test split
- Validation loss monitoring
- Early stopping (patience = 2)
- Dropout regularization
- Confusion Matrix
- ROC Curve
- Loss Curve tracking

This ensures rigorous experimental methodology and avoids optimistic performance estimates.


## 🏗 System Architecture
``````
Frontend (HTML/CSS/JS)
↓
Flask REST API
↓
PyTorch BiLSTM Model
↓
Docker Container
↓
Render Deployment
``````


## 🐳 Docker Deployment

The application is containerized using Docker to ensure reproducibility and environment consistency.

### Dockerfile Highlights:
- Base image: python:3.10-slim
- Gunicorn production server
- Port 5000 exposed
- Model weights bundled inside container


## 🚀 Running Locally

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Flask App
```bash
python app.py
```

### Visit:
```bash
http://localhost:5000
```

## 🐳 Running with Docker
### Build Image
```bash
docker build -t depression-detection-app .
```

### Run Container
```bash
docker run -p 5000:5000 depression-detection-app
```

## 📁 Project Structure
``````
depression-detection/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── best_bilstm_model.pth
├── vocab.pkl
│
├── models/
├── src/
├── templates/
├── static/
└── report/
``````

## 🔬 Key Learnings
- Traditional ML baselines can remain competitive in NLP tasks.
- BiLSTM captures bidirectional context but gains are modest.
- Proper validation prevents overestimated performance.
- Early stopping improves generalization.
- Containerization enables production-ready ML deployment.

## ⚖ Ethical Considerations
- This model does NOT provide medical diagnosis.
- Social media data may contain bias.
- Misclassification risk exists.
- Human oversight is essential in real-world applications.

## 👨‍💻 Author
Mohammad Alquamah Ansari\
B.Sc. Artificial Intelligence\
Capstone Project

## 📜 License

This project is for academic and research purposes only.
