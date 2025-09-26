---

# 🌿 Cotton Leaf Disease Detection

![Project Banner](https://via.placeholder.com/1000x200?text=Cotton+Leaf+Disease+Detection)

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-Vite%2BJSX-blueviolet)](https://reactjs.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4.1-green)](https://tailwindcss.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-success)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-orange)](LICENSE)

---

## 🎯 Project Overview

**Cotton Leaf Disease Detection** is a **desktop web application** that uses **Deep Learning (CNN)** to detect diseases in cotton leaves. The app allows farmers or researchers to **upload leaf images** and receive **disease classification along with confidence scores**.

✨ Features:

* Detects **5–6 types of cotton leaf diseases**
* Shows **confidence levels** for predictions
* Clean, **responsive frontend** with **animations** using React & TailwindCSS v4.1
* Fast **backend API** using FastAPI/Flask
* CSV-based storage for user uploads and predictions
* Easily trainable CNN model

![Frontend GIF](https://via.placeholder.com/800x400?text=Frontend+Animation+GIF)

---

## 🗂️ Folder Structure

```
cotton-leaf-disease-detection/
├── frontend/                 # React Vite frontend
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── utils/
│   │   └── styles/
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── backend/                  # Python backend
│   ├── app/
│   │   ├── models/
│   │   ├── routes/
│   │   ├── utils/
│   │   └── data/
│   ├── training/
│   ├── requirements.txt
│   └── main.py
├── model_training/           # CNN Model training
│   ├── train_model.py
│   ├── requirements.txt
│   └── dataset/
└── README.md
```

---

## 🚀 Tech Stack

**Frontend:**

* React (Vite + JSX)
* TailwindCSS v4.1
* Framer Motion for animations

**Backend:**

* Python 3.7+
* FastAPI / Flask
* TensorFlow/Keras (CNN)
* CSV-based data storage

**Model:**

* Convolutional Neural Network trained on **Kaggle cotton leaf dataset**
* 5–6 disease classes

---

## ⚡ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/cotton-leaf-disease-detection.git
cd cotton-leaf-disease-detection
```

### 2️⃣ Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn main:app --reload   # FastAPI
# OR
python main.py              # Flask
```

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🧠 Model Training

1. Activate virtual environment:

```bash
cd model_training
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

2. Train CNN model:

```bash
python train_model.py
```

* Model saved in `backend/app/models/`
* Uses Kaggle cotton leaf dataset (`dataset/`)

---

## 🖼️ Usage

1. Open the frontend in a browser.
2. Go to **Upload Page**.
3. Upload an image of a cotton leaf.
4. Backend predicts **disease class** + **confidence score**.

![Prediction Example](https://via.placeholder.com/600x400?text=Prediction+Example)

---

## 🛠️ Libraries & Dependencies

**Frontend:**

* react, react-dom, react-router-dom
* tailwindcss@4.1
* framer-motion
* axios

**Backend:**

* fastapi
* flask
* tensorflow>=2.12
* keras
* pandas, numpy
* uvicorn

---

## 👨‍💻 Author

**Shathananda Bhat N** – 3rd Year CS Student

---

## 📄 License

MIT License © 2025

---
