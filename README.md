## ❤️ Heart Disease Prediction using Machine Learning

### 🧠 Overview

This project predicts the likelihood of heart disease in a patient based on medical attributes using various machine learning algorithms.
The goal is to assist early diagnosis and improve preventive healthcare decisions.

---

### ⚙️ Features

* Data preprocessing and cleaning using **Pandas** and **NumPy**
* Exploratory Data Analysis (EDA) and visualizations with **Matplotlib** and **Seaborn**
* Model training using algorithms like:

  * **K-Nearest Neighbors (KNN)**
  * **Logistic Regression**
  * **Random Forest**
  * **Support Vector Machine (SVM)**
* Model evaluation using accuracy, confusion matrix, and classification report
* **Saved trained models** as `.pkl` files for easy reuse
* Interactive **Streamlit web app** for real-time prediction

---

### 🗂️ Project Structure

```
heart_disease_project/
│
├── main.py                    # Main Python script
├── requirements.txt           # All dependencies
├── pkl_files/                 # Folder containing saved models
│   ├── knn_heart_model.pkl
│   ├── logistic_model.pkl
│   └── ...
├── dataset/                   # (optional) Heart disease dataset (CSV)
├── notebooks/                 # Jupyter notebooks for EDA and model training
└── README.md                  # Project documentation
```

---

### 📊 Dataset

Used **Heart Disease UCI Dataset** (available on Kaggle or UCI ML Repository).
It contains patient health records such as:

* Age, Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol Level
* Fasting Blood Sugar
* ECG Results, Heart Rate, etc.

---

### 🧩 How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/heart_disease_project.git
   cd heart_disease_project
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**

   ```bash
   streamlit run main.py
   ```

---

### 🧠 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 87%      |
| KNN                 | 88%      |
| Random Forest       | 85%      |
| SVM                 | 86%      |

---

### 💡 Future Improvements

* Use deep learning models for higher accuracy
* Add explainability with SHAP or LIME
* Deploy on cloud (Streamlit Cloud / Render / Hugging Face Spaces)

---

### 👨‍💻 Author

**Yogi Patel**
📧  patelyogi2635@gmail.com
🔗 https://www.linkedin.com/in/patel-yogi-0a2526346?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
