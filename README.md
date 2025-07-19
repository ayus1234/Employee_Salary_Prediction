# 💼 Employee Salary Prediction

A Streamlit web app to classify employee salaries as `>50K` or `<=50K` based on demographic and employment attributes. This project is designed to assist HR departments and organizations with salary trend analysis and decision-making.  

---

## 🚀 Features

- 📂 Upload CSV datasets for model training (with automatic detection of numerical & categorical columns).
- 👤 Single prediction through dynamic form inputs.
- 📊 Batch prediction by uploading a CSV file and downloading prediction results.
- 💾 Trained model persisted as `models/best_model.pkl` for reuse.
- 🎯 Displays model accuracy after training.

---

## 📦 Tech Stack

- Python 3.x
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- pandas, numpy, joblib

---

## 🛠 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/employee-salary-classification.git
   cd employee-salary-classification

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app:**
    ```bash
    streamlit run app.py
    ```
    
---

## 📸 Results & Screenshots
Below are the key features demonstrated with screenshots of the app:

### 🏠 Streamlit App Home Page

![App Home Page](screenshots/app_home_page.png)

### 📂 Training Dataset Upload & Preview

![Dataset Upload](screenshots/dataset_upload.png)

### 👤 Dynamic Employee Detail Input & Prediction Result

![Single Prediction](screenshots/single_prediction.png)

### 📊 Batch Prediction Upload & Result Download

![Batch Prediction](screenshots/batch_prediction.png)

### 🎯 Model Training Accuracy Shown

![Model Accuracy](screenshots/model_accuracy.png)

---

### 🌐 Demo

🔗 Live Demo

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE (MIT)](LICENSE) file for details.

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit pull requests.
