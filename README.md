# 💼 Employee Salary Classification

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
