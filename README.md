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

<img width="1365" height="631" alt="Screenshot 2025-07-20 004932" src="https://github.com/user-attachments/assets/e2e682b6-744d-4378-851a-db228c10740c" />

---

### 📂 Training Dataset Upload & Preview

<img width="1365" height="633" alt="Screenshot 2025-07-20 005216" src="https://github.com/user-attachments/assets/274004a7-4af9-4ce3-bf48-2718b5c80290" />

---

<img width="1365" height="632" alt="Screenshot 2025-07-19 231820" src="https://github.com/user-attachments/assets/45b4e61b-d711-4d34-b4ae-5c2ebe0a2305" />

---

<img width="1365" height="635" alt="Screenshot 2025-07-20 001735" src="https://github.com/user-attachments/assets/7f1198f4-ee71-413a-8ba2-5cf25d5ad5b4" />

---

### 👤 Dynamic Employee Detail Input & Prediction Result

<img width="1365" height="632" alt="Screenshot 2025-07-20 001937" src="https://github.com/user-attachments/assets/ac9c7837-c88d-47ad-b778-8dfa3d4ba9a0" />

---

<img width="1365" height="632" alt="Screenshot 2025-07-20 002115" src="https://github.com/user-attachments/assets/d26b32a2-0d5c-4e30-89b4-a3df3454db16" />

---

### 📊 Batch Prediction Upload & Result Download

<img width="1365" height="632" alt="Screenshot 2025-07-20 003359" src="https://github.com/user-attachments/assets/f6614048-ff7b-43d2-9a17-9819a2dfe0d7" />

---

<img width="1365" height="632" alt="Screenshot 2025-07-20 003556" src="https://github.com/user-attachments/assets/3d72db50-b34a-4a76-a499-2a7e0211ebb4" />

---

### 🎯 Model Training Accuracy Shown

<img width="1365" height="633" alt="Screenshot 2025-07-20 003204" src="https://github.com/user-attachments/assets/31fcf6fc-a56e-4524-9f5a-3f102d8f9b30" />

---

## 🌐 Live Demo

🔗 [Employee Salary Classification App](https://employeesalaryprediction-zymcnzdci6rswwrhpvj5za.streamlit.app/)  

Click the link above to try out the app directly in your browser.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit pull requests.
