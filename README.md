# 🧠 Loan Prediction Classifier with Frontend & Backend

A full-stack machine learning project that predicts whether a loan application will be approved or not, based on user details. It includes a trained ML model, a FastAPI backend, and a Streamlit frontend for user interaction.

---

## 🚀 Features
- Loan approval prediction using CatBoost Classifier
- FastAPI backend for serving model predictions
- Streamlit frontend for interactive UI
- Dockerized microservices (backend + frontend)
- Deployed with Docker Compose

---

## 📊 Tech Stack
- Python, Pandas, NumPy
- CatBoost, Scikit-learn
- FastAPI
- Streamlit
- Docker, Docker Compose

---

## 🧠 Model Details
- **Algorithm**: CatBoost Classifier
- **Training accuracy**: 96%
- **Test accuracy**: 84%
---

## 📁 Project Structure

├── backend/
│ ├── main.py # FastAPI app
│ └── model.pkl # Trained ML model
├── frontend/
│ └── streamlit_app.py # User interface
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md

yaml
Copy
Edit

---

## 🧪 How to Run Locally

### 🚨 Prerequisites:
- Python 3.8+
- Docker (if using containerized version)

### 🔧 Option 1: Run Locally (without Docker)

# Backend
cd backend
uvicorn main:app --reload

# Frontend
cd ../frontend
streamlit run streamlit_app.py
🐳 Option 2: Run with Docker
bash
Copy
Edit
docker-compose up --build
Visit http://localhost:8501 to access the frontend.

🖼️ Screenshots
Add screenshots of the UI here for visual impact.

📌 Future Improvements
Add SHAP explainability to model output

Add user authentication

Improve UI with better styling

Integrate CI/CD (GitHub Actions)

🙋‍♂️ Author
Ayush Pandey

AI Engineer in training | Deep Learning Enthusiast
