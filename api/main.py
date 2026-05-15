from dotenv import load_dotenv 
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import torch
import pickle
import numpy as np
import torch.nn as nn

import sys
import os
load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/student_db"
)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database.db import SessionLocal, Student


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
class MarksPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

# ── Ab load_model kaam karega ──────────────────────────────
def load_model():
    model = MarksPredictor()

    model.load_state_dict(
        torch.load("../model/model.pth", map_location="cpu")
    )

    model.eval()

    with open("../model/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    return model, scalers["scaler_X"], scalers["scaler_y"]

 
app = FastAPI(title="Student Marks Predictor")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Model Load ───────────────────────────────────────────────
def load_model():
    model = MarksPredictor()
    model.load_state_dict(
        torch.load("../model/model.pth", map_location="cpu")
    )
    model.eval()
 
    with open("../model/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
 
    return model, scalers["scaler_X"], scalers["scaler_y"]
 
ml_model, scaler_X, scaler_y = load_model()
print("Model loaded!")
 
 
# ── Request/Response Models ───────────────────────────────────
class PredictRequest(BaseModel):
    name:        str   = Field(..., example="Rahul")
    study_hours: float = Field(..., ge=0, le=24,  example=6.0)
    attendance:  float = Field(..., ge=0, le=100, example=85.0)
    prev_mark:  float = Field(..., ge=0, le=100, example=72.0)
 
class PredictResponse(BaseModel):
    name:            str
    predicted_marks: float
    grade:           str
    message:         str
 
 
# ── Helper ───────────────────────────────────────────────────
def get_grade(marks: float) -> tuple:
    if marks >= 90:
        return "A+", "Bahut achha! Keep it up!"
    elif marks >= 80:
        return "A",  "Achha performance!"
    elif marks >= 70:
        return "B",  "Theek hai, aur mehnat karo!"
    elif marks >= 60:
        return "C",  "Pass hai, but improve karo!"
    else:
        return "F",  "Zyada padhai karo bhai!"
 
 
# ── Routes ───────────────────────────────────────────────────
 
@app.get("/")
def root():
    return {"message": "Student Marks Predictor API!"}
 
@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}
 
 
@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    db: Session = Depends(get_db)
):
    try:
        # Input prepare karo
        X = np.array([[
            request.study_hours,
            request.attendance,
            request.prev_mark
        ]], dtype=np.float32)
 
        # Normalize karo
        X_scaled = scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
 
        # Predict karo
        with torch.no_grad():
            pred_scaled = ml_model(X_tensor)
            pred        = scaler_y.inverse_transform(
                pred_scaled.numpy()
            )[0][0]
 
        # 0-100 ke beech rakho
        pred = float(np.clip(pred, 0, 100))
 
        # Grade nikalo
        grade, message = get_grade(pred)
 
        # Database mein save karo
        student = Student(
            name        = request.name,
            study_hours = request.study_hours,
            attendance  = request.attendance,
            prev_mark  = request.prev_mark,
            predicted   = pred,
        )
        db.add(student)
        db.commit()
 
        return PredictResponse(
            name            = request.name,
            predicted_marks = round(pred, 1),
            grade           = grade,
            message         = message,
        )
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.get("/students")
def get_students(db: Session = Depends(get_db)):
    """Sab students ki list"""
    students = db.query(Student).order_by(
        Student.created_at.desc()
    ).all()
 
    return [
        {
            "id":           s.id,
            "name":         s.name,
            "study_hours":  s.study_hours,
            "attendance":   s.attendance,
            "prev_mark":   s.prev_mark,
            "predicted":    s.predicted,
            "actual":       s.actual,
        }
        for s in students
    ]
 
 
@app.put("/students/{student_id}/actual")
def update_actual(
    student_id: int,
    actual_marks: float,
    db: Session = Depends(get_db)
):
    """Actual marks update karo"""
    student = db.query(Student).filter(
        Student.id == student_id
    ).first()
 
    if not student:
        raise HTTPException(status_code=404, detail="Student nahi mila")
 
    student.actual = actual_marks
    db.commit()
 
    error = abs(student.predicted - actual_marks)
    return {
        "message":   "Updated!",
        "predicted": student.predicted,
        "actual":    actual_marks,
        "error":     round(error, 1)
    }
 
 
# ── Web UI ───────────────────────────────────────────────────
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Student Marks Predictor</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2rem;
            color: #38bdf8;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 40px;
        }
        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid #334155;
        }
        .card h2 {
            color: #38bdf8;
            margin-bottom: 20px;
            font-size: 1.2rem;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .form-group.full { grid-column: 1 / -1; }
        label { color: #94a3b8; font-size: 0.85rem; }
        input {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 15px;
            color: #e2e8f0;
            font-size: 1rem;
            outline: none;
            transition: border 0.2s;
        }
        input:focus { border-color: #38bdf8; }
        button {
            width: 100%;
            padding: 12px;
            background: #38bdf8;
            color: #0f172a;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            margin-top: 15px;
            transition: background 0.2s;
        }
        button:hover { background: #0ea5e9; }
        .result {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .big-marks {
            font-size: 4rem;
            font-weight: bold;
            color: #38bdf8;
        }
        .grade-badge {
            display: inline-block;
            padding: 5px 20px;
            border-radius: 20px;
            font-size: 1.2rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .msg { color: #94a3b8; margin-top: 10px; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        th {
            background: #0f172a;
            padding: 10px;
            text-align: left;
            color: #38bdf8;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #1e293b;
            color: #94a3b8;
        }
        tr:hover td { background: #1e293b; }
    </style>
</head>
<body>
<div class="container">
    <h1>🎓 Student Marks Predictor</h1>
    <p class="subtitle">PyTorch + FastAPI + PostgreSQL</p>
 
    <!-- Form -->
    <div class="card">
        <h2>📊 Marks Predict Karo</h2>
        <div class="form-grid">
            <div class="form-group full">
                <label>Student Ka Naam</label>
                <input id="name" placeholder="Jaise: Rahul" />
            </div>
            <div class="form-group">
                <label>Roz Kitne Ghante Padhte Ho</label>
                <input id="hours" type="number" placeholder="0-24" min="0" max="24" step="0.5" />
            </div>
            <div class="form-group">
                <label>Attendance (%)</label>
                <input id="attend" type="number" placeholder="0-100" min="0" max="100" />
            </div>
            <div class="form-group full">
                <label>Pichle Exam Ke Marks</label>
                <input id="prev" type="number" placeholder="0-100" min="0" max="100" />
            </div>
        </div>
        <button onclick="predict()">🔮 Predict Karo</button>
 
        <!-- Result -->
        <div class="result" id="result">
            <div class="big-marks" id="marksVal"></div>
            <div class="grade-badge" id="gradeBadge"></div>
            <div class="msg" id="msgVal"></div>
        </div>
    </div>
 
    <!-- History -->
    <div class="card">
        <h2>📋 Prediction History</h2>
        <table>
            <thead>
                <tr>
                    <th>Naam</th>
                    <th>Hours</th>
                    <th>Attendance</th>
                    <th>Prev</th>
                    <th>Predicted</th>
                </tr>
            </thead>
            <tbody id="historyBody"></tbody>
        </table>
    </div>
</div>
 
<script>
    const COLORS = {
        "A+": "#22c55e", "A": "#84cc16",
        "B":  "#eab308", "C": "#f97316", "F": "#ef4444"
    };
 
    async function predict() {
        const name  = document.getElementById("name").value;
        const hours = document.getElementById("hours").value;
        const att   = document.getElementById("attend").value;
        const prev  = document.getElementById("prev").value;
 
        if (!name || !hours || !att || !prev) {
            alert("Sab fields fill karo!");
            return;
        }
 
        const res = await fetch("/predict", {
            method:  "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                name:        name,
                study_hours: parseFloat(hours),
                attendance:  parseFloat(att),
                prev_mark:  parseFloat(prev)
            })
        });
 
        const data = await res.json();
 
        document.getElementById("marksVal").textContent  = data.predicted_marks + "%";
        document.getElementById("gradeBadge").textContent = data.grade;
        document.getElementById("gradeBadge").style.background = COLORS[data.grade] || "#38bdf8";
        document.getElementById("gradeBadge").style.color = "#0f172a";
        document.getElementById("msgVal").textContent    = data.message;
        document.getElementById("result").style.display  = "block";
 
        loadHistory();
    }
 
    async function loadHistory() {
        const res  = await fetch("/students");
        const data = await res.json();
        const body = document.getElementById("historyBody");
 
        body.innerHTML = data.slice(0, 10).map(s => `
            <tr>
                <td>${s.name}</td>
                <td>${s.study_hours}h</td>
                <td>${s.attendance}%</td>
                <td>${s.prev_mark}</td>
                <td><b style="color:#38bdf8">${s.predicted}</b></td>
            </tr>
        `).join("");
    }
 
    loadHistory();
</script>
</body>
</html>
"""
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
 
 
# ============================================================
# STEP 6 — requirements.txt
# ============================================================
REQUIREMENTS = """
torch
fastapi
uvicorn
psycopg2-binary
sqlalchemy
pandas
numpy
scikit-learn
python-dotenv
"""
 
# ============================================================
# RUN ORDER
# ============================================================
"""
1. PostgreSQL install karo + setup.sql run karo
 
2. Libraries install karo:
   pip install -r requirements.txt
 
3. Model train karo:
   python train.py
 
4. API start karo:
   
 
5. Browser mein kholo:
   http://localhost:8000/ui      ← Web Interface
   http://localhost:8000/docs    ← API Docs
"""
