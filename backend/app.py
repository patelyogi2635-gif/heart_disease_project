from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load artifacts
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

app = FastAPI(title="Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Input schema (matches Streamlit inputs)
class HeartInput(BaseModel):
    age: int
    sex: str
    chest_pain: str
    resting_bp: int
    cholesterol: int
    fasting_bs: int
    resting_ecg: str
    max_hr: int
    exercise_angina: str
    oldpeak: float
    st_slope: str

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: HeartInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # 🔹 CATEGORICAL ENCODING (same as training)
    df["sex"] = df["sex"].map({"M": 1, "F": 0})
    df["exercise_angina"] = df["exercise_angina"].map({"Y": 1, "N": 0})

    df = pd.get_dummies(
        df,
        columns=["chest_pain", "resting_ecg", "st_slope"]
    )

    # 🔹 Ensure all columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # 🔹 Reorder columns
    df = df[columns]

    # 🔹 Scaling
    df_scaled = scaler.transform(df)

    # 🔹 Prediction
    prediction = model.predict(df_scaled)[0]

    return {
        "prediction": int(prediction),
        "result": "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    }

