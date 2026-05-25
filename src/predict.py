import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "iris_model.pkl"

model = joblib.load(MODEL_PATH)

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

flower_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

print("Prediction:", flower_names[prediction[0]])