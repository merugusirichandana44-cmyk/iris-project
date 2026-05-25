import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "iris.csv"

def test_dataset_exists():
    assert DATA_PATH.exists()

def test_dataset_not_empty():
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0

def test_target_column_exists():
    df = pd.read_csv(DATA_PATH)
    assert "target" in df.columns