# Iris Flower Classification Project

Machine Learning project using the Iris dataset.

## Project Structure

```text
iris-project/
│
├── data/
├── models/
├── src/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Create virtual environment:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python src/train.py
```

## Predict

```bash
python src/predict.py
```

## Run Tests

```bash
pytest
```