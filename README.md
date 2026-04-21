# Emotion Detection with BiLSTM and Attention

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Accuracy](https://img.shields.io/badge/test%20accuracy-93.2%25-brightgreen)
![F1](https://img.shields.io/badge/macro%20F1-90.63%25-blue)

This project implements an end-to-end emotion classification system for English text using a frozen GloVe embedding layer, a Bidirectional LSTM encoder, and a custom Bahdanau-style attention mechanism.

It supports:

- training on the six-class emotion dataset
- evaluation with saved plots and a text report
- single-text inference from the command line
- REST inference with FastAPI
- uncertainty handling for low-confidence predictions

The implementation is based on the project SRS in `SRS_EmotionDetection_BiLSTM.md`, with a lean MVP focus on the training, evaluation, inference, and API pipeline.

## Current Status

The project has been implemented and validated locally on Windows with a dedicated Conda environment.

Validated checks:

- `flake8` passes
- `pytest` passes
- end-to-end training smoke test passes
- full training and held-out evaluation completed successfully

Latest full evaluation run:

- Test Accuracy: `0.9320`
- Macro F1: `0.9063`
- Weighted F1: `0.9333`

## Model Overview

The classifier architecture is:

1. Token embedding with pre-trained `glove.6B.100d.txt`
2. Frozen embedding layer
3. `Bidirectional(LSTM(128, return_sequences=True))`
4. Custom additive attention layer
5. Dense `64` with ReLU
6. Dropout `0.5`
7. Softmax output over six classes

Target classes:

- `anger`
- `fear`
- `joy`
- `love`
- `sadness`
- `surprise`

## Repository Layout

```text
emotion_detector/
  api/                FastAPI app and response schemas
  data/               CSV loading and preprocessing
  embedding/          GloVe embedding matrix construction
  evaluation/         Metrics, plots, and report generation
  inference/          Predictor service and CLI inference helpers
  model/              Attention layer and model builder
  training/           Training loop and callbacks
  utils/              Logging, file helpers, and seeding
  check_setup.py      External asset validation
  train.py            Training CLI
  evaluate.py         Evaluation CLI
  infer.py            Inference CLI

tests/
  test_preprocessor.py
  test_attention.py
  test_predictor.py
  test_api.py
  test_pipeline_smoke.py
```

## Environment Setup

This repo was validated with:

- Python `3.11.15`
- TensorFlow `2.20.0`
- Windows + Conda

Create the environment:

```bash
conda env create -f environment.yml
conda activate emotion-detector
```

If you prefer direct pip installation inside an existing Python 3.11 environment:

```bash
python -m pip install -r requirements.txt
```

## External Assets

Large assets are intentionally excluded from git history.

Required local files:

- `data/emotion.csv`
- `glove/glove.6B.100d.txt`

They are validated by:

```bash
python -m emotion_detector.check_setup
```

### Dataset

The runtime expects a semicolon-delimited CSV:

```text
text;label
i feel happy today;joy
i am scared right now;fear
```

The implementation was validated using the six-class 20,000-row `dair-ai/emotion` split dataset materialized into `emotion.csv`.

Equivalent sources:

- Kaggle: `praveengovi/emotions-dataset-for-nlp`
- Hugging Face: `dair-ai/emotion` split version

Expected labels:

- `anger`
- `fear`
- `joy`
- `love`
- `sadness`
- `surprise`

### GloVe

Place `glove.6B.100d.txt` inside the `glove/` directory.

Source:

- https://nlp.stanford.edu/projects/glove/

## Commands

All commands below assume the `emotion-detector` environment is active.

### 1. Validate Setup

```bash
python -m emotion_detector.check_setup
```

### 2. Train

```bash
python -m emotion_detector.train --data .\data\emotion.csv --glove .\glove\glove.6B.100d.txt --output .\artefacts
```

### 3. Evaluate

```bash
python -m emotion_detector.evaluate --data .\data\emotion.csv --artefacts .\artefacts
```

### 4. Inference

```bash
python -m emotion_detector.infer --text "I feel happy today" --artefacts .\artefacts
```

### 5. Start API

```bash
python -m uvicorn emotion_detector.api.app:app --host 0.0.0.0 --port 8000
```

Important:

- `0.0.0.0` is the bind address for the server
- in your browser, use `http://localhost:8000/docs` or `http://127.0.0.1:8000/docs`
- the root `/` route is not implemented, so `/docs`, `/health`, `/classes`, and `/predict` are the intended entry points

## API Endpoints

### `GET /health`

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /classes`

Example response:

```json
{
  "classes": ["anger", "fear", "joy", "love", "sadness", "surprise"]
}
```

### `POST /predict`

Request:

```json
{
  "text": "I feel happy today"
}
```

Response:

```json
{
  "input_text": "I feel happy today",
  "predicted_emotion": "joy",
  "top_emotion": "joy",
  "confidence": 0.9999,
  "confidence_threshold": 0.45,
  "is_uncertain": false,
  "probabilities": {
    "anger": 0.0,
    "fear": 0.0,
    "joy": 0.9999,
    "love": 0.0,
    "sadness": 0.0,
    "surprise": 0.0
  }
}
```

## Uncertainty Threshold

The inference layer supports uncertainty handling for low-confidence predictions.

If the top class confidence is below the configured threshold, the response becomes:

- `predicted_emotion: "uncertain"`
- `top_emotion`: still shows the highest-probability class
- `is_uncertain: true`

Threshold precedence:

1. CLI flag `--confidence-threshold`
2. Environment variable `INFERENCE_CONFIDENCE_THRESHOLD`
3. Saved `artefacts/config.json`
4. Default `0.45`

Examples:

```bash
set INFERENCE_CONFIDENCE_THRESHOLD=0.20
python -m emotion_detector.infer --text "I am so happy today" --artefacts .\artefacts
```

```bash
python -m emotion_detector.infer --text "I am so happy today" --artefacts .\artefacts --confidence-threshold 0.45
```

For the API:

```bash
set INFERENCE_CONFIDENCE_THRESHOLD=0.35
python -m uvicorn emotion_detector.api.app:app --host 0.0.0.0 --port 8000
```

## Artefacts

Training writes the following to `artefacts/`:

- `best_model.keras`
- `tokenizer.pkl`
- `label_encoder.pkl`
- `config.json`
- `training_log.csv`

Evaluation additionally writes:

- `evaluation_report.txt`
- `accuracy_curve.png`
- `loss_curve.png`
- `confusion_matrix.png`

These files are gitignored because they are generated artifacts.

## Testing and Quality Checks

Run tests:

```bash
python -m pytest
```

Run lint:

```bash
python -m flake8 emotion_detector tests
```

## Important Implementation Notes

### 1. Data Leakage Prevention

The tokenizer is fit on the training split only.

### 2. Reproducibility

The project sets:

- Python hash seed
- Python `random`
- NumPy
- TensorFlow

### 3. Model Serialization

The saved model uses the `.keras` format and loads the custom attention layer with:

- `custom_objects={"AttentionLayer": AttentionLayer}`

### 4. Phrasing Sensitivity

The training dataset strongly favors `I feel ...` style sentences. That means short paraphrases such as `I am so happy today` can be less stable than `I feel happy today`, even when the overall evaluation metrics are strong.

The uncertainty threshold is included to avoid overconfident bad UX in those cases.

## Known Limitations

- English-only
- sentence-level emotion classification only
- no authentication or persistence layer in the API
- no web demo or Docker packaging in this MVP
- short informal phrases can be sensitive to phrasing shifts

## What Is Git-Tracked vs Ignored

Tracked:

- source code
- tests
- config files
- README
- SRS

Ignored:

- datasets
- GloVe files
- zip archives
- model artifacts
- plots
- caches

## Suggested GitHub Push Workflow

Initialize the repo locally:

```bash
git init
git add .
git commit -m "Initial commit: emotion detector MVP"
```

Then connect a remote and push:

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## References

- GloVe: https://nlp.stanford.edu/projects/glove/
- Kaggle dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
- Hugging Face dataset: https://huggingface.co/datasets/dair-ai/emotion
