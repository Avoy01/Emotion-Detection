# Software Requirements Specification  
## Emotion Detection System Using BiLSTM with Attention Mechanism  
**Version:** 1.0 | **Status:** Draft | **Date:** April 2026

---

## Table of Contents

1. Introduction  
2. Overall System Description  
3. Functional Requirements  
4. Non-Functional Requirements  
5. System Architecture  
6. Data Requirements  
7. Model Requirements  
8. Interface Requirements  
9. Module-Level Implementation Specification  
10. Testing and Validation Requirements  
11. Deployment Requirements  
12. Appendices

---

## 1. Introduction

### 1.1 Purpose

This document provides the complete Software Requirements Specification (SRS) for the Emotion Detection System — a deep learning application that classifies natural language text into one of six emotional categories (joy, sadness, anger, fear, love, surprise) using a Bidirectional Long Short-Term Memory (BiLSTM) network with an attention mechanism.

The intended audience includes developers implementing the system, researchers reviewing the architecture, and evaluators assessing experimental validity.

### 1.2 Scope

The system accepts raw English-language text input and produces a probability distribution over six emotion classes, returning the highest-confidence class as the predicted label. The scope covers:

- End-to-end data preprocessing pipeline
- GloVe-based word embedding
- BiLSTM sequence modeling
- Custom attention mechanism layer
- Softmax classification head
- Training, evaluation, and inference pipelines
- A minimal REST API for inference serving
- A simple web-based demo interface

Out of scope: multi-language support, real-time streaming inference, dialogue-level emotion tracking, user authentication.

### 1.3 Definitions and Abbreviations

| Term | Definition |
|------|------------|
| BiLSTM | Bidirectional Long Short-Term Memory — processes sequences in both forward and backward directions |
| GloVe | Global Vectors for Word Representation — pre-trained static word embeddings from Stanford NLP |
| MNAR | Missing Not At Random — not applicable here; included for completeness from project context |
| Tokenizer | Module that converts raw text into sequences of integer indices |
| Padding | Appending zeros to sequences to enforce uniform length |
| OOV | Out-of-Vocabulary — a word not present in the pre-trained embedding dictionary |
| AUROC | Area Under the Receiver Operating Curve — evaluation metric for binary classifiers |
| Softmax | Normalised exponential function mapping logits to a probability distribution |
| Dropout | Regularisation technique that randomly zeroes activations during training |
| Epoch | One complete pass over the training dataset |
| Confusion Matrix | Table showing correct vs. predicted class counts |

### 1.4 References

- Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
- Pennington, J., Socher, R., Manning, C. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
- Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
- Kaggle Emotion Dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

---

## 2. Overall System Description

### 2.1 Product Perspective

The system is a standalone deep learning pipeline implemented in Python using TensorFlow/Keras. It comprises three operational modes:

**Training Mode:** Reads the dataset, preprocesses text, trains the BiLSTM-Attention model, saves weights and tokenizer artefacts.

**Evaluation Mode:** Loads a trained model, runs inference on the test split, generates classification metrics, confusion matrix, and per-class performance curves.

**Inference Mode:** Accepts a raw text string (via CLI or REST API), returns a JSON response containing predicted emotion and class probability scores.

### 2.2 System Functions Summary

- Text loading and cleaning from semicolon-delimited CSV
- Tokenization with vocabulary size capping
- Sequence padding to fixed length
- GloVe embedding matrix construction
- BiLSTM sequence encoding
- Soft attention-weighted context vector extraction
- Dense classification with dropout regularisation
- Adam optimiser with sparse categorical cross-entropy loss
- Serialisation of model weights, tokenizer, and label encoder
- REST API wrapping inference pipeline
- Evaluation suite producing accuracy, per-class precision/recall/F1, confusion matrix, and training curves

### 2.3 User Characteristics

| User Type | Technical Level | Primary Interaction |
|-----------|----------------|---------------------|
| ML Researcher | Expert | Training, ablation studies, metric analysis |
| Developer | Intermediate | API integration, deployment |
| End User | Non-technical | Web interface for single-sentence prediction |

### 2.4 Constraints

- The system requires Python 3.9 or later
- GPU with CUDA 11.x+ is strongly recommended for training; inference runs on CPU
- GloVe 100d file (glove.6B.100d.txt, ~347 MB) must be downloaded separately
- The model is designed for English text only
- Sequence length is fixed at 100 tokens; texts longer than this are truncated
- The emotion label set is fixed at the six classes in the Kaggle dataset

### 2.5 Assumptions and Dependencies

- The Kaggle Emotion Dataset maintains its current class label scheme
- GloVe 6B 100-dimensional vectors are available from Stanford NLP
- The host environment has TensorFlow 2.x installed
- Flask or FastAPI is available for API serving

---

## 3. Functional Requirements

### FR-01: Data Loading

**Description:** The system shall load the raw dataset from a CSV file where text and emotion fields are separated by a semicolon delimiter.

**Input:** File path to `emotion.csv`  
**Output:** Two Python lists — raw text strings and raw label strings  
**Validation:** Assert that no null values exist in either column after loading. If nulls are found, drop those rows and log a warning with the count of dropped records.  
**Error Handling:** Raise `FileNotFoundError` with a descriptive message if the path is invalid. Raise `ValueError` if the expected columns ('text', 'label') are absent.

---

### FR-02: Text Cleaning

**Description:** The system shall apply a deterministic cleaning function to each raw text string before tokenization.

**Operations (in order):**
1. Convert to lowercase
2. Remove URLs matching pattern `http[s]?://\S+`
3. Remove HTML tags matching pattern `<.*?>`
4. Remove non-alphanumeric characters except spaces (preserve apostrophes for contractions)
5. Collapse multiple whitespace characters into a single space
6. Strip leading and trailing whitespace

**Input:** Raw string  
**Output:** Cleaned string  
**Constraint:** The cleaning function must be deterministic and produce identical output for identical input across all runs.

---

### FR-03: Label Encoding

**Description:** The system shall convert string emotion labels into integer class indices using a `LabelEncoder` and persist the mapping.

**Input:** List of string labels  
**Output:** NumPy array of integer indices; a serialised `LabelEncoder` object  
**Persistence:** The encoder shall be saved to disk as `label_encoder.pkl` using `joblib.dump`.  
**Classes:** ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'] — alphabetical ordering  
**Validation:** Assert that exactly six unique classes are present after encoding. If not, raise `ValueError` listing the found classes.

---

### FR-04: Train/Test Split

**Description:** The system shall split the dataset into training and test sets using a fixed 80:20 ratio with a fixed random seed.

**Parameters:**  
- `test_size = 0.2`  
- `random_state = 42`  
- `stratify = y` (class distribution is preserved in both splits)

**Constraint:** Stratified split is mandatory because the Kaggle dataset is class-imbalanced. Non-stratified splitting is not acceptable.

---

### FR-05: Tokenization

**Description:** The system shall fit a `Tokenizer` on the training corpus only (not the test corpus) and transform both splits using the fitted tokenizer.

**Parameters:**  
- `num_words = 20000` (vocabulary ceiling)  
- `oov_token = '<OOV>'`

**Fitting:** `tokenizer.fit_on_texts(X_train)` — strictly no data leakage from test split.  
**Transformation:** `tokenizer.texts_to_sequences(X)` applied to both splits.  
**Persistence:** Serialise the fitted tokenizer to `tokenizer.pkl` using `joblib.dump` immediately after fitting.

---

### FR-06: Sequence Padding

**Description:** The system shall pad all tokenized sequences to a uniform maximum length.

**Parameters:**  
- `maxlen = 100`  
- `padding = 'post'` (zeros appended at the end)  
- `truncating = 'post'` (excess tokens removed from the end)

**Implementation:** `tf.keras.preprocessing.sequence.pad_sequences`  
**Constraint:** Padding parameters applied to training and test sequences must be identical. The `maxlen` value is a project constant and must be defined in a central `config.py` file, not hard-coded inline.

---

### FR-07: GloVe Embedding Matrix Construction

**Description:** The system shall construct a NumPy embedding matrix of shape `(vocab_size + 1, embedding_dim)` by mapping each word in the tokenizer's vocabulary to its corresponding GloVe vector.

**Parameters:**  
- `embedding_dim = 100`  
- GloVe file: `glove.6B.100d.txt`

**Algorithm:**
1. Initialise matrix as zeros with shape `(min(num_words, len(word_index)) + 1, 100)`
2. Load GloVe file line by line; parse word and vector
3. For each word in `tokenizer.word_index`, if word exists in GloVe, insert vector at `embedding_matrix[index]`
4. Words not found in GloVe retain zero vectors (OOV handling)

**Logging:** After construction, log: total vocabulary size, number of words found in GloVe, number of OOV words.  
**Constraint:** OOV rate should be below 15% for a standard English corpus. Log a warning if it exceeds this.

---

### FR-08: Model Architecture Construction

**Description:** The system shall construct a Keras Sequential (or Functional API) model with the following exact layer specification in order.

**Layer Stack:**

| # | Layer Type | Configuration | Notes |
|---|-----------|---------------|-------|
| 1 | Embedding | `input_dim=vocab_size+1`, `output_dim=100`, `weights=[embedding_matrix]`, `input_length=100`, `trainable=False` | Non-trainable; frozen GloVe |
| 2 | Bidirectional(LSTM) | `units=128`, `return_sequences=True`, `dropout=0.2`, `recurrent_dropout=0.2` | Returns full sequence for attention |
| 3 | Attention (Custom) | Additive (Bahdanau-style); see FR-09 | Produces context vector |
| 4 | Dense | `units=64`, `activation='relu'` | Feature compression |
| 5 | Dropout | `rate=0.5` | Regularisation |
| 6 | Dense (Output) | `units=6`, `activation='softmax'` | Class probability distribution |

**Constraint:** The embedding layer must use `trainable=False`. This is a hard requirement — fine-tuning GloVe vectors on this dataset size risks overfitting.

---

### FR-09: Attention Layer Implementation

**Description:** The system shall implement a custom Keras attention layer as a subclass of `tf.keras.layers.Layer` using Bahdanau (additive) attention.

**Mathematical Specification:**

Given BiLSTM hidden states $H = [h_1, h_2, ..., h_T]$ where each $h_t \in \mathbb{R}^{256}$ (128 units × 2 directions):

$$e_t = \mathbf{v}^\top \tanh(\mathbf{W} h_t + \mathbf{b})$$

$$\alpha_t = \text{softmax}(e_t) = \frac{\exp(e_t)}{\sum_{j=1}^{T} \exp(e_j)}$$

$$\mathbf{c} = \sum_{t=1}^{T} \alpha_t h_t$$

**Trainable Parameters:**  
- $\mathbf{W} \in \mathbb{R}^{128 \times 256}$ (attention weight matrix)  
- $\mathbf{b} \in \mathbb{R}^{128}$ (bias)  
- $\mathbf{v} \in \mathbb{R}^{128}$ (context vector)

**Implementation Requirements:**
- Subclass `tf.keras.layers.Layer`
- Implement `build(self, input_shape)` — initialise $\mathbf{W}$, $\mathbf{b}$, $\mathbf{v}$ using `self.add_weight`
- Implement `call(self, inputs)` — compute $e_t$, $\alpha_t$, $\mathbf{c}$ as above
- Return both `context_vector` and `attention_weights` from `call` (weights needed for visualisation)
- Implement `get_config(self)` — return a dict including parent config for serialisation

**Serialisation:** The custom layer must be registered so that `tf.keras.models.load_model` can reconstruct it using `custom_objects={'AttentionLayer': AttentionLayer}`.

---

### FR-10: Model Compilation

**Description:** The system shall compile the model with the following exact configuration.

**Parameters:**  
- `optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)`  
- `loss = 'sparse_categorical_crossentropy'`  
- `metrics = ['accuracy']`

**Constraint:** The learning rate must be a named parameter, not embedded inside an Adam string shorthand. This allows future tuning.

---

### FR-11: Model Training

**Description:** The system shall train the compiled model with the following configuration.

**Parameters:**  
- `epochs = 10` (configurable via `config.py`)  
- `batch_size = 32`  
- `validation_split = 0.2` (drawn from training set)

**Callbacks (mandatory):**

| Callback | Configuration | Purpose |
|----------|--------------|---------|
| `ModelCheckpoint` | Save best model by `val_accuracy`; `save_best_only=True` | Prevents saving overfit model |
| `EarlyStopping` | `monitor='val_loss'`, `patience=3`, `restore_best_weights=True` | Stops wasted epochs |
| `ReduceLROnPlateau` | `monitor='val_loss'`, `factor=0.5`, `patience=2`, `min_lr=1e-6` | Adaptive learning rate |
| `CSVLogger` | Append epoch metrics to `training_log.csv` | Reproducible curve plotting |

**Constraint:** Training must use `validation_split`, not the held-out test set. The test set must remain completely unseen until the final evaluation step.

---

### FR-12: Model Serialisation

**Description:** After training is complete, the system shall save all artefacts required for inference.

**Required Artefacts:**

| File | Format | Contents |
|------|--------|---------|
| `best_model.h5` | HDF5 | Best model weights (from `ModelCheckpoint`) |
| `tokenizer.pkl` | Pickle | Fitted `Tokenizer` object |
| `label_encoder.pkl` | Pickle | Fitted `LabelEncoder` object |
| `config.json` | JSON | All hyperparameters (maxlen, vocab_size, embedding_dim, etc.) |
| `training_log.csv` | CSV | Per-epoch train/val accuracy and loss |

**Constraint:** All files must be co-located in a `./artefacts/` directory. Inference code must load from this directory without modification.

---

### FR-13: Evaluation Pipeline

**Description:** The system shall evaluate the trained model on the held-out test split and produce a comprehensive evaluation report.

**Required Outputs:**

1. Overall test accuracy (exact percentage, not range)
2. `classification_report` from `sklearn.metrics` covering all six classes with per-class precision, recall, F1-score, and support
3. Confusion matrix as a normalised heatmap (normalised by row — i.e., recall)
4. Macro-averaged and weighted-averaged F1
5. Training and validation accuracy curves (saved as `accuracy_curve.png`)
6. Training and validation loss curves (saved as `loss_curve.png`)
7. Per-class AUROC scores using one-vs-rest strategy (requires converting labels to binary format)

**Constraint:** All numeric metrics must be reported to four decimal places. Ranges like "90–95%" are not acceptable in the evaluation report.

---

### FR-14: Inference Pipeline

**Description:** The system shall provide an `predict_emotion(text: str) -> dict` function usable as a Python callable.

**Input:** A raw string of arbitrary length  
**Output:** A dictionary with the following schema:
```json
{
  "input_text": "I am feeling very happy today",
  "predicted_emotion": "joy",
  "confidence": 0.9823,
  "probabilities": {
    "anger": 0.0021,
    "fear": 0.0018,
    "joy": 0.9823,
    "love": 0.0091,
    "sadness": 0.0034,
    "surprise": 0.0013
  }
}
```

**Processing Steps:**
1. Apply text cleaning (FR-02)
2. Tokenize using loaded tokenizer
3. Pad to `maxlen = 100`
4. Run model forward pass
5. Convert softmax output to probability dict using label encoder class names
6. Return structured dict

**Constraint:** The function must be stateless — it must not modify any global state.

---

### FR-15: REST API

**Description:** The system shall expose the inference pipeline via a REST API.

**Framework:** FastAPI (preferred) or Flask  
**Endpoint:** `POST /predict`  
**Request Body (JSON):**
```json
{ "text": "I am so angry right now" }
```
**Response Body:** Identical to the output schema in FR-14.  
**Error Response (400):** Returned when `text` is missing or empty:
```json
{ "error": "text field is required and must be non-empty" }
```
**Health Check Endpoint:** `GET /health` returns `{ "status": "ok", "model_loaded": true }`

---

### FR-16: Ablation Study Module

**Description:** The system shall include a reusable ablation runner that can train and evaluate two model variants for comparison:

- **Variant A (Full Model):** BiLSTM + Attention (as specified)
- **Variant B (No Attention):** BiLSTM with `GlobalAveragePooling1D` instead of attention layer

**Output:** A side-by-side comparison table of test accuracy and weighted F1 for both variants.

**Purpose:** This is required to satisfy the scientific claim that the attention mechanism contributes positively to classification performance. Without this, the attention layer is an unjustified architectural choice.

---

## 4. Non-Functional Requirements

### NFR-01: Reproducibility

All random seeds must be set before any randomised operation. The following must appear at the top of every training script:
```python
import os, random, numpy as np, tensorflow as tf
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
With these seeds fixed, two independent training runs on identical hardware must produce identical final test accuracy.

### NFR-02: Performance

- Training (10 epochs, GPU): must complete within 30 minutes on NVIDIA T4 or equivalent
- Inference latency: single-text prediction must complete within 200ms on CPU
- Batch inference (1000 texts): must complete within 30 seconds on CPU

### NFR-03: Code Quality

- All modules must pass `flake8` linting with zero errors
- All public functions and classes must have docstrings (Google style)
- Type hints must be used for all function signatures
- No magic numbers in model code — all hyperparameters read from `config.py`

### NFR-04: Modularity

The codebase must be organised as a Python package with separate modules for each concern (see Section 9). No monolithic script. Each module must be independently importable without side effects.

### NFR-05: Error Handling

- All file I/O operations must be wrapped in `try/except` with informative messages
- Model loading must validate that `config.json` hyperparameters match the loaded model's architecture
- API must return HTTP 422 for malformed JSON, 400 for validation errors, 500 with a sanitised message for internal errors

### NFR-06: Logging

Use Python's `logging` module (not `print`). Log level: INFO by default, DEBUG available via environment variable `LOG_LEVEL=DEBUG`. All training steps, data statistics, and evaluation results must be logged.

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        Input Layer                            │
│    Raw Text String (CLI / API / Web Interface)                │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│                    Preprocessing Pipeline                     │
│   Text Cleaning → Tokenization → Padding                      │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│                     Neural Network                            │
│   Embedding Layer (GloVe, frozen)                             │
│        ↓                                                      │
│   Bidirectional LSTM (128 units, return_sequences=True)       │
│        ↓                                                      │
│   Attention Layer (Bahdanau, custom Keras layer)              │
│        ↓                                                      │
│   Dense (64 units, ReLU) → Dropout (0.5)                     │
│        ↓                                                      │
│   Dense (6 units, Softmax)                                    │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│                     Output Layer                              │
│   Predicted Emotion + Probability Distribution                │
└───────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow Diagram

```
emotion.csv
    │
    ▼ (data_loader.py)
[raw_texts, raw_labels]
    │
    ▼ (preprocessor.py)
[cleaned_texts] + [encoded_labels]
    │
    ▼ (train_test_split, stratified)
[X_train, X_test, y_train, y_test]
    │
    ▼ (tokenizer.fit on X_train only)
[X_train_seq, X_test_seq] + tokenizer.pkl
    │
    ▼ (pad_sequences)
[X_train_pad, X_test_pad]
    │
    ▼ (embedding_builder.py)
[embedding_matrix]  (20000+1, 100)
    │
    ▼ (model.py)
[compiled BiLSTM-Attention model]
    │
    ▼ (trainer.py)
[best_model.h5 + training_log.csv]
    │
    ▼ (evaluator.py)
[metrics report + confusion matrix + curves]
```

---

## 6. Data Requirements

### 6.1 Dataset Specification

| Property | Value |
|----------|-------|
| Name | Kaggle Emotion Dataset |
| File | emotion.csv |
| Delimiter | Semicolon (`;`) |
| Columns | `text` (string), `label` (string) |
| Total Records | ~20,000 (verify on download) |
| Classes | joy, sadness, anger, fear, love, surprise |
| Language | English |

### 6.2 Expected Class Distribution

The dataset is known to be imbalanced. Approximate distribution:

| Emotion | Approx. % |
|---------|-----------|
| Joy | ~30% |
| Sadness | ~25% |
| Anger | ~15% |
| Fear | ~10% |
| Love | ~10% |
| Surprise | ~10% |

**Imbalance Handling:** Use `class_weight` in `model.fit()` computed via `sklearn.utils.class_weight.compute_class_weight`. Do not over-sample or under-sample — this would alter the natural distribution. The class weights must be logged.

### 6.3 Data Integrity Checks

Before training, assert:
- [ ] No duplicate (text, label) pairs
- [ ] No empty strings after cleaning
- [ ] Exactly six unique labels
- [ ] Minimum 100 samples per class in training split
- [ ] No label encoded as NaN

---

## 7. Model Requirements

### 7.1 Architecture Hyperparameters (Central Config)

All of the following must be defined in `config.py` as module-level constants:

```python
VOCAB_SIZE       = 20000
MAX_LEN          = 100
EMBEDDING_DIM    = 100
LSTM_UNITS       = 128
ATTENTION_DIM    = 128
DENSE_UNITS      = 64
DROPOUT_RATE     = 0.5
NUM_CLASSES      = 6
BATCH_SIZE       = 32
EPOCHS           = 10
LEARNING_RATE    = 1e-3
VALIDATION_SPLIT = 0.2
SEED             = 42
GLOVE_PATH       = './glove/glove.6B.100d.txt'
DATA_PATH        = './data/emotion.csv'
ARTEFACT_DIR     = './artefacts/'
```

### 7.2 Minimum Acceptable Performance

| Metric | Minimum Threshold |
|--------|-----------------|
| Test Accuracy | ≥ 87% |
| Weighted F1 | ≥ 0.86 |
| Per-class F1 (all classes) | ≥ 0.70 |
| Macro F1 | ≥ 0.82 |

If any metric falls below these thresholds, the training run is considered failed and the failure must be logged with diagnostic information.

### 7.3 Overfitting Detection

Training is considered overfit if:  
`(training_accuracy - validation_accuracy) > 0.10` at any epoch after epoch 3.

The `EarlyStopping` callback addresses this, but the evaluator must also check and log a warning if the final gap exceeds this threshold.

---

## 8. Interface Requirements

### 8.1 Command-Line Interface

The system shall expose the following CLI entry points:

```bash
# Training
python -m emotion_detector.train --data ./data/emotion.csv --output ./artefacts/

# Evaluation
python -m emotion_detector.evaluate --model ./artefacts/best_model.h5

# Inference (single text)
python -m emotion_detector.infer --text "I am so happy today"

# Ablation
python -m emotion_detector.ablation --data ./data/emotion.csv
```

### 8.2 REST API Interface

| Endpoint | Method | Request | Response |
|----------|--------|---------|---------|
| `/predict` | POST | `{"text": "..."}` | Prediction dict (see FR-14) |
| `/health` | GET | — | `{"status": "ok"}` |
| `/classes` | GET | — | `{"classes": ["anger", "fear", ...]}` |

### 8.3 Web Demo Interface

A minimal single-page HTML interface shall be provided (optionally using Flask's template engine) with:
- A text input area
- A "Predict" button
- An output section displaying the predicted emotion label and a bar chart of class probabilities
- No external CSS frameworks required — plain HTML + minimal inline CSS acceptable

---

## 9. Module-Level Implementation Specification

### 9.1 Package Structure

```
emotion_detector/
│
├── config.py                    # All hyperparameters (FR-07, FR-11)
│
├── data/
│   ├── data_loader.py           # FR-01: CSV loading
│   └── preprocessor.py         # FR-02, FR-03, FR-04, FR-05, FR-06
│
├── embedding/
│   └── embedding_builder.py     # FR-07: GloVe matrix construction
│
├── model/
│   ├── attention.py             # FR-09: Custom AttentionLayer class
│   └── model_builder.py         # FR-08, FR-10: Architecture + compilation
│
├── training/
│   └── trainer.py               # FR-11, FR-12: Training loop + callbacks
│
├── evaluation/
│   └── evaluator.py             # FR-13: Metrics, curves, confusion matrix
│
├── inference/
│   └── predictor.py             # FR-14: predict_emotion() function
│
├── api/
│   ├── app.py                   # FR-15: FastAPI/Flask app definition
│   └── schemas.py               # Request/response Pydantic models
│
├── ablation/
│   └── ablation_runner.py       # FR-16: Variant A vs Variant B
│
├── utils/
│   ├── logger.py                # NFR-06: Logging configuration
│   └── seed.py                  # NFR-01: Reproducibility seeds
│
├── __main__.py                  # Entry point dispatcher
└── tests/
    ├── test_preprocessor.py
    ├── test_attention.py
    ├── test_predictor.py
    └── test_api.py
```

### 9.2 Module Contracts

#### `data/data_loader.py`
```python
def load_dataset(filepath: str) -> tuple[list[str], list[str]]:
    """Load emotion CSV and return (texts, labels)."""
```

#### `data/preprocessor.py`
```python
def clean_text(text: str) -> str:
    """Apply deterministic text cleaning pipeline."""

def encode_labels(labels: list[str], save_path: str) -> np.ndarray:
    """Fit LabelEncoder, save to disk, return encoded array."""

def split_data(X, y, test_size=0.2, seed=42) -> tuple:
    """Stratified train/test split."""

def fit_tokenizer(X_train: list[str], save_path: str) -> Tokenizer:
    """Fit tokenizer on training data only and save."""

def tokenize_and_pad(X: list[str], tokenizer: Tokenizer, maxlen: int) -> np.ndarray:
    """Convert text to padded integer sequences."""
```

#### `embedding/embedding_builder.py`
```python
def build_embedding_matrix(glove_path: str, word_index: dict,
                            vocab_size: int, embedding_dim: int) -> np.ndarray:
    """Construct GloVe embedding matrix; log OOV statistics."""
```

#### `model/attention.py`
```python
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs): ...
    def build(self, input_shape): ...
    def call(self, inputs) -> tuple[tf.Tensor, tf.Tensor]:
        """Returns (context_vector, attention_weights)."""
    def get_config(self) -> dict: ...
```

#### `model/model_builder.py`
```python
def build_model(vocab_size: int, embedding_dim: int, embedding_matrix: np.ndarray,
                maxlen: int, lstm_units: int, attention_dim: int,
                dense_units: int, dropout_rate: float, num_classes: int) -> tf.keras.Model:
    """Construct and compile BiLSTM-Attention model."""
```

#### `training/trainer.py`
```python
def train(model: tf.keras.Model, X_train, y_train,
          epochs: int, batch_size: int, val_split: float,
          artefact_dir: str, class_weights: dict) -> tf.keras.callbacks.History:
    """Run training loop with all required callbacks."""
```

#### `evaluation/evaluator.py`
```python
def evaluate(model: tf.keras.Model, X_test, y_test,
             label_encoder, history, artefact_dir: str) -> dict:
    """Generate full evaluation report; save plots."""
```

#### `inference/predictor.py`
```python
def predict_emotion(text: str, model, tokenizer, label_encoder,
                    maxlen: int) -> dict:
    """Run end-to-end inference; return structured dict."""
```

---

## 10. Testing and Validation Requirements

### 10.1 Unit Tests

| Test | Module | What to Assert |
|------|--------|---------------|
| `test_clean_text` | preprocessor | Output is lowercase, no URLs, no HTML, no special chars |
| `test_clean_text_empty` | preprocessor | Empty string input returns empty string, not error |
| `test_stratified_split` | preprocessor | Class distribution in test split within ±2% of training split |
| `test_tokenizer_no_leakage` | preprocessor | Words only in test set are encoded as OOV index |
| `test_padding_shape` | preprocessor | All output sequences have shape (N, 100) |
| `test_embedding_matrix_shape` | embedding_builder | Matrix shape is (vocab_size+1, 100) |
| `test_embedding_oov_rate` | embedding_builder | OOV rate is logged correctly |
| `test_attention_output_shapes` | attention | Context vector shape is (batch, 256); weights shape is (batch, 100, 1) |
| `test_attention_weights_sum_to_one` | attention | Sum of attention weights over time axis ≈ 1.0 per sample |
| `test_model_output_shape` | model_builder | Output shape is (batch, 6) |
| `test_model_output_is_probability` | model_builder | All outputs in [0, 1] and sum to 1.0 per sample |
| `test_predict_emotion_schema` | predictor | Output dict contains exactly the keys specified in FR-14 |
| `test_predict_emotion_deterministic` | predictor | Same input always returns same output |
| `test_api_predict_valid` | api | Returns 200 for valid JSON input |
| `test_api_predict_missing_field` | api | Returns 400 for missing `text` field |
| `test_api_health` | api | Returns 200 with `status: ok` |

### 10.2 Integration Tests

- **End-to-end training test:** Run training for 2 epochs on a 500-sample subset. Assert `best_model.h5` and all artefacts are created. Assert test accuracy is above 40% (random baseline is ~16%).
- **Artefact reload test:** Load saved model, tokenizer, and label encoder. Run inference on 10 samples. Assert output is identical to inference run before saving.

### 10.3 Regression Baseline

After the first successful full training run, record the exact test accuracy and weighted F1. These values become the regression baseline. Any future change to the model or preprocessing pipeline must be justified if it causes a drop in either metric.

---

## 11. Deployment Requirements

### 11.1 Environment

A `requirements.txt` file must be provided and pinned to exact versions:

```
tensorflow==2.13.0
numpy==1.24.3
scikit-learn==1.3.0
pandas==2.0.3
fastapi==0.103.0
uvicorn==0.23.2
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
pydantic==2.3.0
```

### 11.2 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTEFACT_DIR` | `./artefacts/` | Path to saved model and tokenizer |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `PORT` | `8000` | API server port |

### 11.3 API Launch Command

```bash
uvicorn emotion_detector.api.app:app --host 0.0.0.0 --port $PORT
```

### 11.4 Optional Docker Specification

A `Dockerfile` should be provided that:
- Uses `python:3.9-slim` as base
- Copies `requirements.txt` and installs dependencies
- Copies the `artefacts/` directory (pre-trained weights)
- Exposes port 8000
- Sets `CMD` to launch uvicorn

---

## 12. Appendices

### Appendix A: Attention Layer Full Implementation Reference

```python
import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    """Bahdanau-style (additive) attention layer.

    Computes a context vector as a weighted sum of BiLSTM hidden states,
    where weights reflect the relative importance of each timestep.

    Args:
        units: Dimensionality of the attention space.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_W',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_b',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.v = self.add_weight(
            name='attention_v',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        # score: (batch, timesteps, units)
        attention_weights = tf.nn.softmax(
            tf.matmul(score, self.v), axis=1
        )
        # attention_weights: (batch, timesteps, 1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        # context_vector: (batch, features)
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config
```

### Appendix B: Evaluation Report Template

The evaluator must produce a plain-text report file `evaluation_report.txt` with the following structure:

```
======================================================
EMOTION DETECTION MODEL — EVALUATION REPORT
======================================================
Date           : <ISO timestamp>
Model          : best_model.h5
Test samples   : <N>

OVERALL METRICS
---------------
Test Accuracy  : 0.XXXX
Macro F1       : 0.XXXX
Weighted F1    : 0.XXXX

PER-CLASS METRICS
-----------------
Class       Precision  Recall  F1-Score  Support  AUROC
anger       0.XXXX     0.XXXX  0.XXXX    XXXX     0.XXXX
fear        0.XXXX     0.XXXX  0.XXXX    XXXX     0.XXXX
joy         0.XXXX     0.XXXX  0.XXXX    XXXX     0.XXXX
love        0.XXXX     0.XXXX  0.XXXX    XXXX     0.XXXX
sadness     0.XXXX     0.XXXX  0.XXXX    XXXX     0.XXXX
surprise    0.XXXX     0.XXXX  0.XXXX    XXXX     0.XXXX

OVERFITTING CHECK
-----------------
Final train accuracy : 0.XXXX
Final val accuracy   : 0.XXXX
Gap                  : 0.XXXX  [PASS / WARNING]

ARTEFACTS SAVED
---------------
./artefacts/best_model.h5
./artefacts/tokenizer.pkl
./artefacts/label_encoder.pkl
./artefacts/accuracy_curve.png
./artefacts/loss_curve.png
./artefacts/confusion_matrix.png
./artefacts/training_log.csv
======================================================
```

### Appendix C: Known Failure Modes and Mitigations

| Failure Mode | Symptom | Mitigation |
|-------------|---------|-----------|
| Data leakage | Unrealistically high test accuracy (>98%) | Ensure tokenizer is fit on X_train only |
| Gradient vanishing | Loss stops decreasing after epoch 1 | Check recurrent_dropout; reduce LSTM units |
| OOM on GPU | Training crashes mid-epoch | Reduce batch_size to 16; use `tf.data` pipeline |
| GloVe file not found | `FileNotFoundError` at embedding step | Validate `GLOVE_PATH` before training starts |
| Class confusion (love/joy) | Low F1 for love class | Expected; note in paper; consider subword features |
| Model not loading after save | `Unknown layer: AttentionLayer` | Pass `custom_objects` to `load_model` |

### Appendix D: Ablation Experiment Design

Two models must be trained identically except for the pooling strategy after BiLSTM:

| Configuration | Pooling | Expected Weighted F1 |
|--------------|---------|---------------------|
| A: Full Model | Bahdanau Attention | Baseline |
| B: No Attention | GlobalAveragePooling1D | Expected ~2–4% lower |

Both must be trained for the same number of epochs with identical random seeds. The comparison table must include: test accuracy, macro F1, weighted F1, and training time. If Variant B outperforms Variant A, the paper's claim that attention improves performance is falsified and must be revised.

---

*End of SRS — Version 1.0*
