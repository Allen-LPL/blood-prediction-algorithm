# AGENTS.md
Guidance for agentic coding in `blood_bank_predict`.

## Repository Snapshot
- Python ML repository containing two main sub-projects:
  1. `lstm/`: Time-series forecasting pipeline using LSTM and XGBoost for blood bank predictions.
  2. `text_classify/`: NLP pipeline using BERT/HuggingFace and PyTorch for hierarchical text classification.

## Rules Files Check
- Cursor rules: not found (`.cursor/rules/`, `.cursorrules`).
- Copilot instructions: not found (`.github/copilot-instructions.md`).

## Tooling/Config Check
- Not found: `pyproject.toml`, `requirements.txt`, `setup.py`, `pytest.ini`, `tox.ini`, `Makefile`.
- Not found: CI workflows in `.github/workflows/`.
- Practical implication: run scripts directly and use lightweight sanity checks.

## Key Paths
### `lstm/` (Time-Series Forecasting)
- `lstm/0_calendar_features.py`: generate calendar/weather features.
- `lstm/1_prepare_data.py`: read Excel and produce grouped CSV.
- `lstm/2_analyze_data.py`: exploratory plots/anomaly review.
- `lstm/blood_lstm_xgb_v3_2.py`: argparse-based train/eval/report pipeline.
- `lstm/feature/`, `lstm/data/`, `lstm/outputs_v3/`: Data, features, and model artifacts.

### `text_classify/text_classfier/` (NLP Classification)
- `0_prepare_data.py`: Data ingestion and preprocessing.
- `1_bert_train.py`: Main BERT training script.
- `2_bert_predict.py`: Inference pipeline.
- `train_hierarchical.py`, `train.py`: Specialized training approaches.
- Shell scripts: `train.sh`, `validate.sh`, `train_cat1.sh`, etc.

## Runtime Dependencies (Inferred)
- **Time-Series (`lstm`)**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`/`keras`, `xgboost`, `chinese_calendar`, `joblib`
- **NLP (`text_classify`)**: `torch` (PyTorch), `transformers` (HuggingFace), `google-cloud-storage`, `tqdm`, `pandas`, `scikit-learn`

## Build/Lint/Test Commands
There is no formal build or test harness in this checkout.

### Data preparation
```bash
# LSTM
python lstm/0_calendar_features.py
python lstm/1_prepare_data.py
# Text Classification
cd text_classify/text_classfier && python 0_prepare_data.py
```

### Main training
```bash
# LSTM
python lstm/blood_lstm_xgb_v3_2.py
# Text Classification
cd text_classify/text_classfier && ./train.sh
cd text_classify/text_classfier && python 1_bert_train.py
```

### Sanity checks (recommended)
```bash
python -m compileall lstm text_classify
python -m py_compile lstm/blood_lstm_xgb_v3_2.py text_classify/text_classfier/1_bert_train.py
```

## Single-Test Command Reference
No test suite is currently present, but if tests are introduced:

### pytest (preferred)
```bash
pytest tests/test_pipeline.py
pytest tests/test_pipeline.py::test_train_one_group
```

### unittest (fallback)
```bash
python -m unittest tests.test_pipeline
```

## Code Style Guidelines (Repository-Derived)

### Imports
- Order: stdlib -> third-party -> local.
- Prefer explicit imports; avoid wildcard imports.

### Formatting & Structure
- Use 4 spaces for indentation.
- Keep functions small and focused (feature helpers, loaders, trainers).
- Script entrypoints should be guarded with `if __name__ == "__main__":`.
- Use `argparse` for configuring pipeline execution.
- Add comments only for non-obvious logic.

### Naming
- `snake_case` for functions/variables.
- `UPPER_SNAKE_CASE` for module constants.
- `PascalCase` for class/dataclass names.

### Typing
- Typed args and return types are strongly preferred (`Dict`, `List`, `Tuple`).
- Use `dataclass` for configuration or column mappings.
- Prefer concrete types for contracts (`pd.DataFrame`, `np.ndarray`, `torch.Tensor`).

### Error Handling, Logging, and Validation
- Use the `logging` module rather than bare `print()` for training status.
- Validate required columns or parameters early; fail fast with `ValueError`.
- Guard against short datasets before sequence generation/training.
- Never swallow exceptions silently.

### ML & NLP Conventions
- Time-series: Sort by date before sequence generation. Prevent temporal leakage.
- NLP: Set random seeds for reproducibility (`torch.manual_seed`, `np.random.seed`).
- HuggingFace: Set `HF_ENDPOINT` appropriately (e.g., `https://hf-mirror.com`) if required.
- Use `tqdm` for training loops.
- Avoid committing large generated model binaries, checkpoints, or figures.

## Agent Working Rules for This Repo
- Default edits to `lstm/blood_lstm_xgb_v3_2.py` for time-series and `text_classify/text_classfier/1_bert_train.py` for NLP unless specified.
- Keep feature schema changes synchronized across prep and train scripts.
- Do not commit large artifacts or weights.
- If tests are added, create `tests/` and standardize on `pytest`.
