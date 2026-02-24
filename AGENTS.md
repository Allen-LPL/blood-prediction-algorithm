# AGENTS.md
Guidance for agentic coding in `blood_bank_predict`.

## Repository Snapshot
- Python script-style ML repository; main code in `lstm/`.
- Current primary training pipeline: `lstm/blood_lstm_xgb_v3_2.py`.
- Historical/experimental scripts remain in `lstm/` (`v1/v2/v3`, `lstm_train_*`, `predict_v1.py`).
- Large generated artifacts already exist under `lstm/outputs_v3/`.

## Rules Files Check
- Cursor rules: not found (`.cursor/rules/`, `.cursorrules`).
- Copilot instructions: not found (`.github/copilot-instructions.md`).

## Tooling/Config Check
- Not found: `pyproject.toml`, `requirements.txt`, `setup.py`, `pytest.ini`, `tox.ini`, `Makefile`.
- Not found: CI workflows in `.github/workflows/`.
- Practical implication: run scripts directly and use lightweight sanity checks.

## Key Paths
- `lstm/0_calendar_features.py`: generate calendar/weather features.
- `lstm/1_prepare_data.py`: read Excel and produce grouped CSV.
- `lstm/2_analyze_data.py`: exploratory plots/anomaly review.
- `lstm/blood_lstm_xgb_v3_2.py`: argparse-based train/eval/report pipeline.
- `lstm/feature/`: CSV inputs/features.
- `lstm/data/`: source Excel files.

## Runtime Dependencies (Inferred)
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn`
- `tensorflow`/`keras`
- `xgboost`
- `requests`
- `chinese_calendar`
- `joblib`

## Build/Lint/Test Commands
There is no formal build or test harness in this checkout.

### Data preparation
```bash
python lstm/0_calendar_features.py
python lstm/1_prepare_data.py
python lstm/2_analyze_data.py
```

### Main training (default)
```bash
python lstm/blood_lstm_xgb_v3_2.py
```

### Main training (custom)
```bash
python lstm/blood_lstm_xgb_v3_2.py \
  --blood_csv "./lstm/feature/remove_group_data.csv" \
  --calendar_csv "./lstm/feature/blood_calendar_weather_2015_2026.csv" \
  --out_dir "lstm/outputs_v3" \
  --look_back 60 \
  --epochs 60 \
  --batch_size 64 \
  --stations "北京市红十字血液中心" \
  --blood_types "A,B,O,AB"
```

### Legacy scripts
```bash
python lstm/blood_lstm_xgb_v1.py
python lstm/blood_lstm_xgb_v2.py
python lstm/blood_lstm_xgb_v3.py
python lstm/lstm_train_v1.py
python lstm/lstm_train_v2.py
python lstm/predict_v1.py
```

### Sanity checks (recommended)
```bash
python -m compileall lstm
python -m py_compile lstm/blood_lstm_xgb_v3_2.py
```

### Optional lint checks (if installed)
```bash
python -m ruff check lstm
python -m black --check lstm
```

## Single-Test Command Reference
No test suite is currently present, but if tests are introduced:

### pytest (preferred)
```bash
pytest tests/test_pipeline.py
pytest tests/test_pipeline.py::test_train_one_group
pytest tests/test_pipeline.py::TestTrainer::test_split_indices
```

### unittest (fallback)
```bash
python -m unittest tests.test_pipeline
python -m unittest tests.test_pipeline.TestTrainer.test_split_indices
```

## Code Style Guidelines (Repository-Derived)

### Imports
- Order: stdlib -> third-party -> local.
- Prefer explicit imports; avoid wildcard imports.
- Avoid duplicate imports (seen in older files; do not propagate).

### Formatting
- Use 4 spaces for indentation.
- Keep functions small and focused (feature helpers, loaders, trainers).
- Keep script entrypoints in `if __name__ == "__main__":`.
- Add comments only for non-obvious logic.

### Naming
- `snake_case` for functions/variables.
- `UPPER_SNAKE_CASE` for module constants.
- `PascalCase` for class/dataclass names (`Cols` pattern).
- Preserve existing column name schema (`date`, `血站`, `血型`, `总血量`, etc.).

### Typing
- Follow `blood_lstm_xgb_v3_2.py` style: typed args and return types.
- Prefer concrete types for contracts (`pd.DataFrame`, `np.ndarray`, `List[str]`, `Tuple[...]`).
- Use dataclasses for stable column mappings.

### Error Handling and Validation
- Validate required columns early; fail fast with clear `ValueError`.
- Guard against short datasets before sequence generation/training.
- Use numerical guards (`eps`, clipping, floor) for ratio/metric stability.
- Never swallow exceptions silently.

### Time-Series/ML Conventions
- Sort by date before lag/rolling/sequence operations.
- Keep split strategy chronological (no random shuffling for time-series).
- Prevent leakage: do not include future target data in features.
- Keep outputs organized (`reports/`, `figs/`, `models_lstm/`, `models_xgb/`).

### Plotting and Automation
- Existing scripts configure Chinese fonts for plots.
- Prefer save-to-file over blocking `plt.show()` for automation/CI contexts.

## Agent Working Rules for This Repo
- Default edits to `lstm/blood_lstm_xgb_v3_2.py` for active production logic.
- Treat `v1/v2/v3` variants as reference unless task explicitly targets them.
- Keep feature schema changes synchronized across prep and train scripts.
- Do not commit generated model binaries/figures/reports unless explicitly requested.
- If tests are added, create `tests/` and standardize on pytest.

## Basis Files
- `lstm/blood_lstm_xgb_v3_2.py`
- `lstm/0_calendar_features.py`
- `lstm/1_prepare_data.py`
- `lstm/2_analyze_data.py`
- `lstm/blood_lstm_xgb_v1.py`
- `lstm/blood_lstm_xgb_v2.py`
- `lstm/blood_lstm_xgb_v3.py`
- `lstm/lstm_train_v2.py`
- `lstm/predict_v1.py`
