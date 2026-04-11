# AGENTS.md

Blood bank donation volume prediction. Three modeling approaches share a common data pipeline. Related patent drafts in `doc/`.

## Repository Structure

```
lstm/           LSTM + XGBoost ensemble (primary, many versioned scripts)
lstm_cnn/       Conv1D + LSTM hybrid (argparse-driven, most structured)
xgb/            Standalone XGBoost models
doc/            Patent drafts (Chinese, not code)
```

All three consume prepared CSVs from `lstm/feature/` (the shared data hub).

## Critical: Working Directory

Every script uses relative paths. **Run each script from its own directory.**

```bash
# LSTM scripts — CWD must be lstm/
cd lstm && python blood_lstm_xgb_v5\(resid\).py

# CNN+LSTM scripts — CWD must be lstm_cnn/
cd lstm_cnn && python train_2025_forecast.py

# XGBoost scripts — CWD must be xgb/
cd xgb && python xgb_single_v2_n.py
```

`xgb/` and `lstm_cnn/` scripts reference data via `../lstm/feature/...`. Running from the wrong directory silently breaks paths.

## Data Pipeline (must run in order)

```bash
cd lstm
python 0_calendar_features.py          # → feature/blood_calendar_weather_2015_2026.csv
python 1_prepare_data.py               # → feature/all_data.csv  (requires ./data/*.xlsx)
python 1_prepare_data_merge.py         # → feature/all_data_merge.csv, feature/remove_group_merge.csv
```

Raw Excel files go in `lstm/data/` (gitignored). Prepared CSVs in `lstm/feature/` are tracked.

### Feature CSV schemas

| File | Columns |
|------|---------|
| `remove_group_data.csv` | `date, 血站, 血型, 总血量` |
| `all_data.csv` | `date, 血站, 血型, 总血量` |
| `blood_calendar_weather_2015_2026.csv` | `date, weekday, is_weekend, is_holiday, is_workday, year, month, day, dayofyear, weekofyear, weekday_sin, weekday_cos, month_sin, month_cos, 天气, 平均温度, weather_code` |

Column names are Chinese. `血站` = blood station, `血型` = blood type, `总血量`/`血量` = total blood volume, `天气` = weather, `平均温度` = average temperature.

## Recommended Entrypoints

Use the latest stable script in each directory unless the user specifies otherwise.

| Directory | Latest Script | CLI |
|-----------|--------------|-----|
| `lstm/` | `blood_lstm_xgb_v5(resid).py` | argparse |
| `lstm/` | `blood_lstm_xgb_v6(weight).py` | argparse |
| `lstm_cnn/` | `train_2025_forecast.py` | argparse |
| `xgb/` | `xgb_single_v2_n.py` | argparse |

Older `lstm/` scripts (v1–v4) use hardcoded module constants instead of argparse. `blood_lstm_xgb_v3_2.py` is the baseline reference.

### lstm_cnn CLI example (most configurable)

```bash
cd lstm_cnn
python train_2025_forecast.py \
  --blood_csv ../lstm/feature/remove_group_data.csv \
  --weather_csv ../lstm/feature/blood_calendar_weather_2015_2026.csv \
  --out_dir ./out_group_models_2025_forecast \
  --station "北京市红十字血液中心" \
  --blood_types "ALL,A,B,O,AB" \
  --lookback 60 --horizon 7 --epochs 140
```

## Dependencies (no requirements.txt — install manually)

```
numpy pandas matplotlib scikit-learn tensorflow xgboost chinese_calendar joblib seaborn
```

`chinese_calendar` provides `is_holiday`, `get_holiday_detail`, and `Holiday` enum for Chinese public holidays.

Recommended: use a virtualenv. The `1_prepare_data.py` footer references `~/tf_arm_env/bin/activate`.

## Platform Quirk: Chinese Font

Nearly every script hardcodes a macOS Chinese font:

```python
FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
```

This will crash on Linux. `lstm_cnn/train_2025_forecast.py` is the only script that wraps this in a try/except. When adding plots, guard the font path or use `FONT = None` with a fallback.

## Domain-Specific Patterns

- **PKU semester calendar**: v5, v6, and xgb scripts embed Peking University semester start dates (2015–2026) as a feature. The `PKU_CLASS_STARTS` list must be updated for future years.
- **Holiday features**: `chinese_calendar` only covers 2004–2026. Scripts using `get_holiday_detail` will fail for dates outside this range.
- **Station filtering**: Most training scripts filter to `北京市红十字血液中心` (Beijing Red Cross Blood Center). Check before assuming multi-station support.
- **Blood type loop**: Training iterates over `["ALL", "A", "B", "O", "AB"]`. `"ALL"` aggregates all types by summing.
- **Duplicate utilities**: `is_pku_semester_start_season`, `is_holiday_name`, `add_cn_holiday_flags` are copy-pasted across `lstm/v5`, `lstm/v6`, `xgb/xgb_single_v1.py`, `xgb/xgb_single_v2_n.py`. Changes must be applied to all copies.

## Output Directories

| Script family | Default output dir | Contents |
|--------------|-------------------|----------|
| `lstm/v1–v4` | `lstm/models_lstm_xgb/` | `{group}_lstm.keras`, `{group}_xgb.json` |
| `lstm/v5–v6` | `lstm/patent_figures/` | Patent-quality PNGs (tracked in git) |
| `lstm_cnn/train_v1` | `lstm_cnn/out_group_models/` | per-group: `best_model.keras`, `metrics.json`, predictions CSV |
| `lstm_cnn/train_v2` | `lstm_cnn/out_group_models_v2/` | same structure |
| `lstm_cnn/train_2025_forecast` | `lstm_cnn/out_group_models_2025_forecast/` | same + `metrics_summary.json` |

Do not commit model outputs. Only `lstm/patent_figures/` and `lstm/feature/` CSVs are intentionally tracked.

## Sanity Check

```bash
python -m py_compile lstm/blood_lstm_xgb_v5\(resid\).py
python -m py_compile lstm_cnn/train_2025_forecast.py
python -m py_compile xgb/xgb_single_v2_n.py
```

No test suite exists. Use `py_compile` to verify syntax after edits.

## Agent Rules

- Default edits to latest scripts: `lstm/blood_lstm_xgb_v5(resid).py` or `lstm/blood_lstm_xgb_v6(weight).py` for LSTM, `lstm_cnn/train_2025_forecast.py` for CNN, `xgb/xgb_single_v2_n.py` for XGBoost.
- Duplicate utility functions exist across directories. When modifying shared logic (holiday flags, PKU calendar, MAPE functions), update all copies.
- Keep feature schema synchronized between prep scripts (`lstm/1_prepare_data*.py`) and training scripts.
- Do not commit `.keras`, `.h5`, `.pkl` model files or output directories.
- Escape parentheses in filenames: `blood_lstm_xgb_v5\(resid\).py`.
