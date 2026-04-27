import importlib.util
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "lstm" / "blood_lstm_xgb_v5(resid).py"
    spec = importlib.util.spec_from_file_location("blood_lstm_xgb_v5_resid", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_eval_df(start="2023-03-01", periods=500):
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "\u603b\u8840\u91cf": range(periods),
            "y_xgb": range(periods),
            "y_hat": range(periods),
            "y_hat_adaptive": range(periods),
        }
    )


def test_build_trailing_windows_returns_expected_lengths():
    mod = load_module()
    eval_df = make_eval_df()
    windows = mod.build_trailing_windows(eval_df, [7, 30, 90])
    assert [item["window_name"] for item in windows] == [
        "trailing_7d",
        "trailing_30d",
        "trailing_90d",
    ]
    assert [len(item["df"]) for item in windows] == [7, 30, 90]


def test_build_shifted_range_windows_matches_prior_year_dates():
    mod = load_module()
    eval_df = make_eval_df()
    current_df = eval_df[
        (eval_df["date"] >= "2024-03-01") & (eval_df["date"] <= "2024-06-30")
    ].copy()
    windows = mod.build_shifted_range_windows(eval_df, current_df, shifts=[0, 1])
    names = [item["window_name"] for item in windows]
    assert names == ["current_range", "minus_1y_same_range"]
    assert windows[1]["start_date"] == "2023-03-01"
    assert windows[1]["end_date"] == "2023-06-30"


def test_summarize_window_metrics_returns_expected_keys():
    mod = load_module()
    eval_df = make_eval_df(periods=30)
    summary = mod.summarize_window_metrics(
        "demo_group", "trailing_30d", "trailing_length", eval_df
    )
    assert summary["group"] == "demo_group"
    assert summary["window_name"] == "trailing_30d"
    assert summary["sample_count"] == 30
    assert "mae_xgb" in summary
    assert "mape_hat_adaptive" in summary


def test_export_window_outputs_writes_csv_and_json(tmp_path):
    mod = load_module()
    eval_df = make_eval_df(periods=10)
    summary = mod.summarize_window_metrics(
        "demo_group", "trailing_7d", "trailing_length", eval_df.tail(7)
    )
    mod.export_window_tabular_outputs(tmp_path, "demo_group", summary, eval_df.tail(7))
    assert (tmp_path / "trailing_7d_eval.csv").exists()
    assert (tmp_path / "summary_metrics.json").exists() is False


def test_export_window_plots_writes_two_pngs(tmp_path):
    mod = load_module()
    eval_df = make_eval_df(periods=20)
    mod.export_window_plots(
        tmp_path, "demo_group", "trailing_7d", eval_df.tail(7), 1.0, 2.0, 3.0
    )
    assert (tmp_path / "trailing_7d_prediction_comparison.png").exists()
    assert (tmp_path / "trailing_7d_residual_tracking.png").exists()


def test_export_all_eval_windows_creates_expected_artifacts(tmp_path):
    mod = load_module()
    eval_df = make_eval_df(periods=500)
    mod.export_all_eval_windows(tmp_path, "demo_group", eval_df)
    assert (tmp_path / "summary_metrics.csv").exists()
    assert (tmp_path / "current_range_prediction_comparison.png").exists()
    assert (tmp_path / "minus_1y_same_range_eval.csv").exists()
    assert (tmp_path / "trailing_30d_prediction_comparison.png").exists()
