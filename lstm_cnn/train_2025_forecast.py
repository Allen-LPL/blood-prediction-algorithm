import argparse
import json
import os
import re
from datetime import timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models

try:
    from matplotlib.font_manager import FontProperties

    FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
except Exception:
    FONT = None


def slugify(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", s)
    return s[:120]


def masked_mape(y_true: np.ndarray, y_pred: np.ndarray, min_true: float = 1.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true >= min_true
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def build_delta_sample_weights(y_delta: np.ndarray, alpha: float = 2.0, quantile: float = 0.75) -> np.ndarray:
    if y_delta.ndim != 2 or len(y_delta) == 0:
        return np.ones((len(y_delta),), dtype=np.float32)

    mag = np.mean(np.abs(y_delta), axis=1)
    q = float(np.quantile(mag, np.clip(quantile, 0.0, 1.0)))
    if not np.isfinite(q) or q <= 1e-8:
        return np.ones((len(y_delta),), dtype=np.float32)

    extra = np.maximum((mag - q) / (q + 1e-8), 0.0)
    w = 1.0 + float(max(alpha, 0.0)) * extra
    return w.astype(np.float32)


def build_anti_persistence_loss(
        huber_delta: float = 1.0,
        ap_lambda: float = 1.0,
        ap_change_thr: float = 0.6,
        ap_margin: float = 0.6,
):
    huber = tf.keras.losses.Huber(delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)

    def loss_fn(y_true, y_pred):
        base = huber(y_true, y_pred)
        base = tf.reduce_mean(base)

        if ap_lambda <= 0:
            return base

        change_mask = tf.cast(tf.abs(y_true) > ap_change_thr, tf.float32)
        near_zero_penalty = tf.square(tf.nn.relu(ap_margin - tf.abs(y_pred)))
        anti_persist = tf.reduce_sum(change_mask * near_zero_penalty) / (tf.reduce_sum(change_mask) + 1e-6)
        return base + ap_lambda * anti_persist

    return loss_fn


def build_dynamic_feature(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()

    holidays = set(df.loc[df.get("is_holiday", 0) == 1, "date"])
    df["is_before_holiday"] = df["date"].apply(lambda x: 1 if (x + timedelta(days=1)) in holidays else 0)
    df["is_after_holiday"] = df["date"].apply(lambda x: 1 if (x - timedelta(days=1)) in holidays else 0)

    holiday_arr = np.array(sorted(list(holidays)), dtype="datetime64[ns]")
    dates_arr = df["date"].values.astype("datetime64[ns]")
    if len(holiday_arr) > 0:
        idx = np.searchsorted(holiday_arr, dates_arr)
        next_idx = np.clip(idx, 0, len(holiday_arr) - 1)
        prev_idx = np.clip(idx - 1, 0, len(holiday_arr) - 1)
        next_h = holiday_arr[next_idx]
        prev_h = holiday_arr[prev_idx]

        days_until = ((next_h - dates_arr) / np.timedelta64(1, "D")).astype(float)
        days_since = ((dates_arr - prev_h) / np.timedelta64(1, "D")).astype(float)

        days_until = np.where(idx >= len(holiday_arr), 999.0, days_until)
        days_since = np.where(idx <= 0, 999.0, days_since)

        days_until = np.maximum(days_until, 0.0)
        days_since = np.maximum(days_since, 0.0)
    else:
        days_until = np.full(len(df), 999.0, dtype=float)
        days_since = np.full(len(df), 999.0, dtype=float)

    df["days_until_holiday"] = np.clip(days_until, 0, 30)
    df["days_since_holiday"] = np.clip(days_since, 0, 30)
    df["near_holiday7"] = ((df["days_until_holiday"] <= 7) | (df["days_since_holiday"] <= 7)).astype(int)

    if "平均温度" in df.columns:
        df["temp_sq"] = df["平均温度"] ** 2
        df["temp_diff"] = df["平均温度"].diff()
        df["temp_rolling3"] = df["平均温度"].rolling(3).mean()

    if "天气" in df.columns:
        df = pd.get_dummies(df, columns=["天气"])

    y = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = y

    df["trend"] = y.rolling(7).mean()
    df["residual"] = y - df["trend"]
    df["lag1"] = y.shift(1)
    df["lag7"] = y.shift(7)
    df["lag14"] = y.shift(14)
    df["lag28"] = y.shift(28)
    df["rolling7"] = y.rolling(7).mean()
    df["rolling14"] = y.rolling(14).mean()
    df["rolling28"] = y.rolling(28).mean()
    df["ewm7"] = y.ewm(span=7, adjust=False).mean()
    df["ewm14"] = y.ewm(span=14, adjust=False).mean()
    df["absdiff1"] = (y - df["lag1"]).abs()
    df["absdiff7"] = (y - df["lag7"]).abs()
    df["diff7"] = y - df["lag7"]
    df["diff14"] = y - df["lag14"]
    df["rolling_std7"] = y.rolling(7).std()
    df["rolling_std14"] = y.rolling(14).std()
    df["rolling_absdiff7"] = df["absdiff1"].rolling(7).mean()

    return df


def build_group_daily_series(
        blood: pd.DataFrame,
        weather: pd.DataFrame,
        station: str,
        blood_type: str,
        target_col: str = "总血量",
):
    g = blood[(blood["血站"] == station) & (blood["血型"] == blood_type)].copy()
    if blood_type == "ALL":
        g = blood[blood["血站"] == station].copy()
        g = g.drop(columns="血型", errors="ignore")
        g = g.groupby(["date", "血站"], as_index=False).agg(总血量=("总血量", "sum"))

    if g.empty:
        return None

    g["date"] = pd.to_datetime(g["date"])
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"])

    date_min = g["date"].min()
    date_max = g["date"].max()
    all_dates = pd.date_range(date_min, date_max, freq="D")
    base = pd.DataFrame({"date": all_dates})

    g = g[["date", target_col]].copy()
    g[target_col] = pd.to_numeric(g[target_col], errors="coerce")

    df = base.merge(g, on="date", how="left")
    df[target_col] = df[target_col].fillna(0.0)
    df = df.merge(weather, on="date", how="left").sort_values("date").reset_index(drop=True)
    df = build_dynamic_feature(df, target_col)

    y_derived_prefix = (
    "lag", "rolling", "trend", "residual", "ewm", "absdiff", "diff", "rolling_std", "rolling_absdiff")
    y_cols = [c for c in df.columns if (c == target_col) or c.startswith(y_derived_prefix)]
    exo_cols = [c for c in df.columns if c not in (["date"] + y_cols)]

    for c in exo_cols + y_cols:
        if c == "date":
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if exo_cols:
        df[exo_cols] = df[exo_cols].ffill().bfill().fillna(0.0)
    df[y_cols] = df[y_cols].fillna(0.0)

    feature_cols = [c for c in df.columns if c not in ["date", "lag1", target_col]]
    return df, feature_cols


def make_windows_multistep_delta(
        df: pd.DataFrame,
        feature_cols,
        target_col: str,
        lookback: int,
        horizon: int,
):
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    tgt = df[target_col].to_numpy(dtype=np.float32)
    tgt_log = np.log1p(np.maximum(tgt, 0.0)).astype(np.float32)
    dates = df["date"].to_numpy(dtype="datetime64[ns]")

    x_list, y_list, base_list, sd_list, ed_list = [], [], [], [], []
    max_t = len(df) - horizon
    for t in range(lookback, max_t):
        x_list.append(feat[t - lookback:t])
        base = tgt_log[t - 1]
        y_f = tgt_log[t: t + horizon]
        y_delta = y_f - base
        y_list.append(y_delta)
        base_list.append(base)
        sd_list.append(dates[t])
        ed_list.append(dates[t + horizon - 1])

    x = np.asarray(x_list, dtype=np.float32)
    y_delta = np.asarray(y_list, dtype=np.float32)
    base_log = np.asarray(base_list, dtype=np.float32)
    start_dates = np.asarray(sd_list)
    end_dates = np.asarray(ed_list)
    return x, y_delta, base_log, start_dates, end_dates


def split_pre_2024(x, y, base_log, start_dates, end_dates, test_ratio: float):
    cutoff_2024 = np.datetime64("2024-01-01")
    pre_mask = end_dates < cutoff_2024
    if pre_mask.sum() == 0:
        return None

    x_pre = x[pre_mask]
    y_pre = y[pre_mask]
    b_pre = base_log[pre_mask]
    s_pre = start_dates[pre_mask]
    e_pre = end_dates[pre_mask]

    uniq = np.sort(np.unique(e_pre))
    if len(uniq) < 2:
        return None

    cut = int(len(uniq) * (1.0 - test_ratio))
    cut = max(1, min(cut, len(uniq) - 1))
    split_date = uniq[cut]

    train_mask = e_pre < split_date
    test_mask = e_pre >= split_date

    return {
        "x_tr": x_pre[train_mask],
        "y_tr": y_pre[train_mask],
        "b_tr": b_pre[train_mask],
        "s_tr": s_pre[train_mask],
        "x_te": x_pre[test_mask],
        "y_te": y_pre[test_mask],
        "b_te": b_pre[test_mask],
        "s_te": s_pre[test_mask],
        "split_date": split_date,
    }


def prepare_feature_frame(df_input: pd.DataFrame, target_col: str, feature_cols) -> pd.DataFrame:
    df_feat = build_dynamic_feature(df_input.copy(), target_col)

    y_derived_prefix = (
        "lag",
        "rolling",
        "trend",
        "residual",
        "ewm",
        "absdiff",
        "diff",
        "rolling_std",
        "rolling_absdiff",
    )
    y_cols = [c for c in df_feat.columns if (c == target_col) or c.startswith(y_derived_prefix)]
    exo_cols = [c for c in df_feat.columns if c not in (["date"] + y_cols)]

    for c in exo_cols + y_cols:
        if c == "date":
            continue
        if not np.issubdtype(df_feat[c].dtype, np.number):
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    if exo_cols:
        df_feat[exo_cols] = df_feat[exo_cols].ffill().bfill().fillna(0.0)
    df_feat[y_cols] = df_feat[y_cols].fillna(0.0)

    for c in feature_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0.0

    return df_feat.sort_values("date").reset_index(drop=True)


def fit_linear_calibrator(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    x = np.asarray(y_pred, dtype=float).reshape(-1)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return 1.0, 0.0

    x_ok = x[ok]
    y_ok = y[ok]
    x_var = float(np.var(x_ok))
    if x_var <= 1e-8:
        return 1.0, 0.0

    a = float(np.cov(x_ok, y_ok, ddof=0)[0, 1] / x_var)
    b = float(np.mean(y_ok) - a * np.mean(x_ok))
    a = float(np.clip(a, 0.2, 3.0))
    b = float(np.clip(b, -800.0, 800.0))
    return a, b


def naive7_for_dates(df_template: pd.DataFrame, dates: np.ndarray, target_col: str = "总血量") -> np.ndarray:
    s = df_template.copy().sort_values("date").set_index("date")[target_col]
    s = pd.to_numeric(s, errors="coerce")
    out = []
    for d in pd.to_datetime(dates):
        lag7 = s.get(d - pd.Timedelta(days=7), np.nan)
        lag1 = s.get(d - pd.Timedelta(days=1), np.nan)
        if np.isfinite(lag7):
            out.append(float(lag7))
        elif np.isfinite(lag1):
            out.append(float(lag1))
        else:
            out.append(float("nan"))
    return np.asarray(out, dtype=float)


def best_blend_weight(y_true: np.ndarray, y_model: np.ndarray, y_base: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    y_base = np.asarray(y_base, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_model) & np.isfinite(y_base)
    if ok.sum() < 10:
        return 0.7

    yt = y_true[ok]
    ym = y_model[ok]
    yb = y_base[ok]
    grid = np.linspace(0.0, 1.0, 41)
    best_w = 0.7
    best_mae = float("inf")
    for w in grid:
        pred = w * ym + (1.0 - w) * yb
        mae = float(np.mean(np.abs(yt - pred)))
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)
    return best_w


def monthly_bias_map(dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    dt = pd.to_datetime(dates)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(yt) & np.isfinite(yp)
    if ok.sum() < 12:
        return {}

    frame = pd.DataFrame({"date": dt[ok], "res": yt[ok] - yp[ok]})
    frame["m"] = frame["date"].dt.month
    return {int(k): float(v) for k, v in frame.groupby("m")["res"].median().items()}


def build_recursive_naive7_range(
        df_template: pd.DataFrame,
        start_date: str,
        end_date: str,
        target_col: str = "总血量",
) -> pd.DataFrame:
    sim_df = df_template.copy().sort_values("date").reset_index(drop=True)
    sim_df[target_col] = pd.to_numeric(sim_df[target_col], errors="coerce")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask_range = (sim_df["date"] >= start_ts) & (sim_df["date"] < end_ts)
    true_range = sim_df.loc[mask_range, ["date", target_col]].copy()
    if true_range.empty:
        return pd.DataFrame(columns=["date", "y_true_h1", "y_naive7_h1"])

    sim_df.loc[mask_range, target_col] = np.nan
    rows = []
    for current_date in true_range["date"].tolist():
        lag7_date = current_date - pd.Timedelta(days=7)
        lag1_date = current_date - pd.Timedelta(days=1)

        lag7_series = sim_df.loc[sim_df["date"] == lag7_date, target_col]
        lag1_series = sim_df.loc[sim_df["date"] == lag1_date, target_col]
        if (not lag7_series.empty) and np.isfinite(lag7_series.iloc[0]):
            y_pred = float(lag7_series.iloc[0])
        elif (not lag1_series.empty) and np.isfinite(lag1_series.iloc[0]):
            y_pred = float(lag1_series.iloc[0])
        else:
            continue

        y_pred = max(y_pred, 0.0)
        true_series = true_range.loc[true_range["date"] == current_date, target_col]
        if true_series.empty or not np.isfinite(true_series.iloc[0]):
            continue

        rows.append({"date": current_date, "y_true_h1": float(true_series.iloc[0]), "y_naive7_h1": y_pred})
        sim_df.loc[sim_df["date"] == current_date, target_col] = y_pred

    return pd.DataFrame(rows)


def recursive_forecast_h1(
        model,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        df_template: pd.DataFrame,
        feature_cols,
        lookback: int,
        blend_with_baseline: float,
        min_ratio_to_baseline: float,
        absolute_floor: float,
        calibrator_a: float,
        calibrator_b: float,
        monthly_bias: dict[int, float],
        monthly_bias_scale: float,
        horizon: int,
        start_date: str,
        end_date: str,
        target_col: str = "总血量",
) -> pd.DataFrame:
    sim_df = df_template.copy().sort_values("date").reset_index(drop=True)
    sim_df[target_col] = pd.to_numeric(sim_df[target_col], errors="coerce")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    true_2025 = sim_df.loc[
        (sim_df["date"] >= start_ts) & (sim_df["date"] < end_ts),
        ["date", target_col],
    ].copy()
    if true_2025.empty:
        return pd.DataFrame(columns=["date", "y_true_h1", "y_pred_h1"])

    sim_df.loc[(sim_df["date"] >= start_ts) & (sim_df["date"] < end_ts), target_col] = np.nan
    true_map = true_2025.set_index("date")[target_col]

    rows = []
    date_set = set(true_2025["date"].tolist())
    anchor_dates = true_2025["date"].tolist()
    step = max(int(horizon), 1)

    for anchor_i in range(0, len(anchor_dates), step):
        current_date = anchor_dates[anchor_i]
        feat_df = prepare_feature_frame(sim_df, target_col=target_col, feature_cols=feature_cols)
        idx_list = feat_df.index[feat_df["date"] == current_date].tolist()
        if len(idx_list) == 0:
            continue

        idx = int(idx_list[0])
        if idx < lookback or idx <= 0:
            continue

        window = feat_df.loc[idx - lookback: idx - 1, feature_cols].to_numpy(dtype=np.float32)
        if window.shape[0] != lookback:
            continue

        window_s = x_scaler.transform(window).reshape(1, lookback, -1)
        y_pred_s = model.predict(window_s, batch_size=1, verbose=0)
        y_pred_delta = y_scaler.inverse_transform(y_pred_s)[0]

        prev_date = feat_df.loc[idx - 1, "date"]
        prev_val_series = sim_df.loc[sim_df["date"] == prev_date, target_col]
        if prev_val_series.empty or not np.isfinite(prev_val_series.iloc[0]):
            continue

        base_log = float(np.log1p(max(float(prev_val_series.iloc[0]), 0.0)))

        for k in range(step):
            target_date = current_date + pd.Timedelta(days=k)
            if target_date not in date_set:
                continue

            raw_pred = float(np.expm1(base_log + float(y_pred_delta[min(k, len(y_pred_delta) - 1)])))
            raw_pred = max(raw_pred, 0.0)
            raw_pred = max(calibrator_a * raw_pred + calibrator_b, 0.0)
            raw_pred = raw_pred + float(monthly_bias_scale) * float(monthly_bias.get(int(target_date.month), 0.0))

            lag1_series = sim_df.loc[sim_df["date"] == (target_date - pd.Timedelta(days=1)), target_col]
            lag7_series = sim_df.loc[sim_df["date"] == (target_date - pd.Timedelta(days=7)), target_col]
            lag1_val = float(lag1_series.iloc[0]) if (not lag1_series.empty and np.isfinite(lag1_series.iloc[0])) else raw_pred
            lag7_val = float(lag7_series.iloc[0]) if (not lag7_series.empty and np.isfinite(lag7_series.iloc[0])) else lag1_val

            stabilizer = 0.5 * lag1_val + 0.5 * lag7_val
            blend = float(np.clip(blend_with_baseline, 0.0, 1.0))
            pred_blended = blend * raw_pred + (1.0 - blend) * lag7_val
            floor_by_base = float(max(min_ratio_to_baseline, 0.0)) * stabilizer
            y_pred_h1 = max(pred_blended, floor_by_base, float(max(absolute_floor, 0.0)), 0.0)

            rows.append(
                {
                    "date": target_date,
                    "y_true_h1": float(true_map.loc[target_date]),
                    "y_pred_h1": y_pred_h1,
                }
            )

            sim_df.loc[sim_df["date"] == target_date, target_col] = y_pred_h1

    return pd.DataFrame(rows)


def build_cnn_lstm_multistep(
        lookback: int,
        n_features: int,
        horizon: int,
        lr: float = 1e-3,
        ap_lambda: float = 1.0,
        ap_change_thr: float = 0.6,
        ap_margin: float = 0.6,
):
    inp = layers.Input(shape=(lookback, n_features))

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.10)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.10)(x)

    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.15)(x)
    x = layers.LSTM(32, return_sequences=False)(x)

    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(horizon, activation="linear")(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    loss_fn = build_anti_persistence_loss(
        huber_delta=1.0,
        ap_lambda=ap_lambda,
        ap_change_thr=ap_change_thr,
        ap_margin=ap_margin,
    )

    model.compile(optimizer=opt, loss=loss_fn, metrics=[tf.keras.metrics.MAE])
    return model


def plot_true_only(dates, y_true, out_path, title):
    plt.figure(figsize=(12, 4.5))
    plt.plot(pd.to_datetime(dates), y_true, label="true", linewidth=1.3)
    plt.xlabel("Date")
    plt.ylabel("Blood Volume")
    if FONT is not None:
        plt.title(title, fontproperties=FONT)
    else:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_true_vs_pred(dates, y_true, y_pred, out_path, title):
    plt.figure(figsize=(12, 4.5))
    plt.plot(pd.to_datetime(dates), y_true, label="true", linewidth=1.3)
    plt.plot(pd.to_datetime(dates), y_pred, label="pred", linewidth=1.3)
    plt.xlabel("Date")
    plt.ylabel("Blood Volume")
    if FONT is not None:
        plt.title(title, fontproperties=FONT)
    else:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_one_group(
        blood: pd.DataFrame,
        weather: pd.DataFrame,
        station: str,
        blood_type: str,
        out_dir: str,
        lookback: int,
        horizon: int,
        test_ratio: float,
        epochs: int,
        batch_size: int,
        lr: float,
        spike_weight_alpha: float,
        ap_lambda: float,
        ap_change_thr: float,
        ap_margin: float,
        recursive_blend: float,
        recursive_min_ratio: float,
        recursive_abs_floor_ratio: float,
        recursive_monthly_bias_scale: float,
        max_model_weight: float,
        naive_guardrail_ratio: float,
):
    series_pack = build_group_daily_series(blood, weather, station, blood_type, target_col="总血量")
    if series_pack is None:
        return None

    df, feature_cols = series_pack

    x, y_delta, base_log, start_dates, end_dates = make_windows_multistep_delta(
        df=df,
        feature_cols=feature_cols,
        target_col="总血量",
        lookback=lookback,
        horizon=horizon,
    )
    if len(x) < 200:
        return {"skipped": True, "reason": f"too_few_samples={len(x)}"}

    split_pack = split_pre_2024(x, y_delta, base_log, start_dates, end_dates, test_ratio=test_ratio)
    if split_pack is None:
        return {"skipped": True, "reason": "no_enough_pre_2024_data"}

    x_tr = split_pack["x_tr"]
    y_tr = split_pack["y_tr"]
    b_tr = split_pack["b_tr"]
    s_tr = split_pack["s_tr"]
    x_te = split_pack["x_te"]
    y_te = split_pack["y_te"]
    b_te = split_pack["b_te"]
    s_te = split_pack["s_te"]
    split_date = split_pack["split_date"]

    if len(x_tr) < 50 or len(x_te) < 20:
        return {"skipped": True, "reason": f"split_too_small train={len(x_tr)} test={len(x_te)}"}

    x_scaler = StandardScaler()
    x_tr_2d = x_tr.reshape(-1, x_tr.shape[-1])
    x_te_2d = x_te.reshape(-1, x_te.shape[-1])
    x_scaler.fit(x_tr_2d)
    x_tr_s = x_scaler.transform(x_tr_2d).reshape(x_tr.shape)
    x_te_s = x_scaler.transform(x_te_2d).reshape(x_te.shape)

    y_scaler = StandardScaler()
    y_scaler.fit(y_tr)
    y_tr_s = y_scaler.transform(y_tr)
    y_te_s = y_scaler.transform(y_te)

    train_sample_weights = build_delta_sample_weights(y_tr, alpha=spike_weight_alpha, quantile=0.75)
    val_sample_weights = build_delta_sample_weights(y_te, alpha=spike_weight_alpha, quantile=0.75)

    gname = f"{station}__{blood_type}"
    gdir = os.path.join(out_dir, slugify(gname))
    os.makedirs(gdir, exist_ok=True)

    model = build_cnn_lstm_multistep(
        lookback=lookback,
        n_features=x_tr_s.shape[-1],
        horizon=horizon,
        lr=lr,
        ap_lambda=ap_lambda,
        ap_change_thr=ap_change_thr,
        ap_margin=ap_margin,
    )

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        callbacks.ModelCheckpoint(os.path.join(gdir, "best_model.keras"), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        x_tr_s,
        y_tr_s,
        validation_data=(x_te_s, y_te_s, val_sample_weights),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cbs,
        shuffle=False,
        sample_weight=train_sample_weights,
    )

    y_pred_te_s = model.predict(x_te_s, batch_size=batch_size, verbose=0)
    y_pred_te_delta = y_scaler.inverse_transform(y_pred_te_s)
    y_true_te_delta = y_te

    y_pred_te = np.expm1(b_te[:, None] + y_pred_te_delta)
    y_true_te = np.expm1(b_te[:, None] + y_true_te_delta)
    cal_a, cal_b = fit_linear_calibrator(y_pred_te[:, 0], y_true_te[:, 0])
    y_model_cal_te_h1 = np.maximum(cal_a * y_pred_te[:, 0] + cal_b, 0.0)
    y_base_te_h1 = naive7_for_dates(df, s_te, target_col="总血量")
    recursive_blend_star = best_blend_weight(y_true_te[:, 0], y_model_cal_te_h1, y_base_te_h1)
    monthly_bias = monthly_bias_map(s_te, y_true_te[:, 0], y_model_cal_te_h1)

    probe_2024_df = recursive_forecast_h1(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        df_template=df,
        feature_cols=feature_cols,
        lookback=lookback,
        blend_with_baseline=1.0,
        min_ratio_to_baseline=0.0,
        absolute_floor=0.0,
        calibrator_a=cal_a,
        calibrator_b=cal_b,
        monthly_bias={},
        monthly_bias_scale=0.0,
        horizon=horizon,
        start_date="2024-01-01",
        end_date="2025-01-01",
        target_col="总血量",
    )
    naive7_2024_df = build_recursive_naive7_range(
        df_template=df,
        start_date="2024-01-01",
        end_date="2025-01-01",
        target_col="总血量",
    )
    probe_2024_model_mae = float("nan")
    probe_2024_naive7_mae = float("nan")
    guardrail_active = False
    effective_blend_star = float(recursive_blend_star)
    effective_monthly_bias = dict(monthly_bias)
    effective_min_ratio = float(recursive_min_ratio)

    if (not probe_2024_df.empty) and (not naive7_2024_df.empty):
        tune_2024 = probe_2024_df.merge(naive7_2024_df[["date", "y_naive7_h1"]], on="date", how="inner")
        if not tune_2024.empty:
            probe_2024_model_mae = float(mean_absolute_error(tune_2024["y_true_h1"], tune_2024["y_pred_h1"]))
            probe_2024_naive7_mae = float(mean_absolute_error(tune_2024["y_true_h1"], tune_2024["y_naive7_h1"]))
            recursive_blend_star = best_blend_weight(
                tune_2024["y_true_h1"].to_numpy(),
                tune_2024["y_pred_h1"].to_numpy(),
                tune_2024["y_naive7_h1"].to_numpy(),
            )
            blend_2024_pred = (
                    recursive_blend_star * tune_2024["y_pred_h1"].to_numpy()
                    + (1.0 - recursive_blend_star) * tune_2024["y_naive7_h1"].to_numpy()
            )
            monthly_bias = monthly_bias_map(
                tune_2024["date"].to_numpy(),
                tune_2024["y_true_h1"].to_numpy(),
                blend_2024_pred,
            )

            model_weight_cap = float(np.clip(max_model_weight, 0.0, 1.0))
            effective_blend_star = min(float(recursive_blend_star), model_weight_cap)
            if np.isfinite(probe_2024_model_mae) and np.isfinite(probe_2024_naive7_mae):
                if probe_2024_model_mae > probe_2024_naive7_mae * float(max(naive_guardrail_ratio, 0.0)):
                    guardrail_active = True
                    effective_blend_star = min(effective_blend_star, 0.10)
                    effective_monthly_bias = {}
                    effective_min_ratio = max(effective_min_ratio, 0.20)

    if not guardrail_active:
        effective_monthly_bias = dict(monthly_bias)

    train_hist = df.loc[df["date"] < pd.Timestamp("2024-01-01"), "总血量"].to_numpy(dtype=float)
    train_hist = train_hist[np.isfinite(train_hist)]
    if len(train_hist) == 0:
        return {"skipped": True, "reason": "no_train_history_before_2024"}
    abs_floor = float(np.quantile(train_hist, 0.05) * max(recursive_abs_floor_ratio, 0.0))

    forecast_2025_df = recursive_forecast_h1(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        df_template=df,
        feature_cols=feature_cols,
        lookback=lookback,
        blend_with_baseline=effective_blend_star,
        min_ratio_to_baseline=effective_min_ratio,
        absolute_floor=abs_floor,
        calibrator_a=cal_a,
        calibrator_b=cal_b,
        monthly_bias=effective_monthly_bias,
        monthly_bias_scale=recursive_monthly_bias_scale,
        horizon=horizon,
        start_date="2025-01-01",
        end_date="2026-01-01",
        target_col="总血量",
    )
    if forecast_2025_df.empty:
        return {"skipped": True, "reason": "no_2025_recursive_predictions"}

    pre2024_mae = float(mean_absolute_error(y_true_te.reshape(-1), y_pred_te.reshape(-1)))
    pre2024_mape = masked_mape(y_true_te.reshape(-1), y_pred_te.reshape(-1))
    pre2024_smape = smape(y_true_te.reshape(-1), y_pred_te.reshape(-1))

    y_true_2025_h1 = forecast_2025_df["y_true_h1"].to_numpy(dtype=float)
    y_pred_2025_h1 = forecast_2025_df["y_pred_h1"].to_numpy(dtype=float)
    mae_2025_h1 = float(mean_absolute_error(y_true_2025_h1, y_pred_2025_h1))
    mape_2025_h1 = masked_mape(y_true_2025_h1, y_pred_2025_h1)
    smape_2025_h1 = smape(y_true_2025_h1, y_pred_2025_h1)

    naive7_df = build_recursive_naive7_range(
        df_template=df,
        start_date="2025-01-01",
        end_date="2026-01-01",
        target_col="总血量",
    )
    if naive7_df.empty:
        naive7_mae_2025_h1 = float("nan")
        naive7_mape_2025_h1 = float("nan")
        naive7_smape_2025_h1 = float("nan")
    else:
        merged_eval = forecast_2025_df.merge(naive7_df[["date", "y_naive7_h1"]], on="date", how="inner")
        if merged_eval.empty:
            naive7_mae_2025_h1 = float("nan")
            naive7_mape_2025_h1 = float("nan")
            naive7_smape_2025_h1 = float("nan")
        else:
            naive7_mae_2025_h1 = float(mean_absolute_error(merged_eval["y_true_h1"], merged_eval["y_naive7_h1"]))
            naive7_mape_2025_h1 = masked_mape(merged_eval["y_true_h1"].to_numpy(), merged_eval["y_naive7_h1"].to_numpy())
            naive7_smape_2025_h1 = smape(merged_eval["y_true_h1"].to_numpy(), merged_eval["y_naive7_h1"].to_numpy())

    history_mask = df["date"] < pd.Timestamp("2024-01-01")
    plot_true_only(
        dates=df.loc[history_mask, "date"].to_numpy(),
        y_true=df.loc[history_mask, "总血量"].to_numpy(dtype=float),
        out_path=os.path.join(gdir, "true_curve_before_2024.png"),
        title=f"{station}-{blood_type} | True Curve Before 2024",
    )

    plot_true_vs_pred(
        dates=forecast_2025_df["date"].to_numpy(),
        y_true=y_true_2025_h1,
        y_pred=y_pred_2025_h1,
        out_path=os.path.join(gdir, "pred_vs_true_2025_h1.png"),
        title=f"{station}-{blood_type} | 2025 True vs Pred (h1)",
    )

    plt.figure(figsize=(8, 4.5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(gdir, "loss_curve.png"), dpi=200)
    plt.close()

    pred_2025_df = pd.DataFrame(
        {
            "date": pd.to_datetime(forecast_2025_df["date"]).dt.date,
            "y_true_h1": forecast_2025_df["y_true_h1"].to_numpy(dtype=float),
            "y_pred_h1": forecast_2025_df["y_pred_h1"].to_numpy(dtype=float),
        }
    )
    if not naive7_df.empty:
        pred_2025_df = pred_2025_df.merge(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(naive7_df["date"]).dt.date,
                    "y_naive7_h1": naive7_df["y_naive7_h1"].to_numpy(dtype=float),
                }
            ),
            on="date",
            how="left",
        )
    pred_2025_df.to_csv(os.path.join(gdir, "predictions_2025_h1.csv"), index=False, encoding="utf-8-sig")

    model.save(os.path.join(gdir, "final_model.keras"))
    joblib.dump(x_scaler, os.path.join(gdir, "x_scaler.joblib"))
    joblib.dump(y_scaler, os.path.join(gdir, "y_scaler.joblib"))
    with open(os.path.join(gdir, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    metrics = {
        "station": station,
        "blood_type": blood_type,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "train_test_scope": "end_date < 2024-01-01",
        "train_samples": int(len(x_tr)),
        "test_samples": int(len(x_te)),
        "pre2024_split_date": str(pd.to_datetime(split_date).date()),
        "pre2024_test_MAE_all": pre2024_mae,
        "pre2024_test_MAPE_all": float(pre2024_mape),
        "pre2024_test_sMAPE_all": float(pre2024_smape),
        "predict_scope": "recursive h1, 2025-01-01 <= date < 2026-01-01",
        "predict_2025_samples": int(len(forecast_2025_df)),
        "predict_2025_h1_MAE": mae_2025_h1,
        "predict_2025_h1_MAPE": float(mape_2025_h1),
        "predict_2025_h1_sMAPE": float(smape_2025_h1),
        "predict_2025_h1_naive7_MAE": float(naive7_mae_2025_h1),
        "predict_2025_h1_naive7_MAPE": float(naive7_mape_2025_h1),
        "predict_2025_h1_naive7_sMAPE": float(naive7_smape_2025_h1),
        "predict_vs_naive7_mae_ratio": float(mae_2025_h1 / (naive7_mae_2025_h1 + 1e-8)) if np.isfinite(naive7_mae_2025_h1) else float("nan"),
        "recursive_blend": float(recursive_blend),
        "recursive_blend_star": float(recursive_blend_star),
        "effective_blend_star": float(effective_blend_star),
        "recursive_min_ratio": float(recursive_min_ratio),
        "effective_min_ratio": float(effective_min_ratio),
        "recursive_abs_floor": float(abs_floor),
        "recursive_monthly_bias_scale": float(recursive_monthly_bias_scale),
        "guardrail_active": bool(guardrail_active),
        "probe_2024_model_MAE": float(probe_2024_model_mae),
        "probe_2024_naive7_MAE": float(probe_2024_naive7_mae),
        "naive_guardrail_ratio": float(naive_guardrail_ratio),
        "max_model_weight": float(max_model_weight),
        "monthly_bias_map": {str(k): float(v) for k, v in monthly_bias.items()},
        "effective_monthly_bias_map": {str(k): float(v) for k, v in effective_monthly_bias.items()},
        "calibrator_a": float(cal_a),
        "calibrator_b": float(cal_b),
    }
    with open(os.path.join(gdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blood_csv", type=str, default="../lstm/feature/remove_group_data.csv")
    ap.add_argument("--weather_csv", type=str, default="../lstm/feature/blood_calendar_weather_2015_2026.csv")
    ap.add_argument("--out_dir", type=str, default="./out_group_models_2025_forecast")
    ap.add_argument("--station", type=str, default="北京市红十字血液中心")
    ap.add_argument("--blood_types", type=str, default="ALL", help="逗号分隔")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--spike_weight_alpha", type=float, default=6.0)
    ap.add_argument("--ap_lambda", type=float, default=1.0)
    ap.add_argument("--ap_change_thr", type=float, default=0.6)
    ap.add_argument("--ap_margin", type=float, default=0.6)
    ap.add_argument("--recursive_blend", type=float, default=0.7)
    ap.add_argument("--recursive_min_ratio", type=float, default=0.30)
    ap.add_argument("--recursive_abs_floor_ratio", type=float, default=0.15)
    ap.add_argument("--recursive_monthly_bias_scale", type=float, default=0.8)
    ap.add_argument("--max_model_weight", type=float, default=0.35)
    ap.add_argument("--naive_guardrail_ratio", type=float, default=1.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    blood = pd.read_csv(args.blood_csv)
    weather = pd.read_csv(args.weather_csv)

    required = {"date", "血站", "血型", "总血量"}
    if not required.issubset(set(blood.columns)):
        raise ValueError(f"blood_csv 缺字段：{required - set(blood.columns)}")
    if "date" not in weather.columns:
        raise ValueError("weather_csv 必须有 date 列")

    blood["date"] = pd.to_datetime(blood["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    all_metrics = []
    for bt in [x.strip() for x in args.blood_types.split(",") if x.strip()]:
        m = train_one_group(
            blood=blood,
            weather=weather,
            station=args.station,
            blood_type=bt,
            out_dir=args.out_dir,
            lookback=args.lookback,
            horizon=args.horizon,
            test_ratio=args.test_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            spike_weight_alpha=args.spike_weight_alpha,
            ap_lambda=args.ap_lambda,
            ap_change_thr=args.ap_change_thr,
            ap_margin=args.ap_margin,
            recursive_blend=args.recursive_blend,
            recursive_min_ratio=args.recursive_min_ratio,
            recursive_abs_floor_ratio=args.recursive_abs_floor_ratio,
            recursive_monthly_bias_scale=args.recursive_monthly_bias_scale,
            max_model_weight=args.max_model_weight,
            naive_guardrail_ratio=args.naive_guardrail_ratio,
        )
        if m is None:
            continue
        m2 = {"station": args.station, "blood_type": bt}
        if isinstance(m, dict):
            m2.update(m)
        all_metrics.append(m2)

    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(all_metrics).to_csv(os.path.join(args.out_dir, "metrics_summary.csv"), index=False,
                                     encoding="utf-8-sig")

    print("\\n[Done] 输出目录：", args.out_dir)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
