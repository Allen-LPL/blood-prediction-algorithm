import argparse
import json
import math
import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.font_manager import FontProperties
from xgboost import XGBRegressor
from chinese_calendar import get_holiday_detail, Holiday
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error
from datetime import date, timedelta
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
import sklearn

DATE_COL = "date"
STATION_COL = "血站"
BLOOD_COL = "血型"
TARGET_COL = "总血量"  # 你图里是“总血量”

FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")

# 北大“校本部：上课”日期（第一学期 & 第二学期）
PKU_CLASS_STARTS = [
    date(2015, 9, 14),
    date(2016, 2, 22),
    date(2016, 9, 12),
    date(2017, 2, 20),
    date(2017, 9, 11),
    date(2018, 2, 26),
    date(2018, 9, 17),
    date(2019, 2, 18),
    date(2019, 9, 9),
    date(2020, 2, 17),
    date(2020, 9, 21),
    date(2021, 2, 22),
    date(2021, 9, 13),
    date(2022, 2, 21),
    date(2022, 9, 5),
    date(2023, 2, 20),
    date(2023, 9, 11),
    date(2024, 2, 19),
    date(2024, 9, 9),
    date(2025, 2, 17),
    date(2025, 9, 8),
    date(2026, 3, 2),
]


def is_pku_semester_start_season(d, pre_days=14, post_days=21) -> int:
    d = pd.to_datetime(d).date()
    for s in PKU_CLASS_STARTS:
        if s - timedelta(days=pre_days) <= d <= s + timedelta(days=post_days):
            return 1
    return 0


def is_holiday_name(d, target: Holiday) -> int:
    """d: datetime.date/datetime64/str 都行（会被 pd.to_datetime 处理）"""
    d = pd.to_datetime(d).date()
    is_hol, hol = get_holiday_detail(
        d
    )  # hol: Holiday or None  :contentReference[oaicite:1]{index=1}
    return int(hol == target)


def add_cn_holiday_flags(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    s = pd.to_datetime(df[date_col])
    df = df.copy()
    df["is_spring_festival"] = s.apply(
        lambda x: is_holiday_name(x, Holiday.spring_festival)
    )
    df["is_qingming"] = s.apply(lambda x: is_holiday_name(x, Holiday.tomb_sweeping_day))
    return df


def add_pku_season_flag(
    df: pd.DataFrame, date_col: str = "date", pre_days=14, post_days=21
) -> pd.DataFrame:
    s = pd.to_datetime(df[date_col])
    df = df.copy()
    df["is_pku_semester_start_season"] = s.apply(
        lambda x: is_pku_semester_start_season(x, pre_days, post_days)
    )
    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = add_cn_holiday_flags(df, date_col)
    df = add_pku_season_flag(df, date_col, pre_days=14, post_days=21)
    return df


def metric_item(true_y, pred_y):
    result = {}
    rmse_error = root_mean_squared_error(true_y, pred_y)
    result["均方根误差(越小越好)"] = rmse_error
    err_rates = []
    result.setdefault("误差率小于0.1", 0)
    result.setdefault("误差率小于0.2", 0)
    result.setdefault("误差率小于0.3", 0)
    result.setdefault("误差率大于1", 0)
    for y_true, y_pred in zip(true_y, pred_y):
        if y_pred < 0:
            y_pred = 0
        err_rate = abs(y_true - y_pred) / (0.00001 + y_true)
        if err_rate > 1:
            result["误差率大于1"] += 1
        if err_rate < 0.1:
            result["误差率小于0.1"] += 1
        if err_rate < 0.2:
            result["误差率小于0.2"] += 1
        if err_rate < 0.3:
            result["误差率小于0.3"] += 1
        err_rates.append(err_rate)

    avg_error_rate = sum(err_rates) / len(err_rates)
    if isinstance(avg_error_rate, list) or isinstance(avg_error_rate, np.ndarray):
        avg_error_rate = avg_error_rate[0]
    result["平均误差率(越小越好)"] = round(avg_error_rate, 5)
    result["总量"] = len(err_rates)

    result["误差率小于0.1比例"] = round(result["误差率小于0.1"] / len(err_rates), 5)
    result["误差率小于0.2比例"] = round(result["误差率小于0.2"] / len(err_rates), 5)
    result["误差率小于0.3比例"] = round(result["误差率小于0.3"] / len(err_rates), 5)
    result["误差率大于1比例"] = round(result["误差率大于1"] / len(err_rates), 5)
    return result


def load_blood_data():
    all_df = pd.read_csv("../lstm/feature/remove_group_data.csv")
    all_df["date"] = pd.to_datetime(all_df["date"])
    return all_df


def load_weather_data():
    weather_df = pd.read_csv("../lstm/feature/blood_calendar_weather_2015_2026.csv")
    # 节假日前后
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    weather_df["temp_sq"] = weather_df["平均温度"] ** 2
    weather_df["temp_diff"] = weather_df["平均温度"].diff()
    weather_df["temp_rolling3"] = weather_df["平均温度"].rolling(3).mean()
    weather_df = pd.get_dummies(weather_df, columns=["天气"])
    weather_df["temp_rolling3"].fillna(0.0, inplace=True)

    # 补充春节&清明节&十一&开学季
    weather_df = add_calendar_features(weather_df)
    weather_cols = [c for c in weather_df.columns if c.startswith("天气_")]
    for c in weather_cols:
        weather_df[c] = weather_df[c].astype(int)

    return weather_df


def add_prev_rolling_sums(df: pd.DataFrame, windows=(1, 3, 7, 14, 28)) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # 补齐
    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    full_dates = pd.DataFrame({DATE_COL: pd.date_range(min_date, max_date, freq="D")})

    df = pd.merge(full_dates, df, on=DATE_COL, how="left")
    df[TARGET_COL] = df[TARGET_COL].fillna(0.0)

    # 只做排序，不做聚合
    df = df.sort_values(DATE_COL)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    for w in windows:
        # 不含当天：先 rolling 再 shift(1)
        col = f"{w}d_sum"
        df[col] = df[TARGET_COL].rolling(w, min_periods=1).sum().shift(1)
        df[col] = df[col].fillna(0.0)

    return df


def normalize_pred(pred):
    result = []
    for item in pred:
        if item < 0:
            item = 0
        result.append(math.floor(item))
    return np.asarray(result)


def inverse_one_col(
    scaler: MinMaxScaler, y_scaled: np.ndarray, feature_dim: int, y_index: int
):
    tmp = np.zeros((len(y_scaled), feature_dim))
    tmp[:, y_index] = y_scaled
    return scaler.inverse_transform(tmp)[:, y_index]


def make_lstm_xy(values_scaled: np.ndarray, look_back: int, y_index: int):
    X, y = [], []
    for i in range(len(values_scaled) - look_back):
        X.append(values_scaled[i : i + look_back, :])
        y.append(values_scaled[i + look_back, y_index])
    return np.array(X), np.array(y)


def make_demo_series(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = (
        1000
        + 0.05 * t
        + 50 * np.sin(2 * np.pi * t / 7)
        + 120 * np.sin(2 * np.pi * t / 365.25)
        + rng.normal(0, 30, n)
    )
    y = np.clip(y, 0, None)
    return y.astype(np.float32)


def make_supervised(y, lookback=60, horizon=1):
    """
    输入: y[t-lookback:t]
    输出: y[t+horizon-1]   (默认 1-step ahead)
    """
    X, Y = [], []
    for t in range(lookback, len(y) - horizon + 1):
        X.append(y[t - lookback : t])
        Y.append(y[t + horizon - 1])
    X = np.asarray(X, dtype=np.float32)  # (N, lookback)
    Y = np.asarray(Y, dtype=np.float32)  # (N,)
    return X, Y


def time_split(X, Y, train_ratio=0.7, val_ratio=0.1):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    X_tr, y_tr = X[:n_train], Y[:n_train]
    X_va, y_va = X[n_train:n_val], Y[n_train:n_val]
    X_te, y_te = X[n_val:], Y[n_val:]
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def build_lstm(lookback=60, lr=1e-3):
    inp = layers.Input(shape=(lookback, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.15)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mae")
    return m


def train_lstm(X_tr, y_tr, X_va, y_va, lookback=60):
    # LSTM 需要 (N, T, 1)
    X_tr3 = X_tr[..., None]
    X_va3 = X_va[..., None]

    # y 标准化会更稳（不是特征工程，只是尺度变换）
    y_scaler = StandardScaler()
    y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).astype(np.float32)
    y_va_s = y_scaler.transform(y_va.reshape(-1, 1)).astype(np.float32)

    m = build_lstm(lookback=lookback, lr=1e-3)
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]
    m.fit(
        X_tr3,
        y_tr_s,
        validation_data=(X_va3, y_va_s),
        epochs=80,
        batch_size=32,
        verbose=1,
        callbacks=cbs,
        shuffle=True,
    )
    return m, y_scaler


def lstm_predict(model, y_scaler, X):
    pred_s = model.predict(X[..., None], verbose=0).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_s).reshape(-1)
    return pred


# =========================
# 4) 融合：用验证集 MAE 自动算权重
# =========================
def blend_by_val_mae(y_va, pred_xgb_va, pred_lstm_va, eps=1e-8):
    mae_xgb = mean_absolute_error(y_va, pred_xgb_va)
    mae_lstm = mean_absolute_error(y_va, pred_lstm_va)

    # MAE 越小权重越大
    w_xgb = 1.0 / (mae_xgb + eps)
    w_lstm = 1.0 / (mae_lstm + eps)
    w_sum = w_xgb + w_lstm
    w_xgb /= w_sum
    w_lstm /= w_sum
    return w_xgb, w_lstm, mae_xgb, mae_lstm


def build_lstm(timesteps: int, feature_dim: int) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=(timesteps, feature_dim)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def lstm_train(lstm_df: pd.DataFrame, group):
    LOOK_BACK = 60
    feature_cols_lstm = []
    g = lstm_df.copy().sort_values(DATE_COL).reset_index(drop=True)
    g["trend"] = g[TARGET_COL].rolling(7).mean()
    g["residual"] = g[TARGET_COL] - g["trend"]
    lstm_mat = g[feature_cols_lstm + ["trend"]].astype(float).values
    scaler = MinMaxScaler()
    n = len(g)
    split = int(n * (1 - 0.2))
    train_end = split
    scaler.fit(lstm_mat[:train_end])
    lstm_scaled = scaler.transform(lstm_mat)
    y_index = lstm_scaled.shape[1] - 1  # trend 在最后一列

    X_all, y_all = make_lstm_xy(lstm_scaled, LOOK_BACK, y_index)
    sample_split = train_end - LOOK_BACK
    if sample_split <= 50:
        print(f"[SKIP] {group} 训练样本不足 sample_split={sample_split}")
        return

    X_train, X_test = X_all[:sample_split], X_all[sample_split:]
    y_train, y_test = y_all[:sample_split], y_all[sample_split:]

    # val 取训练末尾10%
    val_cut = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_cut], X_train[val_cut:]
    y_tr, y_val = y_train[:val_cut], y_train[val_cut:]

    model = build_lstm(LOOK_BACK, X_all.shape[2])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=64,
        callbacks=[es],
        verbose=0,
    )

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=64,
        callbacks=[es],
        verbose=0,
    )
    trend_pred_scaled = model.predict(X_test, verbose=0).flatten()
    trend_true_scaled = y_test
    feature_dim = lstm_scaled.shape[1]
    trend_pred = inverse_one_col(scaler, trend_pred_scaled, feature_dim, y_index)
    trend_true = inverse_one_col(scaler, trend_true_scaled, feature_dim, y_index)
    return trend_pred, trend_true


def make_residual_sequences(
    feature_mat: np.ndarray, resid_target: np.ndarray, lookback: int
):
    """
    feature_mat: shape (T, num_features)  (包含了滞后残差、当天的天气、XGB预测值等特征)
    resid_target: shape (T,) (只预测残差本身)
    输出:
      X: (N, lookback, num_features)
      y: (N, 1)
      idx: (N,) 每个样本对应的原始时间索引 t
    """
    X, y, idx = [], [], []
    for t in range(lookback, len(feature_mat)):
        # 包含了 t-lookback+1 到 t 的特征（因为 feature_mat 里的残差已经是 lag1，所以不会数据穿越）
        X.append(feature_mat[t - lookback + 1 : t + 1, :])
        y.append(resid_target[t])
        idx.append(t)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)[:, None]
    idx = np.asarray(idx, dtype=np.int64)
    return X, y, idx


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        # x: (batch_size, time_steps, features)
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


def build_lstm_residual_model(lookback: int, feature_dim: int, lr: float = 1e-3):
    inp = layers.Input(shape=(lookback, feature_dim))
    # 恢复稍微轻量化一点的双向LSTM，去掉复杂的自定义Attention防止小样本过拟合
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # 最后一层LSTM直接输出最终隐藏状态即可
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation="linear")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse",
        metrics=[tf.keras.metrics.MAE],
    )
    return model


def add_dynamic_feature(g2: pd.DataFrame):
    g2["lag1"] = g2[TARGET_COL].shift(1).fillna(0)
    g2["lag7"] = g2[TARGET_COL].shift(7).fillna(0)

    g2["rolling7"] = g2[TARGET_COL].rolling(7).mean().fillna(0)

    g2["absdiff1"] = (g2[TARGET_COL] - g2["lag1"]).abs().fillna(0)
    g2["rolling_std7"] = g2[TARGET_COL].rolling(7).std().fillna(0)
    g2["rolling_std14"] = g2[TARGET_COL].rolling(14).std().fillna(0)
    g2["rolling_absdiff7"] = g2["absdiff1"].rolling(7).mean().fillna(0)

    g2["lag14"] = g2[TARGET_COL].shift(14).fillna(0)
    g2["lag28"] = g2[TARGET_COL].shift(28).fillna(0)

    g2["absdiff7"] = (g2[TARGET_COL] - g2["lag7"]).abs().fillna(0)
    g2["diff7"] = (g2[TARGET_COL] - g2["lag7"]).fillna(0)
    g2["diff14"] = (g2[TARGET_COL] - g2["lag14"]).fillna(0)

    g2["rolling14"] = g2[TARGET_COL].rolling(14).mean().fillna(0)
    g2["rolling28"] = g2[TARGET_COL].rolling(28).mean().fillna(0)
    g2["ewm7"] = g2[TARGET_COL].ewm(span=7, adjust=False).mean().fillna(0)
    g2["ewm14"] = g2[TARGET_COL].ewm(span=14, adjust=False).mean().fillna(0)
    return g2


# =========================================================================
# Multi-window evaluation helpers (post-eval_df output layer)
# =========================================================================


def build_trailing_windows(eval_df, lengths=(7, 30, 90, 180, 365)):
    """Slice trailing windows from the end of eval_df."""
    windows = []
    for n in lengths:
        if n > len(eval_df):
            continue
        w_df = eval_df.iloc[-n:].copy()
        windows.append(
            {
                "window_name": f"trailing_{n}d",
                "window_type": "trailing_length",
                "start_date": str(w_df[DATE_COL].iloc[0].date()),
                "end_date": str(w_df[DATE_COL].iloc[-1].date()),
                "df": w_df,
            }
        )
    return windows


def build_shifted_range_windows(full_eval_df, anchor_df, shifts=(0, 1, 2)):
    """Build same-date-range windows shifted back by whole years."""
    anchor_start = pd.to_datetime(anchor_df[DATE_COL].min())
    anchor_end = pd.to_datetime(anchor_df[DATE_COL].max())
    full_eval_df = full_eval_df.copy()
    full_eval_df[DATE_COL] = pd.to_datetime(full_eval_df[DATE_COL])

    names = {0: "current_range"}
    for s in shifts:
        if s > 0:
            names[s] = f"minus_{s}y_same_range"

    windows = []
    for s in shifts:
        try:
            s_start = anchor_start.replace(year=anchor_start.year - s)
            s_end = anchor_end.replace(year=anchor_end.year - s)
        except ValueError:
            # leap-day edge case
            s_start = anchor_start - pd.DateOffset(years=s)
            s_end = anchor_end - pd.DateOffset(years=s)

        mask = (full_eval_df[DATE_COL] >= s_start) & (full_eval_df[DATE_COL] <= s_end)
        w_df = full_eval_df.loc[mask].copy()
        if w_df.empty:
            continue
        windows.append(
            {
                "window_name": names[s],
                "window_type": "historical_same_range",
                "start_date": str(s_start.date()),
                "end_date": str(s_end.date()),
                "df": w_df,
            }
        )
    return windows


def _safe_mape(y_true, y_pred, eps=1e-8):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def summarize_window_metrics(group, window_name, window_type, window_df):
    y = window_df[TARGET_COL].values.astype(float)
    y_xgb = window_df["y_xgb"].values.astype(float)
    y_hat = window_df["y_hat"].values.astype(float)
    y_adp = window_df["y_hat_adaptive"].values.astype(float)
    return {
        "group": group,
        "window_name": window_name,
        "window_type": window_type,
        "start_date": str(pd.to_datetime(window_df[DATE_COL].iloc[0]).date()),
        "end_date": str(pd.to_datetime(window_df[DATE_COL].iloc[-1]).date()),
        "sample_count": len(window_df),
        "mae_xgb": float(mean_absolute_error(y, y_xgb)),
        "mae_hat": float(mean_absolute_error(y, y_hat)),
        "mae_hat_adaptive": float(mean_absolute_error(y, y_adp)),
        "mape_xgb": _safe_mape(y, y_xgb),
        "mape_hat": _safe_mape(y, y_hat),
        "mape_hat_adaptive": _safe_mape(y, y_adp),
    }


def ensure_group_output_dir(base_dir, group):
    path = os.path.join(base_dir, group)
    os.makedirs(path, exist_ok=True)
    return path


def export_window_tabular_outputs(group_dir, group, summary, window_df):
    window_name = summary["window_name"]
    csv_cols = [DATE_COL, TARGET_COL, "y_xgb", "y_hat", "y_hat_adaptive"]
    extra = [c for c in ("resid_pred", "resid_pred_clipped") if c in window_df.columns]
    out_df = window_df[csv_cols + extra].copy()
    out_df.to_csv(
        os.path.join(group_dir, f"{window_name}_eval.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def export_group_summary(group_dir, summary_rows):
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(group_dir, "summary_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    with open(
        os.path.join(group_dir, "summary_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)


def export_window_plots(
    group_dir, group, window_name, window_df, mae_xgb, mae_hat, mae_hat_adp
):
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(14, 6), dpi=300)
    plt.plot(
        window_df[DATE_COL],
        window_df[TARGET_COL],
        label="真实血量",
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    plt.plot(
        window_df[DATE_COL],
        window_df["y_xgb"],
        label="仅使用XGBoost(基线)",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        window_df[DATE_COL],
        window_df["y_hat_adaptive"],
        label="本发明预测(XGBoost+本发明的混合模型自适应)",
        color="#2ca02c",
        linewidth=2.5,
    )
    plt.fill_between(
        window_df[DATE_COL],
        window_df["y_xgb"],
        window_df["y_hat_adaptive"],
        color="gray",
        alpha=0.2,
        label="深度残差修正幅度",
    )
    plt.title(
        f"{group} - 基准模型与本发明混合模型预测效果对比 (MAE: {mae_hat_adp:.1f})",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("总血量 (U)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, f"{window_name}_prediction_comparison.png"))
    plt.close()

    real_resid = window_df[TARGET_COL] - window_df["y_xgb"]
    lstm_resid_pred = window_df["y_hat_adaptive"] - window_df["y_xgb"]

    plt.figure(figsize=(14, 4), dpi=300)
    plt.bar(
        window_df[DATE_COL],
        real_resid,
        label="XGBoost基线真实误差",
        color="#ff9896",
        alpha=0.7,
    )
    plt.plot(
        window_df[DATE_COL],
        lstm_resid_pred,
        label="本发明的混合模型预测的修正残差",
        color="black",
        linewidth=2,
        marker="x",
        markersize=5,
    )
    plt.axhline(0, color="gray", linestyle="-")
    plt.title(
        f"{group} - 本发明 本发明的混合模型 对基线误差的动态捕捉效果", fontsize=16
    )
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("误差幅度", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(group_dir, f"{window_name}_residual_tracking.png"))
    plt.close()


def export_all_eval_windows(
    base_dir,
    group,
    eval_df,
    trailing_lengths=(7, 30, 90, 180, 365),
    year_shifts=(0, 1, 2),
):
    group_dir = str(base_dir)
    os.makedirs(group_dir, exist_ok=True)

    all_summaries = []

    shifted_windows = build_shifted_range_windows(eval_df, eval_df, shifts=year_shifts)
    for w in shifted_windows:
        sm = summarize_window_metrics(
            group, w["window_name"], w["window_type"], w["df"]
        )
        all_summaries.append(sm)
        export_window_tabular_outputs(group_dir, group, sm, w["df"])
        export_window_plots(
            group_dir,
            group,
            w["window_name"],
            w["df"],
            sm["mae_xgb"],
            sm["mae_hat"],
            sm["mae_hat_adaptive"],
        )

    trailing_windows = build_trailing_windows(eval_df, trailing_lengths)
    for w in trailing_windows:
        sm = summarize_window_metrics(
            group, w["window_name"], w["window_type"], w["df"]
        )
        all_summaries.append(sm)
        export_window_tabular_outputs(group_dir, group, sm, w["df"])
        export_window_plots(
            group_dir,
            group,
            w["window_name"],
            w["df"],
            sm["mae_xgb"],
            sm["mae_hat"],
            sm["mae_hat_adaptive"],
        )

    if all_summaries:
        export_group_summary(group_dir, all_summaries)

    return all_summaries


def train_one_group(group, group_blood_df, weather_df):
    lookback = 14  # 缩短窗口期，针对残差，短期信息更有效
    lstm_lr = 1e-3
    # 1.blood特征
    label_col = TARGET_COL
    group_blood_df = add_prev_rolling_sums(group_blood_df, windows=(1, 3, 7, 14, 28))
    group_blood_df = add_dynamic_feature(group_blood_df)

    # 2.合并blood weather
    df = pd.merge(group_blood_df, weather_df, on="date", how="left")
    print(f"all len is {len(df)}")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # 增加时间周期特征，捕获短期波动和残差的季节性
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month"] = df[DATE_COL].dt.month

    # xgb 特征
    feat_cols = []
    for col in df.columns:
        if col in [
            TARGET_COL,
            label_col,
            DATE_COL,
            "date_x",
            "date_y",
            STATION_COL,
            "lag1",
            "1d_sum",
        ]:
            continue
        feat_cols.append(col)

    # 3.切分训练集 测试集
    X, y, target_dates = df[feat_cols].values, df[label_col].values, df[DATE_COL].values
    X_train, X_test, Y_train, Y_test, d_train, d_test = train_test_split(
        X, y, target_dates, test_size=0.1, shuffle=False
    )
    cutoff_date = pd.to_datetime(d_test).min()  # 这时就等价于“测试集第一天”
    print(f"[Split] cutoff_date={cutoff_date}  train={len(X_train)} test={len(Y_test)}")

    # 3.xgb 训练 - 降低复杂度防止过拟合
    n_estimators = 200
    learning_rate = 0.05
    params = {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "seed": 1024,
        "max_depth": 4,
        "eval_metric": mean_squared_error,
        "early_stopping_rounds": 20,
        "missing": 0,
    }

    # 恢复原版手动权重：在业务上这可能确实反映了高血量的极端重要性，平方根不够激进
    sample_weight = []
    for y_val_w in Y_train:
        weight = 3
        if y_val_w >= 45:
            weight = 5
        elif y_val_w >= 36:
            weight = 5
        elif y_val_w >= 26:
            weight = 6
        elif y_val_w >= 18:
            weight = 1
        sample_weight.append(weight)

    # 使用 OOF (Out-of-Fold) 计算训练集上的真实泛化残差
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    for tr_idx, va_idx in kf.split(X_train):
        X_tr_kf, y_tr_kf = X_train[tr_idx], Y_train[tr_idx]
        X_va_kf, y_va_kf = X_train[va_idx], Y_train[va_idx]
        sw_tr = np.array(sample_weight)[tr_idx]

        xgb_fold = XGBRegressor(**params)
        xgb_fold.fit(
            X_tr_kf,
            y_tr_kf,
            eval_set=[(X_va_kf, y_va_kf)],
            sample_weight=sw_tr,
            verbose=0,
        )
        oof_preds[va_idx] = xgb_fold.predict(X_va_kf)

    # 训练最终模型用于预测测试集
    xgb_final = XGBRegressor(**params)
    xgb_final.fit(
        X_train,
        Y_train,
        eval_set=[(X_test, Y_test)],
        sample_weight=sample_weight,
        verbose=0,
    )

    # 拼接 OOF 预测值（训练集）和 最终模型预测值（测试集）
    df["y_xgb_raw"] = 0.0
    df_train_mask = df[DATE_COL] < cutoff_date
    df_test_mask = df[DATE_COL] >= cutoff_date

    df.loc[df_train_mask, "y_xgb_raw"] = oof_preds
    df.loc[df_test_mask, "y_xgb_raw"] = xgb_final.predict(X_test)

    df["y_xgb"] = normalize_pred(df["y_xgb_raw"].to_numpy())
    df["resid"] = (df[TARGET_COL] - df["y_xgb"]).astype(np.float32)

    # 训练集残差去极值（防止LSTM在训练时被个别极其离谱的XGB预测带偏，提高模型收敛稳定性）
    train_resids = df.loc[df_train_mask, "resid"]
    lower, upper = np.percentile(train_resids, [1, 99])
    df.loc[df_train_mask, "resid"] = np.clip(train_resids, lower, upper)

    # -------------------------
    # 2) LSTM：多特征学习 residual 序列
    #    输入包含滞后残差、天气、日期特征，预测 resid[t]
    # -------------------------
    # 挑选 LSTM 辅助特征
    lstm_aux_cols = [
        "y_xgb",
        "平均温度",
        "is_spring_festival",
        "is_qingming",
        "is_pku_semester_start_season",
        "rolling7",
        "lag1",
        "dayofweek",
        "is_weekend",
        "month",
    ]
    lstm_aux_cols = [c for c in lstm_aux_cols if c in df.columns]

    feature_df = df[["resid"] + lstm_aux_cols].copy()

    # 核心改动：把 resid shift 一位，LSTM 就能用 t-1 的残差来预测 t，同时使用 t 的其他特征
    feature_df["resid_lag1"] = feature_df["resid"].shift(1).fillna(0)
    feature_df.drop(columns=["resid"], inplace=True)

    feature_df.fillna(0, inplace=True)

    # 标准化所有特征
    feature_scaler = StandardScaler()
    feature_train = feature_df[df_train_mask]
    feature_scaler.fit(feature_train)
    features_s = feature_scaler.transform(feature_df).astype(np.float32)

    # Target (真实的当期 resid) 标准化 (单独做 scaler 方便 inverse)
    resid_scaler = StandardScaler()
    resid_train = df.loc[df_train_mask, "resid"].to_numpy().reshape(-1, 1)
    resid_scaler.fit(resid_train)
    resid_s_target = (
        resid_scaler.transform(df["resid"].to_numpy().reshape(-1, 1))
        .reshape(-1)
        .astype(np.float32)
    )

    X_seq, y_seq, idx_seq = make_residual_sequences(
        features_s, resid_s_target, lookback
    )

    # 将序列样本按时间切分到 train/test（以 idx 对应的 date 为准）
    sample_dates = df.loc[idx_seq, DATE_COL].to_numpy()
    tr_mask = sample_dates < np.datetime64(cutoff_date)
    te_mask = sample_dates >= np.datetime64(cutoff_date)

    X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
    X_seq_te, y_seq_te = X_seq[te_mask], y_seq[te_mask]
    dates_te = sample_dates[te_mask]
    print(f"[Seq] train_samples={len(X_seq_tr)} test_samples={len(X_seq_te)}")

    feature_dim = X_seq_tr.shape[2]
    lstm = build_lstm_residual_model(lookback, feature_dim=feature_dim, lr=lstm_lr)
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
        callbacks.ModelCheckpoint(
            os.path.join("./", "best_lstm.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    hist = lstm.fit(
        X_seq_tr,
        y_seq_tr,
        validation_data=(X_seq_te, y_seq_te),
        epochs=120,
        batch_size=32,
        verbose=1,
        callbacks=cbs,
    )

    # 预测 residual（标准化空间） -> inverse 回真实 residual
    resid_pred_s = lstm.predict(X_seq_te, batch_size=32).reshape(-1, 1)
    resid_pred = (
        resid_scaler.inverse_transform(resid_pred_s).reshape(-1).astype(np.float32)
    )

    # 对齐到原始 df 的时间索引
    idx_te = idx_seq[te_mask]
    df.loc[idx_te, "resid_pred"] = resid_pred

    # 融合预测：引入自适应Meta-Learner代替硬相加
    # ==========================
    # 在测试集上，我们获取了 y_xgb 和 resid_pred
    df["resid_pred"] = df["resid_pred"].fillna(0)  # 以防有些早期天数没有预测
    df["y_hat"] = df["y_xgb"] + df["resid_pred"]  # 保留 baseline 硬相加

    # 构建 Meta-Learner (基于训练集最后一段验证集的残差预测分布来学习融合权重)
    # 为了简单且不泄露数据，我们直接对 y_xgb 和 resid_pred 进行带保护的硬相加，但对于极端的残差预测进行衰减 (平滑处理)
    # 因为 LSTM 有时候预测出来的残差会过大(尤其是在节假日或者突变日)
    df["resid_pred_clipped"] = np.clip(
        df["resid_pred"], -abs(df["y_xgb"]) * 0.5, abs(df["y_xgb"]) * 0.5
    )

    # 自适应融合预测：y_hat_adaptive
    df["y_hat_adaptive"] = df["y_xgb"] + df["resid_pred_clipped"]

    # 评估：只评估测试区间中，存在 resid_pred 的那些天（因为序列需要 lookback）
    eval_df = df.loc[
        idx_te, [DATE_COL, TARGET_COL, "y_xgb", "y_hat", "y_hat_adaptive"]
    ].copy()
    eval_df = eval_df.sort_values(DATE_COL).reset_index(drop=True)

    mae_xgb = mean_absolute_error(eval_df[TARGET_COL], eval_df["y_xgb"])
    mae_hat = mean_absolute_error(eval_df[TARGET_COL], eval_df["y_hat"])
    mae_hat_adp = mean_absolute_error(eval_df[TARGET_COL], eval_df["y_hat_adaptive"])

    # 计算 MAPE (平均绝对百分比误差/误差率)
    # 为了防止分母为 0 导致报错或无穷大，加上一个极小值 eps
    eps = 1e-8
    mape_xgb = (
        np.mean(
            np.abs(
                (eval_df[TARGET_COL] - eval_df["y_xgb"]) / (eval_df[TARGET_COL] + eps)
            )
        )
        * 100
    )
    mape_hat = (
        np.mean(
            np.abs(
                (eval_df[TARGET_COL] - eval_df["y_hat"]) / (eval_df[TARGET_COL] + eps)
            )
        )
        * 100
    )
    mape_hat_adp = (
        np.mean(
            np.abs(
                (eval_df[TARGET_COL] - eval_df["y_hat_adaptive"])
                / (eval_df[TARGET_COL] + eps)
            )
        )
        * 100
    )

    print(
        f"[MAE] XGB={mae_xgb:.3f}  XGB+LSTM={mae_hat:.3f}  Adaptive={mae_hat_adp:.3f}"
    )
    print(
        f"[误差率 MAPE] XGB={mape_xgb:.2f}%  XGB+LSTM={mape_hat:.2f}%  Adaptive={mape_hat_adp:.2f}%"
    )

    os.makedirs("patent_figures", exist_ok=True)
    group_dir = ensure_group_output_dir("patent_figures", group)
    export_all_eval_windows(group_dir, group, eval_df)
    print(f"[{group}] 多窗口评估已导出至 {group_dir}")


def main():
    # 1.获取数据
    all_blood_df = load_blood_data()
    weather_df = load_weather_data()

    # 2.获取
    station_set = all_blood_df["血站"].unique().tolist()
    blood_types = ["ALL", "A", "B", "O", "AB"]
    for station in station_set:
        if station != "北京市红十字血液中心":
            continue
        for blood_type in blood_types:
            group = station + "_" + blood_type
            group_df = None
            if blood_type == "ALL":
                group_df = all_blood_df[all_blood_df[STATION_COL] == station].copy()
                group_df = group_df.groupby(DATE_COL, as_index=False).agg(
                    {TARGET_COL: "sum", BLOOD_COL: "first", STATION_COL: "first"}
                )
            else:
                group_df = all_blood_df[
                    (all_blood_df[STATION_COL] == station)
                    & (all_blood_df[BLOOD_COL] == blood_type)
                ].copy()
            group_df.drop(columns=[BLOOD_COL, STATION_COL], inplace=True)
            train_one_group(group, group_df, weather_df)

    # 3.整体


if __name__ == "__main__":
    print(sklearn.__version__)
    main()
