import argparse
import math
import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.font_manager import FontProperties
from xgboost import XGBRegressor
from chinese_calendar import get_holiday_detail, Holiday
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
    date(2015, 9, 14), date(2016, 2, 22),
    date(2016, 9, 12), date(2017, 2, 20),
    date(2017, 9, 11), date(2018, 2, 26),
    date(2018, 9, 17), date(2019, 2, 18),
    date(2019, 9, 9), date(2020, 2, 17),
    date(2020, 9, 21), date(2021, 2, 22),
    date(2021, 9, 13), date(2022, 2, 21),
    date(2022, 9, 5), date(2023, 2, 20),
    date(2023, 9, 11), date(2024, 2, 19),
    date(2024, 9, 9), date(2025, 2, 17),
    date(2025, 9, 8), date(2026, 3, 2),
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
    is_hol, hol = get_holiday_detail(d)  # hol: Holiday or None  :contentReference[oaicite:1]{index=1}
    return int(hol == target)


def add_cn_holiday_flags(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    s = pd.to_datetime(df[date_col])
    df = df.copy()
    df["is_spring_festival"] = s.apply(lambda x: is_holiday_name(x, Holiday.spring_festival))
    df["is_qingming"] = s.apply(lambda x: is_holiday_name(x, Holiday.tomb_sweeping_day))
    return df


def add_pku_season_flag(df: pd.DataFrame, date_col: str = "date",
                        pre_days=14, post_days=21) -> pd.DataFrame:
    s = pd.to_datetime(df[date_col])
    df = df.copy()
    df["is_pku_semester_start_season"] = s.apply(lambda x: is_pku_semester_start_season(x, pre_days, post_days))
    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = add_cn_holiday_flags(df, date_col)
    df = add_pku_season_flag(df, date_col, pre_days=14, post_days=21)
    return df


def metric_item(true_y, pred_y):
    result = {}
    rmse_error = root_mean_squared_error(true_y, pred_y)
    result['均方根误差(越小越好)'] = rmse_error
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
            result['误差率大于1'] += 1
        if err_rate < 0.1:
            result['误差率小于0.1'] += 1
        if err_rate < 0.2:
            result['误差率小于0.2'] += 1
        if err_rate < 0.3:
            result['误差率小于0.3'] += 1
        err_rates.append(err_rate)

    avg_error_rate = sum(err_rates) / len(err_rates)
    if isinstance(avg_error_rate, list) or isinstance(avg_error_rate, np.ndarray):
        avg_error_rate = avg_error_rate[0]
    result['平均误差率(越小越好)'] = round(avg_error_rate, 5)
    result['总量'] = len(err_rates)

    result['误差率小于0.1比例'] = round(result['误差率小于0.1'] / len(err_rates), 5)
    result['误差率小于0.2比例'] = round(result['误差率小于0.2'] / len(err_rates), 5)
    result['误差率小于0.3比例'] = round(result['误差率小于0.3'] / len(err_rates), 5)
    result['误差率大于1比例'] = round(result['误差率大于1'] / len(err_rates), 5)
    return result


def load_blood_data():
    all_df = pd.read_csv('../lstm/feature/remove_group_data.csv')
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


def add_prev_rolling_sums(df: pd.DataFrame,
                          windows=(1, 3, 7, 14, 28)) -> pd.DataFrame:
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


def inverse_one_col(scaler: MinMaxScaler, y_scaled: np.ndarray, feature_dim: int, y_index: int):
    tmp = np.zeros((len(y_scaled), feature_dim))
    tmp[:, y_index] = y_scaled
    return scaler.inverse_transform(tmp)[:, y_index]


def make_lstm_xy(values_scaled: np.ndarray, look_back: int, y_index: int):
    X, y = [], []
    for i in range(len(values_scaled) - look_back):
        X.append(values_scaled[i:i + look_back, :])
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
        X.append(y[t - lookback:t])
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
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    m.fit(
        X_tr3, y_tr_s,
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
    model = Sequential([
        Input(shape=(timesteps, feature_dim)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
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
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=64,
        callbacks=[es],
        verbose=0
    )

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=64,
        callbacks=[es],
        verbose=0
    )
    trend_pred_scaled = model.predict(X_test, verbose=0).flatten()
    trend_true_scaled = y_test
    feature_dim = lstm_scaled.shape[1]
    trend_pred = inverse_one_col(scaler, trend_pred_scaled, feature_dim, y_index)
    trend_true = inverse_one_col(scaler, trend_true_scaled, feature_dim, y_index)
    return trend_pred, trend_true


def make_residual_sequences(residual_arr: np.ndarray, lookback: int):
    """
    residual_arr: shape (T,)
    输出:
      X: (N, lookback, 1)
      y: (N, 1)
      idx: (N,) 每个样本对应的原始时间索引 t（预测 residual[t]）
    """
    X, y, idx = [], [], []
    for t in range(lookback, len(residual_arr)):
        X.append(residual_arr[t - lookback:t])
        y.append(residual_arr[t])
        idx.append(t)
    X = np.asarray(X, dtype=np.float32)[..., None]
    y = np.asarray(y, dtype=np.float32)[:, None]
    idx = np.asarray(idx, dtype=np.int64)
    return X, y, idx


def build_lstm_residual_model(lookback: int, lr: float = 1e-3):
    inp = layers.Input(shape=(lookback, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.15)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MAE]
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


def train_one_group(group, group_blood_df, weather_df):
    LOOK_BACK = 60
    lstm_lr = 1e-3

    # 1.blood特征
    label_col = TARGET_COL
    group_blood_df = add_prev_rolling_sums(group_blood_df, windows=(1, 3, 7, 14, 28))
    group_blood_df = add_dynamic_feature(group_blood_df)

    # 2.合并blood weather
    df = pd.merge(group_blood_df, weather_df, on="date", how="left")
    print(f"all len is {len(df)}")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # xgb 特征
    xgb_feat_cols = []
    for col in df.columns:
        if col in [TARGET_COL, label_col, DATE_COL, 'date_x', 'date_y', STATION_COL,
                   'lag1', '1d_sum']:
            continue
        xgb_feat_cols.append(col)

    lstm_feat_cols = []
    for col in df.columns:
        if col in [TARGET_COL, label_col, DATE_COL, 'date_x', 'date_y',STATION_COL,
                   'lag1', '1d_sum']:
            continue
        if (col.startswith('lag') | col.startswith('rolling') | col.startswith('absdiff')
                | col.startswith('diff') | col.startswith('ewm') | col.endswith('sum')):
            continue
        lstm_feat_cols.append(col)

    # lstm
    TEST_RATIO = 0.2
    df["trend"] = df[TARGET_COL].rolling(7).mean()
    df["residual"] = df[TARGET_COL] - df["trend"]
    split = int(len(df) * (1 - TEST_RATIO))
    train_end = split
    lstm_mat = df[lstm_feat_cols + ["trend"]].astype(float).values
    scaler = MinMaxScaler()
    scaler.fit(lstm_mat[:train_end])
    lstm_scaled = scaler.transform(lstm_mat)
    y_index = lstm_scaled.shape[1] - 1  # trend 在最后一列

    X_all, y_all = make_lstm_xy(lstm_scaled, LOOK_BACK, y_index)

    sample_split = train_end - LOOK_BACK

    X_train, X_test = X_all[:sample_split], X_all[sample_split:]
    y_train, y_test = y_all[:sample_split], y_all[sample_split:]

    # val 取训练末尾10%
    val_cut = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_cut], X_train[val_cut:]
    y_tr, y_val = y_train[:val_cut], y_train[val_cut:]

    model = build_lstm(LOOK_BACK, X_all.shape[2])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=64,
        callbacks=[es],
        verbose=0
    )

    trend_pred_scaled = model.predict(X_test, verbose=0).flatten()
    trend_true_scaled = y_test

    feature_dim = lstm_scaled.shape[1]
    trend_pred = inverse_one_col(scaler, trend_pred_scaled, feature_dim, y_index)
    trend_true = inverse_one_col(scaler, trend_true_scaled, feature_dim, y_index)

    # 3.切分训练集 测试集
    X, y, target_dates = df[xgb_feat_cols].values, df[label_col].values, df[DATE_COL].values
    X_train, X_test, Y_train, Y_test, d_train, d_test = train_test_split(
        X,
        y,
        target_dates,
        test_size=0.1,
        shuffle=False)
    cutoff_date = pd.to_datetime(d_test).min()  # 这时就等价于“测试集第一天”
    print(f"[Split] cutoff_date={cutoff_date}  train={len(X_train)} test={len(Y_train)}")

    # 3.xgb 训练
    n_estimators = 100
    learning_rate = 0.4
    params = {
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'min_child_weight': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'scale_pos_weight': 1,
        'seed': 1024,
        'max_depth': 8,
        'eval_metric': mean_squared_error,
        'early_stopping_rounds': 10,
        'missing': 0
    }
    sample_weight = []
    for y in Y_train:
        weight = 3
        if y >= 45:
            weight = 5
        elif y >= 36:
            weight = 5
        elif y >= 26:
            weight = 6
        elif y >= 18:
            weight = 1
        sample_weight.append(weight)
    xgb = XGBRegressor(**params)
    xgb.fit(X_train, Y_train, eval_set=[
        (X_test, Y_test)], sample_weight=sample_weight)

    df["y_xgb"] = xgb.predict(df[xgb_feat_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    df["resid"] = (df[TARGET_COL] - df["y_xgb"]).astype(np.float32)

    # -------------------------
    # 2) LSTM：学习 residual 序列（只用残差，不做额外特征）
    #    预测 resid[t]，用于修正 y_xgb[t]
    # -------------------------
    resid = df["resid"].to_numpy(dtype=np.float32)
    # residual 标准化（只在 train 段 fit）
    resid_scaler = StandardScaler()
    resid_train = resid[df[DATE_COL] < cutoff_date]
    resid_scaler.fit(resid_train.reshape(-1, 1))
    resid_s = resid_scaler.transform(resid.reshape(-1, 1)).reshape(-1).astype(np.float32)

    X_seq, y_seq, idx_seq = make_residual_sequences(resid_s, lookback)

    # 将序列样本按时间切分到 train/test（以 idx 对应的 date 为准）
    sample_dates = df.loc[idx_seq, DATE_COL].to_numpy()
    tr_mask = sample_dates < np.datetime64(cutoff_date)
    te_mask = sample_dates >= np.datetime64(cutoff_date)

    X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
    X_seq_te, y_seq_te = X_seq[te_mask], y_seq[te_mask]
    dates_te = sample_dates[te_mask]
    print(f"[Seq] train_samples={len(X_seq_tr)} test_samples={len(X_seq_te)}")

    lstm = build_lstm_residual_model(lookback, lr=lstm_lr)
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        callbacks.ModelCheckpoint(os.path.join("./", "best_lstm.keras"),
                                  monitor="val_loss", save_best_only=True)
    ]
    hist = lstm.fit(
        X_seq_tr, y_seq_tr,
        validation_data=(X_seq_te, y_seq_te),
        epochs=120,
        batch_size=32,
        verbose=1,
        callbacks=cbs
    )

    # 预测 residual（标准化空间） -> inverse 回真实 residual
    resid_pred_s = lstm.predict(X_seq_te, batch_size=32).reshape(-1, 1)
    resid_pred = resid_scaler.inverse_transform(resid_pred_s).reshape(-1).astype(np.float32)

    # 对齐到原始 df 的时间索引
    idx_te = idx_seq[te_mask]
    df.loc[idx_te, "resid_pred"] = resid_pred

    # 融合预测：y_hat = y_xgb + resid_pred
    df["y_hat"] = df["y_xgb"] + df["resid_pred"]

    # 评估：只评估测试区间中，存在 resid_pred 的那些天（因为序列需要 lookback）
    eval_df = df.loc[idx_te, [DATE_COL, TARGET_COL, "y_xgb", "y_hat"]].copy()
    eval_df = eval_df.sort_values(DATE_COL).reset_index(drop=True)

    mae_xgb = mean_absolute_error(eval_df[TARGET_COL], eval_df["y_xgb"])
    mae_hat = mean_absolute_error(eval_df[TARGET_COL], eval_df["y_hat"])
    print(f"[MAE] XGB={mae_xgb:.3f}  XGB+LSTM={mae_hat:.3f}")

    # -------------------------
    # 输出：loss 曲线、预测对比图、保存模型
    # -------------------------
    # loss curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(hist.history.get("loss", []), label="train_loss")
    plt.plot(hist.history.get("val_loss", []), label="val_loss")
    plt.title(f"{group} LSTM Residual Loss", fontproperties=FONT)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # pred vs true（只画测试最后 120 天里可评估的部分）
    k = min(120, len(eval_df))
    plt.figure(figsize=(12, 4.5))
    plt.plot(eval_df[DATE_COL].iloc[-k:], eval_df[TARGET_COL].iloc[-k:], label="true")
    plt.plot(eval_df[DATE_COL].iloc[-k:], eval_df["y_xgb"].iloc[-k:], label="xgb")
    plt.plot(eval_df[DATE_COL].iloc[-k:], eval_df["y_hat"].iloc[-k:], label="xgb+lstm")
    plt.title(f"{group} predict", fontproperties=FONT)
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL, fontproperties=FONT)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 1.获取数据
    all_blood_df = load_blood_data()
    weather_df = load_weather_data()

    # 2.获取
    station_set = all_blood_df['血站'].unique().tolist()
    blood_types = ["ALL", "A", "B", "O", "AB"]
    for station in station_set:
        if station != '北京市红十字血液中心':
            continue
        for blood_type in blood_types:
            group = station + "_" + blood_type
            group_df = None
            if blood_type == "ALL":
                group_df = all_blood_df[all_blood_df[STATION_COL] == station].copy()
                group_df = group_df.groupby(DATE_COL, as_index=False).agg({
                    TARGET_COL: "sum",
                    BLOOD_COL: "first",
                    STATION_COL: "first"
                })
            else:
                group_df = all_blood_df[
                    (all_blood_df[STATION_COL] == station) &
                    (all_blood_df[BLOOD_COL] == blood_type)
                    ].copy()
            group_df.drop(columns=[BLOOD_COL, STATION_COL], inplace=True)
            train_one_group(group, group_df, weather_df)

    # 3.整体


if __name__ == '__main__':
    print(sklearn.__version__)
    main()
