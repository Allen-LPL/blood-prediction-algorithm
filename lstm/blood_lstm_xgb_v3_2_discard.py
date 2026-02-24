import os
import json
import argparse
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import matplotlib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models, optimizers
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler

FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")


# =========================
# Metrics
# =========================
def safe_mape(y_true, y_pred, eps=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def smape(y_true, y_pred, eps=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# Data config
# =========================
@dataclass
class Cols:
    # blood csv
    date: str = "date"
    station: str = "血站"
    blood_type: str = "血型"
    volume: str = "总血量"

    # calendar csv
    cal_date: str = "date"


# -----------------------------
# Feature engineering
# -----------------------------
def parse_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


def normalize_bool_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        else:
            s = df[c].astype(str).str.lower()
            df[c] = s.map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(df[c])
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def add_temp_features(df: pd.DataFrame, temp_col: str = "平均温度") -> pd.DataFrame:
    df = df.copy()
    if temp_col in df.columns:
        t = pd.to_numeric(df[temp_col], errors="coerce")
        if t.isna().all():
            df[temp_col] = 0.0
        else:
            df[temp_col] = t.fillna(t.mean())
        df["temp_sq"] = df[temp_col].astype(float) ** 2
        df["temp_diff"] = df[temp_col].astype(float).diff().fillna(0.0)
        df["temp_rolling3"] = df[temp_col].astype(float).rolling(3, min_periods=1).mean()
    return df


def add_target_lag_features(df: pd.DataFrame, y_col: str = "y") -> pd.DataFrame:
    df = df.copy()
    df["lag1"] = df[y_col].shift(1)
    df["lag7"] = df[y_col].shift(7)
    df["rolling7"] = df[y_col].rolling(7, min_periods=1).mean()
    return df


def build_group_daily(
        blood_df: pd.DataFrame,
        cal_df: pd.DataFrame,
        cols: Cols,
        station_name: str,
        blood_type: str,
        make_all: bool = False,
) -> pd.DataFrame:
    """
    输出：按 date 的日序列，包含 y、calendar/weather 特征。
    y = 当天总血量；y_next = 下一天总血量(label)。
    """
    b = blood_df.copy()
    c = cal_df.copy()

    b = b[b[cols.station] == station_name].copy()
    if not make_all:
        b = b[b[cols.blood_type] == blood_type].copy()

    # 日聚合
    b_day = b.groupby(cols.date, as_index=False)[cols.volume].sum()
    b_day = b_day.rename(columns={cols.volume: "y"})
    b_day = parse_date_col(b_day, cols.date)

    c = parse_date_col(c, cols.cal_date)
    c = c.rename(columns={cols.cal_date: cols.date})

    # join（以日历为主，缺失 y=0）
    df = pd.merge(b_day, c, on=cols.date, how="left")
    # df = c.merge(b_day, on=cols.date, how="left")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)

    # sort
    df = df.sort_values(cols.date).reset_index(drop=True)

    # 温度衍生
    df = add_temp_features(df, temp_col="平均温度")

    # 目标滞后/rolling（可选）
    df = add_dynamic_features(df, y_col="y")

    # 下一天 label
    df["y_next"] = df["y"].shift(-1)
    df = df.dropna(subset=["y_next"]).reset_index(drop=True)

    return df


def build_lstm_features(df: pd.DataFrame, weather_prefix: str = "天气_") -> List[str]:
    """
    LSTM 输入特征：包含 y（当天）+ 周期 + 温度 + 天气onehot
    注意：不直接放 y_next（会泄漏）
    """
    lstm_features = [
        "y",
        "weekday_sin", "weekday_cos",
        "month_sin", "month_cos",
        "平均温度", "temp_sq", "temp_diff", "temp_rolling3",
        "temp_rolling7", "temp_rolling14"
    ]
    dynamic_features = ["lag1", "lag7", "lag14", "absdiff1", "rolling_std7", "rolling_std14"]
    lstm_features = lstm_features + dynamic_features

    feats = []
    for c in lstm_features:
        if c in df.columns:
            feats.append(c)

    feats += [c for c in df.columns if c.startswith(weather_prefix)]

    # 去重保持顺序
    seen = set()
    out = []
    for x in feats:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_xgb_features(df: pd.DataFrame, weather_prefix: str = "天气_") -> List[str]:
    """
    XGB 输入特征：日历+温度+天气（不含 y / trend_pred，避免模型走捷径）
    """
    base = [
        "weekday_sin", "weekday_cos",
        "month_sin", "month_cos",
        "weekday", "month", "day", "dayofyear", "weekofyear",
        "is_weekend", "is_holiday", "is_workday",
        "is_before_holiday", "is_after_holiday",
        "平均温度", "temp_sq", "temp_diff", "temp_rolling3",
        "temp_rolling7", "temp_rolling14",
    ]
    dynamic_features = ["lag1", "lag7", "lag14", "absdiff1", "rolling_std7", "rolling_std14"]
    base = base + dynamic_features
    feats = [c for c in base if c in df.columns]
    feats += [c for c in df.columns if c.startswith(weather_prefix)]

    seen = set()
    out = []
    for x in feats:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# -----------------------------
# Sequence building
# -----------------------------
def make_sequences(
        df: pd.DataFrame,
        x_cols: List[str],
        y_col: str,
        look_back: int,
        x_scaler: StandardScaler,
        y_scaler: MinMaxScaler,
        fit: bool,
        fit_end_idx: Optional[int] = None,
        y_fit_start_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 [N, look_back, F] -> y_scaled
    y_col 必须是 y_next（下一天）
    """
    X_raw = df[x_cols].astype(float).values
    y_raw = df[y_col].astype(float).values.reshape(-1, 1)

    if fit:
        if fit_end_idx is not None:
            if fit_end_idx < 0 or fit_end_idx >= len(df):
                raise ValueError(f"fit_end_idx out of range: {fit_end_idx}, len={len(df)}")
            if y_fit_start_idx < 0 or y_fit_start_idx > fit_end_idx:
                raise ValueError(
                    f"invalid y_fit_start_idx={y_fit_start_idx} for fit_end_idx={fit_end_idx}"
                )

            x_scaler.fit(X_raw[:fit_end_idx + 1])
            y_scaler.fit(y_raw[y_fit_start_idx:fit_end_idx + 1])
            X_scaled = x_scaler.transform(X_raw)
            y_scaled = y_scaler.transform(y_raw)
        else:
            X_scaled = x_scaler.fit_transform(X_raw)
            y_scaled = y_scaler.fit_transform(y_raw)
    else:
        X_scaled = x_scaler.transform(X_raw)
        y_scaled = y_scaler.transform(y_raw)

    X_seq, y_seq = [], []
    for i in range(look_back - 1, len(df)):
        X_seq.append(X_scaled[i - look_back + 1:i + 1, :])
        y_seq.append(y_scaled[i, 0])

    return np.asarray(X_seq, np.float32), np.asarray(y_seq, np.float32)


def time_split_indices(n: int, train_ratio=0.7, val_ratio=0.15):
    if n < 3:
        raise ValueError(f"not enough sequence samples to split: n={n} (need >= 3)")

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    idx = np.arange(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"bad split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, n={n}"
        )

    return train_idx, val_idx, test_idx


# =========================
# LSTM (trend) — fixed
# =========================
def build_lstm_model(input_shape: Tuple[int, int], lr: float = 1e-3) -> tf.keras.Model:
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    m.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=1.0),
    )
    return m


def train_predict_lstm(
        group_name: str,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        y_scaler: MinMaxScaler,
        save_dir: str,
        epochs: int,
        batch_size: int,
        lr: float = 1e-3,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray, Dict]:
    ensure_dir(save_dir)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), lr=lr)

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
        callbacks=cbs,
    )

    # test pred (scaled) -> inverse
    pred_test_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    trend_pred_test = y_scaler.inverse_transform(pred_test_scaled).reshape(-1).astype(float)

    # y_true test inverse
    y_true_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1).astype(float)

    # “常数预测”硬检测（防你再次遇到 8888 常数线）
    if len(trend_pred_test) >= 12:
        if float(np.std(trend_pred_test[:12])) < 1e-6:
            raise RuntimeError(f"[{group_name}] LSTM trend collapsed to constant. "
                               f"Check feature join / scaler / data validity.")

    info = {
        "trend_mae": float(mean_absolute_error(y_true_test, trend_pred_test)),
        "trend_mape": float(safe_mape(y_true_test, trend_pred_test, eps=1.0)),
        "trend_smape": float(smape(y_true_test, trend_pred_test, eps=1.0)),
        "history": {
            "loss": [float(x) for x in hist.history.get("loss", [])],
            "val_loss": [float(x) for x in hist.history.get("val_loss", [])],
        }
    }

    # save model & curve
    model_path = os.path.join(save_dir, f"{group_name}_lstm.keras")
    model.save(model_path)

    fig = plt.figure()
    plt.plot(info["history"]["loss"], label="train_loss")
    plt.plot(info["history"]["val_loss"], label="val_loss")
    plt.title(f"{group_name} LSTM loss", fontproperties=FONT)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{group_name}_lstm_train.png"), dpi=140)
    plt.close(fig)

    return model, trend_pred_test, y_true_test, info


# =========================
# XGB (resid_ratio)
# =========================
def train_predict_xgb_resid_ratio(
        group_name: str,
        df_seq: pd.DataFrame,
        xgb_features: List[str],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        save_dir: str,
) -> Tuple[XGBRegressor, np.ndarray, Dict]:
    ensure_dir(save_dir)

    if not xgb_features:
        raise ValueError(f"[{group_name}] xgb feature list is empty")

    X = df_seq[xgb_features].astype(float).values
    y = df_seq["resid_ratio_true"].astype(float).values

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    w_train = 1.0 / (1.0 + np.abs(y_train))
    w_train = np.clip(w_train, 0.2, 1.0)

    model = XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror",
    )

    if len(X_val) > 0:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            sample_weight=w_train
        )
    else:
        model.fit(X_train, y_train, verbose=False, sample_weight=w_train)

    pred = model.predict(X_test).astype(float)

    info = {
        "resid_ratio_mae": float(mean_absolute_error(y_test, pred)),
        "resid_ratio_abs_mean": float(np.mean(np.abs(pred))),
    }

    model_path = os.path.join(save_dir, f"{group_name}_xgb.json")
    model.save_model(model_path)

    return model, pred, info


# =========================
# Plot
# =========================
def plot_real_vs_pred(out_path: str, title: str, y_true: np.ndarray, y_pred: np.ndarray):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="real")
    plt.plot(y_pred, label="pred(LSTM+XGB)")
    plt.title(title, fontproperties=FONT)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def add_dynamic_features(all_df: pd.DataFrame, y_col="y") -> pd.DataFrame:
    all_df["lag1"] = all_df[y_col].shift(1)
    all_df["lag7"] = all_df[y_col].shift(7)
    all_df["lag14"] = all_df[y_col].shift(14)

    all_df["absdiff1"] = (all_df[y_col] - all_df["lag1"]).abs()
    all_df["rolling_std7"] = all_df[y_col].rolling(7).std()
    all_df["rolling_std14"] = all_df[y_col].rolling(14).std()
    all_df["rolling_absdiff7"] = all_df["absdiff1"].rolling(7).mean()
    all_df.fillna(0, inplace=True)
    return all_df


# =========================
# Train one group
# =========================
def train_one_group(
        station: str,
        blood_type: str,
        df_group: pd.DataFrame,
        look_back: int,
        out_dir: str,
        epochs: int,
        batch_size: int,
        weather_prefix: str = "天气_",
        eps: float = 1.0,
):
    group_name = f"{station}_{blood_type}"

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "reports"))
    ensure_dir(os.path.join(out_dir, "figs"))
    ensure_dir(os.path.join(out_dir, "models_lstm"))
    ensure_dir(os.path.join(out_dir, "models_xgb"))

    # features
    lstm_feats = build_lstm_features(df_group, weather_prefix=weather_prefix)
    xgb_feats = build_xgb_features(df_group, weather_prefix=weather_prefix)

    # scalers
    x_scaler = StandardScaler()
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    seq_count = len(df_group) - look_back + 1
    if seq_count < 3:
        raise ValueError(
            f"[{group_name}] not enough rows for sequence split: len={len(df_group)}, look_back={look_back}"
        )

    train_idx, val_idx, test_idx = time_split_indices(seq_count, train_ratio=0.7, val_ratio=0.15)
    # 防泄漏：scaler 只用训练时段拟合
    # seq index i 对应原始行结束位置 (look_back - 1 + i)
    fit_end_idx = look_back - 1 + int(train_idx[-1])

    # sequences
    X_seq, y_seq = make_sequences(
        df_group,
        x_cols=lstm_feats,
        y_col="y_next",
        look_back=look_back,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        fit=True,
        fit_end_idx=fit_end_idx,
        y_fit_start_idx=look_back - 1,
    )

    n = len(y_seq)
    if n != seq_count:
        raise ValueError(f"[{group_name}] unexpected sequence count: got {n}, expect {seq_count}")

    X_train, y_train = X_seq[train_idx], y_seq[train_idx]
    X_val, y_val = X_seq[val_idx], y_seq[val_idx]
    X_test, y_test = X_seq[test_idx], y_seq[test_idx]

    # ---- LSTM trend ----
    model_lstm, trend_pred_test, y_true_test, trend_info = train_predict_lstm(
        group_name=group_name,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        y_scaler=y_scaler,
        save_dir=os.path.join(out_dir, "models_lstm"),
        epochs=epochs,
        batch_size=batch_size,
    )

    # ---- Build df_seq aligned to X_seq/y_seq ----
    # X_seq/y_seq corresponds to df_group rows from (look_back-1) to end
    df_seq = df_group.iloc[look_back - 1:].reset_index(drop=True).copy()
    assert len(df_seq) == len(y_seq), (len(df_seq), len(y_seq))

    # predict trend for all samples (needed to build resid_ratio labels consistently)
    pred_all_scaled = model_lstm.predict(X_seq, verbose=0).reshape(-1, 1)
    trend_pred_all = y_scaler.inverse_transform(pred_all_scaled).reshape(-1).astype(float)

    # y_next true in raw scale
    y_next_all = y_scaler.inverse_transform(y_seq.reshape(-1, 1)).reshape(-1).astype(float)

    df_seq["trend_pred"] = trend_pred_all
    df_seq["y_next_true"] = y_next_all

    # ---- Resid ratio label (稳定版) ----
    # FLOOR: 防止 trend_pred 极小导致 ratio 爆炸
    # 业务建议：ALL 的 floor 更高；其它血型低一点
    p5 = float(np.percentile(df_seq["trend_pred"].values, 5))
    if blood_type == "ALL":
        FLOOR = max(p5, 1000.0)
    else:
        FLOOR = max(p5, 200.0)

    denom = np.maximum(df_seq["trend_pred"].values, FLOOR)
    df_seq["resid_ratio_true"] = (df_seq["y_next_true"].values - df_seq["trend_pred"].values) / denom

    # ---- XGB on resid_ratio ----
    model_xgb, resid_ratio_pred_test, resid_info = train_predict_xgb_resid_ratio(
        group_name=group_name,
        df_seq=df_seq,
        xgb_features=xgb_feats,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        save_dir=os.path.join(out_dir, "models_xgb"),
    )

    # ---- Compose total pred (test) ----
    resid_ratio_pred_test = np.clip(resid_ratio_pred_test, -0.8, 2.0)
    trend_test = df_seq["trend_pred"].values[test_idx]
    total_pred_test = trend_test * (1.0 + resid_ratio_pred_test)

    total_mae = float(mean_absolute_error(y_true_test, total_pred_test))
    total_mape = float(safe_mape(y_true_test, total_pred_test, eps=1.0))
    total_smape = float(smape(y_true_test, total_pred_test, eps=1.0))

    print(f"[{group_name}] Total: MAE={total_mae:.2f}, MAPE={total_mape:.4f}, sMAPE={total_smape:.4f}")
    print(
        f"[{group_name}] Trend: MAE={trend_info['trend_mae']:.2f}, MAPE={trend_info['trend_mape']:.4f}, sMAPE={trend_info['trend_smape']:.4f}")
    print(
        f"[{group_name}] resid_ratio MAE={resid_info['resid_ratio_mae']:.4f}, abs_mean={resid_info['resid_ratio_abs_mean']:.4f}")

    # ---- plot ----
    plot_real_vs_pred(
        out_path=os.path.join(out_dir, "figs", f"{group_name}_total_predict.png"),
        title=f"{group_name} total predict",
        y_true=y_true_test,
        y_pred=total_pred_test,
    )

    # ---- report ----
    report = {
        "group": group_name,
        "look_back": look_back,
        "floor": FLOOR,
        "lstm_features": lstm_feats,
        "xgb_features": xgb_feats,
        "metrics": {
            "total_mae": total_mae,
            "total_mape": total_mape,
            "total_smape": total_smape,
            "trend_mae": trend_info["trend_mae"],
            "trend_mape": trend_info["trend_mape"],
            "trend_smape": trend_info["trend_smape"],
            "resid_ratio_mae": resid_info["resid_ratio_mae"],
            "resid_ratio_abs_mean": resid_info["resid_ratio_abs_mean"],
        }
    }
    with open(os.path.join(out_dir, "reports", f"{group_name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"end train: {group_name}\n")
    return report


# -----------------------------
# Loading
# -----------------------------
def load_blood_csv(path: str, cols: Cols) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = parse_date_col(df, cols.date)

    # 统一列名：允许你 CSV 是 “血站/血型/总血量”，也可以传参数改
    needed = [cols.date, cols.station, cols.blood_type, cols.volume]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"blood csv missing column: {c} (have {list(df.columns)})")

    df[cols.volume] = pd.to_numeric(df[cols.volume], errors="coerce").fillna(0.0)
    return df


def load_calendar_csv(path: str, cols: Cols) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = parse_date_col(df, cols.cal_date)

    # 节假日前后
    df["date"] = pd.to_datetime(df["date"])
    holidays = set(df.loc[df["is_holiday"] == 1, "date"])
    df["is_before_holiday"] = df["date"].apply(
        lambda x: 1 if (x + timedelta(days=1)) in holidays else 0
    )
    df["is_after_holiday"] = df["date"].apply(
        lambda x: 1 if (x - timedelta(days=1)) in holidays else 0
    )

    df["temp_sq"] = df["平均温度"] ** 2
    df["temp_diff"] = df["平均温度"].diff()
    df["temp_rolling3"] = df["平均温度"].rolling(3).mean()
    df["temp_rolling7"] = df["平均温度"].rolling(7).mean()
    df["temp_rolling14"] = df["平均温度"].rolling(14).mean()

    df = pd.get_dummies(df, columns=["天气"])

    # 常见 bool 列
    bool_cols = ["is_weekend", "is_holiday", "is_workday", "is_before_holiday", "is_after_holiday"]
    df = normalize_bool_cols(df, bool_cols)

    # 温度列转数值
    if "平均温度" in df.columns:
        df["平均温度"] = pd.to_numeric(df["平均温度"], errors="coerce").fillna(df["平均温度"].mean())

    # bool天气列 -> 0/1
    weather_cols = [c for c in df.columns if c.startswith("天气_")]
    for c in weather_cols:
        df[c] = df[c].astype(int)

    # weekofyear 如果没有，尝试生成
    if "weekofyear" not in df.columns and cols.cal_date in df.columns:
        d = pd.to_datetime(df[cols.cal_date])
        df["weekofyear"] = d.dt.isocalendar().week.astype(int)

    return df


# -----------------------------
# CLI main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blood_csv", type=str, default="./feature/remove_group_data.csv")
    ap.add_argument("--calendar_csv", type=str, default="./feature/blood_calendar_weather_2015_2026.csv")
    ap.add_argument("--out_dir", type=str, default="outputs_v3", help="输出目录")

    # 列名映射
    ap.add_argument("--col_date", type=str, default="date")
    ap.add_argument("--col_station", type=str, default="血站")
    ap.add_argument("--col_blood_type", type=str, default="血型")
    ap.add_argument("--col_volume", type=str, default="总血量")
    ap.add_argument("--col_cal_date", type=str, default="date")

    ap.add_argument("--look_back", type=int, default=60, help="窗口长度（如 14/28/56/84/60）")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--stations", type=str, default="北京市红十字血液中心")
    ap.add_argument("--blood_types", type=str, default="A,B,O,AB", help="逗号分隔；默认A,B,O,AB，会自动加ALL")

    args = ap.parse_args()

    cols = Cols(
        date=args.col_date,
        station=args.col_station,
        blood_type=args.col_blood_type,
        volume=args.col_volume,
        cal_date=args.col_cal_date,
    )

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "reports"))

    blood_df = load_blood_csv(args.blood_csv, cols)
    cal_df = load_calendar_csv(args.calendar_csv, cols)
    cal_df.fillna(0.0, inplace=True)

    # stations
    if args.stations.strip():
        stations = [x.strip() for x in args.stations.split(",") if x.strip()]
    else:
        stations = sorted(blood_df[cols.station].dropna().unique().tolist())

    # blood types
    blood_types = [x.strip() for x in args.blood_types.split(",") if x.strip()]
    # 自动加 ALL
    if "ALL" not in blood_types:
        blood_types = ["ALL"] + blood_types

    all_reports = []
    for st in stations:
        for bt in blood_types:
            make_all = (bt == "ALL")
            df_group = build_group_daily(
                blood_df=blood_df,
                cal_df=cal_df,
                cols=cols,
                station_name=st,
                blood_type=bt,
                make_all=make_all,
            )

            # 防御：数据太短就跳过
            if len(df_group) < args.look_back + 30:
                print(f"[SKIP] {st}_{bt} too short: len={len(df_group)} (need > look_back+30)")
                continue

            rep = train_one_group(
                station=st,
                blood_type=bt,
                df_group=df_group,
                look_back=args.look_back,
                out_dir=args.out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            all_reports.append(rep)

    # 汇总
    if all_reports:
        df_sum = pd.DataFrame([{
            "group": r["group"],
            "look_back": r["look_back"],
            **r["metrics"]
        } for r in all_reports])
        df_sum = df_sum.sort_values(["total_mape", "total_mae"]).reset_index(drop=True)
        df_sum.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False, encoding="utf-8-sig")
        print("saved:", os.path.join(args.out_dir, "summary.csv"))
    else:
        print("No reports generated.")

    print("DONE.")


if __name__ == "__main__":
    main()
