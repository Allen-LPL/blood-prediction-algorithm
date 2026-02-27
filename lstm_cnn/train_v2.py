# -*- coding: utf-8 -*-
"""
train_v2_delta.py
每个血站+血型单独训练 CNN+LSTM，多步预测未来7天（direct multi-output）。

修复 train_v1 中“预测几乎是常数直线”的常见塌陷：
1) 目标改为“相对变化/残差学习”：预测 log1p(y_future) - log1p(y_last_in_window)
2) y_delta 再做 StandardScaler；预测后再反变换 + 复原到原尺度
3) 损失从 MAE 改为 Huber（更不容易学成常数中位数），并保留 MAE 监控
4) 缺失填充：外生天气特征可 ffill/bfill；目标派生的 lag/rolling/diff 等不 bfill（避免用未来值填前面）

运行示例：
python train_v2_delta.py --blood_csv blood.csv --weather_csv weather.csv --out_dir out --station "北京市红十字血液中心"
"""
import argparse
import json
import os
import re
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

try:
    from matplotlib.font_manager import FontProperties

    FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
except Exception:
    FONT = None


# =========================
# Utils
# =========================
def slugify(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", s)
    return s[:120]


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def masked_mape(y_true, y_pred, min_true=1.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true >= min_true
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def wape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps)


def build_delta_sample_weights(y_delta: np.ndarray, alpha: float = 2.0, quantile: float = 0.75) -> np.ndarray:
    """
    给变化幅度大的窗口更高权重，降低模型退化到“恒等于昨天”解的概率。
    y_delta: (N, horizon)
    """
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
    """
    组合损失：Huber + Anti-Persistence
    在真实变化较大的样本上，惩罚 y_pred 过于接近 0（即“等于昨天”）。
    所有阈值都在 y_delta 标准化空间中。
    """
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


# =========================
# Feature engineering
# =========================
def build_dynamic_feature(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """df 必须包含 date, is_holiday, 平均温度, 天气, 以及 target_col(总血量)"""
    df = df.copy()

    holidays = set(df.loc[df.get("is_holiday", 0) == 1, "date"])
    df["is_before_holiday"] = df["date"].apply(lambda x: 1 if (x + timedelta(days=1)) in holidays else 0)
    df["is_after_holiday"] = df["date"].apply(lambda x: 1 if (x - timedelta(days=1)) in holidays else 0)

    # days_until/days_since（clip 到 30 天）
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

    # 温度
    if "平均温度" in df.columns:
        df["temp_sq"] = df["平均温度"] ** 2
        df["temp_diff"] = df["平均温度"].diff()
        df["temp_rolling3"] = df["平均温度"].rolling(3).mean()

    # 天气 one-hot
    if "天气" in df.columns:
        df = pd.get_dummies(df, columns=["天气"])

    # 目标派生滚动/滞后（注意：这里只用历史，不做 bfill）
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
    df["diff7"] = (y - df["lag7"])
    df["diff14"] = (y - df["lag14"])

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
    """构造 (station, blood_type) 的完整日序列，断档 y=0；合并天气+派生特征。"""
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
    df[target_col] = df[target_col].fillna(0.0)  # 断档=0

    df = df.merge(weather, on="date", how="left").sort_values("date").reset_index(drop=True)
    df = build_dynamic_feature(df, target_col)

    # ------- 缺失处理：外生 vs 目标派生分开 -------
    # 目标派生列（不允许 bfill）
    y_derived_prefix = (
    "lag", "rolling", "trend", "residual", "ewm", "absdiff", "diff", "rolling_std", "rolling_absdiff")
    y_cols = [c for c in df.columns if (c == target_col) or c.startswith(y_derived_prefix)]
    exo_cols = [c for c in df.columns if c not in (["date"] + y_cols)]

    # 统一数值化
    for c in exo_cols + y_cols:
        if c == "date":
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 外生：ffill+bfill 再 0
    if exo_cols:
        df[exo_cols] = df[exo_cols].ffill().bfill().fillna(0.0)
    # 目标派生：只 fillna 0（不 bfill）
    df[y_cols] = df[y_cols].fillna(0.0)

    feature_cols = [c for c in df.columns if c not in ["date", "lag1"]]
    return df, feature_cols


# =========================
# Windowing: 目标改为 delta
# =========================
def make_windows_multistep_delta(
        df: pd.DataFrame,
        feature_cols,
        target_col: str,
        lookback: int,
        horizon: int,
):
    """
    用 [t-lookback ... t-1] 预测 [t ... t+horizon-1]
    目标为：y_delta[k] = log1p(y[t+k]) - log1p(y[t-1])  (k=0..h-1)
    同时返回 base_log = log1p(y[t-1]) 供反变换恢复绝对值
    """
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    tgt = df[target_col].to_numpy(dtype=np.float32)
    tgt_log = np.log1p(np.maximum(tgt, 0.0)).astype(np.float32)
    dates = df["date"].to_numpy(dtype="datetime64[ns]")

    X_list, y_list, base_list, sd_list, ed_list = [], [], [], [], []
    max_t = len(df) - horizon
    for t in range(lookback, max_t):
        X_list.append(feat[t - lookback:t])
        base = tgt_log[t - 1]  # 输入窗口最后一天
        y_f = tgt_log[t:t + horizon]
        y_delta = y_f - base
        y_list.append(y_delta)
        base_list.append(base)
        sd_list.append(dates[t])
        ed_list.append(dates[t + horizon - 1])

    X = np.asarray(X_list, dtype=np.float32)
    y_delta = np.asarray(y_list, dtype=np.float32)
    base_log = np.asarray(base_list, dtype=np.float32)  # (N,)
    start_dates = np.asarray(sd_list)
    end_dates = np.asarray(ed_list)
    return X, y_delta, base_log, start_dates, end_dates


def time_split_by_end_date(X, y, base_log, start_dates, end_dates, test_ratio=0.2):
    uniq = np.unique(end_dates)
    uniq = np.sort(uniq)
    cut = int(len(uniq) * (1.0 - test_ratio))
    cut = max(1, min(cut, len(uniq) - 1))
    cutoff = uniq[cut]

    train_mask = end_dates < cutoff
    test_mask = end_dates >= cutoff

    return (
        X[train_mask], y[train_mask], base_log[train_mask], start_dates[train_mask], end_dates[train_mask],
        X[test_mask], y[test_mask], base_log[test_mask], start_dates[test_mask], end_dates[test_mask],
        cutoff
    )


# =========================
# Model
# =========================
def build_cnn_lstm_multistep(
        lookback,
        n_features,
        horizon,
        lr=1e-3,
        ap_lambda=1.0,
        ap_change_thr=0.6,
        ap_margin=0.6,
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

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[tf.keras.metrics.MAE]
    )
    return model


# =========================
# Plotting
# =========================
def plot_loss(history, out_path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pred_vs_true(dates, y_true, y_pred, out_path, title):
    plt.figure(figsize=(10, 4.5))
    plt.plot(pd.to_datetime(dates), y_true, label="true")
    plt.plot(pd.to_datetime(dates), y_pred, label="pred")
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


def plot_pred_vs_true_v2(dates, y_true, y_pred, out_path, title, y_baseline=None):
    k = min(60, len(dates))
    y_true_tmp = np.asarray(y_true).reshape(-1, 1)
    y_pred_tmp = np.asarray(y_pred).reshape(-1, 1)
    y_base_tmp = None if y_baseline is None else np.asarray(y_baseline).reshape(-1, 1)
    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(dates[-k:]), y_true_tmp[-k:, 0], label="true")
    plt.plot(pd.to_datetime(dates[-k:]), y_pred_tmp[-k:, 0], label="pred")
    if y_base_tmp is not None:
        plt.plot(pd.to_datetime(dates[-k:]), y_base_tmp[-k:, 0], label="baseline(yesterday)", linestyle="--")
    plt.title(title, fontproperties=FONT)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# Training per group
# =========================
def train_one_group(
        blood, weather, station, blood_type, out_dir,
        lookback=60, horizon=7, test_ratio=0.2, epochs=120, batch_size=32, lr=1e-3,
        spike_weight_alpha=6.0,
        ap_lambda=1.0,
        ap_change_thr=0.6,
        ap_margin=0.6,
):
    series_pack = build_group_daily_series(blood, weather, station, blood_type, target_col="总血量")
    if series_pack is None:
        return None

    df, feature_cols = series_pack

    # 基础 sanity：lag1 相关性
    if "lag1" in df.columns:
        try:
            print("corr lag1:", df["总血量"].corr(df["lag1"]))
        except Exception:
            pass

    # 构造窗口：delta 目标
    X, y_delta, base_log, sdates, edates = make_windows_multistep_delta(
        df=df, feature_cols=feature_cols, target_col="总血量", lookback=lookback, horizon=horizon
    )
    if len(X) < 200:
        return {"skipped": True, "reason": f"too_few_samples={len(X)}"}

    X_tr, y_tr, b_tr, s_tr, e_tr, X_te, y_te, b_te, s_te, e_te, cutoff = time_split_by_end_date(
        X, y_delta, base_log, sdates, edates, test_ratio=test_ratio
    )
    if len(X_te) < 20 or len(X_tr) < 50:
        return {"skipped": True, "reason": f"split_too_small train={len(X_tr)} test={len(X_te)}"}

    # X 标准化
    x_scaler = StandardScaler()
    X_tr_2d = X_tr.reshape(-1, X_tr.shape[-1])
    X_te_2d = X_te.reshape(-1, X_te.shape[-1])
    x_scaler.fit(X_tr_2d)
    X_tr_s = x_scaler.transform(X_tr_2d).reshape(X_tr.shape)
    X_te_s = x_scaler.transform(X_te_2d).reshape(X_te.shape)

    # y_delta 标准化（非常关键：避免 output bias 主导，导致近常数）
    y_scaler = StandardScaler()
    y_scaler.fit(y_tr)  # (N, horizon)
    y_tr_s = y_scaler.transform(y_tr)
    y_te_s = y_scaler.transform(y_te)
    train_sample_weights = build_delta_sample_weights(y_tr, alpha=spike_weight_alpha, quantile=0.75)
    val_sample_weights = build_delta_sample_weights(y_te, alpha=spike_weight_alpha, quantile=0.75)

    # 输出目录
    gname = f"{station}__{blood_type}"
    gdir = os.path.join(out_dir, slugify(gname))
    os.makedirs(gdir, exist_ok=True)

    # 模型
    model = build_cnn_lstm_multistep(
        lookback,
        X_tr_s.shape[-1],
        horizon,
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
        X_tr_s, y_tr_s,
        validation_data=(X_te_s, y_te_s, val_sample_weights),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cbs,
        shuffle=False,
        sample_weight=train_sample_weights,
    )

    # 预测（delta 反标化）
    y_pred_s = model.predict(X_te_s, batch_size=batch_size)
    y_pred_delta = y_scaler.inverse_transform(y_pred_s)  # 回到 delta(log) 空间
    y_true_delta = y_te  # 真实 delta(log)

    # 复原到原始血量
    y_pred_log = b_te[:, None] + y_pred_delta
    y_true_log = b_te[:, None] + y_true_delta
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_true_log)

    # 持久性基线：未来每一天都预测为输入窗口最后一天
    y_persist = np.expm1(np.repeat(b_te[:, None], horizon, axis=1))

    mae_all = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    mae_all_persist = mean_absolute_error(y_true.reshape(-1), y_persist.reshape(-1))
    mae_ratio_vs_persist = float(mae_all / (mae_all_persist + 1e-8))
    collapse_like_persistence = bool(mae_ratio_vs_persist >= 0.98)
    mape_all = masked_mape(y_true.reshape(-1), y_pred.reshape(-1))
    smape_all = smape(y_true.reshape(-1), y_pred.reshape(-1))

    step_metrics = {}
    for k in range(horizon):
        mae_k = mean_absolute_error(y_true[:, k], y_pred[:, k])
        mape_k = masked_mape(y_true[:, k], y_pred[:, k])
        smape_k = smape(y_true[:, k], y_pred[:, k])
        step_metrics[f"h{k + 1}"] = {"MAE": float(mae_k), "MAPE": float(mape_k), "sMAPE": float(smape_k)}

    metrics = {
        "station": station,
        "blood_type": blood_type,
        "cutoff_end_date": str(pd.to_datetime(cutoff).date()),
        "train_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "MAE_all": float(mae_all),
        "MAE_all_persistence": float(mae_all_persist),
        "MAE_ratio_vs_persistence": mae_ratio_vs_persist,
        "collapse_like_persistence": collapse_like_persistence,
        "MAPE_all": float(mape_all),
        "sMAPE_all": float(smape_all),
        "step_metrics": step_metrics,
        "target": "delta_log1p",
        "loss": "Huber + anti_persistence",
        "spike_weight_alpha": float(spike_weight_alpha),
        "weighted_validation": True,
        "anti_persistence_lambda": float(ap_lambda),
        "anti_persistence_change_thr": float(ap_change_thr),
        "anti_persistence_margin": float(ap_margin),
    }

    # 保存
    model.save(os.path.join(gdir, "final_model.keras"))
    joblib.dump(x_scaler, os.path.join(gdir, "x_scaler.joblib"))
    joblib.dump(y_scaler, os.path.join(gdir, "y_scaler.joblib"))
    with open(os.path.join(gdir, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(os.path.join(gdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 论文图：loss
    plot_loss(history, os.path.join(gdir, "loss_curve.png"))

    if collapse_like_persistence:
        print("[Warn] model is still close to persistence baseline; try larger --ap_lambda / --spike_weight_alpha")

    # 论文图：h1/h7/周总量
    dates_h1 = s_te + np.timedelta64(0, "D")
    dates_h7 = s_te + np.timedelta64(horizon - 1, "D")

    # todo ： start
    eval_df = pd.DataFrame({
        "date": pd.to_datetime(dates_h1),
        "y_true_h1": np.asarray(y_true[:, 0], dtype=float),
        "y_pred_h1": np.asarray(y_pred[:, 0], dtype=float),
    })

    # ✅ 永远用 merge 来拿“真正的真值”，不要用 pos/iloc 去猜
    gt = df[["date", "总血量"]].copy()
    gt["date"] = pd.to_datetime(gt["date"])

    chk = eval_df.merge(gt, on="date", how="left")
    chk["abs_diff_true_vs_df"] = (chk["y_true_h1"] - chk["总血量"]).abs()

    print("merge missing rows:", chk["总血量"].isna().sum())
    print("max abs diff(y_true_h1 vs df):", chk["abs_diff_true_vs_df"].max())
    print("mean abs diff(y_true_h1 vs df):", chk["abs_diff_true_vs_df"].mean())

    # 打印最离谱的那一行
    worst = chk.sort_values("abs_diff_true_vs_df", ascending=False).head(5)
    print(worst[["date", "总血量", "y_true_h1", "y_pred_h1", "abs_diff_true_vs_df"]])
    # todo ： end



    plot_pred_vs_true_v2(
        dates=dates_h1, y_true=y_true[:, 0], y_pred=y_pred[:, 0], y_baseline=y_persist[:, 0],
        out_path=os.path.join(gdir, "pred_vs_true_h1.png"),
        title=f"{station}-{blood_type} | 1-day Ahead"
    )
    plot_pred_vs_true(
        dates=dates_h7, y_true=y_true[:, horizon - 1], y_pred=y_pred[:, horizon - 1],
        out_path=os.path.join(gdir, "pred_vs_true_h7.png"),
        title=f"{station}-{blood_type} | {horizon}-day Ahead"
    )

    true_sum = y_true.sum(axis=1)
    pred_sum = y_pred.sum(axis=1)
    plot_pred_vs_true(
        dates=dates_h1, y_true=true_sum, y_pred=pred_sum,
        out_path=os.path.join(gdir, "pred_week_sum.png"),
        title=f"{station}-{blood_type} | Next {horizon} Days Sum"
    )

    # 输出预测明细
    rows = []
    for i in range(len(s_te)):
        base_date = pd.to_datetime(s_te[i])
        for k in range(horizon):
            d = base_date + pd.Timedelta(days=k)
            rows.append({"date": d.date().isoformat(), "h": k + 1,
                         "y_true": float(y_true[i, k]), "y_pred": float(y_pred[i, k]),
                         "y_persistence": float(y_persist[i, k])})
    pd.DataFrame(rows).to_csv(os.path.join(gdir, "predictions_test.csv"), index=False, encoding="utf-8-sig")

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blood_csv", type=str, default="../lstm/feature/remove_group_data.csv")
    ap.add_argument("--weather_csv", type=str,
                    default="../lstm/feature/blood_calendar_weather_2015_2026.csv")
    ap.add_argument("--out_dir", type=str, default="./out_group_models_v2")
    ap.add_argument("--station", type=str, default="北京市红十字血液中心")
    ap.add_argument("--blood_types", type=str, default="ALL,A,B,O,AB", help="逗号分隔")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--spike_weight_alpha", type=float, default=6.0,
                    help="变化幅度样本加权强度，>0 可降低退化到昨天值基线的概率")
    ap.add_argument("--ap_lambda", type=float, default=1.0,
                    help="anti-persistence 损失系数，越大越抑制预测=昨天")
    ap.add_argument("--ap_change_thr", type=float, default=0.6,
                    help="anti-persistence 生效阈值（标准化后的 |y_delta|）")
    ap.add_argument("--ap_margin", type=float, default=0.6,
                    help="anti-persistence 希望预测离 0 至少的幅度（标准化空间）")
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
            blood=blood, weather=weather,
            station=args.station, blood_type=bt,
            out_dir=args.out_dir,
            lookback=args.lookback, horizon=args.horizon,
            test_ratio=args.test_ratio,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            spike_weight_alpha=args.spike_weight_alpha,
            ap_lambda=args.ap_lambda,
            ap_change_thr=args.ap_change_thr,
            ap_margin=args.ap_margin,
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

    print("\n[Done] 输出目录：", args.out_dir)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
