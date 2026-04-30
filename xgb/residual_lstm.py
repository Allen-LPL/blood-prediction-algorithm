"""
LSTM 残差模块 — XGBoost 输出的残差用 BiLSTM 拟合,
最终预测 y_hat_adaptive = y_xgb + clip(resid_pred, ±0.5·|y_xgb|).

参考实现: lstm/blood_lstm_xgb_v5(resid).py

公开函数:
    compute_oof_y_xgb            5 折 OOF + final XGB
    build_lstm_feature_frame     构造 LSTM 输入 DataFrame
    clip_train_residuals         按 [1, 99] 百分位裁剪训练残差
    train_residual_lstm          全流程: scaler 拟合 + 序列构造 + LSTM 训练
    predict_residual_for_window  单步推理
    clip_resid_pred              输出端自适应裁剪
    save_lstm_artifacts          保存模型 + scalers + meta
    load_lstm_artifacts          加载,缺文件返回 None

模块常量:
    TF_AVAILABLE   tensorflow 是否可用 (False 时所有训练/推理函数会 raise)
    LOOKBACK / EPOCHS / BATCH_SIZE / LSTM_LR / RESID_OUTPUT_CLIP_FRAC
    LSTM_AUX_COLS_COLLECTION / LSTM_AUX_COLS_SUPPLY  规范输入列序
    LSTM_META_SCHEMA_VERSION
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import joblib

log = logging.getLogger(__name__)

# ---------- TensorFlow 延迟导入 ----------
try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models

    TF_AVAILABLE = True
    tf.keras.utils.set_random_seed(1024)
except Exception as exc:  # pragma: no cover - environment-dependent
    log.warning("tensorflow 不可用,LSTM 残差功能将降级为纯 XGB: %s", exc)
    tf = None
    callbacks = layers = models = None
    TF_AVAILABLE = False


# ---------- 常量 ----------
LOOKBACK = 14
LSTM_LR = 1e-3
EPOCHS = 120
BATCH_SIZE = 32
RESID_CLIP_PCT = (1, 99)
RESID_OUTPUT_CLIP_FRAC = 0.5  # ±0.5·|y_xgb|
LSTM_META_SCHEMA_VERSION = 1
MIN_TRAIN_SEQ = 50  # 训练序列少于此值则跳过 LSTM

LSTM_AUX_COLS_COLLECTION = [
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
LSTM_AUX_COLS_SUPPLY = [
    c for c in LSTM_AUX_COLS_COLLECTION if c != "is_pku_semester_start_season"
]


# ---------- 序列构造 ----------


def make_residual_sequences(
    feature_mat: np.ndarray, resid_target: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造滑动窗口序列.

    feature_mat: (T, F) — 已包含 resid_lag1,无未来泄露
    resid_target: (T,)  — 当天的真实残差 (LSTM 目标)
    返回 (X, y, idx): X=(N,lookback,F), y=(N,1), idx=(N,) 对应原始时间索引 t.
    """
    X, y, idx = [], [], []
    for t in range(lookback, len(feature_mat)):
        X.append(feature_mat[t - lookback + 1 : t + 1, :])
        y.append(resid_target[t])
        idx.append(t)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)[:, None]
    idx = np.asarray(idx, dtype=np.int64)
    return X, y, idx


# ---------- 模型架构 ----------


def build_lstm_residual_model(lookback: int, feature_dim: int, lr: float = LSTM_LR):
    """BiLSTM(64) → BN → Dropout → BiLSTM(32) → BN → Dense(32) → Dropout → Dense(1)."""
    if not TF_AVAILABLE:
        raise RuntimeError("tensorflow 不可用,无法构建 LSTM 模型")
    inp = layers.Input(shape=(lookback, feature_dim))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
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


# ---------- 5 折 OOF + final XGB ----------


def compute_oof_y_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    xgb_params: dict,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray, XGBRegressor]:
    """5 折时序 OOF 预测训练段 + final 模型预测测试段.

    返回 (oof_train, pred_test, xgb_final).
    folds: KFold(shuffle=False) 顺序划分,避免泄露未来到过去.
    """
    sample_weight = np.asarray(sample_weight)
    oof = np.zeros(len(X_train), dtype=np.float64)
    splits = max(2, min(n_splits, len(X_train) // 2 if len(X_train) >= 4 else 2))
    kf = KFold(n_splits=splits, shuffle=False)
    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        fold_model = XGBRegressor(**xgb_params)
        fold_model.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            sample_weight=sample_weight[tr_idx],
            verbose=False,
        )
        oof[va_idx] = fold_model.predict(X_train[va_idx])
        log.debug("OOF fold %d/%d done", fold_i + 1, splits)

    xgb_final = XGBRegressor(**xgb_params)
    xgb_final.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        sample_weight=sample_weight,
        verbose=False,
    )
    pred_test = xgb_final.predict(X_test)
    return oof, pred_test, xgb_final


# ---------- 残差处理 ----------


def clip_train_residuals(
    resid: np.ndarray, lo_pct: float = RESID_CLIP_PCT[0], hi_pct: float = RESID_CLIP_PCT[1]
) -> Tuple[np.ndarray, float, float]:
    """按百分位裁剪残差,返回 (clipped, lo, hi)."""
    lo, hi = np.percentile(resid, [lo_pct, hi_pct])
    return np.clip(resid, lo, hi).astype(np.float32), float(lo), float(hi)


def clip_resid_pred(
    resid_pred: np.ndarray, y_xgb: np.ndarray, frac: float = RESID_OUTPUT_CLIP_FRAC
) -> np.ndarray:
    """自适应输出裁剪: clip(resid, -frac·|y_xgb|, +frac·|y_xgb|)."""
    bound = np.abs(np.asarray(y_xgb, dtype=np.float64)) * frac
    return np.clip(resid_pred, -bound, bound)


# ---------- LSTM 输入构造 ----------


def build_lstm_feature_frame(
    df: pd.DataFrame, lstm_aux_cols: list
) -> Tuple[pd.DataFrame, list]:
    """从 df 选取 aux 列 + 派生 resid_lag1,返回 (feature_df, final_cols).

    df 必须含 'resid' 列;返回的 feature_df 不含 'resid' 当天值,只含 'resid_lag1'.
    若 aux_col 在 df 中缺失,会跳过并 log warning,确保下游可用.
    """
    available = [c for c in lstm_aux_cols if c in df.columns]
    missing = [c for c in lstm_aux_cols if c not in df.columns]
    if missing:
        log.warning("LSTM aux 列缺失,自动跳过: %s", missing)
    feature_df = df[available].copy()
    resid_lag1 = df["resid"].shift(1).fillna(0.0).astype(np.float32)
    feature_df["resid_lag1"] = resid_lag1.values
    feature_df = feature_df.fillna(0.0).astype(np.float32)
    final_cols = available + ["resid_lag1"]
    return feature_df, final_cols


# ---------- 训练 LSTM 残差 ----------


def train_residual_lstm(
    feature_df_full: pd.DataFrame,
    resid_target_full: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    lookback: int = LOOKBACK,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LSTM_LR,
) -> Optional[dict]:
    """拟合 scalers (仅训练段)、构造序列、训练 LSTM,返回工件字典.

    若训练序列样本数 < MIN_TRAIN_SEQ,返回 None (调用方降级为纯 XGB).

    返回:
      {
        "lstm_model": Keras Model,
        "feature_scaler": StandardScaler,
        "resid_scaler": StandardScaler,
        "feature_dim": int,
        "lookback": int,
        "history": Keras History,
        "resid_pred_test": np.ndarray (测试段每行的残差预测,未对应行填 NaN),
        "test_idx": np.ndarray (测试段被预测到的原始 df 索引),
      }
    """
    if not TF_AVAILABLE:
        raise RuntimeError("tensorflow 不可用,无法训练残差 LSTM")

    feature_train = feature_df_full[train_mask].to_numpy(dtype=np.float32)
    resid_train = resid_target_full[train_mask].astype(np.float32).reshape(-1, 1)

    feature_scaler = StandardScaler().fit(feature_train)
    resid_scaler = StandardScaler().fit(resid_train)

    features_s = feature_scaler.transform(
        feature_df_full.to_numpy(dtype=np.float32)
    ).astype(np.float32)
    resid_s = (
        resid_scaler.transform(resid_target_full.astype(np.float32).reshape(-1, 1))
        .reshape(-1)
        .astype(np.float32)
    )

    X_seq, y_seq, idx_seq = make_residual_sequences(features_s, resid_s, lookback)
    if len(X_seq) == 0:
        log.warning("序列样本为 0,跳过 LSTM 训练")
        return None

    seq_train_mask = train_mask[idx_seq]
    seq_test_mask = test_mask[idx_seq]
    X_tr, y_tr = X_seq[seq_train_mask], y_seq[seq_train_mask]
    X_te, y_te = X_seq[seq_test_mask], y_seq[seq_test_mask]

    if len(X_tr) < MIN_TRAIN_SEQ:
        log.warning(
            "LSTM 训练序列不足 (%d < %d),跳过", len(X_tr), MIN_TRAIN_SEQ
        )
        return None

    feature_dim = X_seq.shape[2]
    lstm_model = build_lstm_residual_model(lookback, feature_dim, lr=lr)

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]
    val_data = (X_te, y_te) if len(X_te) > 0 else None
    history = lstm_model.fit(
        X_tr,
        y_tr,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=cbs,
        shuffle=True,
    )

    resid_pred_test = np.full(len(feature_df_full), np.nan, dtype=np.float32)
    test_idx = idx_seq[seq_test_mask]
    if len(X_te) > 0:
        pred_s = lstm_model.predict(X_te, batch_size=batch_size, verbose=0).reshape(-1, 1)
        pred = resid_scaler.inverse_transform(pred_s).reshape(-1)
        resid_pred_test[test_idx] = pred.astype(np.float32)

    return {
        "lstm_model": lstm_model,
        "feature_scaler": feature_scaler,
        "resid_scaler": resid_scaler,
        "feature_dim": feature_dim,
        "lookback": lookback,
        "history": history,
        "resid_pred_test": resid_pred_test,
        "test_idx": test_idx,
    }


# ---------- 单步推理 ----------


def predict_residual_for_window(
    lstm_model,
    feature_scaler: StandardScaler,
    resid_scaler: StandardScaler,
    window_df: pd.DataFrame,
    lookback: int,
) -> float:
    """对单个 lookback 窗口预测残差 (反标准化后).

    window_df: (lookback, F) DataFrame,列序必须与训练时 final_cols 一致.
    """
    if len(window_df) != lookback:
        raise ValueError(
            f"window_df 行数 {len(window_df)} 与 lookback {lookback} 不符"
        )
    arr = window_df.to_numpy(dtype=np.float32)
    arr_s = feature_scaler.transform(arr).astype(np.float32)
    pred_s = lstm_model.predict(arr_s[np.newaxis, :, :], verbose=0).reshape(-1, 1)
    pred = resid_scaler.inverse_transform(pred_s).reshape(-1)[0]
    return float(pred)


# ---------- 工件读写 ----------


def lstm_meta_payload(
    lookback: int,
    lstm_aux_cols: list,
    include_pku: bool,
    train_resid_clip: Tuple[float, float],
    business: str,
    output_clip_frac: float = RESID_OUTPUT_CLIP_FRAC,
    feature_dim: Optional[int] = None,
) -> dict:
    return {
        "schema_version": LSTM_META_SCHEMA_VERSION,
        "lookback": lookback,
        "lstm_aux_cols": lstm_aux_cols,
        "include_pku": include_pku,
        "train_resid_clip": [float(train_resid_clip[0]), float(train_resid_clip[1])],
        "output_clip_frac": output_clip_frac,
        "business": business,
        "feature_dim": feature_dim,
        "tf_version": tf.__version__ if TF_AVAILABLE else None,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }


def save_lstm_artifacts(
    group_dir: Path,
    lstm_model,
    feature_scaler: StandardScaler,
    resid_scaler: StandardScaler,
    meta: dict,
) -> None:
    """保存 lstm.keras + feature_scaler.joblib + resid_scaler.joblib + lstm_meta.json."""
    group_dir = Path(group_dir)
    group_dir.mkdir(parents=True, exist_ok=True)
    lstm_model.save(str(group_dir / "lstm.keras"))
    joblib.dump(feature_scaler, group_dir / "feature_scaler.joblib")
    joblib.dump(resid_scaler, group_dir / "resid_scaler.joblib")
    with open(group_dir / "lstm_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("LSTM 工件已保存: %s", group_dir)


# 模块级 LSTM 工件缓存: {abs_group_dir_str: (combined_mtime, artifacts_dict)}
# mtime 变化即缓存失效, 重训会自动失效.
_LSTM_ARTIFACTS_CACHE: dict = {}


def load_lstm_artifacts(group_dir: Path) -> Optional[dict]:
    """加载 LSTM 工件; 任一文件缺失或 schema 不匹配则返回 None.

    带模块级缓存: 同一进程内对同一 group 的重复调用复用已加载模型,
    避免每次 API 请求都重新 `tf.keras.models.load_model` (~1s/次).
    通过文件 mtime 检测重训, 自动失效缓存.
    """
    group_dir = Path(group_dir)
    files = {
        "lstm": group_dir / "lstm.keras",
        "feature_scaler": group_dir / "feature_scaler.joblib",
        "resid_scaler": group_dir / "resid_scaler.joblib",
        "meta": group_dir / "lstm_meta.json",
    }
    for name, p in files.items():
        if not p.exists():
            log.info("LSTM 工件不全 (%s 缺失),走纯 XGB: %s", name, group_dir)
            return None
    if not TF_AVAILABLE:
        log.info("tensorflow 不可用,走纯 XGB: %s", group_dir)
        return None

    # mtime-based 缓存校验
    cache_key = str(group_dir.resolve())
    combined_mtime = max(p.stat().st_mtime for p in files.values())
    cached = _LSTM_ARTIFACTS_CACHE.get(cache_key)
    if cached is not None and cached[0] == combined_mtime:
        return cached[1]

    with open(files["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("schema_version") != LSTM_META_SCHEMA_VERSION:
        log.warning(
            "lstm_meta schema_version 不匹配 (%s),降级为纯 XGB: %s",
            meta.get("schema_version"),
            group_dir,
        )
        return None
    log.info("加载 LSTM 工件 (首次): %s", group_dir)
    lstm_model = tf.keras.models.load_model(str(files["lstm"]))
    feature_scaler = joblib.load(files["feature_scaler"])
    resid_scaler = joblib.load(files["resid_scaler"])
    artifacts = {
        "lstm_model": lstm_model,
        "feature_scaler": feature_scaler,
        "resid_scaler": resid_scaler,
        "meta": meta,
    }
    _LSTM_ARTIFACTS_CACHE[cache_key] = (combined_mtime, artifacts)
    return artifacts
