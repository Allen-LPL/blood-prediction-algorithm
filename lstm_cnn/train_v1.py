# train_group_cnn_lstm_multistep.py
# 每个血站+血型单独训练 CNN+LSTM，多步预测未来7天，并输出论文图

import argparse
import os
import json
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
from matplotlib.font_manager import FontProperties

FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")


# =========================
# Utils
# =========================
def slugify(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", s)  # 保留中文/字母数字/下划线/-
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


def build_dynamic_feature(weather_df, target_col):
    holidays = set(weather_df.loc[weather_df["is_holiday"] == 1, "date"])
    weather_df["is_before_holiday"] = weather_df["date"].apply(
        lambda x: 1 if (x + timedelta(days=1)) in holidays else 0
    )
    weather_df["is_after_holiday"] = weather_df["date"].apply(
        lambda x: 1 if (x - timedelta(days=1)) in holidays else 0
    )

    # ===== 新增：距离节假日的天数特征（提前/滞后效应通常>1天） =====
    # days_until_holiday: 距离下一个节假日还有几天（节假日当天=0）
    # days_since_holiday: 距离上一个节假日过去几天（节假日当天=0）
    holiday_arr = np.array(sorted(list(holidays)), dtype="datetime64[ns]")
    dates_arr = weather_df["date"].values.astype("datetime64[ns]")
    if len(holiday_arr) > 0:
        idx = np.searchsorted(holiday_arr, dates_arr)
        next_idx = np.clip(idx, 0, len(holiday_arr) - 1)
        prev_idx = np.clip(idx - 1, 0, len(holiday_arr) - 1)
        next_h = holiday_arr[next_idx]
        prev_h = holiday_arr[prev_idx]

        days_until = ((next_h - dates_arr) / np.timedelta64(1, "D")).astype(float)
        days_since = ((dates_arr - prev_h) / np.timedelta64(1, "D")).astype(float)

        # 对于 dates > 最后一个节假日，days_until 会是负数（被 clip 到最后一个节假日）-> 设为一个大数
        days_until = np.where(idx >= len(holiday_arr), 999.0, days_until)
        # 对于 dates < 第一个节假日，days_since 会是负数（被 clip 到第一个节假日）-> 设为一个大数
        days_since = np.where(idx <= 0, 999.0, days_since)

        days_until = np.maximum(days_until, 0.0)
        days_since = np.maximum(days_since, 0.0)
    else:
        days_until = np.full(len(weather_df), 999.0, dtype=float)
        days_since = np.full(len(weather_df), 999.0, dtype=float)

    weather_df["days_until_holiday"] = np.clip(days_until, 0, 30)
    weather_df["days_since_holiday"] = np.clip(days_since, 0, 30)
    weather_df["near_holiday7"] = (
            (weather_df["days_until_holiday"] <= 7) | (weather_df["days_since_holiday"] <= 7)
    ).astype(int)

    weather_df["temp_sq"] = weather_df["平均温度"] ** 2
    weather_df["temp_diff"] = weather_df["平均温度"].diff()
    weather_df["temp_rolling3"] = weather_df["平均温度"].rolling(3).mean()

    weather_df = pd.get_dummies(weather_df, columns=["天气"])
    print(weather_df.columns.tolist())

    # 滚动特征
    # trend/residual（不center，避免未来泄露）
    TARGET_COL = target_col
    weather_df["trend"] = weather_df[TARGET_COL].rolling(7).mean()
    weather_df["residual"] = weather_df[TARGET_COL] - weather_df["trend"]

    weather_df["lag1"] = weather_df[TARGET_COL].shift(1)
    weather_df["lag7"] = weather_df[TARGET_COL].shift(7)
    if "rolling7" not in weather_df.columns:
        weather_df["rolling7"] = weather_df[TARGET_COL].rolling(7).mean()

    weather_df["absdiff1"] = (weather_df[TARGET_COL] - weather_df["lag1"]).abs()
    weather_df["rolling_std7"] = weather_df[TARGET_COL].rolling(7).std()
    weather_df["rolling_std14"] = weather_df[TARGET_COL].rolling(14).std()
    weather_df["rolling_absdiff7"] = weather_df["absdiff1"].rolling(7).mean()

    weather_df["lag14"] = weather_df[TARGET_COL].shift(14)
    weather_df["lag28"] = weather_df[TARGET_COL].shift(28)

    weather_df["absdiff7"] = (weather_df[TARGET_COL] - weather_df["lag7"]).abs()
    weather_df["diff7"] = (weather_df[TARGET_COL] - weather_df["lag7"])
    weather_df["diff14"] = (weather_df[TARGET_COL] - weather_df["lag14"])

    weather_df["rolling14"] = weather_df[TARGET_COL].rolling(14).mean()
    weather_df["rolling28"] = weather_df[TARGET_COL].rolling(28).mean()
    weather_df["ewm7"] = weather_df[TARGET_COL].ewm(span=7, adjust=False).mean()
    weather_df["ewm14"] = weather_df[TARGET_COL].ewm(span=14, adjust=False).mean()

    return weather_df


# =========================
# Data processing
# =========================
def build_group_daily_series(
        blood: pd.DataFrame,
        weather: pd.DataFrame,
        station: str,
        blood_type: str,
        target_col: str = "总血量",
):
    """
    为指定 (血站, 血型) 构造完整每日序列：
    - date 按天补齐 min~max
    - 断档总血量填0
    - merge 天气特征（缺失用ffill/bfill，再填0）
    """
    g = blood[(blood["血站"] == station) & (blood["血型"] == blood_type)].copy()
    if blood_type == 'ALL':
        g = blood[blood["血站"] == station]
        g = g.drop(columns='血型')
        g = (
            g.groupby(["date", "血站"], as_index=False)
            .agg(总血量=("总血量", "sum"))
        )
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

    # 合并采血量（断档=0）
    df = base.merge(g, on="date", how="left")
    df[target_col] = df[target_col].fillna(0.0)

    # 合并天气
    df = df.merge(weather, on="date", how="left")

    # 确保按日期排序
    df = df.sort_values("date").reset_index(drop=True)
    df = build_dynamic_feature(df, target_col)

    # 天气列：除 date 外全部作为特征
    weather_cols = [c for c in df.columns if c not in ["date", blood_type, station]]

    # 统一数值化 + 缺失处理
    for c in weather_cols:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[weather_cols] = df[weather_cols].replace([np.inf, -np.inf], np.nan)
    df[weather_cols] = df[weather_cols].ffill().bfill().fillna(0.0)

    print(f"feature cols : {weather_cols}")

    return df, weather_cols


def make_windows_multistep(
        df: pd.DataFrame,
        feature_cols,
        target_col: str,
        lookback: int,
        horizon: int,
):
    """
    输入：df(日序列)
    输出：
      X: (N, lookback, F)
      y: (N, horizon)
      start_dates: (N,) 目标起始日期 t
      end_dates: (N,) 目标结束日期 t+horizon-1（用于时间切分）
    定义：
      用 [t-lookback ... t-1] 预测 [t ... t+horizon-1]
    """
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    tgt = df[target_col].to_numpy(dtype=np.float32)
    dates = df["date"].to_numpy(dtype="datetime64[ns]")

    X_list, y_list, sd_list, ed_list = [], [], [], []

    max_t = len(df) - horizon  # t 的最大起点
    for t in range(lookback, max_t):
        X_list.append(feat[t - lookback:t])
        y_list.append(tgt[t:t + horizon])
        sd_list.append(dates[t])
        ed_list.append(dates[t + horizon - 1])

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    start_dates = np.asarray(sd_list)
    end_dates = np.asarray(ed_list)
    return X, y, start_dates, end_dates


def time_split_by_end_date(X, y, start_dates, end_dates, test_ratio=0.2):
    """
    严格按时间切分：按 end_dates（目标窗口结束日期）取最后 test_ratio 的日期段作为测试集
    """
    uniq = np.unique(end_dates)
    uniq = np.sort(uniq)
    cut = int(len(uniq) * (1.0 - test_ratio))
    cut = max(1, min(cut, len(uniq) - 1))
    cutoff = uniq[cut]

    train_mask = end_dates < cutoff
    test_mask = end_dates >= cutoff

    return (X[train_mask], y[train_mask], start_dates[train_mask], end_dates[train_mask],
            X[test_mask], y[test_mask], start_dates[test_mask], end_dates[test_mask],
            cutoff)


# =========================
# Model
# =========================
def build_cnn_lstm_multistep(lookback, n_features, horizon, lr=1e-3):
    inp = layers.Input(shape=(lookback, n_features))

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.15)(x)

    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)

    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(horizon, activation="linear")(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mae", metrics=[tf.keras.metrics.MAE])
    return model


# =========================
# Plotting (论文图)
# =========================
def plot_loss(history, out_path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pred_vs_true(dates, y_true, y_pred, out_path, title):
    # dates: datetime64 array
    plt.figure(figsize=(10, 4.5))
    plt.plot(pd.to_datetime(dates), y_true, label="true")
    plt.plot(pd.to_datetime(dates), y_pred, label="pred")
    plt.xlabel("Date")
    plt.ylabel("Blood Volume")
    plt.title(title, fontproperties=FONT)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# Main training per group
# =========================
def train_one_group(
        blood, weather, station, blood_type, out_dir,
        lookback, horizon, test_ratio, epochs, batch_size, lr,
        log1p_y=True
):
    series_pack = build_group_daily_series(blood, weather, station, blood_type, target_col="总血量")
    if series_pack is None:
        return None

    df, feature_cols = series_pack

    print(df[["date", "总血量", "lag1", "lag7", "rolling7"]].tail(15))
    print("corr lag1:", df["总血量"].corr(df["lag1"]))

    # 若特征列为空也能跑（但建议至少有日历/温度）
    if len(feature_cols) == 0:
        raise ValueError("天气表除date外没有任何特征列？请检查天气表字段。")

    # 构造窗口
    X, y, sdates, edates = make_windows_multistep(
        df=df,
        feature_cols=feature_cols,
        target_col="总血量",
        lookback=lookback,
        horizon=horizon
    )

    if len(X) < 200:
        # 样本太少不训练
        return {"skipped": True, "reason": f"too_few_samples={len(X)}"}

    # 时间切分
    X_tr, y_tr, s_tr, e_tr, X_te, y_te, s_te, e_te, cutoff = time_split_by_end_date(
        X, y, sdates, edates, test_ratio=test_ratio
    )

    if len(X_te) < 20 or len(X_tr) < 50:
        return {"skipped": True, "reason": f"split_too_small train={len(X_tr)} test={len(X_te)}"}

    # y 变换（论文里可写“对数稳定化”）
    if log1p_y:
        y_tr_t = np.log1p(y_tr)
        y_te_t = np.log1p(y_te)
    else:
        y_tr_t = y_tr.copy()
        y_te_t = y_te.copy()

    # X 标准化（仅训练集fit）
    scaler = StandardScaler()
    X_tr_2d = X_tr.reshape(-1, X_tr.shape[-1])
    X_te_2d = X_te.reshape(-1, X_te.shape[-1])
    scaler.fit(X_tr_2d)
    X_tr_s = scaler.transform(X_tr_2d).reshape(X_tr.shape)
    X_te_s = scaler.transform(X_te_2d).reshape(X_te.shape)

    # 模型
    model = build_cnn_lstm_multistep(lookback, X_tr_s.shape[-1], horizon, lr=lr)

    gname = f"{station}__{blood_type}"
    gdir = os.path.join(out_dir, slugify(gname))
    os.makedirs(gdir, exist_ok=True)

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        callbacks.ModelCheckpoint(os.path.join(gdir, "best_model.keras"), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_tr_s, y_tr_t,
        validation_data=(X_te_s, y_te_t),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cbs
    )

    # 预测 + inverse
    y_pred_t = model.predict(X_te_s, batch_size=batch_size)
    if log1p_y:
        y_pred = np.expm1(y_pred_t)
        y_true = y_te
    else:
        y_pred = y_pred_t
        y_true = y_te

    # 指标（整体 + 分步）
    print("y_pred h1 std/min/max:", y_pred[:, 0].std(), y_pred[:, 0].min(), y_pred[:, 0].max())
    print("y_true h1 std/min/max:", y_true[:, 0].std(), y_true[:, 0].min(), y_true[:, 0].max())
    mae_all = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    mape_all = mape(y_true.reshape(-1), y_pred.reshape(-1))
    smape_all = smape(y_true.reshape(-1), y_pred.reshape(-1))

    step_metrics = {}
    for k in range(horizon):
        mae_k = mean_absolute_error(y_true[:, k], y_pred[:, k])
        mape_k = mape(y_true[:, k], y_pred[:, k])
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
        "log1p_y": bool(log1p_y),
        "MAE_all": float(mae_all),
        "MAPE_all": float(mape_all),
        "sMAPE_all": float(smape_all),
        "step_metrics": step_metrics,
    }

    # 保存模型与元信息
    model.save(os.path.join(gdir, "final_model.keras"))
    joblib.dump(scaler, os.path.join(gdir, "feat_scaler.joblib"))
    with open(os.path.join(gdir, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(os.path.join(gdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 论文图：loss曲线
    plot_loss(history, os.path.join(gdir, "loss_curve.png"))

    # 论文图：预测 vs 真实（1天ahead、7天ahead、7天总量）
    # 对齐日期：第k步对应日期 = start_date + (k-1)天
    # 这里我们用 test 的 start_date 来构造对齐日期序列
    dates_h1 = s_te + np.timedelta64(0, "D")
    dates_h7 = s_te + np.timedelta64(horizon - 1, "D")

    plot_pred_vs_true(
        dates=dates_h1,
        y_true=y_true[:, 0],
        y_pred=y_pred[:, 0],
        out_path=os.path.join(gdir, "pred_vs_true_h1.png"),
        title=f"{station}-{blood_type} | 1-day Ahead"
    )

    plot_pred_vs_true(
        dates=dates_h7,
        y_true=y_true[:, horizon - 1],
        y_pred=y_pred[:, horizon - 1],
        out_path=os.path.join(gdir, "pred_vs_true_h7.png"),
        title=f"{station}-{blood_type} | {horizon}-day Ahead"
    )

    # 未来7天总量（对业务/论文很有解释力）
    true_sum = y_true.sum(axis=1)
    pred_sum = y_pred.sum(axis=1)
    plot_pred_vs_true(
        dates=dates_h1,  # 用窗口起始日期做索引
        y_true=true_sum,
        y_pred=pred_sum,
        out_path=os.path.join(gdir, "pred_week_sum.png"),
        title=f"{station}-{blood_type} | Next {horizon} Days Sum"
    )

    # 输出预测结果明细（用于论文表格/误差分析）
    rows = []
    for i in range(len(s_te)):
        base_date = pd.to_datetime(s_te[i])
        for k in range(horizon):
            d = base_date + pd.Timedelta(days=k)
            rows.append({
                "date": d.date().isoformat(),
                "h": k + 1,
                "y_true": float(y_true[i, k]),
                "y_pred": float(y_pred[i, k]),
            })
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(gdir, "predictions_test.csv"), index=False, encoding="utf-8-sig")

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blood_csv", type=str, default="../lstm/feature/remove_group_data.csv")
    ap.add_argument("--weather_csv", type=str,
                    default="../lstm/feature/blood_calendar_weather_2015_2026.csv")
    ap.add_argument("--out_dir", type=str, default="./out_group_models")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log1p_y", action="store_true", help="对总血量log1p稳定化（建议论文打开）")
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

    # 获取所有组
    groups = blood[["血站", "血型"]].drop_duplicates().reset_index(drop=True)

    all_metrics = []
    station = '北京市红十字血液中心'
    for blood_type in ['ALL', 'A', 'B', 'O', 'AB']:
        m = train_one_group(
            blood=blood,
            weather=weather,
            station=station,
            blood_type=blood_type,
            out_dir=args.out_dir,
            lookback=args.lookback,
            horizon=args.horizon,
            test_ratio=args.test_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            log1p_y=args.log1p_y
        )
        if m is None:
            continue
        # 跳过的也记录
        m2 = {"station": station, "blood_type": blood_type}
        if isinstance(m, dict):
            m2.update(m)
        all_metrics.append(m2)

    # 汇总保存（方便论文里做总体比较）
    summary_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    # 同时给一个csv版（更适合做表格）
    rows = []
    for m in all_metrics:
        if m.get("skipped"):
            rows.append({
                "血站": m.get("station"),
                "血型": m.get("blood_type"),
                "skipped": True,
                "reason": m.get("reason", "")
            })
        else:
            rows.append({
                "血站": m.get("station"),
                "血型": m.get("blood_type"),
                "train_samples": m.get("train_samples"),
                "test_samples": m.get("test_samples"),
                "MAE_all": m.get("MAE_all"),
                "MAPE_all": m.get("MAPE_all"),
                "sMAPE_all": m.get("sMAPE_all"),
                "cutoff_end_date": m.get("cutoff_end_date"),
            })
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metrics_summary.csv"),
                              index=False, encoding="utf-8-sig")

    print("\n[Done] 输出目录：", args.out_dir)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    main()
