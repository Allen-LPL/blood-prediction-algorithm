import os
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

# =========================
# 配置：按你的列名确认
# =========================
CSV_PATH = "daily_join.csv"  # 你 join 后的表

DATE_COL = "date"
STATION_COL = "血站"
BLOOD_COL = "血型"
TARGET_COL = "总血量"  # 你图里是“总血量”

LOOK_BACK = 60  # 10年数据常用 30 或 60
TREND_WIN = 7  # trend=7日均值
TEST_RATIO = 0.2  # 按时间末尾20%测试

OUT_DIR = "models_lstm_xgb"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")


def safe_mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))


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


def make_lstm_xy(values_scaled: np.ndarray, look_back: int, y_index: int):
    X, y = [], []
    for i in range(len(values_scaled) - look_back):
        X.append(values_scaled[i:i + look_back, :])
        y.append(values_scaled[i + look_back, y_index])
    return np.array(X), np.array(y)


def inverse_one_col(scaler: MinMaxScaler, y_scaled: np.ndarray, feature_dim: int, y_index: int):
    tmp = np.zeros((len(y_scaled), feature_dim))
    tmp[:, y_index] = y_scaled
    return scaler.inverse_transform(tmp)[:, y_index]


def train_one_group(g: pd.DataFrame, group_name: str, feature_cols_lstm: list, feature_cols_xgb: list):
    g = g.sort_values(DATE_COL).reset_index(drop=True)

    # trend/residual（不center，避免未来泄露）
    g["trend"] = g[TARGET_COL].rolling(TREND_WIN).mean()
    g["residual"] = g[TARGET_COL] - g["trend"]

    # lag/rolling（如果你表里已有可注释掉；这里做个兜底）
    if "lag1" not in g.columns:
        g["lag1"] = g[TARGET_COL].shift(1)
    if "lag7" not in g.columns:
        g["lag7"] = g[TARGET_COL].shift(7)
    if "rolling7" not in g.columns:
        g["rolling7"] = g[TARGET_COL].rolling(7).mean()

    g = g.dropna().reset_index(drop=True)

    n = len(g)
    if n < LOOK_BACK + 200:
        print(f"[SKIP] {group_name} 数据太少 n={n}")
        return

    split = int(n * (1 - TEST_RATIO))
    train_end = split

    # ========= LSTM：预测 trend(t+1) =========
    lstm_mat = g[feature_cols_lstm + ["trend"]].astype(float).values

    scaler = MinMaxScaler()
    lstm_scaled = scaler.fit_transform(lstm_mat)
    y_index = lstm_scaled.shape[1] - 1  # trend 在最后一列

    X_all, y_all = make_lstm_xy(lstm_scaled, LOOK_BACK, y_index)

    sample_split = train_end - LOOK_BACK
    if sample_split <= 50:
        print(f"[SKIP] {group_name} 训练样本不足 sample_split={sample_split}")
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

    trend_pred_scaled = model.predict(X_test, verbose=0).flatten()
    trend_true_scaled = y_test

    feature_dim = lstm_scaled.shape[1]
    trend_pred = inverse_one_col(scaler, trend_pred_scaled, feature_dim, y_index)
    trend_true = inverse_one_col(scaler, trend_true_scaled, feature_dim, y_index)

    # ========= XGB：预测 相对残差 resid_ratio(t+1) =========
    # resid_ratio = (y - trend) / (trend + 1)
    g2 = g.copy()
    g2["resid_ratio"] = (g2[TARGET_COL] - g2["trend"]) / (g2["trend"] + 1.0)

    # 预测下一天的 resid_ratio
    g2["resid_ratio_tplus1"] = g2["resid_ratio"].shift(-1)
    g2 = g2.dropna().reset_index(drop=True)

    X = g2[feature_cols_xgb].astype(float).values
    y = g2["resid_ratio_tplus1"].astype(float).values

    xgb_train_end = train_end - 1  # 用当天预测次日
    X_train_xgb, X_test_xgb = X[:xgb_train_end], X[xgb_train_end:]
    y_train_xgb, y_test_xgb = y[:xgb_train_end], y[xgb_train_end:]

    xgb = XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        objective="reg:squarederror"
    )
    xgb.fit(X_train_xgb, y_train_xgb)

    resid_ratio_pred = xgb.predict(X_test_xgb)
    resid_ratio_true = y_test_xgb

    # ========= 合成 total =========
    # 用 trend_pred 作为尺度： total_pred = trend_pred * (1 + resid_ratio_pred)
    # 限制一下比例，避免极端值把总量拉爆（可按业务调整）
    resid_ratio_pred = np.clip(resid_ratio_pred, -0.8, 2.0)

    L = min(len(trend_pred), len(resid_ratio_pred))
    total_pred = trend_pred[:L] * (1.0 + resid_ratio_pred[:L])
    total_true = g[TARGET_COL].values[train_end:train_end + L]

    mae = mean_absolute_error(total_true, total_pred)
    mape = safe_mape(total_true, total_pred)
    # Residual 不看 MAPE（会炸），改看 MAE 或 sMAPE
    resid_mae = mean_absolute_error(resid_ratio_true[:L], resid_ratio_pred[:L])

    print(f"\n[{group_name}] : MAE={mae:.2f}, MAPE={mape:.4f}")
    print(f"[{group_name}] Trend MAPE={safe_mape(trend_true[:L], trend_pred[:L]):.4f}")
    print(f"[{group_name}] resid_ratio MAE={resid_mae:.4f}")

    # ========= 画图 =========
    plt.figure(figsize=(11, 4))
    plt.plot(total_true, label="real")
    plt.plot(total_pred, label="predict(LSTM+XGB)")
    plt.title(f"{group_name} predict", fontproperties=FONT)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ========= 保存 =========
    # lstm_path = os.path.join(OUT_DIR, f"{group_name}_lstm.keras".replace("/", "_"))
    # xgb_path = os.path.join(OUT_DIR, f"{group_name}_xgb.json".replace("/", "_"))
    # model.save(lstm_path)
    # xgb.save_model(xgb_path)
    # print("saved:", lstm_path)
    # print("saved:", xgb_path)


def load_data():
    #all_df = pd.read_csv('feature/all_data.csv')
    all_df = pd.read_csv('feature/remove_group_data.csv')
    all_df["date"] = pd.to_datetime(all_df["date"])
    return all_df


def merge_calendar_weather(temp_df):
    weather_df = pd.read_csv("./feature/blood_calendar_weather_2015_2026.csv")
    # 节假日前后
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    holidays = set(weather_df.loc[weather_df["is_holiday"] == 1, "date"])
    weather_df["is_before_holiday"] = weather_df["date"].apply(
        lambda x: 1 if (x + timedelta(days=1)) in holidays else 0
    )
    weather_df["is_after_holiday"] = weather_df["date"].apply(
        lambda x: 1 if (x - timedelta(days=1)) in holidays else 0
    )

    weather_df["temp_sq"] = weather_df["平均温度"] ** 2
    weather_df["temp_diff"] = weather_df["平均温度"].diff()
    weather_df["temp_rolling3"] = weather_df["平均温度"].rolling(3).mean()

    weather_df = pd.get_dummies(weather_df, columns=["天气"])
    print(weather_df.columns.tolist())

    result_df = pd.merge(temp_df, weather_df, on="date", how="left")
    print(result_df.columns.tolist())

    return result_df


def main():
    df = load_data()
    df = merge_calendar_weather(df)
    df = df.sort_values([DATE_COL, STATION_COL, BLOOD_COL]).reset_index(drop=True)

    # bool天气列 -> 0/1
    weather_cols = [c for c in df.columns if c.startswith("天气_")]
    for c in weather_cols:
        df[c] = df[c].astype(int)

    # 这里自动选特征（你也可以手动列出来）
    # LSTM 用：目标历史+周期+温度+天气
    # XGB 用：滞后滚动+节假日/工作日+周期+温度+天气
    base_time_cols = []
    for c in ["weekday_sin", "weekday_cos", "month_sin", "month_cos", "weekday", "month", "day", "dayofyear",
              "weekofyear", "is_weekend", "is_holiday", "is_workday", "is_before_holiday", "is_after_holiday"]:
        if c in df.columns:
            base_time_cols.append(c)

    temp_cols = [c for c in ["平均温度", "temp_sq", "temp_diff", "temp_rolling3"] if c in df.columns]
    lag_cols = [c for c in ["lag1", "lag7", "rolling7", "rolling14", "rolling30"] if c in df.columns]

    # 兜底：如果lag没有，就也不报错（训练函数里会生成lag1/lag7/rolling7）
    feature_cols_lstm = ([TARGET_COL] + lag_cols + [c for c in base_time_cols if "sin" in c or "cos" in c] + temp_cols
                         + weather_cols)
    feature_cols_lstm = list(dict.fromkeys([c for c in feature_cols_lstm if c in df.columns]))

    feature_cols_xgb = lag_cols + base_time_cols + temp_cols + weather_cols
    feature_cols_xgb = list(dict.fromkeys([c for c in feature_cols_xgb if c in df.columns]))

    # 必要列检查
    need = [DATE_COL, STATION_COL, BLOOD_COL, TARGET_COL]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要列：{miss}")

    print("LSTM features:", feature_cols_lstm)
    print("XGB  features:", feature_cols_xgb)

    # 分组训练：血站 + 血型
    blood_types = ["ALL", "O", "B", "A", "AB"]
    station_list = df[STATION_COL].dropna().unique().tolist()
    for station in station_list:
        station_df = df[df[STATION_COL] == station].copy()
        for blood_type in blood_types:
            blood_df = None
            if blood_type == "ALL":
                other_cols = [c for c in station_df.columns if c not in [STATION_COL, TARGET_COL, BLOOD_COL]]
                agg_map = {TARGET_COL: "sum", **{c: "first" for c in other_cols}}
                blood_df = (
                    station_df.groupby(DATE_COL, as_index=False)
                    .agg(agg_map)
                )
            else:
                blood_df = station_df[station_df[BLOOD_COL] == blood_type].copy()
            if blood_df is not None and station == '北京市红十字血液中心':
                group_name = f"{station}_{blood_type}"
                train_one_group(blood_df, group_name, feature_cols_lstm, feature_cols_xgb)


if __name__ == "__main__":
    main()
