from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import chinese_calendar as cc


def safe_is_workday(x):
    if pd.isna(x):
        return 0
    if x.year < 2004 or x.year > 2026:
        return 0
    return int(cc.is_workday(x.date()))


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


if __name__ == '__main__':
    # 1.加载数据
    data = load_data()
    data = merge_calendar_weather(data)

    # 2.编码血站 & 血型
    # 选择一个血站测试（先单血站优化）
    station_name = "北京市红十字血液中心"
    data = data[data["血站"] == station_name]

    look_back = 30
    blood_types = ["ALL", "O", "B", "A", "AB"]

    total_predictions = {}
    total_reals = {}

    features = [
        "总血量",
        "lag1",
        "lag7",
        "rolling7",
        "is_weekend",
        "weekday_sin",
        "weekday_cos",
        "is_holiday",
        "is_workday",
        "is_before_holiday",
        "is_after_holiday",
        "month_sin",
        "month_cos",
        "平均温度",
        "temp_sq",
        "temp_diff",
        "temp_rolling3"
    ]
    weather_cols = [col for col in data.columns if col.startswith("天气_")]

    for col in weather_cols:
        if col not in features:
            features.append(col)

    print(f"features: \n {features}")

    for blood_type in blood_types:
        print(f"\n===== 训练血型 {blood_type} =====")
        df = pd.DataFrame()
        if blood_type == "ALL":
            df = (
                data.groupby(["date"], as_index=False)
                .agg({
                    "总血量": "sum",
                    "is_weekend": "max",
                    "weekday_sin": "max",
                    "weekday_cos": "max",
                    "is_holiday": "max",
                    "is_workday": "max",
                    "is_before_holiday": "max",
                    "is_after_holiday": "max",
                    "month_sin": "max",
                    "month_cos": "max",
                    "平均温度": "max",
                    "temp_sq": "max",
                    "temp_diff": "max",
                    "temp_rolling3": "max",
                    "weather_code": "max",
                    "天气_中雨": "max",
                    "天气_中雪": "max",
                    "天气_多云": "max",
                    "天气_大雨": "max",
                    "天气_大雪": "max",
                    "天气_小雨": "max",
                    "天气_小雪": "max",
                    "天气_少云": "max",
                    "天气_晴": "max",
                    "天气_毛毛雨": "max",
                    "天气_阴": "max"
                })
            )
        else:
            df = (
                data[data["血型"] == blood_type]
                .sort_values("date")  # 时间序列一定要排序
                .copy()  # 关键：防止 warning
            )

        # ====== 新增滞后特征 ======
        df["lag1"] = df["总血量"].shift(1)
        df["lag7"] = df["总血量"].shift(7)
        df["rolling7"] = df["总血量"].rolling(7).mean()
        df = df.dropna()

        values = df[features].values

        # 每个血型单独 scaler
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values)

        X_list, y_list = [], []

        for i in range(len(values_scaled) - look_back):
            X_list.append(values_scaled[i:i + look_back])
            y_list.append(values_scaled[i + look_back][0])

        X = np.array(X_list)
        y = np.array(y_list)

        # 训练 / 验证划分
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = Sequential([
            Input(shape=(look_back, len(features))),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        pred = model.predict(X_val)

        # 反归一化
        pred_full = np.zeros((len(pred), len(features)))
        pred_full[:, 0] = pred.flatten()

        y_full = np.zeros((len(y_val), len(features)))
        y_full[:, 0] = y_val

        pred_real = scaler.inverse_transform(pred_full)[:, 0]
        y_real = scaler.inverse_transform(y_full)[:, 0]

        mae = mean_absolute_error(y_real, pred_real)
        mape = np.mean(np.abs((y_real - pred_real) / y_real))
        print("MAE:", mae)
        print("MAPE:", mape)
        print("平均血量:", np.mean(y_real))
        print("误差占比:", mae / np.mean(y_real))

        # 保存预测结果
        total_predictions[blood_type] = pred_real
        total_reals[blood_type] = y_real

        # ==============================
        # 5️⃣ 画图
        # ==============================

        plt.figure()
        plt.plot(y_real, label="real")
        plt.plot(pred_real, label="predict")
        plt.title(f"{blood_type}_predict")
        plt.legend()
        plt.show()

        # ==============================
        # 6️⃣ 训练曲线
        # ==============================

        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title(f"{blood_type}_train_graph")
        plt.legend()
        plt.show()

    print("\n总量预测完成")
