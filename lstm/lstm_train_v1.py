import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error


def load_data():
    use_cols = [0, 16, 22, 24]
    df = pd.read_excel("./data/采血20250101-0630.xlsx", usecols=use_cols)
    # df2 = pd.read_excel("./data/2024.xlsx", usecols=use_cols)
    # df = pd.concat([df1, df2], axis=0, ignore_index=True)
    df.rename(columns={'采血部门': '血站',
                       '档案血型': '血型',
                       '采血量': '血量'}, inplace=True)
    df["采血日期"] = pd.to_datetime(
        df["采血时间"],
        format="%Y/%m/%d %H:%M:%S"
    ).dt.date

    df["采血日期"] = pd.to_datetime(df["采血日期"])
    df["weekday"] = df["采血日期"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["血型"] = (
        df["血型"]
        .astype("string")  # 统一字符串类型
        .str.strip()  # 去除空格
        .str.upper()  # 转大写
    )
    return df


if __name__ == '__main__':
    # 1.加载数据
    data = load_data()

    # 2.编码血站 & 血型
    # 选择一个血站测试（先单血站优化）
    station_name = data["血站"].unique()[0]
    data = data[data["血站"] == station_name]

    look_back = 14
    blood_types = ["O", "B", "A", "AB"]

    total_predictions = {}
    total_reals = {}

    for blood_type in blood_types:
        print(f"\n===== 训练血型 {blood_type} =====")

        df = data[data["血型"] == blood_type]
        df = (
            df.groupby("采血日期")
            .agg({
                "血量": "sum",
                # "温度": "mean",
                # "是否节假日": "max",
                # "是否大型活动": "max",
                "is_weekend": "max",
                "weekday_sin": "max",
                "weekday_cos": "max"
            })
            .reset_index()
            .sort_values("采血日期")
        )

        features = [
            "血量",
            # "温度",
            # "是否节假日",
            # "是否大型活动",
            "is_weekend",
            "weekday_sin",
            "weekday_cos"
        ]

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
            LSTM(64, return_sequences=False,
                 input_shape=(look_back, len(features))),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=80,
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
        print("MAE:", mae)
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
    # 6️⃣ 总量预测
    # ==============================

    min_len = min(len(v) for v in total_predictions.values())
    min_len_real = min(len(v) for v in total_reals.values())

    total_sum = np.zeros(min_len)
    total_real = np.zeros(min_len_real)

    for v in total_predictions.values():
        total_sum += v[:min_len]
    for v in total_reals.values():
        total_real += v[:min_len_real]

    plt.figure()
    plt.plot(total_sum, label="total_predict")
    plt.plot(total_real, label="total_real")
    plt.title("total_blood")
    plt.legend()
    plt.show()

    print("\n总量预测完成")
