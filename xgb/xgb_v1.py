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

    df = full_dates.merge(df, on=DATE_COL, how="left")
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


def add_dynamic_feature(g2: pd.DataFrame):
    g2["lag1"] = g2[TARGET_COL].shift(1)
    g2["lag7"] = g2[TARGET_COL].shift(7)
    if "rolling7" not in g2.columns:
        g2["rolling7"] = g2[TARGET_COL].rolling(7).mean()

    g2["absdiff1"] = (g2[TARGET_COL] - g2["lag1"]).abs()
    g2["rolling_std7"] = g2[TARGET_COL].rolling(7).std()
    g2["rolling_std14"] = g2[TARGET_COL].rolling(14).std()
    g2["rolling_absdiff7"] = g2["absdiff1"].rolling(7).mean()

    g2["lag14"] = g2[TARGET_COL].shift(14)
    g2["lag28"] = g2[TARGET_COL].shift(28)

    g2["absdiff7"] = (g2[TARGET_COL] - g2["lag7"]).abs()
    g2["diff7"] = (g2[TARGET_COL] - g2["lag7"])
    g2["diff14"] = (g2[TARGET_COL] - g2["lag14"])

    g2["rolling14"] = g2[TARGET_COL].rolling(14).mean()
    g2["rolling28"] = g2[TARGET_COL].rolling(28).mean()
    g2["ewm7"] = g2[TARGET_COL].ewm(span=7, adjust=False).mean()
    g2["ewm14"] = g2[TARGET_COL].ewm(span=14, adjust=False).mean()
    return g2



def train_one_group(group, group_blood_df, weather_df):
    # 1.blood特征
    label_col = "label_tplus1"
    final_valid_days = 60
    test_ratio = 0.1
    group_blood_df = add_prev_rolling_sums(group_blood_df, windows=(1, 3, 7, 14, 28))
    group_blood_df = add_dynamic_feature(group_blood_df)
    group_blood_df[label_col] = group_blood_df[TARGET_COL]#.shift(-1)
    group_blood_df = group_blood_df.dropna(subset=["label_tplus1"])

    # 2.合并blood weather
    df = pd.merge(group_blood_df, weather_df, on="date", how="left")
    feat_cols = []
    for col in df.columns:
        if col in [TARGET_COL, label_col, DATE_COL, 'date_x', 'date_y', 'lag1']:
            continue
        feat_cols.append(col)

    # 3.xgb param
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
    print(f'feat_cols:{feat_cols}')

    # 切分数据
    final_valid_start = df[DATE_COL].max() - pd.Timedelta(days=final_valid_days - 1)
    df_final_valid = df[df[DATE_COL] >= final_valid_start].copy()
    df_train_test = df[df[DATE_COL] < final_valid_start].copy()
    test_size = max(1, int(len(df_train_test) * test_ratio))
    df_train = df_train_test.iloc[:-test_size].copy()
    df_test = df_train_test.iloc[-test_size:].copy()

    print(
        f"[{group}] train={len(df_train)}, "
        f"test={len(df_test)}, "
        f"final_valid={len(df_final_valid)}, "
        f"final_valid_range=({df_final_valid[DATE_COL].min().date()} ~ {df_final_valid[DATE_COL].max().date()})"
    )

    X_train = df_train[feat_cols].values
    y_train = df_train[label_col].values

    X_test = df_test[feat_cols].values
    y_test = df_test[label_col].values

    X_final_valid = df_final_valid[feat_cols].values
    y_final_valid = df_final_valid[label_col].values

    # 4.xgb 训练
    sample_weight = []
    for y in y_train:
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
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[
        (X_test, y_test)], sample_weight=sample_weight)

    # 5.预测
    pred_test = normalize_pred(model.predict(X_test))
    pred_final_valid = normalize_pred(model.predict(X_final_valid))


    # 5.指标
    test_report = metric_item(y_test, pred_test)
    final_valid_report = metric_item(y_final_valid, pred_final_valid)

    eval_result = model.evals_result()

    print(f"\n[{group}] ===== 测试集结果 =====")
    print(test_report)

    print(f"\n[{group}] ===== 最终验证集（最后{final_valid_days}天）结果 =====")
    print(final_valid_report)

    if "validation_0" in eval_result and "rmse" in eval_result["validation_0"]:
        print(f"[{group}] best eval rmse: {eval_result['validation_0']['rmse'][-1]}")

    # ========== 9. 特征重要性 ==========
    feats = []
    for feat, score in zip(feat_cols, model.feature_importances_):
        feats.append((feat, score))
    feats = sorted(feats, key=lambda x: x[1], reverse=True)

    print(f"\n[{group}] ===== Top20 特征重要性 =====")
    for feat, score in feats[:20]:
        print(feat, score)

    train_keep_days = 300
    train_plot_start = df_train[DATE_COL].max() - pd.Timedelta(days=train_keep_days - 1)
    df_train_plot = df_train[df_train[DATE_COL] >= train_plot_start].copy()
    df_plot = pd.concat([df_train_plot, df_test, df_final_valid], axis=0).sort_values(DATE_COL)

    # ========== 10. 画图1：全时间轴 ==========
    plt.figure(figsize=(16, 5))
    plt.plot(df_plot[DATE_COL], df_plot[label_col], label="真实值")
    plt.plot(df_test[DATE_COL], pred_test, label="测试集预测")
    plt.plot(df_final_valid[DATE_COL], pred_final_valid, label=f"最终验证集预测(最后{final_valid_days}天)")

    # 分界线
    plt.axvline(df_test[DATE_COL].iloc[0], linestyle="--", label="测试集开始")
    plt.axvline(df_final_valid[DATE_COL].iloc[0], linestyle="--", label="最终验证集开始")

    plt.title(f"{group} 全时间轴预测效果", fontproperties=FONT)
    plt.xlabel("date")
    plt.ylabel("num")
    plt.legend(prop=FONT)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ========== 11. 画图2：最终验证集近一个月 ==========
    plt.figure(figsize=(14, 5))
    plt.plot(df_final_valid[DATE_COL], y_final_valid, marker="o", label="真实值")
    plt.plot(df_final_valid[DATE_COL], pred_final_valid, marker="o", label="预测值(XGB)")
    plt.title(f"{group} 最终验证集（最后{final_valid_days}天）", fontproperties=FONT)
    plt.xlabel("date")
    plt.ylabel("num")
    plt.legend(prop=FONT)
    plt.xticks(rotation=45)
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

