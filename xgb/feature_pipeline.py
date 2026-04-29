import math
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from chinese_calendar import get_holiday_detail, Holiday
from sklearn.metrics import root_mean_squared_error

# 数据列名 (CSV 中的中文列名)
DATE_COL = "date"
STATION_COL = "血站"
BLOOD_COL = "血型"
TARGET_COL = "总血量"

# 文件路径: 模型保存目录 / 数据源目录 (支持环境变量覆盖)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "lstm" / "feature"

BLOOD_CSV = os.environ.get("BLOOD_CSV", str(DATA_DIR / "remove_group_data.csv"))
WEATHER_CSV = os.environ.get(
    "WEATHER_CSV", str(DATA_DIR / "blood_calendar_weather_2015_2026.csv")
)

# 北京大学校本部开学日期 (2015–2026), 用于生成"开学季"特征
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


# ---------- 特征工程: 节假日 / 开学季 / 日历 ----------


def is_pku_semester_start_season(d, pre_days=14, post_days=21) -> int:
    """判断日期 d 是否落在北大开学季窗口 [开学前pre_days, 开学后post_days] 内."""
    d = pd.to_datetime(d).date()
    for s in PKU_CLASS_STARTS:
        if s - timedelta(days=pre_days) <= d <= s + timedelta(days=post_days):
            return 1
    return 0


def is_holiday_name(d, target: Holiday) -> int:
    """判断日期 d 是否为指定的中国法定节假日 (春节/清明等)."""
    d = pd.to_datetime(d).date()
    try:
        _, hol = get_holiday_detail(d)
        return int(hol == target)
    except Exception:
        return 0


def add_cn_holiday_flags(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """为 DataFrame 添加 is_spring_festival / is_qingming 二值列."""
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


def add_calendar_features(
    df: pd.DataFrame, date_col: str = "date", include_pku: bool = True
) -> pd.DataFrame:
    df = add_cn_holiday_flags(df, date_col)
    if include_pku:
        df = add_pku_season_flag(df, date_col, pre_days=14, post_days=21)
    return df


# ---------- 特征工程: 滞后/滚动统计 ----------


def add_prev_rolling_sums(df: pd.DataFrame, windows=(1, 3, 7, 14, 28)) -> pd.DataFrame:
    """计算前 N 日滚动采血量总和 (shift(1) 避免数据泄露)."""
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    full_dates = pd.DataFrame({DATE_COL: pd.date_range(min_date, max_date, freq="D")})
    df = full_dates.merge(df, on=DATE_COL, how="left")
    df[TARGET_COL] = df[TARGET_COL].fillna(0.0)
    df = df.sort_values(DATE_COL)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    for w in windows:
        col = f"{w}d_sum"
        df[col] = df[TARGET_COL].rolling(w, min_periods=1).sum().shift(1)
        df[col] = df[col].fillna(0.0)
    return df


def add_dynamic_feature(g2: pd.DataFrame) -> pd.DataFrame:
    """添加 lag / rolling_mean / rolling_std / ewm 等动态特征.

    所有滚动/EWM 统计都基于 lag1 (前一天及更早), 避免把当天 target 泄露到特征中.
    在递推预测时, 这保证训练集和推理集的特征分布一致.
    """
    g2["lag1"] = g2[TARGET_COL].shift(1)
    g2["lag7"] = g2[TARGET_COL].shift(7)
    g2["lag14"] = g2[TARGET_COL].shift(14)
    g2["lag28"] = g2[TARGET_COL].shift(28)

    # 所有滚动/EWM 统计都从 lag1 (即 shift(1) 后的序列) 出发, 严格避免泄露
    base = g2["lag1"]  # 截至 t-1 的序列
    if "rolling7" not in g2.columns:
        g2["rolling7"] = base.rolling(7, min_periods=1).mean()
    g2["rolling14"] = base.rolling(14, min_periods=1).mean()
    g2["rolling28"] = base.rolling(28, min_periods=1).mean()
    g2["rolling_std7"] = base.rolling(7, min_periods=2).std()
    g2["rolling_std14"] = base.rolling(14, min_periods=2).std()
    g2["ewm7"] = base.ewm(span=7, adjust=False).mean()
    g2["ewm14"] = base.ewm(span=14, adjust=False).mean()

    # 差分类特征也基于 lag (历史信号), 不含当天 target
    g2["absdiff1"] = (g2["lag1"] - g2["lag1"].shift(1)).abs()
    g2["absdiff7"] = (g2["lag1"] - g2["lag7"]).abs()
    g2["diff7"] = g2["lag1"] - g2["lag7"]
    g2["diff14"] = g2["lag1"] - g2["lag14"]
    g2["rolling_absdiff7"] = g2["absdiff1"].rolling(7, min_periods=1).mean()

    return g2


def normalize_pred(pred):
    """预测值下限裁剪到 0 并向下取整."""
    return np.array([max(0, math.floor(v)) for v in pred])


def metric_item(true_y, pred_y) -> dict:
    """计算 RMSE、平均误差率、各误差阈值达标比例."""
    result = {}
    result["rmse"] = float(root_mean_squared_error(true_y, pred_y))
    err_rates = []
    lt10 = lt20 = lt30 = gt100 = 0
    for yt, yp in zip(true_y, pred_y):
        yp = max(yp, 0)
        er = abs(yt - yp) / (yt + 1e-5)
        err_rates.append(er)
        if er < 0.1:
            lt10 += 1
        if er < 0.2:
            lt20 += 1
        if er < 0.3:
            lt30 += 1
        if er > 1.0:
            gt100 += 1
    n = len(err_rates)
    result["mean_error_rate"] = round(float(np.mean(err_rates)), 5)
    result["within_10pct"] = round(lt10 / n, 5)
    result["within_20pct"] = round(lt20 / n, 5)
    result["within_30pct"] = round(lt30 / n, 5)
    result["over_100pct"] = round(gt100 / n, 5)
    result["total_samples"] = n
    return result


# ---------- 数据加载 ----------


def load_weather_data(path: str = WEATHER_CSV) -> pd.DataFrame:
    """读取天气 CSV 并派生温度/天气 one-hot/节假日/开学季等特征."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["temp_sq"] = df["平均温度"] ** 2
    df["temp_diff"] = df["平均温度"].diff()
    df["temp_rolling3"] = df["平均温度"].rolling(3).mean()
    df = pd.get_dummies(df, columns=["天气"])
    df["temp_rolling3"] = df["temp_rolling3"].fillna(0.0)
    weather_cols = [c for c in df.columns if c.startswith("天气_")]
    for c in weather_cols:
        df[c] = df[c].astype(int)
    df = add_calendar_features(df)
    return df
