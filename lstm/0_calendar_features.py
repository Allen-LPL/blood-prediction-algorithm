import os
import pandas as pd
import numpy as np
import requests
import chinese_calendar as cc
from datetime import datetime

# ----------------------------
# 配置
# ----------------------------
CALENDAR_CACHE_FILE = "./feature/calendar_2015_2026.csv"
WEATHER_CACHE_FILE = "./feature/weather_2015_2026.csv"

BEIJING_LAT = 39.9042
BEIJING_LON = 116.4074

START_YEAR = 2015
END_YEAR = 2025


# ----------------------------
# 1️⃣ 生成日期表和时间特征
# ----------------------------
def generate_calendar_table(start_year=2015, end_year=2026):
    date_range = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D")
    df = pd.DataFrame({"date": date_range})

    # 安全节假日/调休判断
    def safe_is_holiday(d):
        if 2004 <= d.year <= 2026:
            return int(cc.is_holiday(d.date()))
        return 0

    def safe_is_workday(d):
        if 2004 <= d.year <= 2026:
            return int(cc.is_workday(d.date()))
        return 0

    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["date"].apply(safe_is_holiday)
    df["is_workday"] = df["date"].apply(safe_is_workday)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # 周期特征
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ----------------------------
# 2️⃣ 天气 code 转文字
# ----------------------------
def weather_code_to_text(code):
    mapping = {
        0: "晴",
        1: "少云",
        2: "多云",
        3: "阴",
        45: "雾",
        48: "霜雾",

        51: "毛毛雨",
        53: "小雨",
        55: "中雨",
        56: "冻毛毛雨",
        57: "强冻毛毛雨",

        61: "小雨",
        63: "中雨",
        65: "大雨",
        66: "冻雨",
        67: "强冻雨",

        71: "小雪",
        73: "中雪",
        75: "大雪",
        77: "冰粒",

        80: "小阵雨",
        81: "中阵雨",
        82: "大阵雨",

        85: "小阵雪",
        86: "大阵雪",

        95: "雷阵雨",
        96: "雷阵雨伴冰雹",
        99: "强雷暴伴冰雹"
    }
    return mapping.get(code, str(code))


# ----------------------------
# 3️⃣ 获取天气（自动缓存）
# ----------------------------
def fetch_weather_for_date(date_str):
    if os.path.exists(WEATHER_CACHE_FILE):
        cache_df = pd.read_csv(WEATHER_CACHE_FILE, parse_dates=["date"])
        cached = cache_df.loc[cache_df["date"] == date_str]
        if not cached.empty:
            return {"weather": cached["weather"].values[0], "avg_temp": cached["avg_temp"].values[0]}

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={BEIJING_LAT}&longitude={BEIJING_LON}"
        f"&start_date={date_str}&end_date={date_str}"
        "&daily=weathercode,temperature_2m_max,temperature_2m_min"
        "&timezone=Asia/Shanghai"
    )
    res = requests.get(url).json()
    daily = res.get("daily", {})
    codes = daily.get("weathercode", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    if codes and t_max and t_min:
        weather = weather_code_to_text(codes[0])
        avg_temp = (t_max[0] + t_min[0]) / 2
        # 保存缓存
        new_row = pd.DataFrame({"date": [date_str],
                                "weather": [weather],
                                "weather_code": [codes[0]],
                                "avg_temp": [avg_temp]})
        if os.path.exists(WEATHER_CACHE_FILE):
            new_row.to_csv(WEATHER_CACHE_FILE, mode='a', header=False, index=False)
        else:
            new_row.to_csv(WEATHER_CACHE_FILE, index=False)
        return {"weather": weather, "avg_temp": avg_temp, "weather_code": codes[0]}
    return {"weather": None, "avg_temp": None, "weather_code": None}


# ----------------------------
# 4️⃣ 批量加入天气
# ----------------------------
def add_weather_features(df):
    df["天气"] = None
    df["平均温度"] = None
    for idx, row in df.iterrows():
        date = row["date"]
        date_str = date.strftime("%Y-%m-%d")
        w = fetch_weather_for_date(date_str)
        df.at[idx, "天气"] = w["weather"]
        df.at[idx, "平均温度"] = w["avg_temp"]
    return df


def fetch_weather_by_year(year):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={BEIJING_LAT}&longitude={BEIJING_LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=weathercode,temperature_2m_max,temperature_2m_min"
        "&timezone=Asia/Shanghai"
    )

    res = requests.get(url)
    data = res.json()

    daily = data["daily"]

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "weather_code": daily["weathercode"],
        "t_max": daily["temperature_2m_max"],
        "t_min": daily["temperature_2m_min"],
    })

    df["平均温度"] = (df["t_max"] + df["t_min"]) / 2
    df["天气"] = df["weather_code"].apply(weather_code_to_text)

    return df[["date", "天气", "平均温度", "weather_code"]]


def add_weather_features_batch(df):
    weather_all = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"正在获取 {year} 年天气数据...")
        df_year = fetch_weather_by_year(year)
        weather_all.append(df_year)

    weather_df = pd.concat(weather_all)

    df = df.merge(weather_df, on="date", how="left")
    return df


# ----------------------------
# 5️⃣ 主函数
# ----------------------------
def generate_full_dataframe():
    df = generate_calendar_table(START_YEAR, END_YEAR)
    df = add_weather_features_batch(df)
    return df


# ----------------------------
# 6️⃣ 生成完整表
# ----------------------------
if __name__ == "__main__":
    df_all = generate_full_dataframe()
    df_all.to_csv("./feature/blood_calendar_weather_2015_2026.csv", index=False)
    print("生成完成，已保存：blood_calendar_weather_2015_2026.csv")
