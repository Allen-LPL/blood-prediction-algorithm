import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ========== 中文字体支持（Mac） ==========
plt.rcParams["font.sans-serif"] = ["PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

import matplotlib.font_manager as fm


def load_data():
    all_df = pd.read_csv('feature/remove_group_data.csv')
    all_df["date"] = pd.to_datetime(all_df["date"])
    return all_df


if __name__ == '__main__':
    df_station_daily = load_data()
    df_station_daily["date"] = pd.to_datetime(df_station_daily["date"])

    df_station_daily_2024 = df_station_daily[
        (df_station_daily["date"] >= "2024-01-01") &
        (df_station_daily["date"] < "2025-01-01")
        ].copy()

    font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")

    station_name = "北京市红十字血液中心"

    df_plot = (
        df_station_daily_2024[df_station_daily["血站"] == station_name]
        .sort_values("date")
        .copy()
    )

    # 7日滚动均值
    df_plot["7日均线"] = df_plot["总血量"].rolling(7, min_periods=1).mean()

    # 计算异常阈值（2倍标准差）
    mean_val = df_plot["总血量"].mean()
    std_val = df_plot["总血量"].std()
    threshold = mean_val + 2 * std_val

    df_plot["是否异常"] = df_plot["总血量"] > threshold

    # ========== 开始画图 ==========
    plt.figure(figsize=(14, 6))

    # 原始趋势
    plt.plot(df_plot["date"], df_plot["总血量"])

    # 7日均线
    plt.plot(df_plot["date"], df_plot["7日均线"])

    # 异常点标记
    plt.scatter(
        df_plot.loc[df_plot["是否异常"], "date"],
        df_plot.loc[df_plot["是否异常"], "总血量"]
    )

    # 阈值线
    plt.axhline(threshold)

    plt.title(f"{station_name} 每日总血量趋势分析", fontproperties=font)
    plt.xlabel("日期", fontproperties=font)
    plt.ylabel("每日总血量", fontproperties=font)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
