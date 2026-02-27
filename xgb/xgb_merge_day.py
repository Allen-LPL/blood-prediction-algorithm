import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from typing import Dict, List, Tuple, Optional

DATE_COL = "date"
STATION_COL = "血站"
BLOOD_COL = "血型"
TARGET_COL = "血量"
BOTTOM_BLOOD_TYPES = ["A", "B", "O", "AB", "ALL"]  # 训练底层只用这几个，不直接用 ALL

BLOOD_TYPE_MAP = {"A": 2, "B": 3, "O": 4, "AB": 5, "ALL": 1}

STATION_MAP = {
    "ALL": 1,
    "北京市红十字血液中心": 2,
    "北京市昌平区中心血库": 3,
    "北京市大兴区中心血库": 4,
    "北京市房山区中心血库": 5,
    "北京市怀柔区中心血库": 6,
    "北京市门头沟区中心血库": 7,
    "北京市平谷区中心血库": 8,
    "北京市顺义区中心血库": 9,
    "密云区中心血站": 10,
    "通州区中心血站": 11,
    "延庆区中心血站": 12
}

FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")


def _set_cn_title(text: str):
    if FONT is not None:
        plt.title(text, fontproperties=FONT)
    else:
        plt.title(text)


def _set_cn_xlabel(text: str):
    if FONT is not None:
        plt.xlabel(text, fontproperties=FONT)
    else:
        plt.xlabel(text)


def _set_cn_ylabel(text: str):
    if FONT is not None:
        plt.ylabel(text, fontproperties=FONT)
    else:
        plt.ylabel(text)


def _set_cn_legend():
    if FONT is not None:
        plt.legend(prop=FONT)
    else:
        plt.legend()


def normalize_pred(x):
    arr = np.array(x, dtype=float)
    arr = np.maximum(arr, 0)
    return arr


def wmape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return 0.0
    return np.abs(y_true - y_pred).sum() / denom


def metric_item(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    return {
        "RMSE": round(mean_squared_error(y_true, y_pred) ** 0.5, 5),
        "MAE": round(mean_absolute_error(y_true, y_pred), 5),
        "WMAPE": round(wmape(y_true, y_pred), 5),
        "平均误差率": round(float(np.mean(ape)), 5),
        "误差率<10%比例": round(float((ape < 0.1).mean()), 5),
        "误差率<20%比例": round(float((ape < 0.2).mean()), 5),
        "误差率<30%比例": round(float((ape < 0.3).mean()), 5),
        "总量": int(len(y_true)),
    }


def build_bottom_daily_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # 只保留底层血型
    # df = df[df[BLOOD_COL].isin(BOTTOM_BLOOD_TYPES)].copy()

    # 聚合
    df = (
        df.groupby([DATE_COL, STATION_COL, BLOOD_COL], as_index=False)[TARGET_COL]
        .sum()
    )

    # 补齐 date × station × blood_type
    all_dates = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    all_stations = sorted(df[STATION_COL].dropna().astype(str).unique().tolist())
    all_bloods = sorted(df[BLOOD_COL].dropna().astype(str).unique().tolist())

    full_index = pd.MultiIndex.from_product(
        [all_dates, all_stations, all_bloods],
        names=[DATE_COL, STATION_COL, BLOOD_COL]
    )

    df = (
        df.set_index([DATE_COL, STATION_COL, BLOOD_COL])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0)
    return df


def build_summary_with_all(df_bottom: pd.DataFrame, value_col: str = TARGET_COL) -> pd.DataFrame:
    df = df_bottom.copy()
    base = df[[DATE_COL, STATION_COL, BLOOD_COL, value_col]].copy().rename(columns={value_col: TARGET_COL})

    blood_all = df.groupby([DATE_COL, STATION_COL], as_index=False)[value_col].sum()
    blood_all[BLOOD_COL] = "ALL"
    blood_all = blood_all.rename(columns={value_col: TARGET_COL})

    station_all = df.groupby([DATE_COL, BLOOD_COL], as_index=False)[value_col].sum()
    station_all[STATION_COL] = "ALL"
    station_all = station_all.rename(columns={value_col: TARGET_COL})

    grand_all = df.groupby([DATE_COL], as_index=False)[value_col].sum()
    grand_all[STATION_COL] = "ALL"
    grand_all[BLOOD_COL] = "ALL"
    grand_all = grand_all.rename(columns={value_col: TARGET_COL})

    out = pd.concat([base, blood_all, station_all, grand_all], ignore_index=True)
    out = out[[DATE_COL, STATION_COL, BLOOD_COL, TARGET_COL]].copy()
    return out.sort_values([DATE_COL, STATION_COL, BLOOD_COL]).reset_index(drop=True)


def merge_calendar_weather(df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    weather_df = calendar_df.copy()
    weather_df[DATE_COL] = pd.to_datetime(weather_df[DATE_COL])

    weather_df["date"] = pd.to_datetime(weather_df["date"])

    weather_df["temp_sq"] = weather_df["平均温度"] ** 2
    weather_df["temp_diff"] = weather_df["平均温度"].diff()
    weather_df["temp_rolling3"] = weather_df["平均温度"].rolling(3).mean()
    weather_df = pd.get_dummies(weather_df, columns=["天气"])

    for c in ["temp_sq", "temp_diff", "temp_rolling3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = pd.merge(df, weather_df, on=DATE_COL, how="left")

    return df


def add_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["series_id", DATE_COL]).reset_index(drop=True)

    def _add_one_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        s = g[TARGET_COL]

        # lag
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            g[f"lag{lag}"] = s.shift(lag)
            g[f"lag{lag}"] = g[f"lag{lag}"].fillna(0)

        # rolling mean/std: 只看过去，因此先 shift(1)
        s_prev = s.shift(1)
        for win in [3, 7, 14, 28]:
            g[f"rolling_mean_{win}"] = s_prev.rolling(win, min_periods=1).mean()
            g[f"rolling_mean_{win}"] = g[f"rolling_mean_{win}"].fillna(0)
            g[f"rolling_std_{win}"] = s_prev.rolling(win, min_periods=2).std()
            g[f"rolling_std_{win}"] = g[f"rolling_std_{win}"].fillna(0)

        # 差分
        g["prev_diff1"] = s.shift(1) - s.shift(2)
        g["prev_diff1"] = g["prev_diff1"].fillna(0)
        g["prev_diff7"] = s.shift(1) - s.shift(8)
        g["prev_diff7"] = g["prev_diff7"].fillna(0)

        # 目标：下一天
        g["label"] = s#.shift(-1)
        g["label_date"] = g[DATE_COL]#.shift(-1)

        return g

    df = df.groupby("series_id", group_keys=False).apply(_add_one_group)
    return df


def encode_category_columns(df: pd.DataFrame):
    df = df.copy()

    df["station_id"] = df[STATION_COL].map(STATION_MAP).astype(int)
    df["blood_id"] = df[BLOOD_COL].map(BLOOD_TYPE_MAP).astype(int)
    df["series_id"] = df["station_id"].astype(str) + "_" + df["blood_id"].astype(str)

    return df


def prepare_train_df(raw_df: pd.DataFrame, calendar_df: pd.DataFrame):
    # 1. 底层表
    df = build_bottom_daily_df(raw_df)

    # 2. 编码
    df = encode_category_columns(df)

    # 3. 时序特征
    df = add_ts_features(df)

    # 4. 去掉没法训练的样本
    df = df.dropna(subset=["label", "label_date", "lag28"]).reset_index(drop=True)

    # 5. 合并日历天气
    df = merge_calendar_weather(df, calendar_df)

    return df


def split_train_test_final(df: pd.DataFrame, final_valid_days=7, test_days=30):
    """
    按 label_date 切分，避免未来泄漏
    """
    df = df.copy()
    max_label_date = df["label_date"].max()

    final_valid_start = max_label_date - pd.Timedelta(days=final_valid_days - 1)
    test_start = final_valid_start - pd.Timedelta(days=test_days)

    train_df = df[df["label_date"] < test_start].copy()
    test_df = df[(df["label_date"] >= test_start) & (df["label_date"] < final_valid_start)].copy()
    final_df = df[df["label_date"] >= final_valid_start].copy()

    print("train:", train_df["label_date"].min(), "~", train_df["label_date"].max(), len(train_df))
    print("test :", test_df["label_date"].min(), "~", test_df["label_date"].max(), len(test_df))
    print("final:", final_df["label_date"].min(), "~", final_df["label_date"].max(), len(final_df))

    return train_df, test_df, final_df, final_valid_start


def get_feature_cols(df: pd.DataFrame):
    exclude_cols = {
        DATE_COL, "label_date", "label",
        STATION_COL, BLOOD_COL, "series_id", TARGET_COL
    }
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    return feat_cols


def train_global_xgb(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols):
    X_train = train_df[feat_cols].fillna(0).values
    y_train = train_df["label"].values
    X_test = test_df[feat_cols].fillna(0).values
    y_test = test_df["label"].values

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

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    group = "ALL"
    ALL_test_df = test_df[(test_df[STATION_COL] == group) & (test_df[BLOOD_COL] == group)].copy()
    ALL_test_df.sort_values(by=[DATE_COL], inplace=True)
    all_X_test = ALL_test_df[feat_cols].fillna(0).values
    all_Y_text = ALL_test_df["label"].values

    pred_test = normalize_pred(model.predict(all_X_test))
    test_eval = metric_item(all_Y_text, pred_test)

    plt.figure(figsize=(16, 5))
    plt.plot(ALL_test_df[DATE_COL], all_Y_text, label="真实值")
    plt.plot(ALL_test_df[DATE_COL], pred_test, label="预测值(XGB)")
    plt.title(f"{group} predict）", fontproperties=FONT)
    plt.xlabel("date")
    plt.ylabel("num")
    plt.legend(prop=FONT)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\n[测试集结果]", test_eval)
    return model, pred_test, test_eval


def build_future_feature_row(target_date, history_df, station, blood, calendar_df):
    """
    history_df:
        某个 series 的历史数据，包含 date, 血量
        这里面后续会逐步混入预测值
    """
    hist = history_df.copy().sort_values(DATE_COL).reset_index(drop=True)
    vals = hist[TARGET_COL].tolist()

    def lag(n):
        if len(vals) < n:
            return 0.0
        return float(vals[-n])

    def rolling_mean(n):
        if len(vals) == 0:
            return 0.0
        return float(np.mean(vals[-n:]))

    def rolling_std(n):
        arr = vals[-n:]
        if len(arr) < 2:
            return 0.0
        return float(np.std(arr, ddof=1))

    row = {
        DATE_COL: pd.to_datetime(target_date),
        STATION_COL: station,
        BLOOD_COL: blood,
        "series_id": f"{station}_{blood}",
        "station_id": STATION_MAP[station],
        "blood_id": BLOOD_TYPE_MAP[blood],

        "lag1": lag(1),
        "lag2": lag(2),
        "lag3": lag(3),
        "lag7": lag(7),
        "lag14": lag(14),
        "lag21": lag(21),
        "lag28": lag(28),

        "rolling_mean_3": rolling_mean(3),
        "rolling_mean_7": rolling_mean(7),
        "rolling_mean_14": rolling_mean(14),
        "rolling_mean_28": rolling_mean(28),

        "rolling_std_3": rolling_std(3),
        "rolling_std_7": rolling_std(7),
        "rolling_std_14": rolling_std(14),
        "rolling_std_28": rolling_std(28),

        "prev_diff1": lag(1) - lag(2),
        "prev_diff7": lag(1) - lag(8) if len(vals) >= 8 else 0.0,
    }

    cal_row = calendar_df[calendar_df[DATE_COL] == pd.to_datetime(target_date)].copy()
    if len(cal_row) == 0:
        raise ValueError(f"calendar_df 缺少未来日期: {target_date}")

    cal_row = cal_row.iloc[0].to_dict()
    cal_row.pop(DATE_COL, None)
    row.update(cal_row)

    return pd.DataFrame([row])


def recursive_validate_final_n_days(
        model,
        raw_bottom_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        feat_cols: list,
        final_valid_days: int = 7
):
    """
    对每个底层序列做严格递推：
    - 最后 n 天作为最终验证集
    - 第1天用真实历史
    - 第2天开始用前面预测值继续滚
    """
    df = raw_bottom_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    max_date = df[DATE_COL].max()
    final_dates = pd.date_range(max_date - pd.Timedelta(days=final_valid_days - 1), max_date, freq="D")

    result_rows = []

    for (station, blood), g in df.groupby([STATION_COL, BLOOD_COL]):
        g = g.sort_values(DATE_COL).reset_index(drop=True)

        history_df = g[g[DATE_COL] < final_dates.min()][[DATE_COL, TARGET_COL]].copy()
        real_future_df = g[g[DATE_COL].isin(final_dates)][[DATE_COL, TARGET_COL]].copy()

        if len(real_future_df) != final_valid_days:
            continue

        recursive_hist = history_df.copy()

        for d in final_dates:
            feat_row = build_future_feature_row(
                target_date=d,
                history_df=recursive_hist,
                station=station,
                blood=blood,
                calendar_df=calendar_df
            )

            for c in feat_cols:
                if c not in feat_row.columns:
                    feat_row[c] = 0

            feat_row = feat_row[feat_cols].fillna(0)
            pred = float(model.predict(feat_row)[0])
            pred = max(pred, 0)

            real_val = float(real_future_df.loc[real_future_df[DATE_COL] == d, TARGET_COL].iloc[0])

            result_rows.append({
                DATE_COL: d,
                STATION_COL: station,
                BLOOD_COL: blood,
                "real": real_val,
                "pred": pred
            })

            # 回灌预测值，供下一天使用
            recursive_hist = pd.concat([
                recursive_hist,
                pd.DataFrame([{DATE_COL: d, TARGET_COL: pred}])
            ], ignore_index=True)

    return pd.DataFrame(result_rows)


def build_final_valid_summary(final_pred_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    final_pred_df:
        date, 血站, 血型, real/pred
    value_col:
        "real" 或 "pred"
    """
    tmp = final_pred_df[[DATE_COL, STATION_COL, BLOOD_COL, value_col]].copy()
    tmp = tmp.rename(columns={value_col: TARGET_COL})
    return build_summary_with_all(tmp)


def predict_test_df(model, test_df: pd.DataFrame, feat_cols):
    out = test_df[["label_date", STATION_COL, BLOOD_COL, "label"]].copy()
    out = out.rename(columns={"label_date": DATE_COL, "label": "real"})
    out["pred"] = normalize_pred(model.predict(test_df[feat_cols].fillna(0).values))
    return out[[DATE_COL, STATION_COL, BLOOD_COL, "real", "pred"]].copy()


def build_eval_summary(eval_bottom_df: pd.DataFrame):
    real_summary = build_summary_with_all(eval_bottom_df, value_col="real").rename(columns={TARGET_COL: "real"})
    pred_summary = build_summary_with_all(eval_bottom_df, value_col="pred").rename(columns={TARGET_COL: "pred"})
    out = pd.merge(real_summary, pred_summary, on=[DATE_COL, STATION_COL, BLOOD_COL], how="outer").fillna(0)
    return out.sort_values([DATE_COL, STATION_COL, BLOOD_COL]).reset_index(drop=True)


def plot_final_recursive(final_summary: pd.DataFrame, station="ALL", blood="ALL"):
    one = final_summary[(final_summary[STATION_COL] == station) & (final_summary[BLOOD_COL] == blood)].copy()
    if len(one) == 0:
        print(f"[plot_final_recursive] 无数据: {station}_{blood}")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(one[DATE_COL], one["real"], marker="o", label="真实值")
    plt.plot(one[DATE_COL], one["pred"], marker="o", label="递推预测值")
    _set_cn_title(f"{station}_{blood} 最终验证集递推效果")
    _set_cn_xlabel("日期")
    _set_cn_ylabel("血量")
    _set_cn_legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feat_cols, top_n=20):
    scores = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_})
    scores = scores.sort_values("importance", ascending=False).head(top_n)
    scores = scores.iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(scores["feature"], scores["importance"])
    _set_cn_title(f"XGB特征重要性 Top{top_n}")
    _set_cn_xlabel("importance")
    plt.tight_layout()
    plt.show()


def plot_multi_panels_for_key_groups(final_summary: pd.DataFrame, key_groups: Optional[List[Tuple[str, str]]] = None):
    if key_groups is None:
        key_groups = [("ALL", "ALL"), ("ALL", "A"), ("ALL", "B"), ("ALL", "O"), ("ALL", "AB")]

    n = len(key_groups)
    plt.figure(figsize=(14, 4 * n))
    for i, (station, blood) in enumerate(key_groups, start=1):
        one = final_summary[(final_summary[STATION_COL] == station) & (final_summary[BLOOD_COL] == blood)].copy()
        if len(one) == 0:
            continue
        plt.subplot(n, 1, i)
        plt.plot(one[DATE_COL], one["real"], marker="o", label="真实值")
        plt.plot(one[DATE_COL], one["pred"], marker="o", label="递推预测值")
        if FONT is not None:
            plt.title(f"{station}_{blood}", fontproperties=FONT)
            plt.legend(prop=FONT)
        else:
            plt.title(f"{station}_{blood}")
            plt.legend()
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_overview_recent_days(test_summary: pd.DataFrame, final_summary: pd.DataFrame, station="ALL", blood="ALL",
                              keep_days=180):
    test_one = test_summary[(test_summary[STATION_COL] == station) & (test_summary[BLOOD_COL] == blood)].copy()
    final_one = final_summary[(final_summary[STATION_COL] == station) & (final_summary[BLOOD_COL] == blood)].copy()
    if len(test_one) == 0 and len(final_one) == 0:
        print(f"[plot_overview_recent_days] 无数据: {station}_{blood}")
        return

    all_df = pd.concat([test_one, final_one], ignore_index=True).sort_values(DATE_COL)
    if len(all_df) == 0:
        return
    plot_start = pd.to_datetime(all_df[DATE_COL]).max() - pd.Timedelta(days=keep_days - 1)
    all_df = all_df[all_df[DATE_COL] >= plot_start].copy()
    test_one = test_one[test_one[DATE_COL] >= plot_start].copy()
    final_one = final_one[final_one[DATE_COL] >= plot_start].copy()

    plt.figure(figsize=(15, 5))
    plt.plot(all_df[DATE_COL], all_df["real"], label="真实值")
    if len(test_one) > 0:
        plt.plot(test_one[DATE_COL], test_one["pred"], label="测试集预测")
        plt.axvline(test_one[DATE_COL].min(), linestyle="--", label="测试集开始")
    if len(final_one) > 0:
        plt.plot(final_one[DATE_COL], final_one["pred"], label="最终验证递推预测")
        plt.axvline(final_one[DATE_COL].min(), linestyle="--", label="最终验证开始")

    _set_cn_title(f"{station}_{blood} 最近{keep_days}天总览")
    _set_cn_xlabel("日期")
    _set_cn_ylabel("血量")
    _set_cn_legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main(raw_df: pd.DataFrame, calendar_df: pd.DataFrame, final_valid_days=7, test_days=30):
    # 1. 准备训练样本
    train_all_df = prepare_train_df(raw_df, calendar_df)

    # 2. 切分
    train_df, test_df, final_df, final_valid_start = split_train_test_final(
        train_all_df,
        final_valid_days=final_valid_days,
        test_days=test_days
    )

    # 3. 特征列
    feat_cols = get_feature_cols(train_all_df)

    # 4. 训练全局模型
    xgb, _, test_eval = train_global_xgb(train_df, test_df, feat_cols)

    # 5. 测试集预测（单步）
    test_bottom = predict_test_df(xgb, test_df, feat_cols)
    test_summary = build_eval_summary(test_bottom)

    # 6. 最终验证集（未来n天递推）
    raw_bottom_df = build_bottom_daily_df(raw_df)
    raw_bottom_df = merge_calendar_weather(raw_bottom_df, calendar_df)
    final_bottom = recursive_validate_final_n_days(
        model=xgb,
        raw_bottom_df=raw_bottom_df[[DATE_COL, STATION_COL, BLOOD_COL, TARGET_COL]],
        calendar_df=calendar_df.copy(),
        feat_cols=feat_cols,
        final_valid_days=final_valid_days,
    )
    final_summary = build_eval_summary(final_bottom)


if __name__ == '__main__':
    blood_df = pd.read_csv('../lstm/feature/all_data_merge.csv')
    blood_df["date"] = pd.to_datetime(blood_df["date"])
    weather_df = pd.read_csv("../lstm/feature/blood_calendar_weather_2015_2026.csv")
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    main(blood_df, weather_df, final_valid_days=7, test_days=365)
