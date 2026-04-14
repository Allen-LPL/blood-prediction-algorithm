"""
XGBoost 采血量预测 — 模型持久化 + API 服务

基于 xgb_single_v2_n.py 的算法逻辑，增加:
  1. 训练后将 XGBoost 模型 + 特征列表持久化到 xgb/models/
  2. FastAPI 接口：接受时间范围参数，递推预测并返回 JSON

运行 (CWD 必须是 xgb/):
  cd xgb

  # 训练并保存模型
  python xgb_single_v2_n_api.py train

  # 启动 API 服务 (默认 0.0.0.0:8000)
  python xgb_single_v2_n_api.py serve
  python xgb_single_v2_n_api.py serve --port 9000

API:
  GET  /predict?start_date=2025-04-01&end_date=2025-06-30&station=北京市红十字血液中心&blood_type=ALL&date_type=month
       date_type: day=按日(默认) | month=按月 | quarter=按季 | year=按年
  GET  /models
  POST /train

额外依赖 (在原有基础上):
  pip install fastapi uvicorn
"""

import argparse
import json
import logging
import math
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from chinese_calendar import get_holiday_detail, Holiday
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from xgboost import XGBRegressor

# 数据列名 (CSV 中的中文列名)
DATE_COL = "date"
STATION_COL = "血站"
BLOOD_COL = "血型"
TARGET_COL = "总血量"

# 文件路径: 模型保存目录 / 数据源目录 (支持环境变量覆盖)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


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


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = add_cn_holiday_flags(df, date_col)
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
    """添加 lag / rolling_mean / rolling_std / ewm 等动态特征."""
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
    g2["diff7"] = g2[TARGET_COL] - g2["lag7"]
    g2["diff14"] = g2[TARGET_COL] - g2["lag14"]
    g2["rolling14"] = g2[TARGET_COL].rolling(14).mean()
    g2["rolling28"] = g2[TARGET_COL].rolling(28).mean()
    g2["ewm7"] = g2[TARGET_COL].ewm(span=7, adjust=False).mean()
    g2["ewm14"] = g2[TARGET_COL].ewm(span=14, adjust=False).mean()
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


def load_blood_data(path: str = BLOOD_CSV) -> pd.DataFrame:
    """读取采血量 CSV, 返回含 date/血站/血型/总血量 列的 DataFrame."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


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


def _prepare_group_df(
    all_blood_df: pd.DataFrame, station: str, blood_type: str
) -> pd.DataFrame:
    """按 station + blood_type 提取并聚合数据, 返回仅含 date + 总血量 的 DataFrame."""
    if blood_type == "ALL":
        grp = all_blood_df[all_blood_df[STATION_COL] == station].copy()
        grp = grp.groupby(DATE_COL, as_index=False).agg({TARGET_COL: "sum"})
    else:
        grp = all_blood_df[
            (all_blood_df[STATION_COL] == station)
            & (all_blood_df[BLOOD_COL] == blood_type)
        ].copy()
        grp.drop(columns=[BLOOD_COL, STATION_COL], errors="ignore", inplace=True)
    return grp


def group_name(station: str, blood_type: str) -> str:
    return f"{station}_{blood_type}"


# ---------- 模型持久化: 保存/加载 XGBoost 模型 + 特征列 ----------


def _group_dir(group: str) -> Path:
    return MODEL_DIR / group


def save_group_model(group: str, model: XGBRegressor, feat_cols: list) -> Path:
    """保存模型 + 特征列到 xgb/models/<group>/"""
    gdir = _group_dir(group)
    gdir.mkdir(parents=True, exist_ok=True)
    model_path = gdir / "model.json"
    model.save_model(str(model_path))
    with open(gdir / "feat_cols.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False)
    log.info("模型已保存: %s", gdir)
    return gdir


def load_group_model(group: str):
    """加载模型 + 特征列. 返回 (model, feat_cols)."""
    gdir = _group_dir(group)
    model_path = gdir / "model.json"
    cols_path = gdir / "feat_cols.json"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    model = XGBRegressor()
    model.load_model(str(model_path))
    with open(cols_path, "r", encoding="utf-8") as f:
        feat_cols = json.load(f)
    return model, feat_cols


def list_saved_models() -> list:
    """列出所有已保存的模型 group 名."""
    if not MODEL_DIR.exists():
        return []
    return sorted(
        [
            d.name
            for d in MODEL_DIR.iterdir()
            if d.is_dir() and (d / "model.json").exists()
        ]
    )


# ---------- 训练: 按 (血站, 血型) 分组训练 XGBoost ----------


def train_one_group(
    group: str, group_blood_df: pd.DataFrame, weather_df: pd.DataFrame
) -> dict:
    """训练一个 station+blood_type 组合, 保存模型, 返回评估指标."""
    label_col = "label_tplus1"
    final_valid_days = 14
    test_ratio = 0.1

    group_blood_df = add_prev_rolling_sums(group_blood_df, windows=(1, 3, 7, 14, 28))
    group_blood_df = add_dynamic_feature(group_blood_df)
    group_blood_df[label_col] = group_blood_df[TARGET_COL]
    group_blood_df = group_blood_df.dropna(subset=[label_col])

    df = pd.merge(group_blood_df, weather_df, on="date", how="left")
    feat_cols = [
        col
        for col in df.columns
        if col not in (TARGET_COL, label_col, DATE_COL, "date_x", "date_y", "lag1")
    ]

    final_valid_start = df[DATE_COL].max() - pd.Timedelta(days=final_valid_days - 1)
    df_final_valid = df[df[DATE_COL] >= final_valid_start].copy()
    df_train_test = df[df[DATE_COL] < final_valid_start].copy()
    test_size = max(1, int(len(df_train_test) * test_ratio))
    df_train = df_train_test.iloc[:-test_size].copy()
    df_test = df_train_test.iloc[-test_size:].copy()

    log.info(
        "[%s] train=%d, test=%d, final_valid=%d (%s ~ %s)",
        group,
        len(df_train),
        len(df_test),
        len(df_final_valid),
        df_final_valid[DATE_COL].min().date(),
        df_final_valid[DATE_COL].max().date(),
    )

    X_train = df_train[feat_cols].values
    y_train = df_train[label_col].values
    X_test = df_test[feat_cols].values
    y_test = df_test[label_col].values

    sample_weight = []
    for y in y_train:
        if y >= 45:
            w = 5
        elif y >= 36:
            w = 5
        elif y >= 26:
            w = 6
        elif y >= 18:
            w = 1
        else:
            w = 3
        sample_weight.append(w)

    params = dict(
        learning_rate=0.4,
        n_estimators=100,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=1,
        seed=1024,
        max_depth=8,
        eval_metric=mean_squared_error,
        early_stopping_rounds=10,
        missing=0,
    )
    model = XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        sample_weight=sample_weight,
        verbose=False,
    )

    pred_test = normalize_pred(model.predict(X_test))
    test_metrics = metric_item(y_test, pred_test)
    log.info(
        "[%s] 测试集 RMSE=%.2f, 平均误差率=%.4f",
        group,
        test_metrics["rmse"],
        test_metrics["mean_error_rate"],
    )

    new_df = df.copy()
    for cur_date in df_final_valid[DATE_COL].unique():
        cur_row = new_df[new_df[DATE_COL] == cur_date]
        pred_val = normalize_pred(model.predict(cur_row[feat_cols].values))[0]
        new_df.loc[new_df[DATE_COL] == cur_date, TARGET_COL] = pred_val
        new_df = add_prev_rolling_sums(new_df, windows=(1, 3, 7, 14, 28))
        new_df = add_dynamic_feature(new_df)

    save_group_model(group, model, feat_cols)

    return {
        "group": group,
        "train_size": len(df_train),
        "test_size": len(df_test),
        "test_metrics": test_metrics,
    }


def train_all(station: str = "北京市红十字血液中心", blood_types: list = None) -> list:
    """训练所有 group 并保存模型, 返回各组评估指标."""
    if blood_types is None:
        blood_types = ["ALL", "A", "B", "O", "AB"]

    all_blood_df = load_blood_data()
    weather_df = load_weather_data()

    results = []
    for bt in blood_types:
        grp = group_name(station, bt)
        grp_df = _prepare_group_df(all_blood_df, station, bt)
        if grp_df.empty:
            log.warning("[%s] 无数据, 跳过", grp)
            continue
        metrics = train_one_group(grp, grp_df, weather_df)
        results.append(metrics)

    log.info("训练完成, 共 %d 个模型已保存到 %s", len(results), MODEL_DIR)
    return results


# ---------- 预测: 逐日递推 + 按粒度聚合 ----------


def predict_date_range(
    station: str, blood_type: str, start_date: str, end_date: str
) -> list:
    """
    对指定 station + blood_type 在 [start_date, end_date] 范围内做逐日递推预测.
    返回 [{"date": "2025-04-01", "predicted_volume": 42}, ...] 格式的 list.
    """
    grp = group_name(station, blood_type)
    model, feat_cols = load_group_model(grp)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
        raise ValueError(f"end_date ({end_date}) 不能早于 start_date ({start_date})")

    all_blood_df = load_blood_data()
    weather_df = load_weather_data()
    grp_df = _prepare_group_df(all_blood_df, station, blood_type)

    hist_df = grp_df[grp_df[DATE_COL] < start_dt].copy()
    if hist_df.empty:
        raise ValueError(f"无 {start_date} 之前的历史数据")

    pred_dates = pd.date_range(start=start_dt, end=end_dt, freq="D")
    future_df = pd.DataFrame({DATE_COL: pred_dates, TARGET_COL: 0.0})

    full_df = pd.concat([hist_df, future_df], ignore_index=True).sort_values(DATE_COL)
    full_df = add_prev_rolling_sums(full_df, windows=(1, 3, 7, 14, 28))
    full_df = add_dynamic_feature(full_df)
    full_df["label_tplus1"] = full_df[TARGET_COL]

    full_df = pd.merge(full_df, weather_df, on="date", how="left")

    # 补齐训练时的特征列 (天气 one-hot 可能不完全一致)
    for col in feat_cols:
        if col not in full_df.columns:
            full_df[col] = 0

    predictions = []
    for cur_date in pred_dates:
        cur_row = full_df[full_df[DATE_COL] == cur_date]
        if cur_row.empty:
            continue

        x = cur_row[feat_cols].values
        pred_val = int(normalize_pred(model.predict(x))[0])
        predictions.append({"date": str(cur_date.date()), "predicted_volume": pred_val})

        full_df.loc[full_df[DATE_COL] == cur_date, TARGET_COL] = pred_val
        full_df = add_prev_rolling_sums(full_df, windows=(1, 3, 7, 14, 28))
        full_df = add_dynamic_feature(full_df)

    return predictions


def aggregate_predictions(daily_preds: list, date_type: str) -> list:
    """将逐日预测结果按 date_type 聚合: day=原样, month=按月, quarter=按季, year=按年."""
    if date_type == "day" or not daily_preds:
        return daily_preds

    df = pd.DataFrame(daily_preds)
    df["date"] = pd.to_datetime(df["date"])

    period_map = {"month": "M", "quarter": "Q", "year": "Y"}
    freq = period_map.get(date_type)
    if freq is None:
        return daily_preds

    df["period"] = df["date"].dt.to_period(freq).astype(str)
    agg = (
        df.groupby("period", sort=True)
        .agg(
            predicted_volume=("predicted_volume", "sum"),
            days=("predicted_volume", "count"),
        )
        .reset_index()
    )
    return agg.to_dict(orient="records")


def create_app():
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="采血量预测 API",
        description="基于 XGBoost 的血液采集量预测服务",
        version="1.0.0",
    )

    @app.get("/predict")
    def api_predict(
        start_date: str = Query(
            ..., description="开始日期, 如 2025-04-01", pattern=r"^\d{4}-\d{2}-\d{2}$"
        ),
        end_date: str = Query(
            ..., description="结束日期, 如 2025-04-14", pattern=r"^\d{4}-\d{2}-\d{2}$"
        ),
        station: str = Query("北京市红十字血液中心", description="血站名称"),
        blood_type: str = Query("ALL", description="血型: ALL, A, B, O, AB"),
        date_type: str = Query(
            "day", description="聚合粒度: day=日, month=月, quarter=季, year=年"
        ),
    ):
        if date_type not in ("day", "month", "quarter", "year"):
            raise HTTPException(
                400, f"date_type 不合法: {date_type}，可选 day/month/quarter/year"
            )
        grp = group_name(station, blood_type)
        if grp not in list_saved_models():
            raise HTTPException(404, f"模型不存在: {grp}，请先调用 POST /train 训练")
        try:
            daily_preds = predict_date_range(station, blood_type, start_date, end_date)
        except ValueError as e:
            raise HTTPException(400, str(e))
        preds = aggregate_predictions(daily_preds, date_type)
        return {
            "station": station,
            "blood_type": blood_type,
            "date_type": date_type,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(preds),
            "predictions": preds,
        }

    @app.get("/models")
    def api_models():
        models = list_saved_models()
        return {"count": len(models), "models": models}

    @app.post("/train")
    def api_train(
        station: str = Query("北京市红十字血液中心"),
        blood_types: str = Query("ALL,A,B,O,AB", description="逗号分隔血型列表"),
    ):
        bt_list = [x.strip() for x in blood_types.split(",") if x.strip()]
        results = train_all(station=station, blood_types=bt_list)
        return {"trained": len(results), "results": results}

    return app


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost 采血量预测: 训练模型 / 启动 API 服务"
    )
    sub = parser.add_subparsers(dest="mode")

    train_p = sub.add_parser("train", help="训练全部模型并保存到 xgb/models/")
    train_p.add_argument("--station", default="北京市红十字血液中心")
    train_p.add_argument(
        "--blood_types", default="ALL,A,B,O,AB", help="逗号分隔血型列表"
    )

    serve_p = sub.add_parser("serve", help="启动 FastAPI 预测服务")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.mode == "train":
        bt_list = [x.strip() for x in args.blood_types.split(",") if x.strip()]
        results = train_all(station=args.station, blood_types=bt_list)
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    elif args.mode == "serve":
        import uvicorn

        app = create_app()
        log.info("启动 API 服务: http://%s:%d/docs", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
