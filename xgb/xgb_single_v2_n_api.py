"""
XGBoost 采血量 + 供血量预测 — 模型持久化 + API 服务 (双业务版)

基于 xgb_single_v2_n.py 的算法逻辑，增加:
  1. 训练后将 XGBoost 模型 + 特征列表持久化到 xgb/models/
  2. FastAPI 接口：接受时间范围参数，递推预测并返回 JSON
  3. 采血 (collection) + 供血 (supply) 双业务支持

运行 (CWD 必须是 xgb/):
  cd xgb

  # 训练并保存模型 (旧版 CSV 方式, 向后兼容)
  python xgb_single_v2_n_api.py train

  # 训练采血模型 (从数据库)
  python xgb_single_v2_n_api.py train-collection --station 北京市红十字血液中心

  # 训练供血模型 (从数据库)
  python xgb_single_v2_n_api.py train-supply --scope global

  # 启动 API 服务 (默认 0.0.0.0:8000)
  python xgb_single_v2_n_api.py serve
  python xgb_single_v2_n_api.py serve --port 9000

API:
  GET  /predict?start_date=2025-04-01&end_date=2025-06-30&station=北京市红十字血液中心&blood_type=ALL&date_type=month
       date_type: day=按日(默认) | month=按月 | quarter=按季 | year=按年
  GET  /models
  POST /train

  # 新增端点
  GET  /health
  GET  /data/collection?start_date=...&end_date=...
  GET  /data/supply?scope=global&start_date=...&end_date=...
  POST /train/collection
  POST /train/supply
  GET  /predict/collection
  GET  /predict/supply

额外依赖 (在原有基础上):
  pip install fastapi uvicorn sqlalchemy pymysql
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from feature_pipeline import (
    add_prev_rolling_sums,
    add_dynamic_feature,
    add_calendar_features,
    normalize_pred,
    metric_item,
    load_weather_data,
    PKU_CLASS_STARTS,
)
from db_config import get_engine, check_connection, get_masked_url
from data_source import load_collection_daily, load_supply_daily
from product_category import ALL_CATEGORIES

# 数据列名 (CSV 中的中文列名)
DATE_COL = "date"
STATION_COL = "血站"
BLOOD_COL = "血型"
TARGET_COL = "总血量"

SUPPLY_TARGET_COL = "总供血量"
SUPPLY_TYPE_COL = "供血类型"
SUPPLY_ORG_COL = "发血机构"

# 文件路径: 模型保存目录 / 数据源目录 (支持环境变量覆盖)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR.parent / "lstm" / "feature"

BLOOD_CSV = os.environ.get("BLOOD_CSV", str(DATA_DIR / "remove_group_data.csv"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------- 数据加载 (CSV 回退) ----------


def load_blood_data_csv(path: str = BLOOD_CSV) -> pd.DataFrame:
    """读取采血量 CSV, 返回含 date/血站/血型/总血量 列的 DataFrame."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# 向后兼容别名
load_blood_data = load_blood_data_csv


# ---------- 分组数据准备 ----------


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


def _prepare_supply_group_df(
    supply_df: pd.DataFrame, scope: str, category: str, org: Optional[str] = None
) -> pd.DataFrame:
    """按 scope + category (+ org) 提取供血数据, 返回仅含 date + 总血量 的 DataFrame."""
    if scope == "global":
        grp = supply_df[supply_df[SUPPLY_TYPE_COL] == category].copy()
        grp = grp.groupby(DATE_COL, as_index=False).agg({SUPPLY_TARGET_COL: "sum"})
    else:
        grp = supply_df[
            (supply_df[SUPPLY_ORG_COL] == org)
            & (supply_df[SUPPLY_TYPE_COL] == category)
        ].copy()
        grp = grp.groupby(DATE_COL, as_index=False).agg({SUPPLY_TARGET_COL: "sum"})
    grp = grp.rename(columns={SUPPLY_TARGET_COL: TARGET_COL})
    return grp


# ---------- 分组命名 ----------


def group_name(station: str, blood_type: str) -> str:
    """旧版分组命名 (向后兼容)."""
    return f"{station}_{blood_type}"


def collection_group_name(station: str, blood_type: str) -> str:
    return f"collection__{station}__{blood_type}"


def supply_global_group_name(category: str) -> str:
    return f"supply_global__{category}"


def supply_org_group_name(org: str, category: str) -> str:
    return f"supply_org__{org}__{category}"


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


# ---------- 天气特征加载 ----------


def _load_weather_features(include_pku: bool = True) -> pd.DataFrame:
    """Load weather data with optional PKU season flag."""
    df = load_weather_data()
    if not include_pku:
        df = df.drop(columns=["is_pku_semester_start_season"], errors="ignore")
    return df


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


# ---------- 采血训练 ----------


def train_collection(
    station: str = "北京市红十字血液中心",
    blood_types: Optional[list] = None,
    use_db: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list:
    """训练采血模型 (支持 DB 或 CSV 数据源)."""
    if blood_types is None:
        blood_types = ["ALL", "A", "B", "O", "AB"]

    if use_db:
        engine = get_engine()
        all_blood_df = load_collection_daily(
            engine, station=station, start_date=start_date, end_date=end_date
        )
    else:
        all_blood_df = load_blood_data_csv()

    weather_df = _load_weather_features(include_pku=True)

    results = []
    for bt in blood_types:
        grp = collection_group_name(station, bt)
        grp_df = _prepare_group_df(all_blood_df, station, bt)
        if grp_df.empty:
            log.warning("[%s] 无数据, 跳过", grp)
            continue
        metrics = train_one_group(grp, grp_df, weather_df)
        results.append(metrics)

    log.info("Collection training done, %d models saved to %s", len(results), MODEL_DIR)
    return results


# ---------- 供血训练 ----------


def train_supply(
    scope: str = "global",
    categories: Optional[list] = None,
    orgs: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list:
    """训练供血模型 (从数据库)."""
    if categories is None:
        categories = ALL_CATEGORIES

    engine = get_engine()
    weather_df = _load_weather_features(include_pku=False)

    results = []
    if scope == "global":
        supply_df = load_supply_daily(
            engine, scope="global", start_date=start_date, end_date=end_date
        )
        for cat in categories:
            grp = supply_global_group_name(cat)
            grp_df = _prepare_supply_group_df(supply_df, "global", cat)
            if grp_df.empty:
                log.warning("[%s] 无数据, 跳过", grp)
                continue
            metrics = train_one_group(grp, grp_df, weather_df)
            results.append(metrics)
    else:
        if not orgs:
            log.error("scope='org' requires orgs parameter")
            return []
        for org_name in orgs:
            supply_df = load_supply_daily(
                engine,
                scope="org",
                org=org_name,
                start_date=start_date,
                end_date=end_date,
            )
            for cat in categories:
                grp = supply_org_group_name(org_name, cat)
                grp_df = _prepare_supply_group_df(supply_df, "org", cat, org=org_name)
                if grp_df.empty:
                    log.warning("[%s] 无数据, 跳过", grp)
                    continue
                metrics = train_one_group(grp, grp_df, weather_df)
                results.append(metrics)

    log.info("Supply training done (scope=%s), %d models saved", scope, len(results))
    return results


# ---------- 向后兼容: train_all ----------


def train_all(
    station: str = "北京市红十字血液中心", blood_types: Optional[list] = None
) -> list:
    """训练所有 group 并保存模型 (CSV 数据源, 向后兼容)."""
    return train_collection(station=station, blood_types=blood_types, use_db=False)


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

    all_blood_df = load_blood_data_csv()
    weather_df = _load_weather_features(include_pku=True)
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


def predict_supply_range(
    scope: str, category: str, start_date: str, end_date: str, org: Optional[str] = None
) -> list:
    """
    对指定供血类型在 [start_date, end_date] 范围内做逐日递推预测.
    返回 [{"date": "2025-04-01", "predicted_volume": 42}, ...] 格式的 list.
    """
    if scope == "global":
        grp = supply_global_group_name(category)
    else:
        if not org:
            raise ValueError("scope='org' requires org parameter")
        grp = supply_org_group_name(org, category)

    model, feat_cols = load_group_model(grp)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
        raise ValueError(f"end_date ({end_date}) < start_date ({start_date})")

    engine = get_engine()
    supply_df = load_supply_daily(engine, scope=scope, org=org, end_date=start_date)
    grp_df = _prepare_supply_group_df(supply_df, scope, category, org=org)

    hist_df = grp_df[grp_df[DATE_COL] < start_dt].copy()
    if hist_df.empty:
        raise ValueError(f"无 {start_date} 之前的历史数据")

    pred_dates = pd.date_range(start=start_dt, end=end_dt, freq="D")
    future_df = pd.DataFrame({DATE_COL: pred_dates, TARGET_COL: 0.0})

    full_df = pd.concat([hist_df, future_df], ignore_index=True).sort_values(DATE_COL)
    full_df = add_prev_rolling_sums(full_df, windows=(1, 3, 7, 14, 28))
    full_df = add_dynamic_feature(full_df)
    full_df["label_tplus1"] = full_df[TARGET_COL]

    weather_df = _load_weather_features(include_pku=False)
    full_df = pd.merge(full_df, weather_df, on="date", how="left")

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


# ---------- FastAPI 应用 ----------


def create_app():
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="血液预测 API",
        description="基于 XGBoost 的血液采集量 + 供血量预测服务 (双业务版)",
        version="2.0.0",
    )

    # ---- 健康检查 ----

    @app.get("/health")
    def api_health():
        db_ok = check_connection()
        models = list_saved_models()
        return {
            "db_connected": db_ok,
            "db_url": get_masked_url(),
            "model_count": len(models),
        }

    # ---- 数据查询 ----

    @app.get("/data/collection")
    def api_data_collection(
        start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        station: str = Query(None),
        blood_type: str = Query(None),
    ):
        engine = get_engine()
        df = load_collection_daily(
            engine, station=station, start_date=start_date, end_date=end_date
        )
        if blood_type and blood_type != "ALL":
            df = df[df["血型"] == blood_type.upper()]
        return {"count": len(df), "data": df.to_dict(orient="records")}

    @app.get("/data/supply")
    def api_data_supply(
        scope: str = Query("global"),
        start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        category: str = Query(None),
        org: str = Query(None),
    ):
        engine = get_engine()
        df = load_supply_daily(
            engine, scope=scope, org=org, start_date=start_date, end_date=end_date
        )
        if category:
            df = df[df["供血类型"] == category]
        return {"count": len(df), "data": df.to_dict(orient="records")}

    # ---- 采血训练 + 预测 ----

    @app.post("/train/collection")
    def api_train_collection(
        station: str = Query("北京市红十字血液中心"),
        blood_types: str = Query("ALL,A,B,O,AB"),
        start_date: str = Query(None),
        end_date: str = Query(None),
    ):
        bt_list = [x.strip() for x in blood_types.split(",") if x.strip()]
        results = train_collection(
            station=station,
            blood_types=bt_list,
            use_db=True,
            start_date=start_date,
            end_date=end_date,
        )
        return {"trained": len(results), "results": results}

    @app.get("/predict/collection")
    def api_predict_collection(
        start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        station: str = Query("北京市红十字血液中心"),
        blood_type: str = Query("ALL"),
        date_type: str = Query("day"),
    ):
        grp = collection_group_name(station, blood_type)
        if grp not in list_saved_models():
            # fallback to legacy name
            grp = group_name(station, blood_type)
            if grp not in list_saved_models():
                raise HTTPException(404, f"模型不存在: {grp}")
        try:
            daily_preds = predict_date_range(station, blood_type, start_date, end_date)
        except ValueError as e:
            raise HTTPException(400, str(e))
        preds = aggregate_predictions(daily_preds, date_type)
        return {
            "business": "collection",
            "group": grp,
            "date_type": date_type,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(preds),
            "predictions": preds,
        }

    # ---- 供血训练 + 预测 ----

    @app.post("/train/supply")
    def api_train_supply(
        scope: str = Query("global"),
        categories: str = Query("红细胞类,血小板类,血浆类"),
        orgs: str = Query(None),
        start_date: str = Query(None),
        end_date: str = Query(None),
    ):
        cat_list = [x.strip() for x in categories.split(",") if x.strip()]
        org_list = [x.strip() for x in orgs.split(",") if x.strip()] if orgs else None
        results = train_supply(
            scope=scope,
            categories=cat_list,
            orgs=org_list,
            start_date=start_date,
            end_date=end_date,
        )
        return {"trained": len(results), "results": results}

    @app.get("/predict/supply")
    def api_predict_supply(
        scope: str = Query("global"),
        categories: str = Query(
            ..., description="逗号分隔供血类型, 如 红细胞类,血小板类,血浆类"
        ),
        org: str = Query(None),
        start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        date_type: str = Query(
            "day", description="聚合粒度: day=日, month=月, quarter=季, year=年"
        ),
    ):
        if date_type not in ("day", "month", "quarter", "year"):
            raise HTTPException(
                400, f"date_type 不合法: {date_type}，可选 day/month/quarter/year"
            )
        if scope == "org" and not org:
            raise HTTPException(400, "scope='org' requires org parameter")
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        if not cat_list:
            raise HTTPException(400, "categories 不能为空")

        saved = list_saved_models()
        results_by_category = []

        for cat in cat_list:
            if scope == "global":
                grp = supply_global_group_name(cat)
            else:
                grp = supply_org_group_name(org, cat)

            if grp not in saved:
                results_by_category.append(
                    {
                        "category": cat,
                        "group": grp,
                        "error": f"模型不存在: {grp}，请先训练",
                        "predictions": [],
                    }
                )
                continue

            try:
                daily_preds = predict_supply_range(
                    scope, cat, start_date, end_date, org=org
                )
                preds = aggregate_predictions(daily_preds, date_type)
            except ValueError as e:
                results_by_category.append(
                    {
                        "category": cat,
                        "group": grp,
                        "error": str(e),
                        "predictions": [],
                    }
                )
                continue

            results_by_category.append(
                {
                    "category": cat,
                    "group": grp,
                    "count": len(preds),
                    "predictions": preds,
                }
            )

        return {
            "business": "supply",
            "scope": scope,
            "org": org,
            "date_type": date_type,
            "start_date": start_date,
            "end_date": end_date,
            "categories": results_by_category,
        }

    # ---- 模型列表 (增强: 按业务分组) ----

    @app.get("/models")
    def api_models():
        models = list_saved_models()
        grouped = {
            "collection": [],
            "supply_global": [],
            "supply_org": [],
            "legacy": [],
        }
        for m in models:
            if m.startswith("collection__"):
                grouped["collection"].append(m)
            elif m.startswith("supply_global__"):
                grouped["supply_global"].append(m)
            elif m.startswith("supply_org__"):
                grouped["supply_org"].append(m)
            else:
                grouped["legacy"].append(m)
        return {"count": len(models), "models": grouped}

    # ---- 旧版端点 (向后兼容, 标记 Deprecation) ----

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
        response = JSONResponse(
            content={
                "station": station,
                "blood_type": blood_type,
                "date_type": date_type,
                "start_date": start_date,
                "end_date": end_date,
                "count": len(preds),
                "predictions": preds,
            }
        )
        response.headers["Deprecation"] = "true"
        return response

    @app.post("/train")
    def api_train(
        station: str = Query("北京市红十字血液中心"),
        blood_types: str = Query("ALL,A,B,O,AB", description="逗号分隔血型列表"),
    ):
        bt_list = [x.strip() for x in blood_types.split(",") if x.strip()]
        results = train_all(station=station, blood_types=bt_list)
        response = JSONResponse(content={"trained": len(results), "results": results})
        response.headers["Deprecation"] = "true"
        return response

    return app


# ---------- CLI ----------


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost 血液预测: 训练模型 / 启动 API 服务 (双业务版)"
    )
    sub = parser.add_subparsers(dest="mode")

    # 旧版训练 (CSV, 向后兼容)
    train_p = sub.add_parser("train", help="训练全部采血模型并保存 (CSV 数据源)")
    train_p.add_argument("--station", default="北京市红十字血液中心")
    train_p.add_argument(
        "--blood_types", default="ALL,A,B,O,AB", help="逗号分隔血型列表"
    )

    # 采血训练 (DB)
    train_c = sub.add_parser("train-collection", help="Train collection models from DB")
    train_c.add_argument("--station", default="北京市红十字血液中心")
    train_c.add_argument("--blood_types", default="ALL,A,B,O,AB")
    train_c.add_argument("--start_date", default=None)
    train_c.add_argument("--end_date", default=None)

    # 供血训练 (DB)
    train_s = sub.add_parser("train-supply", help="Train supply models from DB")
    train_s.add_argument("--scope", default="global", choices=["global", "org"])
    train_s.add_argument("--categories", default="红细胞类,血小板类,血浆类")
    train_s.add_argument("--orgs", default=None)
    train_s.add_argument("--start_date", default=None)
    train_s.add_argument("--end_date", default=None)

    # API 服务
    serve_p = sub.add_parser("serve", help="启动 FastAPI 预测服务")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.mode == "train":
        bt_list = [x.strip() for x in args.blood_types.split(",") if x.strip()]
        results = train_all(station=args.station, blood_types=bt_list)
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    elif args.mode == "train-collection":
        bt_list = [x.strip() for x in args.blood_types.split(",") if x.strip()]
        results = train_collection(
            station=args.station,
            blood_types=bt_list,
            use_db=True,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    elif args.mode == "train-supply":
        cat_list = [x.strip() for x in args.categories.split(",") if x.strip()]
        org_list = (
            [x.strip() for x in args.orgs.split(",") if x.strip()]
            if args.orgs
            else None
        )
        results = train_supply(
            scope=args.scope,
            categories=cat_list,
            orgs=org_list,
            start_date=args.start_date,
            end_date=args.end_date,
        )
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
