"""
XGBoost + LSTM-残差 采血/供血预测 — 模型持久化 + API 服务

特性:
  1. 训练: XGBoost → 5 折 OOF 残差 → BiLSTM 残差校正
     输出 y_hat_adaptive = y_xgb + clip(resid_pred, ±0.5·|y_xgb|)
  2. 数据源统一为 MySQL (data_source.py + db_config.py)
  3. 模型工件目录: xgb/models/{group}/
       model.json + feat_cols.json (XGB)
       lstm.keras + feature_scaler.joblib + resid_scaler.joblib + lstm_meta.json (LSTM)

运行 (CWD 必须是 xgb/):
  cd xgb

  # 训练采血模型
  python xgb_single_v2_n_api.py train-collection --station 北京市红十字血液中心

  # 训练供血模型
  python xgb_single_v2_n_api.py train-supply --scope global

  # 启动 API 服务 (默认 0.0.0.0:8000)
  python xgb_single_v2_n_api.py serve --port 9000

API:
  GET  /health
  GET  /data/collection?start_date=...&end_date=...
  GET  /data/supply?scope=global&start_date=...&end_date=...
  POST /train/collection
  POST /train/supply
  GET  /predict/collection?start_date=...&end_date=...&station=...&blood_types=ALL&date_type=DAY
  GET  /predict/supply?scope=global&categories=红细胞类&start_date=...&end_date=...
  GET  /models

依赖:
  pip install fastapi uvicorn sqlalchemy pymysql tensorflow joblib
"""

import argparse
import json
import logging
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
from residual_lstm import (
    LSTM_AUX_COLS_COLLECTION,
    LSTM_AUX_COLS_SUPPLY,
    LOOKBACK,
    RESID_OUTPUT_CLIP_FRAC,
    TF_AVAILABLE,
    build_lstm_feature_frame,
    clip_resid_pred,
    clip_train_residuals,
    compute_oof_y_xgb,
    load_lstm_artifacts,
    lstm_meta_payload,
    predict_residual_for_window,
    save_lstm_artifacts,
    train_residual_lstm,
)

# 数据列名 (CSV 中的中文列名)
DATE_COL = "date"
STATION_COL = "血站"
BLOOD_COL = "血型"
TARGET_COL = "总血量"

SUPPLY_TARGET_COL = "总供血量"
SUPPLY_TYPE_COL = "供血类型"
SUPPLY_ORG_COL = "发血机构"

# 文件路径: 模型保存目录
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


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


# ---------- 天气特征加载 (含模块级缓存) ----------

_WEATHER_CACHE: Optional[pd.DataFrame] = None


def _load_weather_features(include_pku: bool = True) -> pd.DataFrame:
    """Load weather data with optional PKU season flag.

    模块级缓存避免每次 API 请求都重读 CSV + 重跑 chinese_calendar 逐行 apply
    (后者是慢点之一). 因为天气/日历数据在进程生命周期内不变, 缓存安全.
    """
    global _WEATHER_CACHE
    if _WEATHER_CACHE is None:
        _WEATHER_CACHE = load_weather_data()
        log.info("天气特征已缓存 (%d 行)", len(_WEATHER_CACHE))
    df = _WEATHER_CACHE
    if not include_pku:
        df = df.drop(columns=["is_pku_semester_start_season"], errors="ignore")
    return df


# ---------- 连续段感知的特征工程 ----------


def _segment_aware_features(df: pd.DataFrame) -> pd.DataFrame:
    """按连续日期段独立做 rolling/dynamic 特征,避免在跨段(如疫情年被剔除)时
    把零填充泄露到 rolling 统计里.

    检测 date diff > 1 即视为新段;每段独立调 add_prev_rolling_sums + add_dynamic_feature
    后再拼回.
    """
    if df.empty:
        return df
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if len(df) == 1:
        df = add_prev_rolling_sums(df, windows=(1, 3, 7, 14, 28))
        return add_dynamic_feature(df)

    gap = df[DATE_COL].diff().dt.days.fillna(1)
    seg_id = (gap > 1).cumsum()
    out = []
    for _, seg_df in df.groupby(seg_id, sort=False, group_keys=False):
        seg_df = add_prev_rolling_sums(seg_df.copy(), windows=(1, 3, 7, 14, 28))
        seg_df = add_dynamic_feature(seg_df)
        out.append(seg_df)
    return pd.concat(out, ignore_index=True)


def _filter_skip_years(df: pd.DataFrame, skip_years: Optional[list]) -> pd.DataFrame:
    """从 df 中删除年份在 skip_years 中的行 (用于训练时排除疫情封控期等异常时段)."""
    if not skip_years:
        return df
    skip_set = set(int(y) for y in skip_years)
    mask = ~pd.to_datetime(df[DATE_COL]).dt.year.isin(skip_set)
    return df.loc[mask].copy()


# ---------- 训练: 按 (血站, 血型) 分组训练 XGBoost ----------


def train_one_group(
    group: str,
    group_blood_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    *,
    business: str,
    include_pku: bool,
    train_lstm: bool = True,
    skip_years: Optional[list] = None,
) -> dict:
    """训练一个 group: XGBoost → 5 折 OOF 残差 → BiLSTM 残差校正,保存所有工件.

    skip_years: 训练时跳过的年份列表 (如 [2019, 2020, 2021] 排除疫情封控).
    跨年的滚动统计会按连续段独立计算,不会把跳过年份当作 0 填进 rolling.
    """
    label_col = "label_tplus1"
    test_ratio = 0.1

    if skip_years:
        before = len(group_blood_df)
        group_blood_df = _filter_skip_years(group_blood_df, skip_years)
        log.info(
            "[%s] skip_years=%s, %d → %d 行",
            group, skip_years, before, len(group_blood_df),
        )

    group_blood_df = _segment_aware_features(group_blood_df)
    group_blood_df[label_col] = group_blood_df[TARGET_COL]
    group_blood_df = group_blood_df.dropna(subset=[label_col])

    df = pd.merge(group_blood_df, weather_df, on="date", how="left")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # LSTM 预期的 dayofweek 列;天气 CSV 提供的是 "weekday".
    if "dayofweek" not in df.columns and "weekday" in df.columns:
        df["dayofweek"] = df["weekday"]

    feat_cols = [
        col
        for col in df.columns
        if col not in (TARGET_COL, label_col, DATE_COL, "date_x", "date_y", "lag1")
    ]

    test_size = max(1, int(len(df) * test_ratio))
    df_train = df.iloc[:-test_size].copy()
    df_test = df.iloc[-test_size:].copy()
    cutoff_date = df_test[DATE_COL].min()

    log.info(
        "[%s] train=%d, test=%d, cutoff=%s",
        group,
        len(df_train),
        len(df_test),
        cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date,
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
    sample_weight = np.asarray(sample_weight)

    params = dict(
        learning_rate=0.05,       # 0.4 → 0.05
        n_estimators=200,         # 100 → 200
        max_depth=4,              # 8 → 4
        min_child_weight=3,       # 1 → 3
        subsample=0.8,            # 0.9 → 0.8
        colsample_bytree=0.8,     # 0.9 → 0.8
        scale_pos_weight=1,
        seed=1024,
        eval_metric=mean_squared_error,
        early_stopping_rounds=20, # 10 → 20
        missing=0,
    )
    # params = dict(
    #     learning_rate=0.4,
    #     n_estimators=100,
    #     min_child_weight=1,
    #     subsample=0.9,
    #     colsample_bytree=0.9,
    #     scale_pos_weight=1,
    #     seed=1024,
    #     max_depth=8,
    #     eval_metric=mean_squared_error,
    #     early_stopping_rounds=10,
    #     missing=0,
    # )

    # 5 折 OOF 训练段 + final XGB 测试段
    oof_train, pred_test_raw, xgb_final = compute_oof_y_xgb(
        X_train, y_train, sample_weight, X_test, y_test, params, n_splits=5
    )
    pred_test_norm = normalize_pred(pred_test_raw)
    test_metrics_xgb = metric_item(y_test, pred_test_norm)
    log.info(
        "[%s] XGB 测试集 RMSE=%.2f, 平均误差率=%.4f",
        group,
        test_metrics_xgb["rmse"],
        test_metrics_xgb["mean_error_rate"],
    )

    # 全序列 y_xgb (训练段 OOF + 测试段 final)
    y_xgb_full = np.concatenate(
        [normalize_pred(oof_train), normalize_pred(pred_test_raw)]
    ).astype(np.float64)
    df["y_xgb"] = y_xgb_full
    df["resid"] = (df[TARGET_COL].astype(np.float64) - df["y_xgb"]).astype(np.float32)

    # 训练段残差按 [1, 99] 百分位裁剪
    train_mask = (df[DATE_COL] < cutoff_date).values
    test_mask = ~train_mask
    train_resid_clipped, lo, hi = clip_train_residuals(
        df.loc[train_mask, "resid"].to_numpy()
    )
    df.loc[train_mask, "resid"] = train_resid_clipped

    save_group_model(group, xgb_final, feat_cols)

    result = {
        "group": group,
        "business": business,
        "train_size": len(df_train),
        "test_size": len(df_test),
        "test_metrics_xgb": test_metrics_xgb,
        "lstm_trained": False,
    }

    # LSTM 残差训练
    if not train_lstm:
        log.info("[%s] train_lstm=False,跳过 LSTM 残差训练", group)
        return result
    if not TF_AVAILABLE:
        log.warning("[%s] tensorflow 不可用,仅保存 XGB", group)
        return result

    aux_cols = (
        LSTM_AUX_COLS_COLLECTION if business == "collection" else LSTM_AUX_COLS_SUPPLY
    )
    feature_df, final_lstm_cols = build_lstm_feature_frame(df, aux_cols)
    resid_target = df["resid"].to_numpy(dtype=np.float32)

    lstm_artifacts = train_residual_lstm(
        feature_df,
        resid_target,
        train_mask,
        test_mask,
        lookback=LOOKBACK,
    )
    if lstm_artifacts is None:
        log.warning("[%s] LSTM 训练被跳过 (数据不足),仅保存 XGB", group)
        return result

    # 测试段评估: y_hat_adaptive = y_xgb + clip(resid_pred)
    test_indices = lstm_artifacts["test_idx"]
    if len(test_indices) > 0:
        y_xgb_eval = df.loc[test_indices, "y_xgb"].to_numpy()
        y_true_eval = df.loc[test_indices, TARGET_COL].to_numpy()
        resid_pred_eval = lstm_artifacts["resid_pred_test"][test_indices]
        resid_pred_clipped = clip_resid_pred(
            resid_pred_eval, y_xgb_eval, frac=RESID_OUTPUT_CLIP_FRAC
        )
        y_hat_adaptive_eval = normalize_pred(y_xgb_eval + resid_pred_clipped)
        test_metrics_adaptive = metric_item(y_true_eval, y_hat_adaptive_eval)
        log.info(
            "[%s] LSTM-adaptive 测试集 RMSE=%.2f (XGB %.2f), 平均误差率=%.4f (XGB %.4f)",
            group,
            test_metrics_adaptive["rmse"],
            test_metrics_xgb["rmse"],
            test_metrics_adaptive["mean_error_rate"],
            test_metrics_xgb["mean_error_rate"],
        )
        result["test_metrics_adaptive"] = test_metrics_adaptive
    else:
        result["test_metrics_adaptive"] = None

    meta = lstm_meta_payload(
        lookback=LOOKBACK,
        lstm_aux_cols=final_lstm_cols,
        include_pku=include_pku,
        train_resid_clip=(lo, hi),
        business=business,
        output_clip_frac=RESID_OUTPUT_CLIP_FRAC,
        feature_dim=lstm_artifacts["feature_dim"],
    )
    save_lstm_artifacts(
        _group_dir(group),
        lstm_artifacts["lstm_model"],
        lstm_artifacts["feature_scaler"],
        lstm_artifacts["resid_scaler"],
        meta,
    )

    history = lstm_artifacts.get("history")
    val_loss_min = (
        float(min(history.history["val_loss"]))
        if history is not None and "val_loss" in history.history
        else None
    )
    result["lstm_trained"] = True
    result["lstm"] = {
        "val_loss": val_loss_min,
        "lookback": LOOKBACK,
        "feature_dim": lstm_artifacts["feature_dim"],
    }

    return result


# ---------- 采血训练 ----------


DEFAULT_SKIP_YEARS = [2019, 2020, 2021]


def train_collection(
    station: str = "北京市红十字血液中心",
    blood_types: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    skip_years: Optional[list] = None,
) -> list:
    """训练采血模型 (始终从数据库加载,XGB + LSTM 残差).

    skip_years: 训练跳过的年份,默认 [2019,2020,2021] 疫情封控期;传 [] 关闭跳过.
    """
    if blood_types is None:
        blood_types = ["ALL", "A", "B", "O", "AB"]
    if skip_years is None:
        skip_years = DEFAULT_SKIP_YEARS

    engine = get_engine()
    all_blood_df = load_collection_daily(
        engine, station=station, start_date=start_date, end_date=end_date
    )

    weather_df = _load_weather_features(include_pku=True)

    results = []
    for bt in blood_types:
        grp = collection_group_name(station, bt)
        grp_df = _prepare_group_df(all_blood_df, station, bt)
        if grp_df.empty:
            log.warning("[%s] 无数据, 跳过", grp)
            continue
        metrics = train_one_group(
            grp, grp_df, weather_df,
            business="collection", include_pku=True,
            skip_years=skip_years,
        )
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
    skip_years: Optional[list] = None,
) -> list:
    """训练供血模型 (从数据库).

    skip_years: 训练跳过的年份,默认 [2019,2020,2021] 疫情封控期;传 [] 关闭跳过.
    """
    if categories is None:
        categories = ALL_CATEGORIES
    if skip_years is None:
        skip_years = DEFAULT_SKIP_YEARS

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
            metrics = train_one_group(
                grp, grp_df, weather_df,
                business="supply", include_pku=False,
                skip_years=skip_years,
            )
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
                metrics = train_one_group(
                    grp, grp_df, weather_df,
                    business="supply", include_pku=False,
                    skip_years=skip_years,
                )
                results.append(metrics)

    log.info("Supply training done (scope=%s), %d models saved", scope, len(results))
    return results


# ---------- 预测: 逐日递推 + 按粒度聚合 ----------

# 方案 A: 长 horizon 递推漂移控制
# 每天 alpha 从 1.0 (纯 recursive) 衰减到 ANCHOR_MIN_ALPHA (anchor 权重最大),
# anchor 来自 start_dt 前 ANCHOR_HISTORY_DAYS 天历史按 day-of-week 的均值.
#
# 默认参数偏"温和":让模型保留更多波峰波谷,但对长 horizon 仍有抗漂移作用.
# 可在 API 层覆盖 (anchor_decay_days / anchor_min_alpha 参数).
ANCHOR_DECAY_DAYS = 120   # alpha 衰减到下限的天数 (越大 → 越长时间保持纯递推)
ANCHOR_MIN_ALPHA = 0.4    # alpha 下限 (越大 → 模型权重越大, anchor 越弱)
ANCHOR_HISTORY_DAYS = 56  # 用 start_dt 前 N 天历史构建 DOW 锚 (8 周, 减少局部偏差)

# 预测时从 DB 拉取的历史窗口 (天数). rolling28 + anchor 56 + 安全余量.
PREDICT_HISTORY_DAYS = 365


def _compute_dow_anchors(hist_df: pd.DataFrame, last_n: int = ANCHOR_HISTORY_DAYS) -> tuple:
    """从历史 hist_df 末尾 last_n 天按 day-of-week 计算锚点均值.

    返回 (dow_anchor_dict, overall_anchor):
      dow_anchor_dict: {0..6 → 该 DOW 的均值}
      overall_anchor: 整段均值, DOW 缺失时的 fallback
    """
    if hist_df is None or hist_df.empty:
        return {}, 0.0
    sub = hist_df.sort_values(DATE_COL).tail(last_n)
    if sub.empty:
        return {}, 0.0
    dows = pd.to_datetime(sub[DATE_COL]).dt.dayofweek
    dow_means = sub.groupby(dows)[TARGET_COL].mean().to_dict()
    overall = float(sub[TARGET_COL].mean())
    return {int(k): float(v) for k, v in dow_means.items()}, overall


def _anchor_alpha(
    days_ahead: int,
    decay_days: int = ANCHOR_DECAY_DAYS,
    min_alpha: float = ANCHOR_MIN_ALPHA,
) -> float:
    """随预测距离衰减的递推权重: day 0 → 1.0, day decay_days → min_alpha, 之后保持.

    min_alpha=1.0 等价于关闭 anchor (纯递推).
    """
    if days_ahead <= 0:
        return 1.0
    if min_alpha >= 1.0:
        return 1.0
    decay = max(0.0, 1.0 - days_ahead / max(1, decay_days))
    return max(min_alpha, decay)


def _resolve_collection_group(station: str, blood_type: str) -> str:
    """解析采血模型 group 名: 优先新格式 collection__*,缺失时回退旧格式 station_blood."""
    saved = set(list_saved_models())
    primary = collection_group_name(station, blood_type)
    if primary in saved:
        return primary
    legacy = group_name(station, blood_type)
    if legacy in saved:
        log.info("使用旧版命名模型: %s", legacy)
        return legacy
    raise FileNotFoundError(
        f"模型不存在: 已查找 {primary} 和 {legacy},请先运行 train-collection"
    )


def predict_date_range(
    station: str,
    blood_type: str,
    start_date: str,
    end_date: str,
    *,
    all_blood_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    anchor_decay_days: int = ANCHOR_DECAY_DAYS,
    anchor_min_alpha: float = ANCHOR_MIN_ALPHA,
    anchor_history_days: int = ANCHOR_HISTORY_DAYS,
) -> list:
    """
    对指定 station + blood_type 在 [start_date, end_date] 范围内做逐日递推预测.
    数据从数据库加载;若该 group 训练过 LSTM 残差,会自动叠加 y_hat_adaptive.
    返回 [{"date": "2025-04-01", "predicted_volume": 42}, ...].

    可选 all_blood_df / weather_df: 调用方预先加载好的数据 (如 API 一次请求多血型时
    复用), 避免重复 DB 查询和天气加载.
    """
    grp = _resolve_collection_group(station, blood_type)
    model, feat_cols = load_group_model(grp)
    lstm_pack = load_lstm_artifacts(_group_dir(grp))

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
        raise ValueError(f"end_date ({end_date}) 不能早于 start_date ({start_date})")

    include_pku = (
        bool(lstm_pack["meta"].get("include_pku", True)) if lstm_pack else True
    )
    if all_blood_df is None:
        # 只拉 start_dt 前 PREDICT_HISTORY_DAYS 天的历史, 大幅压缩 DB 拉取量
        hist_start = (start_dt - pd.Timedelta(days=PREDICT_HISTORY_DAYS)).strftime("%Y-%m-%d")
        engine = get_engine()
        all_blood_df = load_collection_daily(
            engine, station=station, start_date=hist_start, end_date=start_date
        )
    if weather_df is None:
        weather_df = _load_weather_features(include_pku=include_pku)
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
    if "dayofweek" not in full_df.columns and "weekday" in full_df.columns:
        full_df["dayofweek"] = full_df["weekday"]

    # 补齐训练时的特征列 (天气 one-hot 可能不完全一致)
    for col in feat_cols:
        if col not in full_df.columns:
            full_df[col] = 0

    predictions = _recursive_predict(
        full_df=full_df,
        pred_dates=pred_dates,
        model=model,
        feat_cols=feat_cols,
        lstm_pack=lstm_pack,
        start_dt=start_dt,
        anchor_decay_days=anchor_decay_days,
        anchor_min_alpha=anchor_min_alpha,
        anchor_history_days=anchor_history_days,
    )
    return predictions


def _recursive_predict(
    full_df: pd.DataFrame,
    pred_dates: pd.DatetimeIndex,
    model: XGBRegressor,
    feat_cols: list,
    lstm_pack: Optional[dict],
    start_dt: pd.Timestamp,
    *,
    anchor_decay_days: int = ANCHOR_DECAY_DAYS,
    anchor_min_alpha: float = ANCHOR_MIN_ALPHA,
    anchor_history_days: int = ANCHOR_HISTORY_DAYS,
) -> list:
    """逐日递推预测.若提供 lstm_pack,则在 XGB 输出基础上叠加 LSTM 残差校正.

    方案 A 锚定混合: 长 horizon 时,回写到 TARGET_COL 的值不是纯 y_hat,而是
        blend = α·y_hat + (1-α)·anchor[DOW]
    其中 α 随 days_ahead 线性衰减 (1.0 → 0.2),anchor 来自 start_dt 前 28 天
    按 DOW 的真实历史均值. 这样递推到第 60 天后,lag/rolling 特征会被锚点拉住,
    避免单调下漂崩溃. 输出给用户的 predicted_volume 仍是模型原始 y_hat.
    """
    full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)

    # 计算 DOW 锚点 (基于 start_dt 前 N 天真实历史)
    hist_for_anchor = full_df[full_df[DATE_COL] < start_dt]
    dow_anchor, overall_anchor = _compute_dow_anchors(
        hist_for_anchor, last_n=anchor_history_days
    )
    log.info(
        "锚点(DOW)|history=%dd|decay=%dd|min_alpha=%.2f: %s, 整体=%.1f",
        anchor_history_days, anchor_decay_days, anchor_min_alpha,
        {k: round(v, 1) for k, v in dow_anchor.items()}, overall_anchor,
    )

    use_lstm = lstm_pack is not None
    if use_lstm:
        lstm_meta = lstm_pack["meta"]
        lstm_model = lstm_pack["lstm_model"]
        feature_scaler = lstm_pack["feature_scaler"]
        resid_scaler = lstm_pack["resid_scaler"]
        lookback = int(lstm_meta["lookback"])
        lstm_aux_cols = list(lstm_meta["lstm_aux_cols"])
        clip_lo, clip_hi = lstm_meta.get("train_resid_clip", [-1e9, 1e9])
        output_clip_frac = float(
            lstm_meta.get("output_clip_frac", RESID_OUTPUT_CLIP_FRAC)
        )

        # 历史段一次性 y_xgb,据此计算历史残差并按训练时的百分位裁剪
        hist_mask = full_df[DATE_COL] < start_dt
        hist_y_xgb = normalize_pred(
            model.predict(full_df.loc[hist_mask, feat_cols].values)
        )
        full_df.loc[hist_mask, "y_xgb"] = hist_y_xgb
        full_df["y_xgb"] = full_df.get("y_xgb", pd.Series(0.0, index=full_df.index))

        hist_resid = (
            full_df.loc[hist_mask, TARGET_COL].astype(np.float64)
            - full_df.loc[hist_mask, "y_xgb"].astype(np.float64)
        ).clip(clip_lo, clip_hi)
        # resid_lag1[s] = resid[s-1]
        full_df["resid_lag1"] = 0.0
        # 仅对历史段(含 start_dt 这一天的 resid_lag1)做 shift
        resid_series = pd.Series(0.0, index=full_df.index, dtype=np.float64)
        resid_series.loc[hist_mask] = hist_resid.values
        full_df["resid_lag1"] = resid_series.shift(1).fillna(0.0).values
    else:
        lstm_meta = None
        lstm_model = feature_scaler = resid_scaler = None
        lookback = 0
        lstm_aux_cols = []
        output_clip_frac = RESID_OUTPUT_CLIP_FRAC

    predictions = []
    for cur_date in pred_dates:
        cur_mask = full_df[DATE_COL] == cur_date
        if not cur_mask.any():
            continue

        x = full_df.loc[cur_mask, feat_cols].values
        y_xgb = float(normalize_pred(model.predict(x))[0])
        if use_lstm:
            full_df.loc[cur_mask, "y_xgb"] = y_xgb

        resid_pred_clipped = 0.0
        if use_lstm:
            window_start = cur_date - pd.Timedelta(days=lookback - 1)
            window = (
                full_df[
                    (full_df[DATE_COL] >= window_start)
                    & (full_df[DATE_COL] <= cur_date)
                ]
                .sort_values(DATE_COL)
            )
            if len(window) == lookback:
                window_input = window.reindex(columns=lstm_aux_cols).fillna(0.0)
                resid_pred = predict_residual_for_window(
                    lstm_model,
                    feature_scaler,
                    resid_scaler,
                    window_input,
                    lookback,
                )
                resid_pred_clipped = float(
                    clip_resid_pred(
                        np.array([resid_pred]),
                        np.array([y_xgb]),
                        frac=output_clip_frac,
                    )[0]
                )

        y_hat_arr = normalize_pred(np.array([y_xgb + resid_pred_clipped]))
        y_hat = int(y_hat_arr[0])
        predictions.append({"date": str(cur_date.date()), "predicted_volume": y_hat})

        # 锚定混合: 回写到 TARGET_COL 的值用于下一天的 lag/rolling 计算,
        # 用 blend 而不是纯 y_hat,防止递推漂移
        days_ahead = (cur_date - start_dt).days
        alpha = _anchor_alpha(
            days_ahead,
            decay_days=anchor_decay_days,
            min_alpha=anchor_min_alpha,
        )
        anchor_val = dow_anchor.get(int(cur_date.dayofweek), overall_anchor)
        blended_target = alpha * y_hat + (1.0 - alpha) * anchor_val
        full_df.loc[cur_mask, TARGET_COL] = blended_target

        if use_lstm:
            next_date = cur_date + pd.Timedelta(days=1)
            next_mask = full_df[DATE_COL] == next_date
            if next_mask.any():
                full_df.loc[next_mask, "resid_lag1"] = resid_pred_clipped

        # 重算滚动/动态特征(LSTM aux 列 lag1, rolling7 也在此更新)
        full_df = add_prev_rolling_sums(full_df, windows=(1, 3, 7, 14, 28))
        full_df = add_dynamic_feature(full_df)

    return predictions


def predict_supply_range(
    scope: str,
    category: str,
    start_date: str,
    end_date: str,
    org: Optional[str] = None,
    *,
    supply_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    anchor_decay_days: int = ANCHOR_DECAY_DAYS,
    anchor_min_alpha: float = ANCHOR_MIN_ALPHA,
    anchor_history_days: int = ANCHOR_HISTORY_DAYS,
) -> list:
    """
    对指定供血类型在 [start_date, end_date] 范围内做逐日递推预测.
    数据从数据库加载;若该 group 训练过 LSTM 残差,会自动叠加 y_hat_adaptive.

    可选 supply_df / weather_df: 调用方预加载,避免一次请求多类目重复查 DB.
    """
    if scope == "global":
        grp = supply_global_group_name(category)
    else:
        if not org:
            raise ValueError("scope='org' requires org parameter")
        grp = supply_org_group_name(org, category)

    model, feat_cols = load_group_model(grp)
    lstm_pack = load_lstm_artifacts(_group_dir(grp))

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
        raise ValueError(f"end_date ({end_date}) < start_date ({start_date})")

    if supply_df is None:
        hist_start = (start_dt - pd.Timedelta(days=PREDICT_HISTORY_DAYS)).strftime("%Y-%m-%d")
        engine = get_engine()
        supply_df = load_supply_daily(
            engine, scope=scope, org=org, start_date=hist_start, end_date=start_date
        )
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

    include_pku = (
        bool(lstm_pack["meta"].get("include_pku", False)) if lstm_pack else False
    )
    if weather_df is None:
        weather_df = _load_weather_features(include_pku=include_pku)
    full_df = pd.merge(full_df, weather_df, on="date", how="left")
    if "dayofweek" not in full_df.columns and "weekday" in full_df.columns:
        full_df["dayofweek"] = full_df["weekday"]

    for col in feat_cols:
        if col not in full_df.columns:
            full_df[col] = 0

    predictions = _recursive_predict(
        full_df=full_df,
        pred_dates=pred_dates,
        model=model,
        feat_cols=feat_cols,
        lstm_pack=lstm_pack,
        start_dt=start_dt,
        anchor_decay_days=anchor_decay_days,
        anchor_min_alpha=anchor_min_alpha,
        anchor_history_days=anchor_history_days,
    )
    return predictions


# ---------- 回测评估 (one-step-ahead with ground truth) ----------
# 与 /predict/* 的关键区别:
#   /predict: 递推未来,昨天的 lag1 = 昨天的预测值 → 误差累积 (真预测)
#   /eval:    单步回测,昨天的 lag1 = 昨天的真实值 → 无误差累积 (调试/汇报)
# /eval 等价于 lstm/blood_lstm_xgb_v5(resid).py 测试段评估的产出 (v5 那种 90% 拟合图).
# 任何模型在这种设定下都会比真预测好看, 不能据此外推未来预测能力.


def _one_step_evaluate(
    full_df: pd.DataFrame,
    eval_dates: list,
    model: XGBRegressor,
    feat_cols: list,
    lstm_pack: Optional[dict],
) -> list:
    """单步回测: 每天用截至 t-1 的真实特征预测 t, 不递推.

    full_df 必须包含 eval_dates 范围内的真实 TARGET_COL 值.
    返回 [{date, predicted_volume, actual_volume, y_xgb, resid_pred, error, error_rate}, ...].
    """
    full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)

    # 一次性用 XGB 预测全序列 (特征已基于真实 lag, 无递推)
    all_pred = normalize_pred(model.predict(full_df[feat_cols].values))
    full_df["y_xgb"] = all_pred

    use_lstm = lstm_pack is not None
    if use_lstm:
        lstm_meta = lstm_pack["meta"]
        lstm_model = lstm_pack["lstm_model"]
        feature_scaler = lstm_pack["feature_scaler"]
        resid_scaler = lstm_pack["resid_scaler"]
        lookback = int(lstm_meta["lookback"])
        lstm_aux_cols = list(lstm_meta["lstm_aux_cols"])
        clip_lo, clip_hi = lstm_meta.get("train_resid_clip", [-1e9, 1e9])
        output_clip_frac = float(
            lstm_meta.get("output_clip_frac", RESID_OUTPUT_CLIP_FRAC)
        )
        # resid_lag1 用真实残差的滞后 (训练时也是这样)
        resid = (
            full_df[TARGET_COL].astype(np.float64)
            - full_df["y_xgb"].astype(np.float64)
        ).clip(clip_lo, clip_hi)
        full_df["resid_lag1"] = resid.shift(1).fillna(0.0).values
    else:
        lstm_meta = None
        lstm_model = feature_scaler = resid_scaler = None
        lookback = 0
        lstm_aux_cols = []
        output_clip_frac = RESID_OUTPUT_CLIP_FRAC

    results = []
    for cur_date in eval_dates:
        cur_mask = full_df[DATE_COL] == cur_date
        if not cur_mask.any():
            continue
        actual = float(full_df.loc[cur_mask, TARGET_COL].iloc[0])
        y_xgb = float(full_df.loc[cur_mask, "y_xgb"].iloc[0])

        resid_pred_clipped = 0.0
        if use_lstm:
            window_start = cur_date - pd.Timedelta(days=lookback - 1)
            window = (
                full_df[
                    (full_df[DATE_COL] >= window_start)
                    & (full_df[DATE_COL] <= cur_date)
                ]
                .sort_values(DATE_COL)
            )
            if len(window) == lookback:
                window_input = window.reindex(columns=lstm_aux_cols).fillna(0.0)
                resid_pred = predict_residual_for_window(
                    lstm_model, feature_scaler, resid_scaler, window_input, lookback
                )
                resid_pred_clipped = float(
                    clip_resid_pred(
                        np.array([resid_pred]),
                        np.array([y_xgb]),
                        frac=output_clip_frac,
                    )[0]
                )

        y_hat_arr = normalize_pred(np.array([y_xgb + resid_pred_clipped]))
        y_hat = int(y_hat_arr[0])
        err = y_hat - actual
        err_rate = abs(err) / max(actual, 1e-6) if actual > 0 else 0.0

        results.append({
            "date": str(cur_date.date()),
            "predicted_volume": y_hat,
            "actual_volume": int(round(actual)),
            "y_xgb": int(round(y_xgb)),
            "resid_pred": int(round(resid_pred_clipped)),
            "error": int(err),
            "error_rate": round(err_rate, 4),
        })

    return results


def _eval_summary(rows: list) -> dict:
    """计算回测的汇总指标: MAE / MAPE / 误差分布."""
    if not rows:
        return {"sample_count": 0}
    actuals = np.array([r["actual_volume"] for r in rows], dtype=float)
    preds = np.array([r["predicted_volume"] for r in rows], dtype=float)
    err = preds - actuals
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(abs_err / np.maximum(actuals, 1e-6))) * 100
    err_rates = abs_err / np.maximum(actuals, 1e-6)
    return {
        "sample_count": len(rows),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "within_10pct_ratio": round(float(np.mean(err_rates < 0.1)), 4),
        "within_20pct_ratio": round(float(np.mean(err_rates < 0.2)), 4),
        "within_30pct_ratio": round(float(np.mean(err_rates < 0.3)), 4),
        "over_100pct_ratio": round(float(np.mean(err_rates > 1.0)), 4),
        "actual_mean": round(float(actuals.mean()), 1),
        "predicted_mean": round(float(preds.mean()), 1),
        "actual_std": round(float(actuals.std()), 1),
        "predicted_std": round(float(preds.std()), 1),
    }


def evaluate_date_range(
    station: str, blood_type: str, start_date: str, end_date: str,
    *,
    all_blood_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    anchor_decay_days: int = ANCHOR_DECAY_DAYS,
    anchor_min_alpha: float = ANCHOR_MIN_ALPHA,
    anchor_history_days: int = ANCHOR_HISTORY_DAYS,
) -> list:
    """混合预测 (eval + forecast):
    - 真实历史覆盖范围内 [start_dt, min(end_dt, max_real_date)]: 单步预测,使用真实 lag/rolling
    - 超出真实数据范围 (d > max_real_date): 走递推未来预测 (复用 predict_date_range)
    返回 [{"date": "...", "predicted_volume": int}, ...] (简化字段, 不含真实值/诊断信息).
    """
    grp = _resolve_collection_group(station, blood_type)
    model, feat_cols = load_group_model(grp)
    lstm_pack = load_lstm_artifacts(_group_dir(grp))

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
        raise ValueError(f"end_date ({end_date}) 不能早于 start_date ({start_date})")

    include_pku = (
        bool(lstm_pack["meta"].get("include_pku", True)) if lstm_pack else True
    )
    if all_blood_df is None:
        # 拉历史 + eval 区间的真实数据 (end_dt 超出真实数据末尾时, DB 自然返回到末尾)
        hist_start = (start_dt - pd.Timedelta(days=PREDICT_HISTORY_DAYS)).strftime("%Y-%m-%d")
        engine = get_engine()
        all_blood_df = load_collection_daily(
            engine, station=station, start_date=hist_start, end_date=end_date,
        )
    if weather_df is None:
        weather_df = _load_weather_features(include_pku=include_pku)

    grp_df = _prepare_group_df(all_blood_df, station, blood_type)
    if grp_df.empty:
        raise ValueError(f"无 {station}/{blood_type} 数据")

    # 真实数据时间末尾 (超过这个的就走递推预测)
    max_real_date = pd.to_datetime(grp_df[DATE_COL]).max()

    rows: list = []

    # ---- Part 1: 真实数据覆盖范围 → 单步预测 (lag/rolling 基于真实历史) ----
    eval_end = min(end_dt, max_real_date)
    if start_dt <= eval_end:
        eval_grp_df = _segment_aware_features(grp_df)
        eval_grp_df["label_tplus1"] = eval_grp_df[TARGET_COL]
        eval_grp_df = eval_grp_df.dropna(subset=["label_tplus1"])
        full_df = pd.merge(eval_grp_df, weather_df, on="date", how="left")
        if "dayofweek" not in full_df.columns and "weekday" in full_df.columns:
            full_df["dayofweek"] = full_df["weekday"]
        for col in feat_cols:
            if col not in full_df.columns:
                full_df[col] = 0

        eval_dates = [
            d for d in pd.date_range(start=start_dt, end=eval_end, freq="D")
            if (full_df[DATE_COL] == d).any()
        ]
        diag_rows = _one_step_evaluate(full_df, eval_dates, model, feat_cols, lstm_pack)
        rows.extend(
            {"date": r["date"], "predicted_volume": r["predicted_volume"]}
            for r in diag_rows
        )

    # ---- Part 2: 超出真实数据末尾 → 递推未来预测 ----
    if end_dt > max_real_date:
        forecast_start = (max_real_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        forecast_preds = predict_date_range(
            station, blood_type, forecast_start, end_date,
            all_blood_df=all_blood_df,
            weather_df=weather_df,
            anchor_decay_days=anchor_decay_days,
            anchor_min_alpha=anchor_min_alpha,
            anchor_history_days=anchor_history_days,
        )
        rows.extend(forecast_preds)

    return rows


def evaluate_supply_range(
    scope: str, category: str, start_date: str, end_date: str,
    org: Optional[str] = None,
    *,
    supply_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    anchor_decay_days: int = ANCHOR_DECAY_DAYS,
    anchor_min_alpha: float = ANCHOR_MIN_ALPHA,
    anchor_history_days: int = ANCHOR_HISTORY_DAYS,
) -> list:
    """混合预测 (eval + forecast) 供血版本.

    与 evaluate_date_range 同义: 真实历史覆盖范围走单步, 超出走递推.
    返回 [{"date": "...", "predicted_volume": int}, ...].
    """
    if scope == "global":
        grp = supply_global_group_name(category)
    else:
        if not org:
            raise ValueError("scope='org' requires org parameter")
        grp = supply_org_group_name(org, category)

    model, feat_cols = load_group_model(grp)
    lstm_pack = load_lstm_artifacts(_group_dir(grp))

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if end_dt < start_dt:
        raise ValueError(f"end_date ({end_date}) < start_date ({start_date})")

    if supply_df is None:
        hist_start = (start_dt - pd.Timedelta(days=PREDICT_HISTORY_DAYS)).strftime("%Y-%m-%d")
        engine = get_engine()
        supply_df = load_supply_daily(
            engine, scope=scope, org=org, start_date=hist_start, end_date=end_date,
        )

    grp_df = _prepare_supply_group_df(supply_df, scope, category, org=org)
    if grp_df.empty:
        raise ValueError(f"无 {scope}/{category} 数据")

    include_pku = (
        bool(lstm_pack["meta"].get("include_pku", False)) if lstm_pack else False
    )
    if weather_df is None:
        weather_df = _load_weather_features(include_pku=include_pku)

    max_real_date = pd.to_datetime(grp_df[DATE_COL]).max()

    rows: list = []

    # ---- Part 1: 真实数据覆盖范围 → 单步预测 ----
    eval_end = min(end_dt, max_real_date)
    if start_dt <= eval_end:
        eval_grp_df = _segment_aware_features(grp_df)
        eval_grp_df["label_tplus1"] = eval_grp_df[TARGET_COL]
        eval_grp_df = eval_grp_df.dropna(subset=["label_tplus1"])
        full_df = pd.merge(eval_grp_df, weather_df, on="date", how="left")
        if "dayofweek" not in full_df.columns and "weekday" in full_df.columns:
            full_df["dayofweek"] = full_df["weekday"]
        for col in feat_cols:
            if col not in full_df.columns:
                full_df[col] = 0

        eval_dates = [
            d for d in pd.date_range(start=start_dt, end=eval_end, freq="D")
            if (full_df[DATE_COL] == d).any()
        ]
        diag_rows = _one_step_evaluate(full_df, eval_dates, model, feat_cols, lstm_pack)
        rows.extend(
            {"date": r["date"], "predicted_volume": r["predicted_volume"]}
            for r in diag_rows
        )

    # ---- Part 2: 超出真实数据末尾 → 递推未来预测 ----
    if end_dt > max_real_date:
        forecast_start = (max_real_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        forecast_preds = predict_supply_range(
            scope, category, forecast_start, end_date, org=org,
            supply_df=supply_df,
            weather_df=weather_df,
            anchor_decay_days=anchor_decay_days,
            anchor_min_alpha=anchor_min_alpha,
            anchor_history_days=anchor_history_days,
        )
        rows.extend(forecast_preds)

    return rows


def aggregate_predictions(daily_preds: list, date_type: str) -> list:
    """将逐日预测结果按 date_type 聚合: day=原样, month=按月, quarter=按季, year=按年."""
    if date_type == "DAY" or not daily_preds:
        return daily_preds

    df = pd.DataFrame(daily_preds)
    df["date"] = pd.to_datetime(df["date"])

    period_map = {"MONTH": "M", "QUARTER": "Q", "YEAR": "Y"}
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
        skip_years: str = Query(
            "2019,2020,2021",
            description="逗号分隔训练时跳过的年份 (默认排除疫情封控期);传空字符串关闭",
        ),
    ):
        bt_list = [x.strip() for x in blood_types.split(",") if x.strip()]
        skip_y = [int(y.strip()) for y in skip_years.split(",") if y.strip()] if skip_years else []
        results = train_collection(
            station=station,
            blood_types=bt_list,
            start_date=start_date,
            end_date=end_date,
            skip_years=skip_y,
        )
        return {"trained": len(results), "results": results}

    @app.get("/predict/collection")
    def api_predict_collection(
        start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        station: str = Query("北京市红十字血液中心"),
        blood_types: str = Query("ALL", description="逗号分隔血型, 如 ALL,A,B,O,AB"),
        date_type: str = Query(
            "DAY", description="聚合粒度: DAY=日, MONTH=月, QUARTER=季, YEAR=年"
        ),
        anchor_decay_days: int = Query(
            ANCHOR_DECAY_DAYS,
            description="anchor 衰减天数, 越大越长保留波动 (默认 120)",
        ),
        anchor_min_alpha: float = Query(
            ANCHOR_MIN_ALPHA,
            description="anchor 最低权重 (默认 0.4); 设为 1.0 关闭 anchor",
        ),
        anchor_history_days: int = Query(
            ANCHOR_HISTORY_DAYS,
            description="anchor 历史窗口天数 (默认 56)",
        ),
    ):
        if date_type not in ("DAY", "MONTH", "QUARTER", "YEAR"):
            raise HTTPException(
                400, f"date_type 不合法: {date_type}，可选 DAY/MONTH/QUARTER/YEAR"
            )

        bt_list = [b.strip() for b in blood_types.split(",") if b.strip()]
        if not bt_list:
            raise HTTPException(400, "blood_types 不能为空")

        saved = list_saved_models()
        results_by_type = []

        # 一次拉 DB + 加载天气, 各血型共享 (DB ~1-3s + 天气 ~1-2s 只付一次)
        start_dt = pd.to_datetime(start_date)
        hist_start = (start_dt - pd.Timedelta(days=PREDICT_HISTORY_DAYS)).strftime("%Y-%m-%d")
        engine = get_engine()
        shared_blood_df = load_collection_daily(
            engine, station=station, start_date=hist_start, end_date=start_date
        )
        shared_weather_df = _load_weather_features(include_pku=True)

        for bt in bt_list:
            grp = collection_group_name(station, bt)
            if grp not in saved:
                # fallback to legacy name
                grp = group_name(station, bt)
                if grp not in saved:
                    results_by_type.append(
                        {
                            "blood_type": bt,
                            "group": grp,
                            "error": f"模型不存在: {grp}，请先训练",
                            "predictions": [],
                        }
                    )
                    continue

            try:
                daily_preds = predict_date_range(
                    station, bt, start_date, end_date,
                    all_blood_df=shared_blood_df,
                    weather_df=shared_weather_df,
                    anchor_decay_days=anchor_decay_days,
                    anchor_min_alpha=anchor_min_alpha,
                    anchor_history_days=anchor_history_days,
                )
                preds = aggregate_predictions(daily_preds, date_type)
            except ValueError as e:
                results_by_type.append(
                    {
                        "blood_type": bt,
                        "group": grp,
                        "error": str(e),
                        "predictions": [],
                    }
                )
                continue

            results_by_type.append(
                {
                    "blood_type": bt,
                    "group": grp,
                    "count": len(preds),
                    "predictions": preds,
                }
            )

        return {
            "business": "collection",
            "station": station,
            "date_type": date_type,
            "start_date": start_date,
            "end_date": end_date,
            "blood_types": results_by_type,
        }

    # ---- 供血训练 + 预测 ----

    @app.post("/train/supply")
    def api_train_supply(
        scope: str = Query("global"),
        categories: str = Query("红细胞类,血小板类,血浆类"),
        orgs: str = Query(None),
        start_date: str = Query(None),
        end_date: str = Query(None),
        skip_years: str = Query(
            "2019,2020,2021",
            description="逗号分隔训练时跳过的年份 (默认排除疫情封控期);传空字符串关闭",
        ),
    ):
        cat_list = [x.strip() for x in categories.split(",") if x.strip()]
        org_list = [x.strip() for x in orgs.split(",") if x.strip()] if orgs else None
        skip_y = [int(y.strip()) for y in skip_years.split(",") if y.strip()] if skip_years else []
        results = train_supply(
            scope=scope,
            categories=cat_list,
            orgs=org_list,
            start_date=start_date,
            end_date=end_date,
            skip_years=skip_y,
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
            "DAY", description="聚合粒度: DAY=日, MONTH=月, QUARTER=季, YEAR=年"
        ),
        anchor_decay_days: int = Query(
            ANCHOR_DECAY_DAYS,
            description="anchor 衰减天数, 越大越长保留波动 (默认 120)",
        ),
        anchor_min_alpha: float = Query(
            ANCHOR_MIN_ALPHA,
            description="anchor 最低权重 (默认 0.4); 设为 1.0 关闭 anchor",
        ),
        anchor_history_days: int = Query(
            ANCHOR_HISTORY_DAYS,
            description="anchor 历史窗口天数 (默认 56)",
        ),
    ):
        if date_type not in ("DAY", "MONTH", "QUARTER", "YEAR"):
            raise HTTPException(
                400, f"date_type 不合法: {date_type}，可选 DAY/MONTH/QUARTER/YEAR"
            )
        if scope == "org" and not org:
            raise HTTPException(400, "scope='org' requires org parameter")
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        if not cat_list:
            raise HTTPException(400, "categories 不能为空")

        saved = list_saved_models()
        results_by_category = []

        # 一次拉 DB + 加载天气, 各类别共享
        start_dt = pd.to_datetime(start_date)
        hist_start = (start_dt - pd.Timedelta(days=PREDICT_HISTORY_DAYS)).strftime("%Y-%m-%d")
        engine = get_engine()
        shared_supply_df = load_supply_daily(
            engine, scope=scope, org=org, start_date=hist_start, end_date=start_date
        )
        shared_weather_df = _load_weather_features(include_pku=False)

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
                    scope, cat, start_date, end_date, org=org,
                    supply_df=shared_supply_df,
                    weather_df=shared_weather_df,
                    anchor_decay_days=anchor_decay_days,
                    anchor_min_alpha=anchor_min_alpha,
                    anchor_history_days=anchor_history_days,
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

    # ---- 混合预测端点 (in-data 单步 + 超出范围递推) ----

    @app.get("/eval/collection")
    def api_eval_collection(
        start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        station: str = Query("北京市红十字血液中心"),
        blood_types: str = Query("ALL", description="逗号分隔血型, 如 ALL,A,B,O,AB"),
        date_type: str = Query(
            "DAY", description="聚合粒度: DAY=日, MONTH=月, QUARTER=季, YEAR=年"
        ),
        anchor_decay_days: int = Query(
            ANCHOR_DECAY_DAYS,
            description="anchor 衰减天数 (仅用于超出真实数据的递推部分)",
        ),
        anchor_min_alpha: float = Query(
            ANCHOR_MIN_ALPHA,
            description="anchor 最低权重 (仅用于超出真实数据的递推部分);1.0 关闭",
        ),
        anchor_history_days: int = Query(
            ANCHOR_HISTORY_DAYS,
            description="anchor 历史窗口天数 (仅用于超出真实数据的递推部分)",
        ),
    ):
        """混合预测端点:
          - 真实历史覆盖范围内: 单步预测 (lag/rolling 基于真实值, 形态准确)
          - 超出真实数据末尾: 自动切换到递推未来预测
        响应仅含日期 + 预测值, 不含真实数据.
        """
        if date_type not in ("DAY", "MONTH", "QUARTER", "YEAR"):
            raise HTTPException(
                400, f"date_type 不合法: {date_type}, 可选 DAY/MONTH/QUARTER/YEAR"
            )

        bt_list = [b.strip() for b in blood_types.split(",") if b.strip()]
        if not bt_list:
            raise HTTPException(400, "blood_types 不能为空")

        saved = list_saved_models()
        results_by_type = []

        start_dt = pd.to_datetime(start_date)
        hist_start = (start_dt - pd.Timedelta(days=PREDICT_HISTORY_DAYS)).strftime("%Y-%m-%d")
        engine = get_engine()
        shared_blood_df = load_collection_daily(
            engine, station=station, start_date=hist_start, end_date=end_date
        )
        shared_weather_df = _load_weather_features(include_pku=True)

        for bt in bt_list:
            grp = collection_group_name(station, bt)
            if grp not in saved:
                grp = group_name(station, bt)
                if grp not in saved:
                    results_by_type.append({
                        "blood_type": bt, "group": grp,
                        "error": f"模型不存在: {grp}, 请先训练",
                        "predictions": [],
                    })
                    continue

            try:
                daily_preds = evaluate_date_range(
                    station, bt, start_date, end_date,
                    all_blood_df=shared_blood_df,
                    weather_df=shared_weather_df,
                    anchor_decay_days=anchor_decay_days,
                    anchor_min_alpha=anchor_min_alpha,
                    anchor_history_days=anchor_history_days,
                )
                preds = aggregate_predictions(daily_preds, date_type)
            except ValueError as e:
                results_by_type.append({
                    "blood_type": bt, "group": grp,
                    "error": str(e), "predictions": [],
                })
                continue

            results_by_type.append({
                "blood_type": bt,
                "group": grp,
                "count": len(preds),
                "predictions": preds,
            })

        return {
            "business": "collection",
            "station": station,
            "date_type": date_type,
            "start_date": start_date,
            "end_date": end_date,
            "blood_types": results_by_type,
        }

    @app.get("/eval/supply")
    def api_eval_supply(
        scope: str = Query("global"),
        categories: str = Query(
            ..., description="逗号分隔供血类型, 如 红细胞类,血小板类,血浆类"
        ),
        org: str = Query(None),
        start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        date_type: str = Query(
            "DAY", description="聚合粒度: DAY=日, MONTH=月, QUARTER=季, YEAR=年"
        ),
        anchor_decay_days: int = Query(
            ANCHOR_DECAY_DAYS,
            description="anchor 衰减天数 (仅用于超出真实数据的递推部分)",
        ),
        anchor_min_alpha: float = Query(
            ANCHOR_MIN_ALPHA,
            description="anchor 最低权重 (仅用于超出真实数据的递推部分);1.0 关闭",
        ),
        anchor_history_days: int = Query(
            ANCHOR_HISTORY_DAYS,
            description="anchor 历史窗口天数 (仅用于超出真实数据的递推部分)",
        ),
    ):
        """混合预测端点 (供血): in-data 单步 + 超出范围递推. 响应仅含预测值."""
        if date_type not in ("DAY", "MONTH", "QUARTER", "YEAR"):
            raise HTTPException(
                400, f"date_type 不合法: {date_type}, 可选 DAY/MONTH/QUARTER/YEAR"
            )
        if scope == "org" and not org:
            raise HTTPException(400, "scope='org' requires org parameter")
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        if not cat_list:
            raise HTTPException(400, "categories 不能为空")

        saved = list_saved_models()
        results_by_category = []

        start_dt = pd.to_datetime(start_date)
        hist_start = (start_dt - pd.Timedelta(days=PREDICT_HISTORY_DAYS)).strftime("%Y-%m-%d")
        engine = get_engine()
        shared_supply_df = load_supply_daily(
            engine, scope=scope, org=org, start_date=hist_start, end_date=end_date
        )
        shared_weather_df = _load_weather_features(include_pku=False)

        for cat in cat_list:
            if scope == "global":
                grp = supply_global_group_name(cat)
            else:
                grp = supply_org_group_name(org, cat)

            if grp not in saved:
                results_by_category.append({
                    "category": cat, "group": grp,
                    "error": f"模型不存在: {grp}, 请先训练",
                    "predictions": [],
                })
                continue

            try:
                daily_preds = evaluate_supply_range(
                    scope, cat, start_date, end_date, org=org,
                    supply_df=shared_supply_df,
                    weather_df=shared_weather_df,
                    anchor_decay_days=anchor_decay_days,
                    anchor_min_alpha=anchor_min_alpha,
                    anchor_history_days=anchor_history_days,
                )
                preds = aggregate_predictions(daily_preds, date_type)
            except ValueError as e:
                results_by_category.append({
                    "category": cat, "group": grp,
                    "error": str(e), "predictions": [],
                })
                continue

            results_by_category.append({
                "category": cat,
                "group": grp,
                "count": len(preds),
                "predictions": preds,
            })

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

    return app


# ---------- CLI ----------


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost 血液预测: 训练模型 / 启动 API 服务 (双业务版)"
    )
    sub = parser.add_subparsers(dest="mode")

    # 采血训练 (DB)
    train_c = sub.add_parser("train-collection", help="Train collection models from DB")
    train_c.add_argument("--station", default="北京市红十字血液中心")
    train_c.add_argument("--blood_types", default="ALL,A,B,O,AB")
    train_c.add_argument("--start_date", default=None)
    train_c.add_argument("--end_date", default=None)
    train_c.add_argument(
        "--skip_years", default="2019,2020,2021",
        help="训练跳过的年份(默认疫情封控期);传空字符串关闭",
    )

    # 供血训练 (DB)
    train_s = sub.add_parser("train-supply", help="Train supply models from DB")
    train_s.add_argument("--scope", default="global", choices=["global", "org"])
    train_s.add_argument("--categories", default="红细胞类,血小板类,血浆类")
    train_s.add_argument("--orgs", default=None)
    train_s.add_argument("--start_date", default=None)
    train_s.add_argument("--end_date", default=None)
    train_s.add_argument(
        "--skip_years", default="2019,2020,2021",
        help="训练跳过的年份(默认疫情封控期);传空字符串关闭",
    )

    # API 服务
    serve_p = sub.add_parser("serve", help="启动 FastAPI 预测服务")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.mode == "train-collection":
        bt_list = [x.strip() for x in args.blood_types.split(",") if x.strip()]
        skip_y = (
            [int(y.strip()) for y in args.skip_years.split(",") if y.strip()]
            if args.skip_years else []
        )
        results = train_collection(
            station=args.station,
            blood_types=bt_list,
            start_date=args.start_date,
            end_date=args.end_date,
            skip_years=skip_y,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    elif args.mode == "train-supply":
        cat_list = [x.strip() for x in args.categories.split(",") if x.strip()]
        org_list = (
            [x.strip() for x in args.orgs.split(",") if x.strip()]
            if args.orgs
            else None
        )
        skip_y = (
            [int(y.strip()) for y in args.skip_years.split(",") if y.strip()]
            if args.skip_years else []
        )
        results = train_supply(
            scope=args.scope,
            categories=cat_list,
            orgs=org_list,
            start_date=args.start_date,
            end_date=args.end_date,
            skip_years=skip_y,
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
