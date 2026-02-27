import os
import pandas as pd


def build_summary_with_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 原始唯一值
    station_list = sorted(df["血站"].dropna().astype(str).unique().tolist())
    blood_list = sorted(df["血型"].dropna().astype(str).unique().tolist())

    # 加上 ALL
    station_all_list = station_list + ["ALL"]
    blood_all_list = blood_list + ["ALL"]

    # 1) 原始明细聚合：date + 血站 + 血型
    base = (
        df.groupby(["date", "血站", "血型"], as_index=False)["血量"]
        .sum()
    )

    # 2) 血型汇总：date + 血站 + 血型=ALL
    blood_all = (
        df.groupby(["date", "血站"], as_index=False)["血量"]
        .sum()
    )
    blood_all["血型"] = "ALL"

    # 3) 血站汇总：date + 血站=ALL + 血型
    station_all = (
        df.groupby(["date", "血型"], as_index=False)["血量"]
        .sum()
    )
    station_all["血站"] = "ALL"

    # 4) 总汇总：date + 血站=ALL + 血型=ALL
    grand_all = (
        df.groupby(["date"], as_index=False)["血量"]
        .sum()
    )
    grand_all["血站"] = "ALL"
    grand_all["血型"] = "ALL"

    # 合并所有已存在结果
    result = pd.concat(
        [base, blood_all, station_all, grand_all],
        ignore_index=True,
        sort=False
    )

    result = result[["date", "血站", "血型", "血量"]].copy()

    # ---------------------------
    # 补齐所有 date × 血站(含ALL) × 血型(含ALL)
    # ---------------------------
    all_dates = sorted(result["date"].dropna().unique().tolist())

    full_index = pd.MultiIndex.from_product(
        [all_dates, station_all_list, blood_all_list],
        names=["date", "血站", "血型"]
    )

    result = (
        result.set_index(["date", "血站", "血型"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # 血量转成数值
    result["血量"] = pd.to_numeric(result["血量"], errors="coerce").fillna(0)

    # 排序
    result = result.sort_values(["date", "血站", "血型"]).reset_index(drop=True)
    return result


if __name__ == '__main__':
    folder_path = "./data"  # 你的目录

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    df_list = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        use_cols = [0, 2, 3, 16, 22, 24]
        df = pd.read_excel(file_path, usecols=use_cols)
        df.rename(columns={'采血部门': '血站',
                           '档案血型': '血型',
                           '采血量': '血量',
                           '组织方式': 'group_type'}, inplace=True)
        df["date"] = pd.to_datetime(df["采血时间"], format="%Y/%m/%d %H:%M:%S").dt.date
        df["date"] = pd.to_datetime(df["date"])

        df["血型"] = (
            df["血型"]
            .astype("string")  # 统一字符串类型
            .str.strip()  # 去除空格
            .str.upper()  # 转大写
        )

        df["血量"] = df["血量"].where(df["血量"] <= 100.0, df["血量"] / 200.0)

        df_list.append(df)

    # 合并成一个大表
    df_all = pd.concat(df_list, ignore_index=True)
    print(df_all["date"].isna().sum())
    print(df_all["date"].dt.year.unique())

    # 组装包含团体
    group_result_df = build_summary_with_fill(df_all)
    group_result_df.to_csv("./feature/all_data_merge.csv", index=False)
    print("总数据量:", group_result_df.shape)

    # 组装不包含团体的
    df_all_remove_group = df_all[~df_all["group_type"].isin(["团体预约", "团体无偿"])]
    remove_group_result_df = build_summary_with_fill(df_all_remove_group)
    remove_group_result_df.to_csv("./feature/remove_group_merge.csv", index=False)
    print("总数据量:", remove_group_result_df.shape)
