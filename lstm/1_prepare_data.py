import os
import pandas as pd

if __name__ == '__main__':
    folder_path = "./data"  # 你的目录

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    df_list = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        use_cols = [0, 2, 16, 22, 24]
        df = pd.read_excel(file_path, usecols=use_cols)
        df.rename(columns={'采血部门': '血站',
                           '档案血型': '血型',
                           '采血量': '血量'}, inplace=True)
        #df = df[~df["组织方式"].isin(["团体预约", "团体无偿"])]
        df["date"] = pd.to_datetime(
            df["采血时间"],
            format="%Y/%m/%d %H:%M:%S"
        ).dt.date

        df["date"] = pd.to_datetime(df["date"])

        df["血型"] = (
            df["血型"]
            .astype("string")  # 统一字符串类型
            .str.strip()  # 去除空格
            .str.upper()  # 转大写
        )

        df["血量"] = df["血量"].where(df["血量"] <= 100.0, df["血量"] / 200.0)

        df_grouped = (
            df.groupby(["date", "血站", "血型"], as_index=False)
            .agg(总血量=("血量", "sum"))
        )

        df_list.append(df_grouped)

    # 合并成一个大表
    df_all = pd.concat(df_list, ignore_index=True)
    print(df_all["date"].isna().sum())
    print(df_all["date"].dt.year.unique())

    print("总数据量:", df_all.shape)

    df_all.to_csv("./feature/all_data.csv", index=False)
