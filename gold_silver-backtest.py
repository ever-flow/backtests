import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 📌 데이터 불러오기
file_path = "/mnt/data/금_은_금은비.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 📌 데이터 정리
df["날짜"] = pd.to_datetime(df["날짜"])
df = df.sort_values("날짜").reset_index(drop=True)
df.rename(columns={"종가_XAU": "금_가격", "종가_XAG": "은_가격"}, inplace=True)
initial_cash = 100  # 초기 투자금
transaction_cost = 0.002  # 거래 비용 (0.2%)

# 📌 1. 금 보유 전략 (Buy & Hold Gold)
gold_units = initial_cash / df.loc[0, "금_가격"]
df["금_보유_전략"] = gold_units * df["금_가격"]

# 📌 2. 주기적 리밸런싱 전략 (Periodic Rebalancing)
def periodic_rebalancing(df, rebalancing_period=26):
    gold_alloc = (initial_cash / 2) / df.loc[0, "금_가격"]
    silver_alloc = (initial_cash / 2) / df.loc[0, "은_가격"]
    portfolio_values = [initial_cash]

    for i, row in df.iterrows():
        if i == 0:
            continue

        gold_value = gold_alloc * row["금_가격"]
        silver_value = silver_alloc * row["은_가격"]
        total_value = gold_value + silver_value

        if i % rebalancing_period == 0:
            new_gold_alloc = (total_value / 2) / row["금_가격"]
            new_silver_alloc = (total_value / 2) / row["은_가격"]
            cost = (abs(new_gold_alloc - gold_alloc) * row["금_가격"] +
                    abs(new_silver_alloc - silver_alloc) * row["은_가격"]) * transaction_cost
            gold_alloc, silver_alloc = new_gold_alloc, new_silver_alloc
            total_value -= cost

        portfolio_values.append(total_value)
    
    df["주기적_리밸런싱"] = portfolio_values
    return df

df = periodic_rebalancing(df)

# 📌 3. 동적 리밸런싱 전략 (Dynamic Rebalancing)
def dynamic_rebalancing(df, rebalance_interval=90):
    q1, q3 = df["금은비"].quantile([0.25, 0.75])
    gold_alloc = (initial_cash / 2) / df.loc[0, "금_가격"]
    silver_alloc = (initial_cash / 2) / df.loc[0, "은_가격"]
    last_rebalance = df.loc[0, "날짜"]
    portfolio_values = [initial_cash]

    for i, row in df.iterrows():
        if i == 0:
            continue

        gold_value = gold_alloc * row["금_가격"]
        silver_value = silver_alloc * row["은_가격"]
        total_value = gold_value + silver_value

        if (row["날짜"] - last_rebalance).days >= rebalance_interval:
            if row["금은비"] >= q3:
                new_silver_alloc = total_value / row["은_가격"]
                new_gold_alloc = 0
            elif row["금은비"] <= q1:
                new_gold_alloc = total_value / row["금_가격"]
                new_silver_alloc = 0
            else:
                new_gold_alloc, new_silver_alloc = gold_alloc, silver_alloc

            cost = (abs(new_gold_alloc - gold_alloc) * row["금_가격"] +
                    abs(new_silver_alloc - silver_alloc) * row["은_가격"]) * transaction_cost
            gold_alloc, silver_alloc = new_gold_alloc, new_silver_alloc
            total_value -= cost
            last_rebalance = row["날짜"]

        portfolio_values.append(total_value)
    
    df["동적_리밸런싱"] = portfolio_values
    return df

df = dynamic_rebalancing(df)

# 📌 CAGR & 샤프지수 계산 함수
def calculate_cagr(df, strategy_col):
    years = len(df) / 52
    return (df[strategy_col].iloc[-1] / initial_cash) ** (1 / years) - 1

def calculate_sharpe_ratio(df, strategy_col, risk_free_rate=0.02):
    weekly_returns = df[strategy_col].pct_change().dropna()
    excess_returns = weekly_returns - (risk_free_rate / 52)
    return excess_returns.mean() * 52 / (excess_returns.std() * np.sqrt(52))

# 📌 성과 비교
df_results = pd.DataFrame({
    "CAGR": {
        "금 보유 전략": calculate_cagr(df, "금_보유_전략"),
        "주기적 리밸런싱": calculate_cagr(df, "주기적_리밸런싱"),
        "동적 리밸런싱": calculate_cagr(df, "동적_리밸런싱")
    },
    "샤프지수": {
        "금 보유 전략": calculate_sharpe_ratio(df, "금_보유_전략"),
        "주기적 리밸런싱": calculate_sharpe_ratio(df, "주기적_리밸런싱"),
        "동적 리밸런싱": calculate_sharpe_ratio(df, "동적_리밸런싱")
    }
})

# 📌 성과 테이블 출력
import ace_tools as tools
tools.display_dataframe_to_user(name="거래비용 반영된 투자 전략 비교", dataframe=df_results)
