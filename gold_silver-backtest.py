import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 📌 데이터 불러오기
file_path = "/content/금_은_금은비.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 📌 데이터 정리
df["날짜"] = pd.to_datetime(df["날짜"])
df = df.sort_values("날짜").reset_index(drop=True)
df.rename(columns={"종가_XAU": "금_가격", "종가_XAG": "은_가격"}, inplace=True)

initial_cash = 100          # 초기 투자금
transaction_cost = 0.002     # 거래 비용 (0.2%)

# 📌 1. 금 보유 전략 (Buy & Hold Gold)
gold_units = initial_cash / df.loc[0, "금_가격"]
df["금_보유_전략"] = gold_units * df["금_가격"]

# 📌 2. 주기적 리밸런싱 전략 (26주 50:50)
def periodic_rebalancing(df, rebalancing_period=26):
    gold_alloc   = (initial_cash / 2) / df.loc[0, "금_가격"]
    silver_alloc = (initial_cash / 2) / df.loc[0, "은_가격"]
    portfolio_values = [initial_cash]

    for i, row in df.iterrows():
        if i == 0:
            continue

        gold_value   = gold_alloc   * row["금_가격"]
        silver_value = silver_alloc * row["은_가격"]
        total_value  = gold_value + silver_value

        # 26주마다 50:50로 재조정
        if i % rebalancing_period == 0:
            new_gold_alloc   = (total_value / 2) / row["금_가격"]
            new_silver_alloc = (total_value / 2) / row["은_가격"]
            cost = (abs(new_gold_alloc - gold_alloc)   * row["금_가격"] +
                    abs(new_silver_alloc - silver_alloc) * row["은_가격"]) * transaction_cost
            gold_alloc, silver_alloc = new_gold_alloc, new_silver_alloc
            total_value -= cost

        portfolio_values.append(total_value)

    df["주기적_리밸런싱"] = portfolio_values
    return df

df = periodic_rebalancing(df)

# 📌 3. 동적 리밸런싱 전략 (월 1회 체크, q1/q3 돌파 시 100% 전환)
def dynamic_rebalancing(df):
    q1, q3 = df["금은비"].quantile([0.25, 0.75])  # 분위수 (전체 구간 기준)
    gold_alloc   = (initial_cash / 2) / df.loc[0, "금_가격"]
    silver_alloc = (initial_cash / 2) / df.loc[0, "은_가격"]

    last_check_date = df.loc[0, "날짜"]
    next_check_date = last_check_date + pd.DateOffset(months=1)

    portfolio_values = [initial_cash]

    for i, row in df.iterrows():
        if i == 0:
            continue

        gold_value   = gold_alloc   * row["금_가격"]
        silver_value = silver_alloc * row["은_가격"]
        total_value  = gold_value + silver_value

        # 한 달 경과 시 리밸런싱 여부 판단
        if row["날짜"] >= next_check_date:
            if row["금은비"] >= q3:          # 금 대비 은 비싸짐 → 은 100%
                new_silver_alloc = total_value / row["은_가격"]
                new_gold_alloc   = 0
            elif row["금은비"] <= q1:        # 금 싸짐 → 금 100%
                new_gold_alloc   = total_value / row["금_가격"]
                new_silver_alloc = 0
            else:                            # 범위 내 → 유지
                new_gold_alloc, new_silver_alloc = gold_alloc, silver_alloc

            cost = (abs(new_gold_alloc - gold_alloc)   * row["금_가격"] +
                    abs(new_silver_alloc - silver_alloc) * row["은_가격"]) * transaction_cost
            gold_alloc, silver_alloc = new_gold_alloc, new_silver_alloc
            total_value -= cost

            last_check_date = row["날짜"]
            next_check_date = last_check_date + pd.DateOffset(months=1)

        portfolio_values.append(total_value)

    df["동적_리밸런싱"] = portfolio_values
    return df

df = dynamic_rebalancing(df)

# 📌 지표 계산 함수
def calculate_cagr(df, col):
    years = (df["날짜"].iloc[-1] - df["날짜"].iloc[0]).days / 365.25
    return (df[col].iloc[-1] / initial_cash) ** (1 / years) - 1

def calculate_sharpe_ratio(df, col, risk_free_rate=0.02):
    r = df[col].pct_change().dropna()
    excess = r - (risk_free_rate / 52)
    return excess.mean() * 52 / (excess.std() * np.sqrt(52))

def calculate_sortino_ratio(df, col, risk_free_rate=0.02):
    r = df[col].pct_change().dropna()
    excess = r - (risk_free_rate / 52)
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return excess.mean() * 52 / (downside_std * np.sqrt(52))

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
    },
    "소르티노지수": {
        "금 보유 전략": calculate_sortino_ratio(df, "금_보유_전략"),
        "주기적 리밸런싱": calculate_sortino_ratio(df, "주기적_리밸런싱"),
        "동적 리밸런싱": calculate_sortino_ratio(df, "동적_리밸런싱")
    }
})

# 📌 성과 테이블 출력
display(df_results)
