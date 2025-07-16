import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import yfinance as yf

# 📌 데이터 불러오기 및 전처리
def load_and_preprocess_data(tickers, start_date, end_date):
    """
    yfinance에서 데이터를 불러오고 전처리합니다.
    """
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    data.columns = ["금_가격", "은_가격"]
    df = data.reset_index()
    df.rename(columns={"Date": "날짜"}, inplace=True)
    df = df.sort_values("날짜").reset_index(drop=True)

    # 초기 데이터에 NaN이 있는 행 제거 (두 자산 모두 데이터가 있는 시점부터 시작)
    df.dropna(subset=["금_가격", "은_가격"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        print("Warning: No valid data available after removing NaNs.")
        return None

    df["금은비"] = df["금_가격"] / df["은_가격"] # 금은비 계산 추가
    return df

# 📌 전략 백테스팅 함수
def backtest_strategy(df, initial_cash, transaction_cost, strategy_type="buy_hold_gold", rebalancing_period=26, quantile_thresholds=[0.25, 0.75]):
    """
    주어진 데이터프레임과 전략 유형에 따라 백테스팅을 수행합니다.
    """
    df_strategy = df.copy()

    if df_strategy.empty:
        return pd.Series([], dtype=float) # 빈 DataFrame 처리

    if strategy_type == "buy_hold_gold":
        # 초기 투자금으로 첫날 금 가격에 맞춰 금 ETF 구매
        gold_units = initial_cash / df_strategy.loc[0, "금_가격"]
        df_strategy["금_보유_전략"] = gold_units * df_strategy["금_가격"]
        return df_strategy["금_보유_전략"]

    elif strategy_type == "periodic_rebalancing":
        strategy_col_name = "주기적_리밸런싱"
        df_strategy[strategy_col_name] = np.nan # 컬럼 초기화

        # 초기 투자금의 절반으로 금 ETF, 나머지 절반으로 은 ETF 구매
        gold_alloc   = (initial_cash / 2) / df_strategy.loc[0, "금_가격"]
        silver_alloc = (initial_cash / 2) / df_strategy.loc[0, "은_가격"]
        df_strategy.loc[0, strategy_col_name] = initial_cash # 첫날 가치 설정

        for i in range(1, len(df_strategy)):
            row = df_strategy.loc[i]
            prev_value = df_strategy.loc[i-1, strategy_col_name] # 이전 날의 포트폴리오 가치

            # 현재 시점의 포트폴리오 가치 계산
            gold_value   = gold_alloc   * row["금_가격"]
            silver_value = silver_alloc * row["은_가격"]
            total_value  = gold_value + silver_value

            # 26주마다 50:50 비율로 재조정
            if i % rebalancing_period == 0:
                 # 리밸런싱 후의 금, 은 할당량 계산
                new_gold_alloc   = (total_value / 2) / row["금_가격"]
                new_silver_alloc = (total_value / 2) / row["은_가격"]

                # 리밸런싱에 따른 거래 비용 계산
                cost = (abs(new_gold_alloc - gold_alloc)   * row["금_가격"] +
                        abs(new_silver_alloc - silver_alloc) * row["은_가격"]) * transaction_cost

                # 할당량 업데이트 및 거래 비용 차감
                gold_alloc, silver_alloc = new_gold_alloc, new_silver_alloc
                total_value -= cost

            df_strategy.loc[i, strategy_col_name] = total_value # 현재 날짜에 포트폴리오 가치 할당

        return df_strategy[strategy_col_name]

    elif strategy_type == "dynamic_rebalancing":
        strategy_col_name = "동적_리밸런싱"
        df_strategy[strategy_col_name] = np.nan # 컬럼 초기화

        q1, q3 = df_strategy["금은비"].quantile(quantile_thresholds)

        gold_alloc   = (initial_cash / 2) / df_strategy.loc[0, "금_가격"]
        silver_alloc = (initial_cash / 2) / df_strategy.loc[0, "은_가격"]

        last_check_date = df_strategy.loc[0, "날짜"]
        next_check_date = last_check_date + pd.DateOffset(months=1)

        df_strategy.loc[0, strategy_col_name] = initial_cash # 첫날 가치 설정


        for i in range(1, len(df_strategy)):
            row = df_strategy.loc[i]
            prev_value = df_strategy.loc[i-1, strategy_col_name] # 이전 날의 포트폴리오 가치


            # 현재 시점의 포트폴리오 가치 계산
            gold_value   = gold_alloc   * row["금_가격"]
            silver_value = silver_alloc * row["은_가격"]
            total_value  = gold_value + silver_value


            # 한 달 경과 시 리밸런싱 여부 판단
            current_date = row["날짜"]
            if current_date >= next_check_date:
                # 금은비가 Q3 이상이면 금 대비 은이 비싸므로 은 100%
                if row["금은비"] >= q3:
                    new_silver_alloc = total_value / row["은_가격"]
                    new_gold_alloc   = 0
                # 금은비가 Q1 이하면 금 대비 은이 싸므로 금 100%
                elif row["금은비"] <= q1:
                    new_gold_alloc   = total_value / row["금_가격"]
                    new_silver_alloc = 0
                # 금은비가 Q1과 Q3 사이면 기존 비율 유지
                else:
                    new_gold_alloc, new_silver_alloc = gold_alloc, silver_alloc

                # 리밸런싱에 따른 거래 비용 계산
                cost = (abs(new_gold_alloc - gold_alloc)   * row["금_가격"] +
                        abs(new_silver_alloc - silver_alloc) * row["은_가격"]) * transaction_cost

                # 할당량 업데이트 및 거래 비용 차감
                gold_alloc, silver_alloc = new_gold_alloc, new_silver_alloc
                total_value -= cost

                # 다음 리밸런싱 체크 날짜 업데이트
                last_check_date = current_date
                next_check_date = last_check_date + pd.DateOffset(months=1)

            df_strategy.loc[i, strategy_col_name] = total_value # 현재 날짜에 포트폴리오 가치 할당

        return df_strategy[strategy_col_name]
    else:
        return None


# 📌 지표 계산 함수
def calculate_metrics(df, initial_cash):
    """
    백테스팅 결과 데이터프레임을 바탕으로 성과 지표를 계산합니다.
    """
    metrics = {}
    for col in df.columns:
        if col != "날짜" and col != "금은비": # 전략 컬럼에 대해서만 계산
            # CAGR 계산
            first_value = df[col].iloc[0] # 첫 날의 포트폴리오 가치
            last_value = df[col].iloc[-1]
            years = (df["날짜"].iloc[-1] - df["날짜"].iloc[0]).days / 365.25
            cagr = (last_value / first_value) ** (1 / years) - 1 if first_value > 0 and years > 0 else np.nan

            # 일별 수익률 계산 (NaN 값 제거)
            returns = df[col].pct_change().dropna()

            # 샤프 지수 계산 (무위험 이자율 가정: 2%)
            risk_free_rate = 0.02
            # 일별 데이터로 샤프 지수 계산 시 연율화 계수는 np.sqrt(252)를 사용합니다.
            excess_returns = returns - (risk_free_rate / 252) # 일별 수익률 기준 연간 무위험 이자율을 일별로 변환 (약 252 거래일)
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else np.nan

            # 소르티노 지수 계산
            downside_returns = excess_returns[excess_returns < 0]
            downside_std = downside_returns.std()
            # 일별 데이터로 소르티노 지수 계산 시 연율화 계수는 np.sqrt(252)를 사용합니다.
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else np.nan

            metrics[col] = {"CAGR": cagr, "샤프지수": sharpe_ratio, "소르티노지수": sortino_ratio}

    return pd.DataFrame(metrics).T # DataFrame 형태로 반환


# 📌 메인 실행 부분
tickers = ["GLD", "SLV"]
start_date = "2004-11-18"
end_date = pd.to_datetime("today").strftime("%Y-%m-%d")
initial_cash = 100
transaction_cost = 0.002

# 데이터 로드 및 전처리
df = load_and_preprocess_data(tickers, start_date, end_date)

if df is not None:
    # 전략별 백테스팅 수행
    df["금_보유_전략"]     = backtest_strategy(df.copy(), initial_cash, transaction_cost, strategy_type="buy_hold_gold")
    df["주기적_리밸런싱"]  = backtest_strategy(df.copy(), initial_cash, transaction_cost, strategy_type="periodic_rebalancing", rebalancing_period=26)
    df["동적_리밸런싱"]    = backtest_strategy(df.copy(), initial_cash, transaction_cost, strategy_type="dynamic_rebalancing", quantile_thresholds=[0.25, 0.75])

    # 성과 지표 계산 및 출력
    df_results = calculate_metrics(df, initial_cash)
    display(df_results)

    # 📌 과적합 방지에 대한 고려 사항
    # 1. **데이터 기간:** 사용 가능한 전체 데이터를 사용하여 분석했습니다. 특정 기간에만 최적화된 전략이 아닌지 확인하기 위해 다양한 기간으로 테스트해 볼 수 있습니다.
    # 2. **변수 선택:** 현재 금은비만 사용하고 있습니다. 다른 관련 지표(예: 이동 평균, 변동성 지수 등)를 추가하여 더 견고한 전략을 만들 수 있지만, 너무 많은 변수는 과적합을 유발할 수 있습니다.
    # 3. **전략 단순성:** 현재 전략은 비교적 간단합니다. 복잡한 규칙은 과거 데이터에 과적합될 가능성이 높습니다.
    # 4. **Out-of-Sample 테스트:** 가장 좋은 과적합 방지 방법은 데이터를 훈련 세트와 테스트 세트로 나누는 것입니다. 훈련 세트로 전략을 개발하고, 테스트 세트로 성능을 검증해야 합니다. 현재 코드는 전체 데이터로 백테스팅하므로, 실제 미래 성능을 보장하지 않습니다.
    # 5. **거래 비용:** 현재 거래 비용을 고려했지만, 실제 거래 환경에서는 슬리피지(Slippage) 등 추가 비용이 발생할 수 있습니다.

else:
    print("데이터 로드 및 전처리 중 오류 발생 또는 유효한 데이터 부족.")
