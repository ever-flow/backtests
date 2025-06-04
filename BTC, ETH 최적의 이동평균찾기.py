!pip install koreanize-matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm import tqdm
import warnings
import koreanize_matplotlib
# from scipy.stats import skew # skew는 현재 코드에서 사용되지 않으므로 제거해도 무방합니다.
import datetime
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

def fetch_crypto_data():
    """
    BTC-USD, ETH-USD의 종가 데이터를 수집하여 하나의 DataFrame으로 반환
    """
    try:
        # Download data
        btc = yf.download("BTC-USD", start="2016-01-01", end="2025-05-25", progress=False)
        eth = yf.download("ETH-USD", start="2017-01-10", end="2025-05-25", progress=False)

        # Extract Close prices
        btc_close = btc["Close"]
        eth_close = eth["Close"]

        # Create aligned DataFrame
        df = pd.DataFrame()
        df["BTC"] = btc_close
        df["ETH"] = eth_close

        return df.dropna()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_cagr(cumulative_returns_factor, num_years):
    """
    누적 수익률 팩터(예: 최종자산/초기자산)와 투자 기간(년)을 이용하여 CAGR 계산
    """
    if cumulative_returns_factor is None or num_years == 0 or cumulative_returns_factor <= 0: # 0 또는 음수 누적수익률 팩터 방지
        return -1.0 if cumulative_returns_factor == 0 else 0.0 # 전액손실시 -100%
    return (cumulative_returns_factor ** (1 / num_years)) - 1

def calculate_sortino_ratio(net_returns, risk_free_rate=0.0):
    """
    하방 변동성을 고려한 Sortino Ratio 계산
    """
    negative_returns = net_returns[net_returns < risk_free_rate] # 무위험수익률보다 낮은 수익률을 하방으로 간주
    if len(negative_returns) == 0:
        mean_annual_return = net_returns.mean() * 365
        if mean_annual_return > risk_free_rate:
            return np.inf
        return 0.0

    downside_deviation_annual = negative_returns.std() * np.sqrt(365)
    if downside_deviation_annual == 0:
        mean_annual_return = net_returns.mean() * 365
        if mean_annual_return > risk_free_rate:
             return np.inf
        return 0.0

    annualized_mean_return = net_returns.mean() * 365
    return (annualized_mean_return - risk_free_rate) / downside_deviation_annual

def calculate_mdd(cumulative_series):
    """주어진 누적 수익률 시리즈에 대한 최대 낙폭(MDD)을 계산합니다."""
    if cumulative_series.empty or len(cumulative_series) < 2:
        return 0.0
    cummax = cumulative_series.cummax()
    drawdown = (cumulative_series / cummax) - 1
    return drawdown.min()

def evaluate_strategy(price_series, ma_window, fee=0.0025):
    """
    이동평균(ma_window) 대비 가격 위치를 이용한 트렌드 추종 전략 평가
    """
    if len(price_series) < ma_window :
        return {
            "window": ma_window, "sharpe": 0.0, "sortino": np.nan, "combined_sortino": np.nan,
            "cagr": 0.0, "final_value": 1.0, "drawdown": 0.0, "volatility": 0.0,
            "cumulative_series": pd.Series([1.0], index=[price_series.index[0] if not price_series.empty else pd.Timestamp('1970-01-01')])
        }

    ma = price_series.rolling(window=ma_window, min_periods=ma_window).mean()
    valid_indices = ma.dropna().index
    if len(valid_indices) < 2:
        return {
            "window": ma_window, "sharpe": 0.0, "sortino": np.nan, "combined_sortino": np.nan, "cagr": 0.0,
            "final_value": 1.0, "drawdown": 0.0, "volatility": 0.0,
            "cumulative_series": pd.Series([1.0], index=[price_series.index[0] if not price_series.empty else pd.Timestamp('1970-01-01')])
        }

    price_series_eval = price_series.loc[valid_indices]
    ma_eval = ma.loc[valid_indices]

    signal = (price_series_eval > ma_eval).astype(int)
    position = signal.shift(1).fillna(0)
    returns = price_series_eval.pct_change().fillna(0)
    trades = position.diff().fillna(0).abs()
    net_returns = position * returns - trades * fee
    cumulative = (1 + net_returns).cumprod()

    if cumulative.empty or cumulative.iloc[-1] <= 0: # 0 이하 자본 처리
        final_value = cumulative.iloc[-1] if not cumulative.empty else 0.0
        cagr_val = -1.0 if final_value == 0 else calculate_cagr(final_value, 1/365.25) # 임시 num_years
        mdd_val = calculate_mdd(cumulative) if not cumulative.empty else (-1.0 if final_value == 0 else 0.0)

        return {
            "window": ma_window, "sharpe": 0.0, "sortino": np.nan, "combined_sortino": np.nan,
            "cagr": cagr_val,
            "final_value": final_value,
            "drawdown": mdd_val,
            "volatility": net_returns.std() * np.sqrt(365) if len(net_returns) > 1 else 0.0,
            "cumulative_series": cumulative if not cumulative.empty else pd.Series([final_value if final_value > 0 else 1.0], index=[price_series_eval.index[0] if not price_series_eval.empty else pd.Timestamp('1970-01-01')])
        }

    num_years = max((price_series_eval.index[-1] - price_series_eval.index[0]).days / 365.25, 1/365.25)
    cagr = calculate_cagr(cumulative.iloc[-1], num_years)
    std_dev = net_returns.std()
    overall_sharpe = (net_returns.mean() * 365) / (std_dev * np.sqrt(365)) if std_dev > 0 else 0.0
    overall_sortino = calculate_sortino_ratio(net_returns)

    recent_sortino = np.nan
    if len(net_returns) >= 1000:
        recent_sortino = calculate_sortino_ratio(net_returns.iloc[-1000:])

    if not np.isnan(overall_sortino) and not np.isnan(recent_sortino):
        combined_sortino = 0.7 * overall_sortino + 0.3 * recent_sortino
    elif not np.isnan(overall_sortino):
        combined_sortino = overall_sortino
    elif not np.isnan(recent_sortino):
        combined_sortino = recent_sortino
    else:
        combined_sortino = np.nan

    max_dd = calculate_mdd(cumulative)
    vol = std_dev * np.sqrt(365) if std_dev > 0 else 0.0

    return {
        "window": ma_window, "sharpe": overall_sharpe, "sortino": overall_sortino,
        "combined_sortino": combined_sortino, "cagr": cagr,
        "final_value": cumulative.iloc[-1], "drawdown": max_dd, "volatility": vol,
        "cumulative_series": cumulative
    }

def evaluate_rebalancing_strategy(data, ma_window, rebalance_freq='M', weight_btc=0.5, weight_eth=0.5, fee=0.0025):
    if data.empty or len(data) < ma_window:
        start_index_for_default = data.index[0] if not data.empty else pd.Timestamp('1970-01-01')
        return {
            "window": ma_window, "sharpe": 0.0, "sortino": np.nan, "combined_sortino": np.nan, "cagr": 0.0,
            "final_value": 1.0, "drawdown": 0.0, "volatility": 0.0,
            "cumulative_series": pd.Series([1.0], index=[start_index_for_default])
        }

    btc_ma = data['BTC'].rolling(window=ma_window, min_periods=ma_window).mean()
    eth_ma = data['ETH'].rolling(window=ma_window, min_periods=ma_window).mean()

    valid_btc_idx = btc_ma.dropna().index
    valid_eth_idx = eth_ma.dropna().index
    
    if valid_btc_idx.empty or valid_eth_idx.empty: # 둘 중 하나라도 MA 계산 불가시
        return {
            "window": ma_window, "sharpe": 0.0, "sortino": np.nan, "combined_sortino": np.nan, "cagr": 0.0,
            "final_value": 1.0, "drawdown": 0.0, "volatility": 0.0,
            "cumulative_series": pd.Series([1.0], index=[data.index[0] if not data.empty else pd.Timestamp('1970-01-01')])
        }
    start_idx = max(valid_btc_idx[0], valid_eth_idx[0])
    
    if len(data.loc[start_idx:]) < 2:
         return {
            "window": ma_window, "sharpe": 0.0, "sortino": np.nan, "combined_sortino": np.nan, "cagr": 0.0,
            "final_value": 1.0, "drawdown": 0.0, "volatility": 0.0,
            "cumulative_series": pd.Series([1.0], index=[data.index[0] if not data.empty else pd.Timestamp('1970-01-01')])
        }

    eval_data = data.loc[start_idx:].copy() # 여기서도 .copy()
    btc_ma_eval = btc_ma.loc[start_idx:]
    eth_ma_eval = eth_ma.loc[start_idx:]

    btc_signal = (eval_data['BTC'] > btc_ma_eval).astype(int)
    eth_signal = (eval_data['ETH'] > eth_ma_eval).astype(int)
    btc_position = btc_signal.shift(1).fillna(0)
    eth_position = eth_signal.shift(1).fillna(0)
    
    btc_returns_daily = eval_data['BTC'].pct_change().fillna(0)
    eth_returns_daily = eval_data['ETH'].pct_change().fillna(0)
    
    btc_trades = btc_position.diff().fillna(0).abs()
    eth_trades = eth_position.diff().fillna(0).abs()
    
    btc_strategy_returns = btc_position * btc_returns_daily - btc_trades * fee
    eth_strategy_returns = eth_position * eth_returns_daily - eth_trades * fee

    eval_data['month'] = eval_data.index.to_period(rebalance_freq)
    eval_data['rebalance_signal'] = eval_data['month'].ne(eval_data['month'].shift(1)).astype(int)
    if not eval_data.empty:
        eval_data.iloc[0, eval_data.columns.get_loc('rebalance_signal')] = 0


    portfolio_value = pd.Series(index=eval_data.index, dtype=float)
    if eval_data.empty: # eval_data가 비어있을 극단적 경우 대비
        return {
            "window": ma_window, "sharpe": 0.0, "sortino": np.nan, "combined_sortino": np.nan, "cagr": 0.0,
            "final_value": 1.0, "drawdown": 0.0, "volatility": 0.0,
            "cumulative_series": pd.Series([1.0], index=[data.index[0] if not data.empty else pd.Timestamp('1970-01-01')])
        }
    portfolio_value.iloc[0] = 1.0

    current_btc_weight = weight_btc
    current_eth_weight = weight_eth

    for i in range(1, len(eval_data)):
        prev_total_value = portfolio_value.iloc[i-1]
        if prev_total_value <= 0: # 이전 가치가 0 이하면 더 이상 진행 불가
            portfolio_value.iloc[i:] = prev_total_value 
            break

        btc_value_after_growth = prev_total_value * current_btc_weight * (1 + btc_strategy_returns.iloc[i])
        eth_value_after_growth = prev_total_value * current_eth_weight * (1 + eth_strategy_returns.iloc[i])
        current_total_value_before_rebalance = btc_value_after_growth + eth_value_after_growth

        if current_total_value_before_rebalance <= 0:
            portfolio_value.iloc[i:] = current_total_value_before_rebalance
            break
            
        if eval_data['rebalance_signal'].iloc[i] == 1:
            temp_btc_weight = btc_value_after_growth / current_total_value_before_rebalance
            temp_eth_weight = eth_value_after_growth / current_total_value_before_rebalance
            rebalancing_cost = (abs(temp_btc_weight - weight_btc) + abs(temp_eth_weight - weight_eth)) * fee * current_total_value_before_rebalance
            
            portfolio_value.iloc[i] = current_total_value_before_rebalance - rebalancing_cost
            current_btc_weight = weight_btc
            current_eth_weight = weight_eth
        else:
            portfolio_value.iloc[i] = current_total_value_before_rebalance
            current_btc_weight = btc_value_after_growth / current_total_value_before_rebalance
            current_eth_weight = eth_value_after_growth / current_total_value_before_rebalance
            
    portfolio_value = portfolio_value.fillna(method='ffill').fillna(0) # 전파 후 0으로 채움

    net_returns_portfolio = portfolio_value.pct_change().fillna(0)
    cumulative_portfolio = portfolio_value

    if cumulative_portfolio.empty or cumulative_portfolio.iloc[-1] <= 0:
        final_value = cumulative_portfolio.iloc[-1] if not cumulative_portfolio.empty else 0.0
        cagr_val = -1.0 if final_value == 0 else calculate_cagr(final_value, 1/365.25)
        mdd_val = calculate_mdd(cumulative_portfolio) if not cumulative_portfolio.empty else (-1.0 if final_value == 0 else 0.0)
        return {
            "window": ma_window, "sharpe": 0.0, "sortino": np.nan, "combined_sortino": np.nan,
            "cagr": cagr_val, "final_value": final_value, "drawdown": mdd_val,
            "volatility": net_returns_portfolio.std() * np.sqrt(365) if len(net_returns_portfolio) > 1 else 0.0,
            "cumulative_series": cumulative_portfolio if not cumulative_portfolio.empty else pd.Series([final_value if final_value > 0 else 1.0], index=[eval_data.index[0] if not eval_data.empty else pd.Timestamp('1970-01-01')])
        }

    num_years = max((eval_data.index[-1] - eval_data.index[0]).days / 365.25, 1/365.25)
    cagr = calculate_cagr(cumulative_portfolio.iloc[-1], num_years)
    std_dev = net_returns_portfolio.std()
    overall_sharpe = (net_returns_portfolio.mean() * 365) / (std_dev * np.sqrt(365)) if std_dev > 0 else 0.0
    overall_sortino = calculate_sortino_ratio(net_returns_portfolio)

    recent_sortino = np.nan
    if len(net_returns_portfolio) >= 1000:
        recent_sortino = calculate_sortino_ratio(net_returns_portfolio.iloc[-1000:])

    if not np.isnan(overall_sortino) and not np.isnan(recent_sortino):
        combined_sortino = 0.7 * overall_sortino + 0.3 * recent_sortino
    elif not np.isnan(overall_sortino):
        combined_sortino = overall_sortino
    elif not np.isnan(recent_sortino):
        combined_sortino = recent_sortino
    else:
        combined_sortino = np.nan
        
    max_dd = calculate_mdd(cumulative_portfolio)
    vol = std_dev * np.sqrt(365) if std_dev > 0 else 0.0

    return {
        "window": ma_window, "sharpe": overall_sharpe, "sortino": overall_sortino,
        "combined_sortino": combined_sortino, "cagr": cagr,
        "final_value": cumulative_portfolio.iloc[-1], "drawdown": max_dd, "volatility": vol,
        "cumulative_series": cumulative_portfolio
    }

def run_backtest():
    print("🕒 암호화폐 데이터 수집 중...")
    data = fetch_crypto_data()

    if data.empty:
        print("❌ 데이터 수집 실패. 프로그램을 종료합니다.")
        return

    print(f"✅ 데이터 수집 완료: {data.index[0].date()} ~ {data.index[-1].date()}")
    windows = list(range(10, 201, 10))

    print("\n📈 BTC 전략 평가:")
    btc_results = []
    if 'BTC' in data and not data['BTC'].empty:
        for w in tqdm(windows, desc="BTC MA windows"):
            btc_results.append(evaluate_strategy(data["BTC"].copy(), w))

    print("\n📈 ETH 전략 평가:")
    eth_results = []
    if 'ETH' in data and not data['ETH'].empty:
        for w in tqdm(windows, desc="ETH MA windows"):
            eth_results.append(evaluate_strategy(data["ETH"].copy(), w))

    print("\n📈 월별 50:50 리밸런싱 전략 평가:")
    rebal_5050 = []
    for w in tqdm(windows, desc="Rebalancing 50:50 MA windows"):
        rebal_5050.append(evaluate_rebalancing_strategy(data.copy(), w, rebalance_freq='M', weight_btc=0.5, weight_eth=0.5))

    print("\n📈 월별 60:40 리밸런싱 전략 평가:")
    rebal_6040 = []
    for w in tqdm(windows, desc="Rebalancing 60:40 MA windows"):
        rebal_6040.append(evaluate_rebalancing_strategy(data.copy(), w, rebalance_freq='M', weight_btc=0.6, weight_eth=0.4))

    btc_df = pd.DataFrame(btc_results).sort_values("combined_sortino", ascending=False).reset_index(drop=True) if btc_results else pd.DataFrame()
    eth_df = pd.DataFrame(eth_results).sort_values("combined_sortino", ascending=False).reset_index(drop=True) if eth_results else pd.DataFrame()
    rebal_5050_df = pd.DataFrame(rebal_5050).sort_values("combined_sortino", ascending=False).reset_index(drop=True) if rebal_5050 else pd.DataFrame()
    rebal_6040_df = pd.DataFrame(rebal_6040).sort_values("combined_sortino", ascending=False).reset_index(drop=True) if rebal_6040 else pd.DataFrame()

    best_strategies = {}
    if not btc_df.empty: best_strategies["BTC"] = btc_df.loc[0]
    if not eth_df.empty: best_strategies["ETH"] = eth_df.loc[0]
    if not rebal_5050_df.empty: best_strategies["Rebal 50:50"] = rebal_5050_df.loc[0]
    if not rebal_6040_df.empty: best_strategies["Rebal 60:40"] = rebal_6040_df.loc[0]

    if not best_strategies:
        print("\n❌ 유효한 백테스팅 결과가 없습니다.")
        return

    plt.figure(figsize=(12,6))
    legend_labels = []
    if "BTC" in best_strategies:
        best_btc = best_strategies["BTC"]
        if not best_btc["cumulative_series"].empty :
            plt.plot(best_btc["cumulative_series"], label=f"BTC Best MA={int(best_btc['window'])}일")
            legend_labels.append(f"BTC Best MA={int(best_btc['window'])}일")
    if "ETH" in best_strategies:
        best_eth = best_strategies["ETH"]
        if not best_eth["cumulative_series"].empty :
            plt.plot(best_eth["cumulative_series"], label=f"ETH Best MA={int(best_eth['window'])}일")
            legend_labels.append(f"ETH Best MA={int(best_eth['window'])}일")
    if "Rebal 50:50" in best_strategies:
        best_r5050 = best_strategies["Rebal 50:50"]
        if not best_r5050["cumulative_series"].empty :
            plt.plot(best_r5050["cumulative_series"], label=f"50:50 Best MA={int(best_r5050['window'])}일")
            legend_labels.append(f"50:50 Best MA={int(best_r5050['window'])}일")
    if "Rebal 60:40" in best_strategies:
        best_r6040 = best_strategies["Rebal 60:40"]
        if not best_r6040["cumulative_series"].empty :
            plt.plot(best_r6040["cumulative_series"], label=f"60:40 Best MA={int(best_r6040['window'])}일")
            legend_labels.append(f"60:40 Best MA={int(best_r6040['window'])}일")
    
    if legend_labels: # 플롯할 데이터가 있는 경우에만 제목 및 범례 표시
        plt.title("최적 MA 전략 누적 수익률")
        plt.xlabel("Date"); plt.ylabel("Cumulative Return")
        plt.legend(); plt.grid(alpha=0.3)
        plt.show()
    else:
        print("\n📈 시각화할 누적 수익률 데이터가 없습니다.")


    print("\n=== 최적 전략 요약 (Combined Sortino 기준) ===")
    strategy_names_map = {
        "BTC": "🐂 BTC", "ETH": "🐂 ETH",
        "Rebal 50:50": "⚖️ 50:50 리밸런싱", "Rebal 60:40": "⚖️ 60:40 리밸런싱"
    }

    for key, best_strat_data in best_strategies.items():
        name = strategy_names_map.get(key, key)
        series = best_strat_data["cumulative_series"]
        
        if series.empty or len(series) < 2:
            print(f"{name} 최적 MA: {int(best_strat_data.get('window', 0))}일 - 데이터 부족으로 상세 분석 불가")
            continue

        last_date = series.index[-1]
        total_return_factor = series.iloc[-1]
        total_return_pct = (total_return_factor - 1) * 100
        total_cagr = best_strat_data["cagr"] * 100
        total_mdd = best_strat_data['drawdown'] * 100

        print(f"{name} 최적 MA (Combined Sortino): {int(best_strat_data['window'])}일")
        print(f"  - 전체 기간: 수익률 {total_return_pct:.1f}%, CAGR {total_cagr:.2f}%, MDD {total_mdd:.1f}%")
        
        # 지난 5년 성과
        ret5, cagr5, mdd5 = np.nan, np.nan, np.nan
        start_5y = last_date - relativedelta(years=5)
        s5 = series[series.index >= start_5y]
        if len(s5) > 1:
            ret5_factor = s5.iloc[-1] / s5.iloc[0]
            ret5 = (ret5_factor - 1) * 100
            years5 = max((s5.index[-1] - s5.index[0]).days / 365.25, 1/365.25)
            if years5 > 0 : cagr5 = calculate_cagr(ret5_factor, years5) * 100
            mdd5 = calculate_mdd(s5) * 100
            print(f"  - 지난 5년: 수익률 {ret5:.1f}%, CAGR {cagr5:.2f}%, MDD {mdd5:.1f}%")
        else:
            print("  - 지난 5년: 데이터 부족 또는 계산 불가")
            
        # 지난 1년 성과
        ret1, cagr1, mdd1 = np.nan, np.nan, np.nan
        start_1y = last_date - relativedelta(years=1)
        s1 = series[series.index >= start_1y]
        if len(s1) > 1:
            ret1_factor = s1.iloc[-1] / s1.iloc[0]
            ret1 = (ret1_factor - 1) * 100
            years1 = max((s1.index[-1] - s1.index[0]).days / 365.25, 1/365.25)
            if years1 > 0 : cagr1 = calculate_cagr(ret1_factor, years1) * 100
            mdd1 = calculate_mdd(s1) * 100
            print(f"  - 지난 1년: 수익률 {ret1:.1f}%, CAGR {cagr1:.2f}%, MDD {mdd1:.1f}%")
        else:
            print("  - 지난 1년: 데이터 부족 또는 계산 불가")
            
        print(f"  - 지표: 결합 소르티노 {best_strat_data['combined_sortino']:.3f}, 샤프 {best_strat_data['sharpe']:.3f}, 소르티노 {best_strat_data['sortino']:.3f}\n")

if __name__ == "__main__":
    run_backtest()
