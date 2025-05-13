# 시각화 추가
# 필요한 라이브러리 설치
!pip install bayesian-optimization
!pip install yfinance

# 경고 메시지 억제
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 라이브러리 임포트
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from bayes_opt import BayesianOptimization
from IPython.display import display

# 종목 리스트 정의
tickers = ["CRWD", "MELI", "SMCI", "PLTR", "IONQ", "TMDX", "LEU", "CRSP", "HIMS", "RKLB"]

all_results = []

# 백테스트 함수 (거래수수료 0.002 적용, 트레일링 스톱 포함)
def backtest_strategy(price, signal, sma_filter=None, tc=0.002, initial_capital=1, trailing_stop_pct=0):
    position = 0.0
    cash = float(initial_capital)
    equity_curve = np.zeros(len(price))
    buy_points, sell_points = [], []
    high_since_buy = 0.0

    for i in range(len(price)):
        if i == 0:
            equity_curve[i] = cash
            continue

        if sma_filter is not None and float(price.iloc[i]) < float(sma_filter.values[i]):
            if position > 0:
                proceeds = position * float(price.iloc[i])
                cost = proceeds * tc
                cash += proceeds - cost
                position = 0.0
                sell_points.append((price.index[i], float(price.iloc[i])))
            equity_curve[i] = cash
            continue

        if position > 0 and trailing_stop_pct > 0:
            if float(price.iloc[i]) > high_since_buy:
                high_since_buy = float(price.iloc[i])
            elif float(price.iloc[i]) < high_since_buy * (1 - trailing_stop_pct / 100):
                proceeds = position * float(price.iloc[i])
                cost = proceeds * tc
                cash += proceeds - cost
                position = 0.0
                sell_points.append((price.index[i], float(price.iloc[i])))
                equity_curve[i] = cash
                continue

        if signal.values[i] == 1 and position == 0:
            cost = cash * tc
            cash -= cost
            position = cash / float(price.iloc[i])
            cash = 0.0
            high_since_buy = float(price.iloc[i])
            buy_points.append((price.index[i], float(price.iloc[i])))
        elif signal.values[i] == -1 and position > 0:
            proceeds = position * float(price.iloc[i])
            cost = proceeds * tc
            cash += proceeds - cost
            position = 0.0
            sell_points.append((price.index[i], float(price.iloc[i])))

        equity_curve[i] = cash + position * float(price.iloc[i])

    equity_curve = pd.Series(equity_curve, index=price.index)
    daily_returns = equity_curve.pct_change().dropna()
    period = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (equity_curve.iloc[-1] / initial_capital) ** (1/period) - 1 if period > 0 else 0
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    mdd = abs(drawdown.min())

    return equity_curve, cagr, mdd, sharpe, period, position, cash, buy_points, sell_points

# MACD 및 SIGNAL 계산 함수
def generate_filtered_macd_signals(data, short_span, long_span, signal_span):
    ema_short = data['Close'].ewm(span=short_span, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_span, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_span, adjust=False).mean()

    signal = np.zeros(len(data))
    for i in range(1, len(data)):
        if ema_short.values[i] > ema_long.values[i]:
            if (macd.values[i] > macd_signal.values[i]) and (macd.values[i-1] <= macd_signal.values[i-1]):
                signal[i] = 1
            elif (macd.values[i] < macd_signal.values[i]) and (macd.values[i-1] >= macd_signal.values[i-1]):
                signal[i] = -1
    return pd.Series(signal, index=data.index)

# Buy and Hold 전략 평가 함수
def buy_and_hold(price, initial_capital=1, tc=0.002):
    cash = initial_capital
    cost = cash * tc
    cash -= cost
    position = cash / float(price.iloc[0])
    cash = 0.0
    equity_curve = [initial_capital]
    for i in range(1, len(price)):
        equity_curve.append(position * float(price.iloc[i]))
    equity_curve = pd.Series(equity_curve, index=price.index)
    daily_returns = equity_curve.pct_change().dropna()
    period = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (equity_curve.iloc[-1] / initial_capital) ** (1/period) - 1 if period > 0 else 0
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    mdd = abs(drawdown.min())
    return equity_curve, cagr, mdd, sharpe, period

# 각 종목별 최적화 및 시각화
for ticker in tickers:
    print(f"\n{ticker} 분석 중...")
    today = date.today()
    twenty_years_ago = today - timedelta(days=int(20 * 365.25))

    df = yf.download(ticker, start=twenty_years_ago, end=today, auto_adjust=True)
    if df.empty:
        print(f"{ticker}에 대한 20년 데이터가 없습니다. 가능한 모든 데이터를 다운로드합니다.")
        df = yf.download(ticker, period="max", auto_adjust=True)
    df.sort_index(inplace=True)
    df.rename(columns={'Volume': 'Vol.'}, inplace=True)

    three_years_ago = pd.Timestamp(today - timedelta(days=int(3 * 365.25)))
    df_recent = df[df.index >= three_years_ago]

    def objective(short_span, delta, signal_span, sma_window, trailing_stop_pct):
        short_span = int(round(short_span / 5) * 5)
        delta = int(round(delta / 5) * 5)
        signal_span = int(round(signal_span / 5) * 5)
        sma_window = int(round(sma_window / 5) * 5)
        trailing_stop_pct = int(round(trailing_stop_pct / 5) * 5)
        long_span = short_span + delta

        if short_span >= long_span or sma_window < 5:
            return -100

        sma_filter_overall = df['Close'].rolling(window=sma_window).mean()
        signals_overall = generate_filtered_macd_signals(df, short_span, long_span, signal_span)
        _, _, _, sharpe_overall, _, _, _, _, _ = backtest_strategy(
            df['Close'], signals_overall, sma_filter=sma_filter_overall, tc=0.002, initial_capital=1, trailing_stop_pct=trailing_stop_pct
        )

        sma_filter_recent = df_recent['Close'].rolling(window=sma_window).mean()
        signals_recent = generate_filtered_macd_signals(df_recent, short_span, long_span, signal_span)
        _, _, _, sharpe_recent, _, _, _, _, _ = backtest_strategy(
            df_recent['Close'], signals_recent, sma_filter=sma_filter_recent, tc=0.002, initial_capital=10000, trailing_stop_pct=trailing_stop_pct
        )

        avg_sharpe = (sharpe_overall + sharpe_recent) / 2.0
        return avg_sharpe

    seeds = [1, 2, 3]
    best_run = None
    for seed in seeds:
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={
                'short_span': (5, 55),
                'delta': (5, 210),
                'signal_span': (5, 35),
                'sma_window': (10, 300),
                'trailing_stop_pct': (0, 40)
            },
            random_state=seed,
            verbose=2
        )
        optimizer.maximize(init_points=3, n_iter=80)
        if best_run is None or optimizer.max['target'] > best_run['target']:
            best_run = optimizer.max

    best_params = best_run['params']
    best_short_span = int(round(best_params['short_span'] / 5) * 5)
    best_delta = int(round(best_params['delta'] / 5) * 5)
    best_signal_span = int(round(best_params['signal_span'] / 5) * 5)
    best_sma_window = int(round(best_params['sma_window'] / 5) * 5)
    best_trailing_stop_pct = int(round(best_params['trailing_stop_pct'] / 5) * 5)
    best_long_span = best_short_span + best_delta

    print(f"최적 파라미터: Short EMA = {best_short_span}, Long EMA = {best_long_span}, Signal EMA = {best_signal_span}, SMA Window = {best_sma_window}, Trailing Stop (%) = {best_trailing_stop_pct}")

    # 전체 데이터에 대해 최적 전략 평가 및 시각화 데이터 준비
    sma_filter_best = df['Close'].rolling(window=best_sma_window).mean()
    signals_best = generate_filtered_macd_signals(df, best_short_span, best_long_span, best_signal_span)
    equity_curve, cagr, mdd, sharpe, period, final_position, final_cash, buy_points, sell_points = backtest_strategy(
        df['Close'], signals_best, sma_filter=sma_filter_best, tc=0.002, initial_capital=1, trailing_stop_pct=best_trailing_stop_pct
    )

    # 최근 3년 데이터에 대해 평가
    sma_filter_recent = df_recent['Close'].rolling(window=best_sma_window).mean()
    signals_recent = generate_filtered_macd_signals(df_recent, best_short_span, best_long_span, best_signal_span)
    _, cagr_recent, mdd_recent, sharpe_recent, period_recent, _, _, _, _ = backtest_strategy(
        df_recent['Close'], signals_recent, sma_filter=sma_filter_recent, tc=0.002, initial_capital=10000, trailing_stop_pct=best_trailing_stop_pct
    )

    # Buy and Hold 전략 평가
    equity_curve_bh, cagr_bh, mdd_bh, sharpe_bh, period_bh = buy_and_hold(df['Close'], initial_capital=1, tc=0.002)

    # 시각화: 최적 전략의 매수/매도 시점
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Close Price', alpha=0.5)
    plt.plot(sma_filter_best, label=f'SMA {best_sma_window}', alpha=0.5)
    if buy_points:
        buy_dates, buy_prices = zip(*buy_points)
        plt.scatter(buy_dates, buy_prices, marker='^', color='g', label='Buy', alpha=1)
    if sell_points:
        sell_dates, sell_prices = zip(*sell_points)
        plt.scatter(sell_dates, sell_prices, marker='v', color='r', label='Sell', alpha=1)
    plt.title(f'{ticker} - Buy and Sell Points with Optimal Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # 시각화: 자본 곡선 비교
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve, label='Strategy Equity Curve')
    plt.plot(equity_curve_bh, label='Buy and Hold Equity Curve')
    plt.title(f'{ticker} - Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.show()

    all_results.append({
        'Ticker': ticker,
        'Short EMA': best_short_span,
        'Long EMA': best_long_span,
        'Signal EMA': best_signal_span,
        'SMA Window': best_sma_window,
        'Trailing Stop (%)': best_trailing_stop_pct,
        'CAGR (%)_Overall': round(cagr * 100, 2),
        'MDD (%)_Overall': round(mdd * 100, 2),
        'Sharpe_Overall': round(sharpe, 2),
        'CAGR (%)_Recent': round(cagr_recent * 100, 2),
        'MDD (%)_Recent': round(mdd_recent * 100, 2),
        'Sharpe_Recent': round(sharpe_recent, 2),
        'Sharpe_Average': round((sharpe + sharpe_recent) / 2, 2),
        'CAGR (%)_BuyHold': round(cagr_bh * 100, 2),
        'MDD (%)_BuyHold': round(mdd_bh * 100, 2),
        'Sharpe_BuyHold': round(sharpe_bh, 2),
        'Data Period (Years)_Overall': round(period, 2)
    })

# 결과 DataFrame 생성 및 가중치 계산
all_results_df = pd.DataFrame(all_results)
all_results_df['Weighted_Sharpe'] = all_results_df['Sharpe_Average'] * np.log(all_results_df['Data Period (Years)_Overall'])
all_results_df = all_results_df.sort_values(by='Weighted_Sharpe', ascending=False).reset_index(drop=True)

# 최종 결과 출력
print("\n모든 종목의 최적 전략 (Weighted_Sharpe 기준 정렬):")
display_columns = [
    'Ticker', 'Short EMA', 'Long EMA', 'Signal EMA', 'SMA Window', 'Trailing Stop (%)',
    'CAGR (%)_Overall', 'MDD (%)_Overall', 'Sharpe_Overall',
    'CAGR (%)_Recent', 'MDD (%)_Recent', 'Sharpe_Recent', 'Sharpe_Average',
    'CAGR (%)_BuyHold', 'MDD (%)_BuyHold', 'Sharpe_BuyHold',
    'Data Period (Years)_Overall', 'Weighted_Sharpe'
]
display(all_results_df[display_columns])
