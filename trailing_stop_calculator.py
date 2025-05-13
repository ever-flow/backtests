!pip install finance-datareader pandas matplotlib

import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 파라미터 설정
ticker = 'TSLA'
start_date = '2025-04-21'
stop_percentage = 13  # 트레일링 스탑 비율(%)

# 데이터 불러오기
data = fdr.DataReader(ticker, start=start_date)
if data.empty:
    raise ValueError("데이터 없음. 종목 코드 또는 시작일을 확인하세요.")

# 누적 최고가 및 트레일링 스탑 계산
data['Rolling_High'] = data['High'].cummax()
data['Trailing_Stop'] = data['Rolling_High'] * (1 - stop_percentage / 100)

# 최신 정보 출력 (한글 유지)
latest_high = data['Rolling_High'].iloc[-1]
latest_close = data['Close'].iloc[-1]
drop_pct = (latest_high - latest_close) / latest_high * 100
stop_price = latest_high * (1 - stop_percentage / 100)

print(f"\n[{ticker}] {start_date} 이후 분석 결과:")
print(f"최고가: {latest_high:.2f}")
print(f"현재 종가: {latest_close:.2f}")
print(f"최고가 대비 하락률: {drop_pct:.2f}%")
print(f"트레일링 스탑 가격: {stop_price:.2f}")
print("포지션:", "!!! 매도 !!!" if drop_pct >= stop_percentage else "보유")

# --- 시각화 (영문 표기) ---
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)
plt.plot(data.index, data['Rolling_High'], '--', label='Rolling High', linewidth=1)
plt.plot(data.index, data['Trailing_Stop'], '--', label=f'Trailing Stop ({stop_percentage}%)', linewidth=1)

# 트레일링 밴드 음영 처리
plt.fill_between(
    data.index,
    data['Trailing_Stop'],
    data['Rolling_High'],
    color='gray',
    alpha=0.3,
    label='Trailing Band'
)

plt.title(f'{ticker} Trailing Stop Visualization')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
