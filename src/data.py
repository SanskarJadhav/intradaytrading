# src/data.py

import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime, time

def fetch_intraday_data(ticker: str) -> pd.DataFrame:
    # 1mo of 5m data is heavy; 
    # we fetch slightly more than TRAIN_DAYS to ensure a full buffer
    df = yf.download(ticker, period="1mo", interval="5m", progress=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data available for {ticker}.")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.columns = [str(c).strip() for c in df.columns]
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().apply(pd.to_numeric)
    
    # Standardize to NY Time
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
        
    # Strictly Regular Trading Hours (RTH)
    df = df.between_time('09:30', '15:55')
    return df

def split_train_test(df: pd.DataFrame):
    df = df.copy()
    df['date'] = df.index.date
    unique_days = sorted(df['date'].unique())
    
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    
    # Logic: Is today's session "complete" enough to test?
    # We define "Incomplete" as any time before market close (4:00 PM)
    market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if unique_days[-1] == now_ny.date() and now_ny < market_close:
        # Today is in the data but the market is still open/just closed
        # Use yesterday as the final test day to ensure "Complete Session Alpha"
        test_date = unique_days[-2]
    else:
        # Market is closed or today isn't in the data yet
        test_date = unique_days[-1]
    
    train_df = df[df['date'] < test_date].drop(columns=['date'])
    test_df = df[df['date'] == test_date].drop(columns=['date'])
    
    return train_df, test_df
