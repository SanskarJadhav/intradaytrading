# src/data.py

import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime

def fetch_intraday_data(ticker: str) -> pd.DataFrame:
    """
    Fetches and cleans intraday data with strict Timezone localization 
    and Regular Trading Hours (RTH) filtering.
    """
    df = yf.download(ticker, period="1mo", interval="5m", progress=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"Null data returned for {ticker}. Check ticker symbol or API connectivity.")
        
    # Handle yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.columns = [str(c).strip() for c in df.columns]
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().apply(pd.to_numeric)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
    df = df.between_time('09:30', '15:55')
    
    return df

def split_train_test(df: pd.DataFrame):
    """
    Implements a 'Last-Complete-Session' split. 
    This ensures that testing is performed on the most recent full trading day, 
    preventing look-ahead bias and incomplete sample sets.
    """
    df = df.copy()
    df['date'] = df.index.date
    unique_days = sorted(df['date'].unique())
    ny_tz = pytz.timezone('America/New_York')
    today_ny = datetime.now(ny_tz).date()
    
    if unique_days[-1] == today_ny:
        test_date = unique_days[-2]
    else:
        test_date = unique_days[-1]
    
    train_df = df[df['date'] < test_date].drop(columns=['date'])
    test_df = df[df['date'] == test_date].drop(columns=['date'])
    
    return train_df, test_df
