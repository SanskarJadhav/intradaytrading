# src/data.py

import yfinance as yf
import pandas as pd

def fetch_intraday_data(ticker: str) -> pd.DataFrame:
    # Fetch 30 days of 5-min data
    df = yf.download(ticker, period="1mo", interval="5m", progress=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data fetched for {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.columns = [str(c).strip() for c in df.columns]
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().apply(pd.to_numeric)
    
    # FIX: Convert Timezone to EST to prevent "20:50" weirdness
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
        
    # FIX: Restrict to Regular Trading Hours ONLY
    df = df.between_time('09:30', '15:55')
    
    return df

def split_train_test(df: pd.DataFrame):
    df = df.copy()
    df['date'] = df.index.date
    unique_days = sorted(df['date'].unique())
    
    # FIX: Strictly test on YESTERDAY (the second to last day in the dataset)
    # This guarantees we don't use today's incomplete data.
    yesterday = unique_days[-2]
    
    train_df = df[df['date'] < yesterday].drop(columns=['date'])
    test_df = df[df['date'] == yesterday].drop(columns=['date'])
    
    return train_df, test_df