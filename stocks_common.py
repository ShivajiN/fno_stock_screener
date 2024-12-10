import pandas as pd
import numpy as np
import yfinance as yf
import yfinance.shared as shared
import talib
from datetime import datetime, timedelta
import ta  # Technical Analysis Library
import os
#from datetime import datetime, time
import inspect

def current_line():
    return inspect.currentframe().f_lineno

def weekdays():
    today = datetime.today()

    # Check if today is not Saturday or Sunday
    if today.weekday() not in (5, 6):  # 5: Friday, 6: Saturday
        #print("Today is not a weekend day.")
        return True
    
    return False

def detect_insidebar(df):
    return (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))


# Function to detect inside bar and breakouts
def detect_inside_bar(df):
    df['Inside_Bar'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))


    breakout_up = (df['Close'] > df['High'].shift(1)) & df['Inside_Bar']
    breakout_down = (df['Close'] < df['Low'].shift(1)) & df['Inside_Bar']
    print(df[df['Inside_Bar'] == True])
    print(breakout_up)

    # Filter for true breakout values and include date
    breakout_up_true = df[breakout_up]
    breakout_down_true = df[breakout_down]

    return breakout_up_true, breakout_down_true

def detect_relative_volume(df, threshold=3):

    avg_volume = df['Volume'].rolling(window=20).mean()

    relative_volume = df['Volume'] / avg_volume
    print(relative_volume)
    breakout = relative_volume > threshold
    return breakout

def detect_triangle(df, window=5):
    # Create lists to store slopes
    high_slope = []
    low_slope = []

    # Calculate the slope for 'High' and 'Low' columns manually
    for i in range(len(df)):
        if i < window:
            high_slope.append(None)  # Not enough data for slope calculation
            low_slope.append(None)
        else:
            high_slope.append(df['High'].iloc[i] - df['High'].iloc[i - window])
            low_slope.append(df['Low'].iloc[i] - df['Low'].iloc[i - window])

    # Assign these lists to new columns in the DataFrame
    df['High_Slope'] = high_slope
    df['Low_Slope'] = low_slope

    # Ascending and descending triangle conditions
    ascending_triangle = (df['High_Slope'] == 0) & (df['Low_Slope'] > 0)
    descending_triangle = (df['Low_Slope'] == 0) & (df['High_Slope'] < 0)

    # Manually calculate the rolling max and min for breakout detection
    breakout_up = []
    breakout_down = []

    for i in range(len(df)):
        if i < window:
            breakout_up.append(False)
            breakout_down.append(False)
        else:
            max_high = max(df['High'].iloc[i - window:i])
            min_low = min(df['Low'].iloc[i - window:i])
            breakout_up.append(df['Close'].iloc[i] > max_high and ascending_triangle.iloc[i])
            breakout_down.append(df['Close'].iloc[i] < min_low and descending_triangle.iloc[i])

    # Convert to pandas Series to maintain consistency
    breakout_up = pd.Series(breakout_up, index=df.index)
    breakout_down = pd.Series(breakout_down, index=df.index)

    return breakout_up, breakout_down


def fetch_yf_data(symbol, start_time, end_time, tf):
    try:
        df = yf.download(
            symbol,
            start=start_time,
            end=end_time,
            interval=tf)
        # print(df)
        #exit(1);
        return df
        print(list(shared._ERRORS.keys()))
    except:
        return False

# Fetch stock data
# period '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
def fetch_stock_data(ticker,period='5d', interval='15m'):
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data

# Function to calculate returns after crossover
def calculate_returns(df, crossover_index, periods=[5, 10, 30, 365]):
    """Calculate percentage return after crossover for multiple periods."""
    pos = df.index.get_loc(crossover_index)

    returns = {}

    for period in periods:
        # Check if there are enough days after the crossover
        if pos + period < len(df):
            start_price = df['Close'].iloc[pos]
            end_price = df['Close'].iloc[pos + period]
            returns[f'{period}-day Return'] = (end_price - start_price) / start_price * 100  # Return in %
        else:
            # Not enough data for the period
            returns[f'{period}-day Return'] = np.nan  # Return NaN if not enough data

    return returns


# Detect if stock has been in a range for more than 2 days
def detect_range(data, window_size=2*24*4, tolerance=0.01):
    avg_price = sum(data['Close']) / len(data['Close'])  # Manually calculate mean
    low_band = avg_price * (1 - tolerance)
    high_band = avg_price * (1 + tolerance)

    # Create a list to store whether the data is within the range
    in_range = []

    # Iterate through the data and manually calculate the rolling window max and min
    for i in range(len(data)):
        if i < window_size:
            in_range.append(False)  # Not enough data for a full window
        else:
            max_high = max(data['High'][i - window_size:i])
            min_low = min(data['Low'][i - window_size:i])
            in_range.append(max_high <= high_band and min_low >= low_band)

    # Convert the list to a pandas Series for consistency
    in_range = pd.Series(in_range, index=data.index)

    return in_range, low_band, high_band

def detect_atr_breakout(df, atr_multiplier=2):
    # True Range (TR)
    df['TR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Breakout conditions
    breakout_up = df['Close'] > (df['Close'].shift(1) + atr_multiplier * df['ATR'])
    breakout_down = df['Close'] < (df['Close'].shift(1) - atr_multiplier * df['ATR'])

    return breakout_up, breakout_down


# Calculate relative volume
def calculate_relative_volume(data, volume_window=20):
    avg_volume = data['Volume'].shift(1).rolling(window=volume_window).mean()
    relative_volume = data['Volume'] / avg_volume
    return relative_volume

# Calculate RSI
def calculate_rsi(data, period=14):
    rsi = talib.RSI(data['Close'], timeperiod=period)
    return rsi

# Calculate EMA
def calculate_ema(data, period):
    ema = talib.EMA(data['Close'], timeperiod=period)
    return ema

# Detect breakout from range with high relative volume, RSI, and EMA
def detect_breakout_with_rsi_ema(data, low_band, high_band, relative_volume_threshold=2):
    """
    Detect breakout from range with high relative volume, RSI, and EMA conditions.

    :param data: DataFrame containing stock data.
    :param low_band: The lower price band of the range.
    :param high_band: The upper price band of the range.
    :param relative_volume_threshold: Threshold for relative volume to consider breakout (default 2).
    :return: DataFrame rows where breakout occurs.
    """
    # Calculate RSI
    data['RSI'] = calculate_rsi(data)

    # Calculate 20 EMA and 50 EMA
    data['EMA20'] = calculate_ema(data, 20)
    data['EMA50'] = calculate_ema(data, 50)

    # Define breakout conditions (ensure conditions are enclosed in parentheses)
    upper_breakout = (
        (data['Close'] > high_band) &
        (data['Relative_Volume'] > relative_volume_threshold) &
        (data['RSI'] > 50) &
        (data['EMA20'] > data['EMA50'])
    )

    lower_breakout = (
        (data['Close'] < low_band) &
        (data['Relative_Volume'] > relative_volume_threshold) &
        (data['RSI'] < 60) &
        (data['EMA20'] < data['EMA50'])
    )

    # Combine upper and lower breakouts
    breakout = upper_breakout | lower_breakout

    return data[breakout]

# Let's modify the breakout detection function to include the additional criteria for RSI, VWAP, and EMAs

# Modified detect_breakout_with_rsi_ema function
def detect_breakout_with_rsi_ema_vwap(data, daily_data, low_band, high_band, relative_volume_threshold=2):
    """
    Detect breakout from range with high relative volume, RSI, EMA, and VWAP conditions.
    Daily and 15-min timeframes are considered for RSI, EMA, and VWAP.

    :param data: DataFrame containing 15-min stock data.
    :param daily_data: DataFrame containing Daily stock data.
    :param low_band: The lower price band of the range.
    :param high_band: The upper price band of the range.
    :param relative_volume_threshold: Threshold for relative volume to consider breakout (default 2).
    :return: DataFrame rows where breakout occurs.
    """
    # Calculate 15-min RSI, EMA, and VWAP
    data['RSI_15min'] = calculate_rsi(data, period=14)
    data['EMA20_15min'] = calculate_ema(data, period=20)
    data['EMA50_15min'] = calculate_ema(data, period=50)
    data['VWAP_15min'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

    # Calculate Daily RSI, EMA, and VWAP
    daily_data['RSI_Daily'] = calculate_rsi(daily_data, period=14)
    daily_data['EMA20_Daily'] = calculate_ema(daily_data, period=20)
    daily_data['EMA50_Daily'] = calculate_ema(daily_data, period=50)
    daily_data['VWAP_Daily'] = (daily_data['Volume'] * (daily_data['High'] + daily_data['Low'] + daily_data['Close']) / 3).cumsum() / daily_data['Volume'].cumsum()

    # Define breakout conditions (upper and lower) with RSI, VWAP, and EMA in both timeframes

    # Upper breakout conditions:
    upper_breakout = (
        (data['Close'] > high_band) &
        (data['Relative_Volume'] > relative_volume_threshold) &
        (data['RSI_15min'] > 50) & (daily_data['RSI_Daily'].iloc[-1] > 50) &  # RSI for both timeframes
        (data['Close'] > data['VWAP_15min']) & (daily_data['Close'].iloc[-1] > daily_data['VWAP_Daily'].iloc[-1]) &  # VWAP check
        (data['EMA20_15min'] > data['EMA50_15min']) & (daily_data['EMA20_Daily'].iloc[-1] > daily_data['EMA50_Daily'].iloc[-1])  # EMA check
    )

    # Lower breakout conditions:
    lower_breakout = (
        (data['Close'] < low_band) &
        (data['Relative_Volume'] > relative_volume_threshold) &
        (data['RSI_15min'] < 60) & (daily_data['RSI_Daily'].iloc[-1] < 60) &  # RSI for both timeframes
        (data['Close'] < data['VWAP_15min']) & (daily_data['Close'].iloc[-1] < daily_data['VWAP_Daily'].iloc[-1]) &  # VWAP check
        (data['EMA20_15min'] < data['EMA50_15min']) & (daily_data['EMA20_Daily'].iloc[-1] < daily_data['EMA50_Daily'].iloc[-1])  # EMA check
    )

    # Combine upper and lower breakouts
    breakout = upper_breakout | lower_breakout

    return data[breakout]

def fo_stocks(mode = 'dev'):
    if mode == 'dev':
        fo_stocks = pd.read_csv('fo_stocks_dev.csv')
    else:
        fo_stocks = pd.read_csv('fo_stocks.csv')

    fo_stocks = list(fo_stocks['Name'])

    return fo_stocks

def detect_crossover(df, short, long):
    return (df[short].shift(1) > df[long].shift(1)) & (df[short].shift(2) <= df[long].shift(2))

def detect_crossdown(df, short, long):
    return (df[short].shift(1) < df[long].shift(1)) & (df[short].shift(2) >= df[long].shift(2))

def detect_trends(df):
    """Detect trends in stock prices."""
    df = calculate_moving_average(df)

    # Initialize trend column using .loc

    # Detect Uptrend and Downtrend using .loc
    df.loc[(df['Close'] > df[f'SMA_{20}']) & (df[f'SMA_{20}'].diff() > 0), 'Trend'] = 'Uptrend'

    df.loc[ (df['Close'] < df[f'SMA_{20}']) & (df[f'SMA_{20}'].diff() < 0), 'Trend'] = 'Downtrend'

    df.loc[ (df['Trend'] != 'Uptrend') & (df['Trend'] != 'Downtrend'), 'Trend'] = 'Sideways'

    print(df)
    exit(1)
    return df


def calculate_moving_average(df, window=20):
    """Calculate Simple Moving Average."""
    df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    return df

def pre_open_conditions(df_row_pre_open, df_15m, pre_open_index, last_row_index, prev_row_index):
    today = datetime.today()
    today_9_15 = today.strftime('%Y-%m-%d 09:15:00')
    today_9_15 = pd.to_datetime(today_9_15)
    # print(pre_open_index)
    # print(df_row_pre_open.loc[pre_open_index, 'gap_up'])
    # exit(1)
    conditions_pre_check = conditions(df_15m, last_row_index, prev_row_index)
    
    
    pre_open_condition = {}
    pre_open_condition['gap_up_first_red_O=H'] = (
        #(conditions_pre_check['high_vol']) &
        (df_row_pre_open.loc[pre_open_index, 'gap_up']) &
        (df_15m.loc[last_row_index,'Candle_Color'] == 'Red') &
        (df_15m.loc[last_row_index,'High'] == df_15m.loc[last_row_index,'Open']) &
        (last_row_index == today_9_15)
    )
    
    pre_open_condition['gap_down_first_green_O=L'] = (
        (df_row_pre_open.loc[pre_open_index, 'gap_down']) &
        (df_15m.loc[last_row_index,'Candle_Color'] == 'Green') &
        #(conditions_pre_check['high_vol']) &
        (df_15m.loc[last_row_index,'Low'] == df_15m.loc[last_row_index,'Open']) &
        (last_row_index == today_9_15)
    )
    
    pre_open_condition['gap_up_first_red'] = (
        (pre_open_condition['gap_up_first_red_O=H'] == False) &
        (df_row_pre_open.loc[pre_open_index, 'gap_up']) &
        (df_15m.loc[last_row_index,'Candle_Color'] == 'Red') &
        #(conditions_pre_check['high_vol']) &
        (last_row_index == today_9_15)
    )
    
    pre_open_condition['gap_down_first_green'] = (
        (pre_open_condition['gap_down_first_green_O=L'] == False) &
        (df_row_pre_open.loc[pre_open_index, 'gap_down']) &
        (df_15m.loc[last_row_index,'Candle_Color'] == 'Green') &
        #(conditions_pre_check['high_vol']) &
        (last_row_index == today_9_15)
    )
    
    pre_open_condition['gap_up_green_red_high_vol'] = (
        #(conditions_pre_check['high_vol']) &
        (df_row_pre_open.loc[pre_open_index, 'gap_up']) &
        (df_15m.loc[last_row_index,'Opposite_Color_Pattern'] == 'Green/Red') &
        ((df_15m.loc[last_row_index,'is_doji'] == False) | (df_15m.loc[prev_row_index,'is_doji'] == False)) &
        ((df_15m.loc[last_row_index,'consecutive_green'] > 4) | (df_15m.loc[prev_row_index,'consecutive_green'] > 4) )
    )
    
    pre_open_condition['gap_down_red_green_high_vol'] = (
         #(conditions_pre_check['high_vol']) &
         (df_row_pre_open.loc[pre_open_index, 'gap_down']) &
        (df_15m.loc[last_row_index,'Opposite_Color_Pattern'] == 'Red/Green') &
        ((df_15m.loc[last_row_index,'is_doji'] == False) | (df_15m.loc[prev_row_index,'is_doji'] == False)) &
        ((df_15m.loc[last_row_index,'consecutive_red'] > 4) | (df_15m.loc[prev_row_index,'consecutive_red'] > 4) ) 
    )
    
    if df_row_pre_open.loc[pre_open_index, 'high_vol'] :
        strategy = []
        for key, value in conditions_pre_check.items():
            if value:
                strategy.append(key)
        if len(strategy):
             strategy_str = "_".join(strategy)
             pre_open_condition[f'pre_open_high_vol_{strategy_str}'] = True
        
    
    return pre_open_condition


def conditions(df, last_row_index, prev_row_index, vol_threshold=1.2):

    condition = {}

    condition['high_vol'] = (
        (   (
                (df.loc[last_row_index,'vol_10d'] > vol_threshold) |
                (df.loc[last_row_index, 'vol_20d'] > vol_threshold) |
                (df.loc[last_row_index, 'vol_5d']  > vol_threshold)
            )
          ) &
        (df.loc[last_row_index, 'Volume_Change']  > 1 )
    )

    condition['pin_bearish_engulf_high_vol'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'is_doji'] == False) &
        ((df.loc[last_row_index, 'Below_EMA_After_Consecutive']) | (df.loc[prev_row_index, 'Below_EMA_After_Consecutive']) |
         (df.loc[last_row_index, 'consecutive_green'] > 3)  |  (df.loc[prev_row_index, 'consecutive_green'] > 3)
         ) &
        (df.loc[last_row_index, 'Open'] > df.loc[last_row_index, 'Close']) &
        (df.loc[last_row_index, 'upper_wick'] >  (df.loc[last_row_index, 'body_size']) / 2) &
        (((df.loc[last_row_index, 'Open'] > max(df.loc[prev_row_index, 'Open'],df.loc[prev_row_index, 'Close'])) )) &
        (((df.loc[last_row_index, 'Close'] < min(df.loc[prev_row_index, 'Open'],df.loc[prev_row_index, 'Close'])) ))
    )

    condition['pin_bearish_engulf'] = (
        (condition['pin_bearish_engulf_high_vol'] == False) &
        (df.loc[last_row_index, 'is_doji'] == False) &
        ((df.loc[last_row_index, 'Below_EMA_After_Consecutive']) | (df.loc[prev_row_index, 'Below_EMA_After_Consecutive']) |
         (df.loc[last_row_index, 'consecutive_green'] > 3)  |  (df.loc[prev_row_index, 'consecutive_green'] > 3)
         ) &
        (df.loc[last_row_index, 'Open'] > df.loc[last_row_index, 'Close']) &
        (df.loc[last_row_index, 'upper_wick'] >  (df.loc[last_row_index, 'body_size']) / 2) &
        (((df.loc[last_row_index, 'Open'] > max(df.loc[prev_row_index, 'Open'],df.loc[prev_row_index, 'Close'])) )) &
        (((df.loc[last_row_index, 'Close'] < min(df.loc[prev_row_index, 'Open'],df.loc[prev_row_index, 'Close'])) ))
    )
    
    condition['bearish_engulf'] = (
        (condition['pin_bearish_engulf_high_vol'] == False) &
        (condition['pin_bearish_engulf'] == False) &
        ((df.loc[last_row_index, 'Below_EMA_After_Consecutive']) | (df.loc[prev_row_index, 'Below_EMA_After_Consecutive']) |
         (df.loc[last_row_index, 'consecutive_green'] > 3)  |  (df.loc[prev_row_index, 'consecutive_green'] > 3)
         ) &
        (df.loc[last_row_index, 'is_doji'] == False) &
        (df.loc[last_row_index, 'Open'] > df.loc[last_row_index, 'Close'] ) &
        (df.loc[prev_row_index, 'Open'] < df.loc[prev_row_index, 'Close'] ) &
        (df.loc[last_row_index, 'Open'] > df.loc[prev_row_index, 'Close'] ) &
        (df.loc[last_row_index, 'Close'] < df.loc[prev_row_index, 'Open'] )
    )
    
    condition['bullish_engulf'] = (
        ((df.loc[last_row_index, 'Above_EMA_After_Consecutive']) | (df.loc[prev_row_index, 'Above_EMA_After_Consecutive']) |
         (df.loc[last_row_index, 'consecutive_red'] > 3)  |  (df.loc[prev_row_index, 'consecutive_red'] > 3)
         ) &
        (df.loc[last_row_index, 'is_doji'] == False) &
        (df.loc[last_row_index, 'Open'] < df.loc[last_row_index, 'Close'] ) &
        (df.loc[prev_row_index, 'Open'] > df.loc[prev_row_index, 'Close'] ) &
        (df.loc[last_row_index, 'Open'] < df.loc[prev_row_index, 'Close'] ) &
        (df.loc[last_row_index, 'Close'] > df.loc[prev_row_index, 'Open'] )
    )

    condition['up'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'Close']   > df.loc[last_row_index,'vwap']) &
        (df.loc[last_row_index, 'Close']   > df.loc[last_row_index,'ema_10']) &
        (df.loc[last_row_index, 'ema_10']  > df.loc[last_row_index,'ema_20']) &
        (df.loc[last_row_index, 'Close']   > df.loc[prev_row_index, 'Close']) &
        (df.loc[last_row_index, 'rsi']     > 50)
    )

    condition['down'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'Close']   <= df.loc[last_row_index,'vwap']) &
        (df.loc[last_row_index, 'Close']   <= df.loc[last_row_index,'ema_10']) &
        (df.loc[last_row_index, 'ema_10']  < df.loc[last_row_index,'ema_20']) &
        (df.loc[last_row_index, 'Close']   < df.loc[prev_row_index, 'Close']) &
        (df.loc[last_row_index, 'rsi']     < 60)
    )

    condition['inside'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'inside'] | df.loc[prev_row_index, 'inside']) &
        (df.loc[prev_row_index, 'is_doji'] == False)
    )

    condition['crossover'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'crossover'] | df.loc[prev_row_index, 'crossover'])
    )

    condition['crossdown'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'crossdown'] | df.loc[prev_row_index, 'crossdown'])
    )

    condition['dojis_group'] = (
        (df.loc[last_row_index, 'consecutive_doji'] > 4 | df.loc[prev_row_index, 'consecutive_doji'] > 4)
    )

    condition['vwap_rejected'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'vwap_rejected'] | df.loc[prev_row_index, 'vwap_rejected']) &
        (df.loc[last_row_index, 'is_doji'] == False)
    )

    condition['vwap_accepted'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'vwap_accepted'] | df.loc[prev_row_index, 'vwap_accepted']) &
        (df.loc[last_row_index, 'is_doji'] == False)
    )

    condition['green_red'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'Opposite_Color_Pattern'] == 'Green/Red') &
        (df.loc[last_row_index, 'is_doji'] == False) &
        (df.loc[prev_row_index, 'is_doji'] == False)
    )
    condition['red_green'] = (
        (condition['high_vol']) &
        (df.loc[last_row_index, 'Opposite_Color_Pattern'] == 'Red/Green') &
        (df.loc[last_row_index, 'is_doji'] == False) &
        (df.loc[prev_row_index, 'is_doji'] == False)
    )
    
    condition['PIB'] = (
        ((df.loc[last_row_index, 'Below_EMA_After_Consecutive']) | (df.loc[prev_row_index, 'Below_EMA_After_Consecutive']) |
         (df.loc[last_row_index, 'consecutive_green'] > 3)  |  (df.loc[prev_row_index, 'consecutive_green'] > 3)
         ) &
        ((df.loc[last_row_index, 'inside']) | (df.loc[prev_row_index, 'inside'])) &
        (df.loc[last_row_index, 'High'] < df.loc[prev_row_index, 'High'] ) &
        (df.loc[last_row_index, 'Low'] > df.loc[prev_row_index, 'Low'] ) &
        (df.loc[last_row_index, 'upper_wick'] > df.loc[last_row_index, 'body_size'] * 3 ) &
        (df.loc[prev_row_index, 'upper_wick'] > df.loc[prev_row_index, 'body_size'] * 3 )
    )

    condition['hanging_man'] = (
        ((df.loc[last_row_index, 'Below_EMA_After_Consecutive']) | (df.loc[prev_row_index, 'Below_EMA_After_Consecutive']) |
         (df.loc[last_row_index, 'consecutive_green'] > 3)  |  (df.loc[prev_row_index, 'consecutive_green'] > 3)
         ) &
        ((df.loc[last_row_index, 'is_doji']) | (df.loc[prev_row_index, 'is_doji'])) &
        (df.loc[last_row_index, 'Close'] < df.loc[last_row_index, 'Open'] ) &
        (df.loc[last_row_index, 'lower_wick'] > df.loc[last_row_index, 'body_size'] * 3 ) &
        (df.loc[last_row_index, 'upper_wick'] < df.loc[last_row_index, 'body_size'] )
    )
    
    condition['hammer'] = (
        ((df.loc[last_row_index, 'Above_EMA_After_Consecutive']) | (df.loc[prev_row_index, 'Above_EMA_After_Consecutive']) |
         (df.loc[last_row_index, 'consecutive_red'] > 3)  |  (df.loc[prev_row_index, 'consecutive_red'] > 3)
         ) &
        ((df.loc[last_row_index, 'is_doji']) | (df.loc[prev_row_index, 'is_doji'])) &
        (df.loc[last_row_index, 'Close'] > df.loc[last_row_index, 'Open'] ) &
        (df.loc[last_row_index, 'lower_wick'] > df.loc[last_row_index, 'body_size'] * 3 ) &
        (df.loc[last_row_index, 'upper_wick'] < df.loc[last_row_index, 'body_size'] )
    )
    
    condition['hammer_inverted'] = (
        ((df.loc[last_row_index, 'Above_EMA_After_Consecutive']) | (df.loc[prev_row_index, 'Above_EMA_After_Consecutive']) |
         (df.loc[last_row_index, 'consecutive_red'] > 3)  |  (df.loc[prev_row_index, 'consecutive_red'] > 3)
         ) &
        ((df.loc[last_row_index, 'is_doji']) | (df.loc[prev_row_index, 'is_doji'])) &
        (df.loc[last_row_index, 'Close'] > df.loc[last_row_index, 'Open'] ) &
        (df.loc[last_row_index, 'lower_wick'] < df.loc[last_row_index, 'body_size'] ) &
        (df.loc[last_row_index, 'upper_wick'] > df.loc[last_row_index, 'body_size'] * 3)
    )


    prev2_row_index = df.index[df.index.get_loc(last_row_index) - 2]
    condition['SOW_result_effort_diversion'] = (
        (condition['high_vol']) &
        ( (df.loc[last_row_index, 'consecutive_green'] > 3)  |  (df.loc[prev_row_index, 'consecutive_green'] > 3) ) &
        ( (df.loc[last_row_index, 'vsa_r_gt_e'])  |  (df.loc[prev_row_index, 'vsa_r_gt_e']) | (df.loc[last_row_index, 'vsa_r_le_e']) | (df.loc[prev_row_index, 'vsa_r_le_e']))
    )

    condition['SOW_Up_Thrust'] = (
        (condition['high_vol']) &
        ( (df.loc[last_row_index, 'consecutive_green'] > 3)  |  (df.loc[prev_row_index, 'consecutive_green'] > 3) ) &
        ( (df.loc[last_row_index, 'is_doji'])  |  (df.loc[prev_row_index, 'is_doji']) )
    )

    condition['SOW_no_demand'] = (
        ( (df.loc[last_row_index, 'consecutive_green'] > 3)  |  (df.loc[prev_row_index, 'consecutive_green'] > 3) ) &
        ( df.loc[last_row_index, 'is_doji'] ) &
        ( df.loc[last_row_index, 'body_size'] < df.loc[prev_row_index, 'body_size'] ) &
        ( (df.loc[last_row_index, 'Volume'])  <  (df.loc[prev_row_index, 'Volume']) ) &
        ( (df.loc[last_row_index, 'Volume'])  <  (df.loc[prev2_row_index, 'Volume']) )
    )


    condition['SOS_result_effort_diversion'] = (
        (condition['high_vol']) &
        ( (df.loc[last_row_index, 'consecutive_red'] > 3)  |  (df.loc[prev_row_index, 'consecutive_red'] > 3) ) &
        ( (df.loc[last_row_index, 'vsa_r_gt_e'])  |  (df.loc[prev_row_index, 'vsa_r_gt_e'])  | (df.loc[last_row_index, 'vsa_r_le_e'])  |  (df.loc[prev_row_index, 'vsa_r_le_e']))
    )

    condition['SOS_Down_Thrust'] = (
        (condition['high_vol']) &
        ( (df.loc[last_row_index, 'consecutive_red'] > 3)  |  (df.loc[prev_row_index, 'consecutive_red'] > 3) ) &
        ( (df.loc[last_row_index, 'is_doji'])  |  (df.loc[prev_row_index, 'is_doji']) )
    )

    condition['SOS_no_supply'] = (
        ( (df.loc[last_row_index, 'consecutive_red'] > 3)  |  (df.loc[prev_row_index, 'consecutive_red'] > 3) ) &
        ( df.loc[last_row_index, 'is_doji'] ) &
        ( df.loc[last_row_index, 'body_size'] < df.loc[prev_row_index, 'body_size'] ) &
        ( (df.loc[last_row_index, 'Volume'])  <  (df.loc[prev_row_index, 'Volume']) ) &
        ( (df.loc[last_row_index, 'Volume'])  <  (df.loc[prev2_row_index, 'Volume']) )
    )
    
    condition['big_red_near_inside_n_doji'] = (
        (condition['high_vol']) &
        ( df.loc[last_row_index, 'big_red_near_inside_n_doji'] ) 
    )

    
    # marubaju big candle change > 2% with high volume, next candle doji

    #1# yest green candle, today gap up, first candle green with high volume, second candle close below 50% of first casndle
    #2# candle body > avg body and volume > volume avg


    return condition

def create_features(df, symbol):
    min_consecutive_days = 5
    size_threshold = 0.01

    df['symbol']  = symbol
    
    df['change']  = df['Close'].diff()
    df['variance_high_5d'] = df['High'].rolling(3).var()
    df['variance_low_5d'] = df['Low'].rolling(3).var()
    df['last_cdl_vol'] = df['Volume'].shift(1)
    df['Volume_Change'] = df['Volume'].pct_change()
    df['max_3d'] = df['High'].rolling(window=3).max()
    df['max_5d'] = df['High'].rolling(window=5).max()
    df['max_20d'] = df['High'].rolling(window=20).max()
    df['min_3d'] = df['High'].rolling(window=3).min()
    df['min_5d'] = df['High'].rolling(window=5).min()
    df['min_20d'] = df['High'].rolling(window=20).min()
    df['vol_avg_5d']  = df['Volume'].shift(1).rolling(window=5).mean()
    df['vol_avg_10d']  = df['Volume'].shift(1).rolling(window=10).mean()
    df['vol_avg_20d']  = df['Volume'].shift(1).rolling(window=20).mean()
    df['vol_5d']  = calculate_relative_volume(df,5)  #  1 week
    df['vol_10d'] = calculate_relative_volume(df,20) # one month
    df['vol_20d'] = calculate_relative_volume(df,60) # three month
    df['rsi']     = calculate_rsi(df)
    df['ema_10']  = calculate_ema(df, 10)
    df['ema_20']  = calculate_ema(df, 20)

    # For example, body sizes are similar if they are within 5% of each other
    df['body_size'] = np.abs(df['Close'] - df['Open'])
    df['Prev_Body_Size'] = df['body_size'].shift(1)
    # Determine if the current candle's body size is almost similar to the previous candle's body size
    df['Similar_Body_Size'] = df.apply(
        lambda row: abs(row['body_size'] - row['Prev_Body_Size']) / max(row['Prev_Body_Size'],1) <= size_threshold
                    if pd.notnull(row['Prev_Body_Size']) else False,
        axis=1
    )
    
    df['big_red'] = ((df['Open'] - df['Close']) > (0.02 * df['Open']))
    
    # Calculate the 50% level of the big red candle (previous candle)
    df['big_red_50_per_level'] = np.where(
        df['big_red'],
        df['Close'] + (df['Open'] - df['Close']) * 0.5,
        np.nan
    )

    
    df['is_doji'] = np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
    df['consecutive_doji'] = (df['is_doji'].groupby((df['is_doji'] != df['is_doji'].shift()).cumsum()).cumsum())

    df['is_spinning_top'] = (np.abs(df['Open'] - df['Close']) < 0.2 * (df['High'] - df['Low'])) & ~df['is_doji']
    df['consecutive_top'] = (df['is_spinning_top'].groupby((df['is_spinning_top'] != df['is_spinning_top'].shift()).cumsum()).cumsum())

    # df['big_red_near_doji'] = np.where(
    #     (df['is_doji'] | df['is_spinning_top']) &
    #     (df['big_red'].shift(1).rolling(window=5).max() == 1), 
    #     True, 
    #     False
    # )

    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    df['Above_EMA'] = np.where((df['Open'] > df['ema_10']) & (df['Close'] > df['ema_10']), 1, 0)
    # Find consecutive days above the EMA
    df['Consecutive_Above_EMA'] = (df['Above_EMA'].groupby((df['Above_EMA'] != df['Above_EMA'].shift()).cumsum()).cumsum())
    df['Below_EMA_After_Consecutive'] = np.where((df['Consecutive_Above_EMA'].shift(1) >= min_consecutive_days) &
                                             ((df['Open'] < df['ema_10']) | (df['Close'] < df['ema_10'])), 1, 0)

    df['Below_EMA'] = np.where((df['Open'] < df['ema_10']) & (df['Close'] < df['ema_10']), 1, 0)
    # Find consecutive days above the EMA
    df['Consecutive_Below_EMA'] = (df['Below_EMA'].groupby((df['Below_EMA'] != df['Below_EMA'].shift()).cumsum()).cumsum())
    df['Above_EMA_After_Consecutive'] = np.where((df['Consecutive_Below_EMA'].shift(1) >= min_consecutive_days) &
                                             ((df['Open'] < df['ema_10']) | (df['Close'] < df['ema_10'])), 1, 0)

    # Detecting color of each candle: Green if Close > Open, Red if Open > Close
    df['Candle_Color'] = ['Green' if row['Close'] > row['Open'] else 'Red' for _, row in df.iterrows()]
    df['Prev_Candle_Color'] = df['Candle_Color'].shift(1)
    # Detecting Red followed by Green (R/G) and Green followed by Red (G/R)
    df['Opposite_Color_Pattern'] = df.apply(lambda row: 'Red/Green' if row['Prev_Candle_Color'] == 'Red' and row['Candle_Color'] == 'Green' else
                'Green/Red' if row['Prev_Candle_Color'] == 'Green' and row['Candle_Color'] == 'Red' else
                None,
                axis=1
    )

    # detect consectuive green candles
    df['is_green'] = df['Close'] > df['Open']
    df['consecutive_green'] = (df['is_green'].groupby((df['is_green'] != df['is_green'].shift()).cumsum()).cumsum())

    df['is_red'] = df['Close'] < df['Open']
    df['consecutive_red'] = (df['is_red'].groupby((df['is_red'] != df['is_red'].shift()).cumsum()).cumsum())


    df['open_eg_high'] = np.where((df['Open'] == df['High']) & (df['Candle_Color'] == 'Red') & (df['is_doji'] == 0) , 1, 0)
    df['open_eg_low'] = np.where((df['Open'] == df['Low']) & (df['Candle_Color'] == 'Green') & (df['is_doji'] == 0), 1, 0)

    df['inside']  = detect_insidebar(df)
    df['crossover']  = detect_crossover(df, 'ema_10', 'ema_20')
    df['crossdown']  = detect_crossdown(df, 'ema_10', 'ema_20')
    df['vwap']    = calculate_vwap(df)
    df['vwap_rejected'] = np.where( ((df['Open'] > df['vwap']) | (df['High'] > df['vwap'])) & (df['Close'] < df['vwap']) & (df['Candle_Color'] == 'Red') & (df['is_doji'] == 0) , 1, 0)
    df['vwap_accepted'] = np.where( ((df['Open'] < df['vwap']) | (df['High'] < df['vwap'])) & (df['Close'] > df['vwap']) & (df['Candle_Color'] == 'Green') & (df['is_doji'] == 0) , 1, 0)

    #df['near_brn'] = df['Close'].apply(is_near_big_round_number)
    
    df['big_red_near_inside_n_doji'] = np.where(
        (df['is_doji'] | df['is_spinning_top'] | df['inside']) &
        (df['big_red'].shift(1).rolling(window=3).max() == 1), 
        True, 
        False
    )

    #VSA
    df['vsa_r_gt_e'] = (df['body_size'] > df['body_size'].shift(1)) & (df['Volume'] < df['Volume'].shift(1))
    df['vsa_r_le_e'] = (df['body_size'] < df['body_size'].shift(1)) & (df['Volume'] > df['Volume'].shift(1))

    #print(df)
    #return df.dropna()
    return df

def is_near_big_round_number(price, threshold=0.5):
    # Determine the number of digits in the price
    if price < 100:
        # For 2-digit numbers
        lower_brn = round(price // 10) * 10  # Nearest lower multiple of 10
        upper_brn = lower_brn + 10             # Nearest upper multiple of 10
    elif price < 1000:
        # For 3-digit numbers
        lower_brn = round(price // 50) * 50    # Nearest lower multiple of 50
        upper_brn = lower_brn + 50               # Nearest upper multiple of 50
    else:
        # For 4-digit numbers
        lower_brn = round(price // 100) * 100   # Nearest lower multiple of 100
        upper_brn = lower_brn + 100              # Nearest upper multiple of 100

    # Check if the price is within the threshold of either BRN
    return abs(price - lower_brn) <= threshold or abs(price - upper_brn) <= threshold





# =============================================================================
# def is_current_time_between(start_time: str, end_time: str) -> bool:
#     """
#     Check if the current time is between the start_time and end_time.
#
#     :param start_time: Start time as a string in 'HH:MM' format (24-hour format)
#     :param end_time: End time as a string in 'HH:MM' format (24-hour format)
#     :return: True if current time is between start_time and end_time, False otherwise
#         # Example usage
#         start = "22:00"
#         end = "06:00"
#
#         if is_current_time_between(start, end):
#             print("Current time is between the given time range.")
#         else:
#             print("Current time is outside the given time range.")
#     """
#     # Get current time
#     current_time = datetime.now().time()
#
#     # Convert start_time and end_time strings to time objects
#     start_time_obj = time.fromisoformat(start_time)
#     end_time_obj = time.fromisoformat(end_time)
#
#     # Check if the current time is between start and end time
#     if start_time_obj <= end_time_obj:
#         # Case where the start and end times are on the same day
#         return start_time_obj <= current_time <= end_time_obj
#     else:
#         # Case where the time window crosses midnight
#         return current_time >= start_time_obj or current_time <= end_time_obj
#
# =============================================================================


def calculate_vwap(df_15m):
    return (df_15m['Volume'] * (df_15m['High'] + df_15m['Low'] + df_15m['Close']) / 3).cumsum() / df_15m['Volume'].cumsum()

def detect_patterns(df):
    """Detect various candlestick patterns."""
    df = calculate_moving_average(df)

    # Initialize pattern columns
    df['Bearish_Doji'] = False
    df['Bearish_Marubozu'] = False
    df['Hanging_Man'] = False
    df['Double_Top'] = False
    df['Triple_Top'] = False
    df['Morning_Star'] = False

    # Bearish Doji: Doji with close < open, following a bullish candle
    df['is_doji'] = np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
    df['Bearish_Doji'] = df['is_doji'] & (df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1))

    # Bearish Marubozu: No wicks, open == high and close == low
    df['Bearish_Marubozu'] = (df['Open'] == df['High']) & (df['Close'] == df['Low']) & (df['Open'] > df['Close'])

    # Hanging Man: Small body with long lower wick, after an uptrend
    df['Hanging_Man'] = (
        df['Close'] < df['Open'] &
        ((df['Open'] - df['Low']) / (df['High'] - df['Low']) > 0.6) &
        (df['Close'] > df['SMA_20'])
    )

    # Double Top: Peaks at the same price level with a trough in between
    df['Double_Top'] = False  # Implement logic for double top here

    # Triple Top: Three peaks at the same price level with troughs in between
    df['Triple_Top'] = False  # Implement logic for triple top here

    # Morning Star: Three candle pattern - large bearish, small body, large bullish
    df['Morning_Star'] = (
        (df['Close'].shift(2) < df['Open'].shift(2)) &  # Previous bearish candle
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # Small body candle
        (df['Close'] > df['Open']) &  # Current bullish candle
        (df['Close'] > df['Open'].shift(2)) &
        (df['Close'] > df['Close'].shift(1))
    )

    return df

def detect_doji_with_long_upper_wick(df):
    # Calculate the body size (difference between open and close)
    df['body_size'] = np.abs(df['Close'] - df['Open'])

    # Calculate the upper wick size (difference between high and max(open, close))
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)

    # Calculate the previous candle low for comparison
    df['previous_low'] = df['Low'].shift(1)

    # Detect doji candles (open and close prices are almost equal)
    df['is_doji'] = df['body_size'] / (df['High'] - df['Low']) < 0.1  # Tweak the threshold as needed

    # Detect long upper wick (upper wick is significantly larger than the body)
    df['has_long_upper_wick'] = df['upper_wick'] > (2 * df['body_size'])

    # Detect if both open and close are below the previous candle's low
    df['is_below_previous'] = (df['Open'] < df['previous_low']) & (df['Close'] < df['previous_low'])

    # Combine all conditions
    df['doji_with_long_upper_wick'] = df['is_doji'] & df['has_long_upper_wick'] & df['is_below_previous']

    return df

# Function to fetch data and calculate momentum indicators
def get_momentum_stocks(ticker):
    # Fetch historical data
    stock_data = yf.download(ticker, period='6mo', interval='1d')

    # Calculate RSI
    stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()

    # Calculate MACD
    macd = ta.trend.MACD(stock_data['Close'], window_slow=26, window_fast=12, window_sign=9)
    stock_data['MACD'] = macd.macd()
    stock_data['MACD_Signal'] = macd.macd_signal()

    # Condition for detecting bullish momentum (RSI > 70, MACD > MACD Signal)
    stock_data['Bullish_Momentum'] = np.where((stock_data['RSI'] > 70) & (stock_data['MACD'] > stock_data['MACD_Signal']), True, False)

    # Condition for detecting bearish momentum (RSI < 30, MACD < MACD Signal)
    stock_data['Bearish_Momentum'] = np.where((stock_data['RSI'] < 30) & (stock_data['MACD'] < stock_data['MACD_Signal']), True, False)

    # Filter out stocks in bullish or bearish momentum
    bullish_stocks = stock_data[stock_data['Bullish_Momentum'] == True]
    bearish_stocks = stock_data[stock_data['Bearish_Momentum'] == True]

    return stock_data, bullish_stocks, bearish_stocks

def csv_write_append(df, filename):
    # Check if the file exists and append accordingly
    if not os.path.exists(filename):
        df.to_csv(filename, index=True)  # Create a new CSV if it doesn't exist
    else:
        # Append to an existing CSV, preserving the header
        df.to_csv(filename, mode='a', header=False, index=True)

def last_week_monday_date():

    # Get today's date
    today = datetime.today()

    # Calculate the day of the week (Monday is 0 and Sunday is 6)
    current_weekday = today.weekday()

    # Calculate the first day of last week
    # Subtract current weekday + 7 days to get the previous Monday
    last_week_first_day = today - timedelta(days=current_weekday + 7)

    # Format the date as 'year-month-date'
    formatted_date = last_week_first_day.strftime('%Y-%m-%d')


    return formatted_date

def last_week_fri_mon_date():
    # Get today's date
    today = datetime.today()

    # Calculate how many days to subtract to get last Monday and Friday
    days_to_monday = today.weekday() + 7  # `weekday()` gives 0 for Monday, so add 7 to get last Monday
    days_to_friday = today.weekday() - 4 + 7  # `weekday() == 4` for Friday, adjust to get last Friday

    # Get last Monday and Friday
    last_monday = today - timedelta(days=days_to_monday)
    last_friday = today - timedelta(days=days_to_friday)

    # Print dates in format year-month-date
    return last_monday.strftime('%Y-%m-%d'), last_friday.strftime('%Y-%m-%d')

def print_colored_rows(df):
    for _, row in df.iterrows():
        # Condition: Print in green if Score >= 80, otherwise print in red
        if row['%CHNG'] > 1:
            print(f"\033[1;32m{row.to_string(index=False)}\033[0m")  # Green
        else:
            print(f"\033[1;31m{row.to_string(index=False)}\033[0m")  # Red

def is_file_updated_today(file_path):
    # Get the current date
    today = datetime.today()
    
    # Get the modification time of the file
    file_mod_time = os.path.getmtime(file_path)
    
    # Convert the modification time to a date object
    file_mod_date = datetime.fromtimestamp(file_mod_time)
    
    # Check if the modification date is today
    return file_mod_date == today

def print_groped(df):
    grouped = df.groupby('color')
    # Separator
    separator = '-' * 120  # 20 dashes
    
    # Iterate through the groups and print them
    for name, group in grouped:
        print(f"Group: {name}")
        #df_dropped = group.drop('id', axis=1)
        # df_dropped = df.set_index('date', drop=True)
        sorted_group = group.sort_values(by='FINALQUANTITY', ascending=False)
        sorted_group = sorted_group.set_index('SYMBOL', drop=True)
        sorted_group['FINALQUANTITY'] = sorted_group['FINALQUANTITY'].apply(lambda x: f"{x:,}".replace(",", "X").replace("X", ","))

        print(sorted_group)
        print(separator)
