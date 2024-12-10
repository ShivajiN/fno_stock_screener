# -*- coding: utf-8 -*-

import stocks_common as stk
import pytz
from datetime import datetime, timedelta,time
import os
import pandas as pd
import numpy as np
import winsound



if(stk.weekdays() == False): #testing
    print('Today is weekend')
    exit(1)

mode = 'dev'
mode = 'live'
fo_stocks = stk.fo_stocks(mode)

today = datetime.today()
today_ymd = today.strftime('%Y-%m-%d')
today_9_15 = today.strftime('%Y-%m-%d 09:15:00')
today_3_15 = today.strftime('%Y-%m-%d 15:15:00')
formatted_date = datetime.today().strftime('%d-%b-%Y')
# formatted_date = datetime.today().strftime('19-%b-%Y')

pre_open_file_name = f'pre_open/MW-Pre-Open-Market-{formatted_date}.csv'


if os.path.exists(pre_open_file_name):
    df_pre_open = pd.read_csv(pre_open_file_name)
    df_pre_open.columns = df_pre_open.columns.str.replace(' ', '').str.replace('\n', '')
    
    df_pre_open['FINALQUANTITY'] = df_pre_open['FINALQUANTITY'].str.replace(',', '')
    df_pre_open['FINAL'] = df_pre_open['FINAL'].str.replace(',', '')

    cols_to_convert = ['%CHNG', 'FINALQUANTITY', 'FINAL']  # List of columns to convert

    df_pre_open[cols_to_convert] = df_pre_open[cols_to_convert].apply(pd.to_numeric, errors='coerce') 
    df_pre_open['color'] = np.where( (df_pre_open['%CHNG'] > 0.0), 'Green', 'Red')
    df_pre_open['high_vol'] = np.where( (df_pre_open['FINALQUANTITY'] > 100000), 1, 0)
    df_pre_open['strategy'] = ''
    df_pre_open['RSI'] = 0
    df_pre_open['vol_5d'] = 0
    df_pre_open['vol_10d'] = 0
    df_pre_open['doji'] = 0
    df_pre_open['PDC_below_VWAP'] = 0
    df_pre_open['PDC_below_EMA10'] = 0
    df_pre_open['gap_up'] = np.where( ( (df_pre_open['%CHNG'] > 1) | ( (df_pre_open['%CHNG'] >= 0.40) & (df_pre_open['%CHNG'] < 1.0) & (df_pre_open['FINALQUANTITY'] > 100000)) ), 1, 0)
    df_pre_open['gap_down'] = np.where( ( (df_pre_open['%CHNG'] < -1) | ( (df_pre_open['%CHNG'] <= -0.30) & (df_pre_open['%CHNG'] > -1.0) & (df_pre_open['FINALQUANTITY'] > 100000)) ), 1, 0)
    df_pre_open['flat'] = np.where((abs(df_pre_open['%CHNG']) <= 0.40) & (abs(df_pre_open['%CHNG']) >= 0.0), 1, 0)
    
    df_pre_open_with_gap = df_pre_open[(df_pre_open['FINAL'] >= 200)  & ( (df_pre_open['gap_up'] == 1)  | (df_pre_open['gap_down'] == 1) | (df_pre_open['high_vol']  == 1))]
    df_pre_open_with_gap_sort = df_pre_open_with_gap.sort_values(['color','FINALQUANTITY'], ascending=[0,0])
    
    # print(df_pre_open_with_gap_sort)
    # exit(1)

#today_ymd = today.strftime('%Y-%m-29') #testing
print(f'Today : {today_ymd}')

last_week_monday, last_week_fri = stk.last_week_fri_mon_date()

filtered_ticker_old = pd.read_csv("filtered_ticker.csv")
filtered_ticker_old_last_week = filtered_ticker_old[filtered_ticker_old['date'] > last_week_monday]
filtered_ticker_old_unique = filtered_ticker_old_last_week.drop_duplicates(subset=['ticker']).tail(50)
filtered_ticker_old_unique_ticker = filtered_ticker_old_unique['ticker'].values
# print(len(filtered_ticker_old_unique_ticker))
# exit(1)

# print(last_week_fri)
# exit(1)

vol_threshold = 1.5

# Create an empty DataFrame with two columns
filtered_ticker = pd.DataFrame(columns=['date','strategy', 'ticker', 'Candle', 'Pre_open_%chang', 'Pre_open_vol','Pre_open','in_pre_open'])
filtered_ticker_intra = pd.DataFrame(columns=[
    'datetime','strategy', 'ticker', 'Candle', 'Pre_open_vol',  'vol_D_5d','vol_10d','rsi_D_15m','ema10_D_15m', 'vwap_D_15m','comment',
    # sort 
    'rsi_d', 'rsi_15', 'ema10_D','ema10_15m', 'vwap_D', 'vwap_15m'
])

# print(df_pre_open)
# df_pre_open.loc[177,'SYMBOL'] = 'CUB'
# print(df_pre_open)
# exit(1)
for symbol in fo_stocks:
    
    symbol_clean = symbol.replace('.NS', '')
    print('Fetching..',symbol_clean)
    
    row_pre_open = df_pre_open.loc[df_pre_open['SYMBOL'] == symbol_clean]
   
    if len(row_pre_open):
        pre_open_index = row_pre_open.index[-1]
    else:
        pre_open_index = 0
    
    daily_data_file_path = f'data/daily_data_{symbol_clean}_{today_ymd}.csv'
    if os.path.exists(daily_data_file_path):
        df_daily = pd.read_csv(daily_data_file_path)
        df_daily = df_daily.set_index('Date', drop=True)
    else:
        df_daily = stk.fetch_stock_data(symbol, "3mo", "1d")
        df_daily.to_csv(daily_data_file_path, sep=',', encoding='utf-8')
    
    index_exist = today_ymd in df_daily.index
    
    if(index_exist & stk.weekdays()):
        # Remove the last row using iloc
        df_yest = df_daily.iloc[:-1].copy()
    else:
        df_yest = df_daily.copy()
   
    last_row_index = df_yest.index[-1] # testing
    prev_row_index = df_daily.index[df_daily.index.get_loc(last_row_index) - 1]
    
    # print(last_row_index, df_yest.tail(2))
    # exit(1)
    
    df_daily = stk.create_features(df_yest, symbol)
    df_daily['in_pre_open'] = 0
    
    # df_daily = df_daily[(df_daily['is_doji'] == True) | (df_daily['is_spinning_top'] == True) | (df_daily['big_red'] == True) | (df_daily['big_red_near_inside_n_doji'] == True)]
    # print(df_daily[['Close', 'Open','is_doji','big_red_50_per_level',  'big_red', 'big_red_near_inside_n_doji']])
    # exit(1)
        
    # above_ema = df_daily[df_daily['Below_EMA_After_Consecutive'] == 1].copy()
    
    # stk.csv_write_append(above_ema, 'ema_crossdown.csv')
    
    # print(above_ema[['Close','Volume', 'Below_EMA_After_Consecutive']])
    # continue
    
    vol_ratio_threshold = 1.5
    conditions = stk.conditions(df_daily, last_row_index, prev_row_index, vol_ratio_threshold)
    
    # if(df_daily.loc[last_row_index, 'variance_high_5d'] < 5):
    #     print(last_row_index, symbol)
    # else:
    #     continue
    indices = (
        (symbol == '^NSEI') | (symbol == '^NSEBANK') | (symbol == '^BSESN')
    )
    
    vwap10_daily =  1 if df_daily.loc[last_row_index,'Close']     > df_daily.loc[last_row_index,'vwap']  else 0 
    ema10_daily  =  1 if df_daily.loc[last_row_index,'Close']     > df_daily.loc[last_row_index,'ema_10']  else 0 
    doji_daily_check = 1 if df_daily.loc[last_row_index,'is_doji'] or df_daily.loc[last_row_index,'consecutive_doji']  else 0 
    df_pre_open.loc[pre_open_index,'RSI'] = round(df_daily.loc[last_row_index,'rsi'])
    df_pre_open.loc[pre_open_index,'vol_5d'] = round(df_daily.loc[last_row_index,'vol_5d'])
    df_pre_open.loc[pre_open_index,'vol_10d'] = round(df_daily.loc[last_row_index,'vol_10d'])
    df_pre_open.loc[pre_open_index,'doji'] = doji_daily_check
    df_pre_open.loc[pre_open_index,'PDC_below_VWAP'] = vwap10_daily
    df_pre_open.loc[pre_open_index,'PDC_below_EMA10'] = ema10_daily

    
    filtered_old_ticker = symbol in filtered_ticker_old_unique_ticker
    
    filtered_pre_open = symbol_clean in df_pre_open_with_gap_sort['SYMBOL'].values
    
    add_row_daily = False
    
    strategy_daily = []
    for key, value in conditions.items():
        #print(key, value)
        # if key == 'high_vol':
        #     continue
        
        if(conditions[key]):
            strategy_daily.append(key)
            add_row_daily = True
            # if key == 'bearish_engulf':
            df_pre_open.loc[pre_open_index,'strategy'] = key

       
    
    # if not any other strategy detected then check for high vol
    if(not add_row_daily and conditions['high_vol']):
        strategy_daily.append('high_vol')
        add_row_daily = True
    

    min_vol_threshold = 200000
    strategy_concat =''
    if(len(strategy_daily) and add_row_daily and df_daily.loc[last_row_index, 'vol_avg_5d'] > min_vol_threshold):
        strategy_concat = ", ".join(strategy_daily)    
        new_row = pd.DataFrame({'date': [last_row_index],'strategy': [strategy_concat], 'ticker': [symbol], 'Candle': [df_daily.loc[last_row_index,'Candle_Color']]})
        
        
        if len(row_pre_open):
            new_row['Pre_open_%chang'] = df_pre_open.loc[pre_open_index, '%CHNG']
            new_row['Pre_open_vol'] = df_pre_open.loc[pre_open_index, 'FINALQUANTITY']
            new_row['in_pre_open'] = 1
                
            if df_pre_open.loc[pre_open_index, 'gap_up'] :
                new_row['Pre_open'] = 'up'
            elif df_pre_open.loc[pre_open_index, 'gap_down'] :
                new_row['Pre_open'] = 'down'
            elif df_pre_open.loc[pre_open_index, 'flat'] :
                new_row['Pre_open'] = 'flat'
            else:
                new_row['Pre_open'] = '-'
        filtered_ticker = pd.concat([filtered_ticker, new_row], ignore_index=True, axis=0)
                

    #exit(1)
    ###### Intra Day Check ######
    # conditions = [indices, up, down, inside, crossover, crossdown]
    for key, value in conditions.items():
        if(conditions[key]):
            break
        
    if any([conditions[key], indices, filtered_old_ticker, filtered_pre_open]):
    # if any([conditions[key], indices]):
        #print(f'\n\nFetching..15m..{symbol_clean}')

        comment =  []
        
        df_15m_org = stk.fetch_stock_data(symbol, "5d", "15m")
        #print(df_15m_org)
        index_exist_intra = today_3_15 in df_15m_org.index
        index_exist_intra_9_15 = today_9_15 in df_15m_org.index
        last_row_index_intra = df_15m_org.index[-1]
        
        df_slice = ( (last_row_index_intra.time() == time(9, 15)) | (last_row_index_intra.time() == time(15, 15)) )
          
        if df_slice:
            df_15m = df_15m_org.copy()
            #df_15m = df_15m_org.iloc[:-19].copy() #testing
        else:
            df_15m = df_15m_org.iloc[:-1].copy()
            
        last_row_index_intra = df_15m.index[-1]
        
        df_15m = stk.create_features(df_15m, symbol)
        
        #df_15m =df_15m[df_15m['consecutive_green']>4]
        # print(df_15m)
        # exit(1)
     
        prev_row_index_intra = df_15m.index[df_15m.index.get_loc(last_row_index_intra) - 1]
        
        #print(stk.detect_vsa(df_15m, last_row_index_intra, prev_row_index_intra))
       # check intraday volume spurt, last row excluded
        end_index_loc = df_15m.index.get_loc(last_row_index_intra)
            
        today_date = pd.Timestamp.today().normalize()
        df_today = df_15m[df_15m.index.normalize() == today_date]
        
        if(len(df_today)):
            df_today = df_today.copy()
        else:
            last_row_index = prev_row_index
            df_today = df_15m.tail(25).copy()
       
        today_9_15 = df_today.index[0]
        
        df_today['vol_avg_ratio'] = stk.calculate_relative_volume(df_15m, 24)
 
        
        ema10_15m    =  1 if df_15m.loc[last_row_index_intra,'Close'] > df_15m.loc[last_row_index_intra,'ema_10']  else 0
        vwap10_15m   =  1 if df_15m.loc[last_row_index_intra,'Close'] > df_15m.loc[last_row_index_intra,'vwap']  else 0
        
        rsi_daily_15m =f"{round(df_daily.loc[last_row_index,'rsi'], 0)}_{round(df_15m.loc[last_row_index_intra, 'rsi'],0)}"
        ema_daily_15m = f"{ema10_daily}_{ema10_15m}" 
        vwap_daily_15m = f"{vwap10_daily}_{vwap10_15m}" 

        if(mode == 'dev'):
            stk.csv_write_append(df_daily, 'debug_code1.csv')
            stk.csv_write_append(df_15m, 'debug_code1.csv')
            #print(df_daily[df_daily['variance_high_5d'] < 5])
            #print(df_daily[['Close','Volume','vol_20d', 'vol_10d', 'rsi', 'ema_10', 'ema_20']].tail(2)) #print selected columns
            print(f'\nlast_row_index_intra:{last_row_index_intra}, prev_row_index_intra:{prev_row_index_intra}')
            #print(df_15m[['Close','Volume','High', 'variance_high_5d']]) #print selected columns
        
        
        if(len(df_today) > 2 and df_today.iloc[-1]['Volume'] > df_today.iloc[-2]['Volume']  and df_today.iloc[-1]['Volume'] > df_today.iloc[-1]['vol_avg_ratio'] and df_today.iloc[-1]['vol_avg_ratio'] > 4):
            comment.append('Vol Spurt')
        
        if(df_daily.loc[last_row_index, 'crossover'] ):
            comment.append('crossover')
        
        if(df_daily.loc[last_row_index, 'open_eg_high'] ):
            comment.append('open_eg_high')
        
        if(df_daily.loc[last_row_index, 'open_eg_low'] ):
            comment.append('open_eg_low')

        #print(today_9_15, last_row_index, df_15m.loc[today_9_15,'Open'],df_daily.loc[last_row_index, 'Open'], df_daily.loc[last_row_index, 'change'])
        if(df_daily.loc[last_row_index, 'change'] > 0):
            if(df_15m.loc[today_9_15,'Open'] >= df_daily.loc[last_row_index, 'Close']):
                comment.append('+ Open')
            elif(df_15m.loc[today_9_15,'Open'] < df_daily.loc[last_row_index, 'Open']):
                comment.append('InvBearishOpen') #Inverse Bearish Open
        else:
            if(df_15m.loc[today_9_15,'Open'] < df_daily.loc[last_row_index, 'Close']):
                comment.append('- Open')
            elif(df_15m.loc[today_9_15,'Open'] >= df_daily.loc[last_row_index, 'Open']):
                comment.append('InvBullishOpen') #Inverse Bullish Open
                
        if(df_daily.loc[last_row_index,'Below_EMA_After_Consecutive']):
            num_cndl = df_daily.loc[last_row_index,'Consecutive_Above_EMA']
            comment.append(f'< EMA10-D({num_cndl})')
            
        if(df_daily.loc[last_row_index,'Above_EMA_After_Consecutive']):
            num_cndl = df_daily.loc[last_row_index,'Consecutive_Below_EMA']
            comment.append(f'> EMA10-D({num_cndl})')
            
        if(df_15m.loc[last_row_index_intra,'Below_EMA_After_Consecutive']):
            num_cndl = df_15m.loc[last_row_index_intra,'Consecutive_Above_EMA']
            comment.append(f'< EMA10-D({num_cndl})')
            
        if(df_15m.loc[last_row_index_intra,'Above_EMA_After_Consecutive']):
            num_cndl = df_15m.loc[last_row_index_intra,'Consecutive_Below_EMA']
            comment.append(f'> EMA10-15({num_cndl})')
        
        comment_concat =''
        if(len(comment)):
            comment_concat = ", ".join(comment)
        
        vol_ratio_threshold = 3
        conditions_intra = stk.conditions(df_15m, last_row_index_intra, prev_row_index_intra, vol_ratio_threshold)
       
        if(mode == 'dev'):
            print(conditions_intra)
                        
        new_row_intra = pd.DataFrame({
            'datetime': [last_row_index_intra],
            'strategy': '', 
            'ticker': [symbol_clean], 
            'Candle': df_15m.loc[last_row_index_intra,'Candle_Color'],
            'Pre_open_vol': df_pre_open.loc[pre_open_index, 'FINALQUANTITY'],
            'vol_D_5d': round(df_daily.loc[last_row_index,'vol_5d'],2),
            'vol_10d': round(df_15m.loc[last_row_index_intra,'vol_10d'],2),
            'rsi_D_15m' : rsi_daily_15m,
            'ema10_D_15m' : ema_daily_15m,
            'vwap_D_15m' : vwap_daily_15m,
            'comment': [comment_concat],
            
            #sort
            'rsi_d': round(df_daily.loc[last_row_index,'rsi'],2),
            'rsi_15':round(df_15m.loc[last_row_index_intra,'rsi'],2),
            'ema10_D' : ema10_daily,
            'ema10_15m' : ema10_15m,
            'vwap_D' : vwap10_daily,
            'vwap_15m' : vwap10_15m,
           
        })
         
        strategy_intra = []
        
        
        #pre-open check
        if pre_open_index:
            conditions_intra_pre = stk.pre_open_conditions(row_pre_open, df_15m, pre_open_index, last_row_index_intra, prev_row_index_intra)
            for key_pre, value_pre in conditions_intra_pre.items():
                if value_pre:
                    new_row_intra['strategy'] = key_pre
                    filtered_ticker_intra = pd.concat([filtered_ticker_intra, new_row_intra], ignore_index=True, axis=0)
       
             
        for key, value in conditions_intra.items():
            add_row = False
            if key == 'high_vol': # Discarded initally to detecct any other strategy there
                continue
            
            if( conditions_intra[key] and (key == 'green_red' or key == 'red_green') and df_today.loc[last_row_index_intra,'vol_avg_ratio'] > 3):
                strategy_intra.append(key + ' + IHV')  # Intra High Vol
            elif(conditions_intra[key] and df_today.loc[last_row_index_intra,'vol_avg_ratio'] > 3):
                strategy_intra.append(key + ' + IHV')
            elif(conditions_intra[key] and  (key != 'green_red' and key != 'red_green')):
                strategy_intra.append(key)
                
        if(len(strategy_intra) < 1 and (conditions_intra['high_vol'] and  df_today.loc[last_row_index_intra,'vol_avg_ratio'] > 3)):
            strategy_intra.append(key)
            
        if len(strategy_intra):
            new_row_intra ['strategy'] = ", ".join(strategy_intra)          
            filtered_ticker_intra = pd.concat([filtered_ticker_intra, new_row_intra], ignore_index=True, axis=0)
        
       
        if(df_15m.loc[last_row_index_intra, 'Volume'] > df_daily.loc[last_row_index, 'vol_avg_5d'] * 0.3 and df_daily.loc[last_row_index, 'vol_avg_5d'] > 2000000):
             new_row_intra['strategy'] = 'Vol > 30% Avg Vol'
             filtered_ticker_intra = pd.concat([filtered_ticker_intra, new_row_intra], ignore_index=True, axis=0)
 
        if(conditions_intra['high_vol'] and df_daily.loc[last_row_index, 'variance_high_5d'] < 5 and (df_15m.loc[today_9_15, 'Open']  > df_daily.loc[last_row_index, 'max_5d'] or df_15m.loc[today_9_15, 'Open']  > df_daily.loc[last_row_index, 'max_3d']) ):
              new_row_intra['strategy'] = 'Open > high'
              filtered_ticker_intra = pd.concat([filtered_ticker_intra, new_row_intra], ignore_index=True, axis=0)
        
        if(conditions_intra['high_vol'] and df_daily.loc[last_row_index, 'variance_low_5d'] < 5 and (df_15m.loc[today_9_15, 'Open']  < df_daily.loc[last_row_index, 'min_5d'] or df_15m.loc[today_9_15, 'Open']  < df_daily.loc[last_row_index, 'min_3d']) ):
              new_row_intra['strategy'] = 'Open < low'
              filtered_ticker_intra = pd.concat([filtered_ticker_intra, new_row_intra], ignore_index=True, axis=0)

        inside_cross_up = (
            (conditions_intra['high_vol']) &
            (conditions['inside']) &
            (df_15m.loc[last_row_index_intra, 'Close'] > df_daily.loc[last_row_index, 'High']) &
            ((df_15m.loc[last_row_index_intra, 'Open'] < df_daily.loc[last_row_index, 'High']) |
              (df_15m.loc[last_row_index_intra, 'Low'] < df_daily.loc[last_row_index, 'High']) )            
        )
        
        inside_cross_down = (
            (conditions_intra['high_vol']) &
            (conditions['inside']) &
            (df_15m.loc[last_row_index_intra, 'Close'] < df_daily.loc[last_row_index, 'Low']) &
            ((df_15m.loc[last_row_index_intra, 'Open'] > df_daily.loc[last_row_index, 'Low']) |
              (df_15m.loc[last_row_index_intra, 'High'] > df_daily.loc[last_row_index, 'Low']) )            
        )
        
        df_15m_prev_row_index = df_15m.index[df_15m.index.get_loc(last_row_index_intra) - 1]
        inside_intra_cross_up = (
            (conditions_intra['high_vol']) &
            (df_15m.loc[df_15m_prev_row_index, 'inside']) &
            (df_15m.loc[last_row_index_intra, 'Close'] > df_15m.loc[df_15m_prev_row_index, 'High']) &
            ((df_15m.loc[last_row_index_intra, 'Open'] < df_15m.loc[df_15m_prev_row_index, 'High']) |
              (df_15m.loc[last_row_index_intra, 'Low'] < df_15m.loc[df_15m_prev_row_index, 'High']) )            
        )
        
        inside_intra_cross_down = (
            (conditions_intra['high_vol']) &
            (df_15m.loc[df_15m_prev_row_index, 'inside']) &
            (df_15m.loc[last_row_index_intra, 'Close'] < df_15m.loc[df_15m_prev_row_index, 'Low']) &
            ((df_15m.loc[last_row_index_intra, 'Open'] > df_15m.loc[df_15m_prev_row_index, 'Low']) |
              (df_15m.loc[last_row_index_intra, 'High'] > df_15m.loc[df_15m_prev_row_index, 'Low']) )            
        )
        
  
        if(inside_cross_up):
            new_row_intra['strategy'] = 'Inside Crossed Up'
            filtered_ticker_intra = pd.concat([filtered_ticker_intra, new_row_intra], ignore_index=True, axis=0)
        if(inside_cross_down):
            new_row_intra['strategy'] = 'Inside Crossed Down'
            new_row_intra = pd.DataFrame({'datetime': [last_row_index_intra],'strategy': ['Inside Crossed Down'], 'ticker': [symbol], 'comment': [comment_concat]})
            filtered_ticker_intra = pd.concat([filtered_ticker_intra, new_row_intra], ignore_index=True, axis=0)
        if(inside_intra_cross_up):
            new_row_intra['strategy'] = '15m Inside Crossed Up'
            filtered_ticker_intra = pd.concat([filtered_ticker_intra, new_row_intra], ignore_index=True, axis=0)
        if(inside_intra_cross_down):
            new_row_intra['strategy'] = '15m Inside Crossed Down'
            filtered_ticker_intra = pd.concat([filtered_ticker_intra, new_row_intra], ignore_index=True, axis=0)

df_pre_open_with_gap = df_pre_open[  (df_pre_open['FINAL'] > 200)  & ((df_pre_open['strategy'] != '')  |  (df_pre_open['gap_up'] == 1)  | (df_pre_open['gap_down'] == 1) | (df_pre_open['high_vol']  == 1))]
df_pre_open_with_gap_sort = df_pre_open_with_gap.sort_values(['color','FINALQUANTITY'], ascending=[0,0])
# df_pre_open_with_gap_sort['FINALQUANTITY'] = df_pre_open_with_gap_sort['FINALQUANTITY'].apply(lambda x: f"{x:,}".replace(",", "X").replace("X", ","))

df_pre_open_with_gap_sort = df_pre_open_with_gap_sort[['SYMBOL', 'color','CHNG', '%CHNG','FINALQUANTITY', 'RSI', 'vol_5d','vol_10d','doji','PDC_below_VWAP', 'PDC_below_EMA10', 'strategy']]
df_pre_open_with_gap_sort['RSI'] = df_pre_open_with_gap_sort['RSI'].astype(int)
df_pre_open_with_gap_sort['PDC_below_VWAP'] = df_pre_open_with_gap_sort['PDC_below_VWAP'].astype(int)
df_pre_open_with_gap_sort['PDC_below_EMA10'] = df_pre_open_with_gap_sort['PDC_below_EMA10'].astype(int)
# print(df_pre_open_with_gap_sort)
# exit(1)

stk.print_groped(df_pre_open_with_gap_sort)
#exit(1)

# print(df_pre_open_with_gap_sort)
#stk.print_colored_rows(df_pre_open_with_gap_sort)
df_pre_open_with_gap_sort['date'] = today_ymd
df_pre_open_with_gap_sort = df_pre_open_with_gap_sort.set_index('date', drop=True)
pre_open_tickers = 'pre_open_ticker.csv'

current_time = pd.Timestamp.now()
given_time_pre = pd.to_datetime(today.strftime('%Y-%m-%d 09:30:00'))
is_less_pre = current_time < given_time_pre

if is_less_pre:
    stk.csv_write_append(df_pre_open_with_gap_sort, pre_open_tickers)

# Remove duplicate values from the "column1" column
filtered_ticker = filtered_ticker.drop_duplicates(subset='ticker')   
filtered_ticker = filtered_ticker.set_index('date', drop=True)
filtered_ticker = filtered_ticker.sort_values(['Candle','Pre_open_vol','Pre_open'], ascending=[0,0,0])
filtered_ticker_intra = filtered_ticker_intra.set_index('datetime', drop=True)
filtered_ticker_intra = filtered_ticker_intra.drop_duplicates(subset='ticker')
columns_to_display = ['ticker', 'Candle', 'vol_10d', 'Pre_open_vol','vol_D_5d','rsi_D_15m','ema10_D_15m', 'vwap_D_15m', 'strategy', 'comment']
filtered_ticker_intra = filtered_ticker_intra.sort_values(['Candle','vol_10d','Pre_open_vol', 'rsi_d', 'rsi_15', 'ema10_D','ema10_15m', 'vwap_D', 'vwap_15m'], ascending=[0,0,0,0,0,1,1,1,1])
filtered_ticker_intra['Pre_open_vol'] = filtered_ticker_intra['Pre_open_vol'].apply(lambda x: f"{x:,}".replace(",", "X").replace("X", ","))


given_time = pd.to_datetime(today.strftime('%Y-%m-%d 15:40:00'))
is_less = current_time < given_time

daily_columns_to_display = ['ticker', 'Candle',  'Pre_open_%chang', 'Pre_open_vol', 'Pre_open', 'strategy']


# if(is_less == False):
if(is_less):
    if(len(filtered_ticker) and mode != 'dev' ):
        print("#### Yestrday #### Date : ", last_row_index )
        print(filtered_ticker[daily_columns_to_display])
        
        print("#### Intraday #### Time : ", last_row_index_intra)
        print(filtered_ticker_intra[columns_to_display])
        filtered_ticker_file = 'filtered_ticker.csv'
        stk.csv_write_append(filtered_ticker, filtered_ticker_file)
            
    else:
        print("#### Yestrday #### Date : ", last_row_index )
        print(filtered_ticker)
            
    if(len(filtered_ticker_intra) and mode != 'dev'):
        filtered_ticker_file_intra = 'filtered_ticker_intra.csv'
        stk.csv_write_append(filtered_ticker_intra, filtered_ticker_file_intra)  
    else:
        print("#### Intraday #### Time : ", last_row_index_intra)
        print(filtered_ticker_intra[columns_to_display])
else:
    print("Print only")
    print("#### Yestrday #### Date : ", last_row_index )
    print(filtered_ticker)
    print("#### Intraday #### Time : ", last_row_index_intra)
    print(filtered_ticker_intra[columns_to_display])
    
duration = 500  # milliseconds
freq = 2500  # Hz
winsound.Beep(freq, duration)
exit(1)
