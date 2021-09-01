import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import functions as func

def read_df_2016(filename):
    df_2016 = pd.read_csv(input_path.format(filename))
    # if df_16_F['RETAIL_ACTUALS_AND_FORECAST'].dtype == 'O':
    #     df_16_F['RETAIL_ACTUALS_AND_FORECAST'] = pd.to_numeric(
    #         df_16_F['RETAIL_ACTUALS_AND_FORECAST'].str.replace(',', ''), errors='coerce')
    # if df_16_F['RETAIL_FORECAST'].dtype == 'O':
    #     df_16_F['RETAIL_FORECAST'] = pd.to_numeric(df_16_F['RETAIL_FORECAST'].str.replace(',', ''), errors='coerce')
    return(df_2016)

def read_df_2013(filename):
    df_14_16 = pd.read_csv(input_path.format(filename))
    df_14_16 = df_14_16.rename(columns={'Level 5 - Region': 'LEVEL_5_REPORTING',
                                        'Level 3 - Markets': 'LEVEL_3_REPORTING',
                                        'Brand': 'BRAND',
                                        'Model Family': 'MODEL_FAMILY',
                                        'Actual Date': 'ACTUAL_DATE',
                                        'Retail Actuals And Forecast': 'RETAIL_ACTUALS_AND_FORECAST',
                                        'Retail Forecast': 'RETAIL_FORECAST'})
    df_14_16['LEVEL_5_REPORTING'] = df_14_16['LEVEL_5_REPORTING'].replace({'UK': 'United Kingdom'})
    if df_14_16['RETAIL_ACTUALS_AND_FORECAST'].dtype == 'O':
        df_14_16['RETAIL_ACTUALS_AND_FORECAST'] = pd.to_numeric(
            df_14_16['RETAIL_ACTUALS_AND_FORECAST'].str.replace(',', ''), errors='coerce')
    if df_14_16['RETAIL_FORECAST'].dtype == 'O':
        df_14_16['RETAIL_FORECAST'] = pd.to_numeric(df_14_16['RETAIL_FORECAST'].str.replace(',', ''), errors='coerce')
    return(df_14_16)
    
if __name__ == '__main__':
    import os
    os.chdir('C:\\Users\\hchang\\Documents\\ACoE_Forecasting\\Order Intake\\SPRINT 10\\py')
    input_path = '../input_vault/csv/{}'
    output_path = '../input_vault/pickle/{}'
    
    #2013-2019 forecast_OIDB.csv
    df_16_F = read_df_2016('SPRINT_OIDB_HISTORICAL_20180316_2116.csv')
    df_16_F['ACTUAL_DATE'] = pd.to_datetime(df_16_F['ACTUAL_DATE'], format="%Y-%m-%d")
    
    df_14_16 =read_df_2013('OIDB_FORECAST_HISTORY.csv')
    date_cols = ['ACTUAL_DATE']
    df = df_14_16
    for x in date_cols:
        print(x)
        df[x] = df[x].str.split(' ').str[0]
        dt_list = df[x].str.split('/').tolist()
        dt_list = [x if isinstance(x, (list)) else [None, None, None] for x in dt_list]
        df_date = pd.DataFrame(dt_list, columns=['day', 'month', 'year'])
        ##hcc##
        df_date['date'] = df_date['day'] + "/" + df_date['month'] + "/" + df_date['year']
        df[x] = pd.to_datetime(df_date['date'], format="%d/%m/%Y")
    df_14_16 = df

    df = pd.concat([df_16_F, df_14_16], ignore_index=True)
    df['DAILYPREDICTEDRETAILS'] = df['DAILYPREDICTEDRETAILS'].fillna(0)
    name_code_dict = func.get_name_code(input_path.format('name_code_mapping.csv'))
    df['MODEL_FAMILY'] = df['MODEL_FAMILY'].replace(name_code_dict).str.upper()
    df['BRAND'] = df['BRAND'].str.upper()
    df['reporting_date'] = df['ACTUAL_DATE'] + MonthEnd(0)
    
    #test = df[(df['LEVEL_3_REPORTING'] == 'Netherlands') & (df['MODEL_FAMILY'] == 'XF') & (df['ACTUAL_DATE'] =='2017-08-01')]
    ##Agg after fixing MODEL_FAMILY column.    
    group_by = ['LEVEL_5_REPORTING', 'LEVEL_3_REPORTING', 'BRAND', 'MODEL_FAMILY', 'ACTUAL_DATE', 'reporting_date']
    df = df.groupby(group_by, as_index=False).aggregate(np.sum)

    ##hcc##When was the forecast number predicted? Is it the number one month before the actual date?
    df = df[df['RETAIL_FORECAST']>0]
    df['RETAIL_FORECAST_cutoff'] = df['RETAIL_FORECAST'] * 0.9

    df['Target_Met'] = df['RETAIL_ACTUALS_AND_FORECAST'] >= (df['RETAIL_FORECAST'] * 0.9)

    df.to_pickle(output_path.format('OIDB_FORECAST_HISTORY.pkl'))
    df.to_csv(output_path.format('OIDB_FORECAST_HISTORY.csv'))
    
    print('Null in dates:')
    for x in date_cols:
        num_null = df[pd.isnull(df[x])].shape[0]
        print('{}:{}'.format(x,num_null))