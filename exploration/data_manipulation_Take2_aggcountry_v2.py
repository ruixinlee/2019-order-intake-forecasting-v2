import pandas as pd
import numpy as np
import datetime as dt
from calendar import monthrange
from pandas.tseries.offsets import MonthEnd

def find_eom(data, country, date_to_use):
    vista_dict = []
    df_mend = create_month_end(data, date_to_use)
    df_sub = data[data['LEVEL_3_REPORTING'] == country].sort_values([date_to_use])
    df_sub['forwarddiff'] = df_sub['SOLD_ORDER_NUMBERS'].shift(-1) / df_sub['SOLD_ORDER_NUMBERS']
    # print( df_sub['SOLD_ORDER_NUMBERS'].pct_change(-1))
    firstdate = df_sub[date_to_use].min()
    lastdate = df_sub[date_to_use].max()

    df_mend_sub = df_mend[(df_mend['MonthEnd'] >= firstdate) & (df_mend['MonthEnd'] <= lastdate)]

    df_mend_sub = df_mend_sub.to_dict('record')
    df_mend_sub = sorted(df_mend_sub, key=lambda k: k['MonthEnd'])


    for me in df_mend_sub:
        df_sub2 = df_sub[
            (df_sub[date_to_use] >= me['prior']) & (df_sub[date_to_use] <= me['post'])]
        df_sub2 = df_sub2.to_dict('record')
        if df_sub2:
            max_sales = max([i['SOLD_ORDER_NUMBERS'] for i in df_sub2])
            min_forw = min([i['forwarddiff'] for i in df_sub2])
            me['eom'] = max([i[date_to_use] for i in df_sub2 if
                             i['SOLD_ORDER_NUMBERS'] == max_sales or i['forwarddiff'] == min_forw])
            # print(me['eom'])
        else:

            df_sub2 = df_sub[(df_sub[date_to_use].map(lambda x: x.month) == me['prior'].month)
                             & (df_sub[date_to_use].map(lambda x: x.year) == me['prior'].year)]
            me['eom'] = df_sub2[date_to_use].max()
           # print('error')
    eom_list = [me['eom'] for me in df_mend_sub]
    df_sub['eom'] = [i in eom_list for i in df_sub[date_to_use].tolist()]

    # create reporting month
    df_sub['reporting_date'] = np.nan
    reporting_month = df_sub[date_to_use].min() + MonthEnd(0)
    for i, row in df_sub.iterrows():
        df_sub.set_value(i, 'reporting_date', reporting_month)
        if row['eom']:
            reporting_month =  next_end_of_month(reporting_month,1)

    df_sub['reporting_date'] =  pd.to_datetime(df_sub['reporting_date'])
    df_max_date = df_sub[[date_to_use, 'reporting_date']].groupby('reporting_date',
                                                 as_index=False)[date_to_use].max()
    df_max_date = pd.DataFrame(df_max_date.rename(columns={date_to_use:'reporting_max_date'}))

    df_sub = df_sub.merge(df_max_date, on= ['reporting_date'], how='outer')

    df_sub = df_sub.to_dict('records')
    df_sub = sorted(df_sub, key=lambda k: k[date_to_use])
    vista_dict.extend(df_sub)
    vista_final_eom = pd.DataFrame.from_records(vista_dict)

    return(vista_final_eom)

def next_end_of_month(date, delta):
    mth = (date.month + delta) % 12
    if not mth:
        mth = 12
    yr = date.year + (date.month + delta - 1) // 12
    lastday = monthrange(yr, mth)[1]
    return(pd.to_datetime(dt.datetime(year=yr, month = mth, day = lastday)) + MonthEnd(0) )

def aggregte_orders(data, groupby_cols):
    data = data.groupby(groupby_cols, as_index=False)['ORDER_NO'].count()
    data = data.rename(columns = {'ORDER_NO':'SOLD_ORDER_NUMBERS'})
    return(data)

def create_month_end(data, date_to_use):
    df_mend= pd.DataFrame(data[date_to_use] + MonthEnd(0)).drop_duplicates().rename(columns={date_to_use:'MonthEnd'})
    df_mend = df_mend.sort_values('MonthEnd')
    df_mend['prior'] = df_mend['MonthEnd'] - dt.timedelta(3)
    df_mend['post'] = df_mend['MonthEnd'] + dt.timedelta(3) # not 6 for differencing
    return(df_mend)

#vista_curves_master = vista_curves
#vista_curves = vista_curves_master
#vista_eom = vista_final_eom
def create_curves(vista_curves, vista_eom, country):
    df_list = []
    vista_eom = vista_eom.drop(['SOLD_ORDER_NUMBERS','forwarddiff'], axis = 1)
    vista_curves = pd.merge(vista_curves,vista_eom, how='left', on = ['LEVEL_5_REPORTING','LEVEL_3_REPORTING',
                                                                       'CUSTOMER_HANDOVER_DATE'] )
    vista_curves = vista_curves[vista_curves['LEVEL_3_REPORTING'] == country]
    vista_curves = vista_curves.rename(columns={'reporting_date': 'CHD_reporting_date', 'reporting_max_date': 'CHD_reporting_max_date'})
    vista_curves['CPDD_reporting_date'] = vista_curves['CURRENT_PLANNED_DELIVERY_DATE'] + MonthEnd(0)
    
    vista_retail_total = vista_curves.groupby(['MODEL_FAMILY', 'CHD_reporting_date']).agg({'SOLD_ORDER_NUMBERS':'sum'}).reset_index()
    vista_retail_total = vista_retail_total.rename(columns = {'SOLD_ORDER_NUMBERS': 'VISTA_RETAIL_ACTUALS_AND_FORECAST'})       
    
    #vista_curves = vista_curves.drop(['CUSTOMER_HANDOVER_DATE', 'CHD_reporting_max_date', 'CURRENT_PLANNED_DELIVERY_DATE'], axis = 1)
    col_name = vista_curves.columns.tolist()
    vista_curves = vista_curves[vista_curves['SOLD_ORDER_DATE'] <= vista_curves['CURRENT_PLANNED_DELIVERY_DATE']]
    vista_curves = vista_curves.groupby([i for i in col_name if i not in ['eom', 'CUSTOMER_HANDOVER_DATE', 'CHD_reporting_max_date', 'CURRENT_PLANNED_DELIVERY_DATE', 'SOLD_ORDER_NUMBERS']]).agg({'SOLD_ORDER_NUMBERS': 'sum'}).reset_index()    
    vista_curves['pre_order_days'] = (vista_curves['SOLD_ORDER_DATE'] - vista_curves['CPDD_reporting_date']).dt.days
    vista_curves = vista_curves.drop(['SOLD_ORDER_DATE', 'LEVEL_5_REPORTING', 'LEVEL_3_REPORTING'], axis=1)

    for model in vista_curves['MODEL_FAMILY'].unique().tolist():
        #print(model)
        vista_curves_sub = vista_curves[vista_curves['MODEL_FAMILY'] == model ]
                
        #Following will also remove stock orders
        vista_curves_sub = vista_curves_sub[(vista_curves_sub['pre_order_days']<=0) & (vista_curves_sub['pre_order_days']>=-365)]
        min_order_days = vista_curves_sub.pre_order_days.min()
        range_order_day = [i for i in np.arange(-2000, 1)]
        
        handover_months = pd.Series()
        for num_months in range(4):
            #print((pd.Series(vista_curves_sub['CPDD_reporting_date'].unique()) + MonthEnd(num_months))[0])
            handover_months = handover_months.append((pd.Series(vista_curves_sub['CPDD_reporting_date'].unique()) + MonthEnd(num_months)))
        

        xx, yy, zz = np.meshgrid(range_order_day, vista_curves_sub['CPDD_reporting_date'].unique().tolist(), handover_months.unique().tolist())
        df_new = {'pre_order_days': xx.flatten(), 'CPDD_reporting_date': yy.flatten(), 'CHD_reporting_date': zz.flatten()}
        df_new = pd.DataFrame(df_new)
        df_new['CPDD_reporting_date'] = pd.to_datetime(df_new['CPDD_reporting_date'])
        df_new['CHD_reporting_date'] = pd.to_datetime(df_new['CHD_reporting_date'])

        #df_new = vista_curves_sub['BRAND'].unique()[0]
        #df_new = vista_curves_sub['SEGMENT'].unique()[0]
        #df_new = vista_curves_sub['MODEL_FAMILY'].unique()[0]
        
        ##hcc##: similar to yy.
        df_dates = vista_curves_sub[['CPDD_reporting_date', 'BRAND', 'SEGMENT', 'MODEL_FAMILY']].drop_duplicates()
        df_new = df_new.merge(df_dates, on=['CPDD_reporting_date'], how='left')
        df_new['months_before_handover'] = - (df_new['CPDD_reporting_date'].dt.month + df_new['CPDD_reporting_date'].dt.year*12)\
                                            + (df_new['CHD_reporting_date'].dt.month + df_new['CHD_reporting_date'].dt.year*12)    
        df_new = df_new[(df_new['months_before_handover'] >= 0) & (df_new['months_before_handover'] <= 3)]
        
        #vista_curves_sub['reporting_date'] = pd.to_datetime(vista_curves_sub['reporting_date'])
        df_new = df_new.merge(vista_curves_sub, how='left',
                              on=['CHD_reporting_date', 'CPDD_reporting_date', 'pre_order_days',
                                  'BRAND','SEGMENT', 'MODEL_FAMILY']).fillna(0)
        df_new = pd.merge(df_new,vista_retail_total, on = ['MODEL_FAMILY', 'CHD_reporting_date'], how = 'left')

        # reporting_date = df_new['reporting_date'].unique()[5]
        df_model_all = []
        for CPDD_reporting_date in df_new['CPDD_reporting_date'].unique():
            df_new2 = df_new[df_new['CPDD_reporting_date'] == CPDD_reporting_date]
            for CHD_reporting_date in df_new2['CHD_reporting_date'].unique():
                df_new3 = df_new2[df_new2['CHD_reporting_date'] == CHD_reporting_date]
                df_new3 = df_new3.sort_values(by=['pre_order_days'])
                df_new3['Cum_Sold_Order_Numbers'] = df_new3['SOLD_ORDER_NUMBERS'].cumsum()
                df_model_all.append(df_new3)
                
        if len(df_model_all) != 0:
            df_model = pd.concat(df_model_all)
        else:
            df_model = pd.DataFrame()
        df_list.append(df_model)
        #import time
        #time.sleep(1)
    df_all = pd.concat(df_list)
    df_all = df_all[df_all['pre_order_days'] >= -360] #-180
    #df_all_2 = df_all.pivot_table(index=['CHD_reporting_date', 'CPDD_reporting_date', 'BRAND', 'MODEL_FAMILY','SEGMENT', 'months_before_handover', 'VISTA_RETAIL_ACTUALS_AND_FORECAST'],
    #                            columns='pre_order_days', values='Cum_Sold_Order_Numbers').reset_index()
    #df_all_1 = df_all.pivot_table(index=['CHD_reporting_date', 'CPDD_reporting_date', 'BRAND', 'MODEL_FAMILY','SEGMENT', 'months_before_handover'],
    #                            columns='pre_order_days', values='Cum_Sold_Order_Numbers').reset_index()
    #test = pd.merge(df_all_1, df_all_2, how = 'left', on = ['CHD_reporting_date', 'CPDD_reporting_date', 'BRAND', 'MODEL_FAMILY','SEGMENT', 'months_before_handover', -360])
    df_all['VISTA_RETAIL_ACTUALS_AND_FORECAST'] = df_all['VISTA_RETAIL_ACTUALS_AND_FORECAST'].fillna(0)    
    df_all = df_all.pivot_table(index=['CHD_reporting_date', 'CPDD_reporting_date', 'BRAND', 'MODEL_FAMILY','SEGMENT', 'months_before_handover', 'VISTA_RETAIL_ACTUALS_AND_FORECAST'],
                                columns='pre_order_days', values='Cum_Sold_Order_Numbers').reset_index()
    return (df_all)

def write_cut_off_point_comparision(vista_curves, vista_df, vista_final_eom):
    month_end_comparison = vista_curves[
        ['reporting_date', 'reporting_max_date', 'BRAND', 'MODEL_FAMILY', 'RETAIL_ACTUALS_AND_FORECAST',
         'RETAIL_FORECAST', 'Target_Met',
         'RETAIL_FORECAST_cutoff', 0]]
    month_end_comparison = month_end_comparison.rename(columns={0: 'new_total_sold'})
    mec2 = vista_df.copy(deep=True)
    mec2['end_of_month_date'] = mec2['CUSTOMER_HANDOVER_DATE'] + MonthEnd(0)
    mec2 = mec2.groupby(['LEVEL_5_REPORTING', 'LEVEL_3_REPORTING', 'BRAND', 'MODEL_FAMILY',
                         'end_of_month_date'], as_index=False).ORDER_NO.count()
    mec2 = mec2.rename(columns={'end_of_month_date': 'reporting_date', 'ORDER_NO': 'eom_total_sold'})
    mec3 = mec2.merge(month_end_comparison, on=['BRAND', 'MODEL_FAMILY', 'reporting_date'], how='inner')
    mec3.to_csv('precentage_total_sold_to_forecast.csv')

    vista_final_eom2 = vista_final_eom[['CUSTOMER_HANDOVER_DATE', 'eom']]
    vista_df2 = vista_df.merge(vista_final_eom2, on='CUSTOMER_HANDOVER_DATE', how='inner')
    vista_df2.to_csv('vista_with_calculated_oem.csv')

    return()

def aggregate_countries (df_VISTA, df_SPRINT, aggregate_market_map_path):
    mkt_mapping = pd.read_csv(aggregate_market_map_path)
    level3_mapping = dict(zip(mkt_mapping['LEVEL_3_REPORTING'],mkt_mapping['TEMP_LEVEL_3_REPORTING']))
    region_mapping = dict(zip(mkt_mapping['LEVEL_3_REPORTING'],mkt_mapping['TEMP_LEVEL_5_REPORTING']))

    df_VISTA['LEVEL_5_REPORTING'] = df_VISTA['LEVEL_3_REPORTING'].map(region_mapping)
    df_VISTA['LEVEL_3_REPORTING'] = df_VISTA['LEVEL_3_REPORTING'].map(level3_mapping)
    #df_VISTA.groupby(['LEVEL_5_REPORTING', 'LEVEL_3_REPORTING']).size().reset_index(name='Freq')

    df_SPRINT['LEVEL_5_REPORTING'] = df_SPRINT['LEVEL_3_REPORTING'].map(region_mapping)
    df_SPRINT['LEVEL_3_REPORTING'] = df_SPRINT['LEVEL_3_REPORTING'].map(level3_mapping)
    df_SPRINT = df_SPRINT.groupby(['LEVEL_5_REPORTING', 'LEVEL_3_REPORTING', 'BRAND', 'MODEL_FAMILY', 'ACTUAL_DATE', 'reporting_date']).sum().reset_index()
    df_SPRINT['RETAIL_FORECAST_cutoff'] = df_SPRINT['RETAIL_FORECAST'] * 0.9
    df_SPRINT['Target_Met'] = df_SPRINT['RETAIL_ACTUALS_AND_FORECAST'] >= (df_SPRINT['RETAIL_FORECAST'] * 0.9)


    #df_SPRINT.groupby(['LEVEL_5_REPORTING', 'LEVEL_3_REPORTING']).size().reset_index(name='Freq')
        
    #df_SPRINT[(df_SPRINT['MODEL_FAMILY']=='RANGE ROVER EVOQUE') & (df_SPRINT['LEVEL_3_REPORTING'].isin(overseas_list))].groupby('ACTUAL_DATE').count()
    #test = df_VISTA[df_VISTA['LEVEL_3_REPORTING'] == 'Europe_LowVol']
    return df_VISTA, df_SPRINT
    
#country_list = ['Singapore']
#country_list = ['UK NSC','Chery JLR (China Sales)','USA','France','Spain']
#country = 'UK NSC'
#country = 'Overseas'
    
def Create_Historical_Glide_Paths(vista_df_master, GPnS_df_master, country, CHD_reporting_date_cutoff = '2018-02-28'):
        vista_df = vista_df_master
        vista_df = vista_df[vista_df['LEVEL_3_REPORTING'] == country]
        Level_5_Reporting = vista_df['LEVEL_5_REPORTING'].unique()[0]

        GPnS_df = GPnS_df_master
        GPnS_df = GPnS_df[GPnS_df['LEVEL_3_REPORTING'] == country]
        GPnS_df = GPnS_df.drop(['ACTUAL_DATE', 'LEVEL_5_REPORTING', 'LEVEL_3_REPORTING', 'DAILYPREDICTEDRETAILS'],
                               axis=1)
        GPnS_df = GPnS_df.rename(columns = {'reporting_date': 'CHD_reporting_date'})
        #reporting_date = EOM of ACTUAL_DATE (Date of actual retail number);

        #Get the sales EOM date. Use all data, regardless of model, to spot spikes/troughs.
        vista_df = vista_df.drop_duplicates('ORDER_NO')
        vista_eom = aggregte_orders(vista_df, ['LEVEL_5_REPORTING', 'LEVEL_3_REPORTING', 'CUSTOMER_HANDOVER_DATE'])
        vista_final_eom = find_eom(vista_eom, country, 'CUSTOMER_HANDOVER_DATE')
        #vista_final_eom.to_csv('vista_final_eom.csv')

        #Create Glide Path
        vista_curves = aggregte_orders(vista_df,
                                       ['LEVEL_5_REPORTING', 'LEVEL_3_REPORTING', 'CURRENT_PLANNED_DELIVERY_DATE',
                                        'BRAND',
                                        'SEGMENT',
                                        'MODEL_FAMILY',
                                        'SOLD_ORDER_DATE', 'CUSTOMER_HANDOVER_DATE'])
        vista_curves = create_curves(vista_curves, vista_final_eom, country)
        ##End of month of VISTA data until which data is compelete and that of SPRINT data that has actual retail.   
        vista_curves = vista_curves[vista_curves['CHD_reporting_date'] <= CHD_reporting_date_cutoff]


        ### produce cut off point comparision
        # write_cut_off_point_comparision(vista_curves, vista_df, vista_final_eom)
        vista_curves = pd.merge(vista_curves, GPnS_df, on=['BRAND', 'MODEL_FAMILY', 'CHD_reporting_date'], how='left')
        #vista_curves = vista_curves[~pd.isnull(vista_curves['RETAIL_ACTUALS_AND_FORECAST'])]

        i_cols = [i for i in vista_curves.columns.values.tolist() if isinstance(i, int)]
        for i in i_cols:
            vista_curves[i] = vista_curves[i] / vista_curves['VISTA_RETAIL_ACTUALS_AND_FORECAST'] * vista_curves['RETAIL_ACTUALS_AND_FORECAST']
            vista_curves[i] = vista_curves[i] / vista_curves[['RETAIL_FORECAST', 'RETAIL_ACTUALS_AND_FORECAST']].max(axis=1)
        vista_curves['0day_sold_order_level%'] = vista_curves[0]
        vista_curves = vista_curves[vista_curves['0day_sold_order_level%'].notnull()]
        vista_curves.to_pickle('../pickle_vault/pre_orders_curves/CPDD_{}-{}.pkl'.format(Level_5_Reporting, country))
    return()