import os
import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import Dataset
import functions as func
from pandas.tseries.offsets import MonthEnd
import numpy as np

def ClassFeat_Future(num_future_months = 15, model_code_name_map_path, recent_sprint_bq_billing_project, aggregate_market_map_path):
    num_future_months
    :param model_code_name_map_path: Corrects the model family column (e.g. '../input_vault/csv/name_code_mapping.csv')
    :param recent_sprint_bq_billing_project: Google BigQuery Account Project ID (e.g. 'jlr-dl-cat')
        :param aggregate_market_map_path: Path of the Level3 & Level5 to new Level3 & new Level5 relation table (e.g. '../input_vault/csv/LEVEL3_LEVEL5_TEMPLEVEL3_mapping.csv')
    :param prediction_cases_mkt_unagg_path: Path of the prediction dataset, the one whose Level3 is unaltered to the new Level3 for country aggregation (e.g. '../input_vault/csv/prediction_cases_mkt_unagg.csv')
    :param prediction_file_path: Path to the file containing the prediction cases (e.g. '../input_vault/csv/prediction_cases.csv').

    :type recent_sprint_bq_billing_project: str.

    segment_inc_col = True


    # Query for data, which also includes (most) data manipulations.
    months = num_future_months - 1
    client = bigquery.Client(project = recent_sprint_bq_billing_project)
    query_job = client.query(
    ('''
    #standardSQL
    with table1 as
    (SELECT
    DATE(CAST(SUBSTR(CALENDAR_PERIOD, 0, 4) AS INT64), CAST(SUBSTR(CALENDAR_PERIOD, -2) AS INT64), 1) as CALENDAR_PERIOD,
    DATE(CAST(SUBSTR(YEAR_MON_ACT , 0, 4) AS INT64), CAST(SUBSTR(YEAR_MON_ACT, -2) AS INT64), 1) as YEAR_MON_ACT,
    LEVEL_3_REPORTING,
    MODEL_FAMILY_DESC,
    RETAIL_FORECAST,
    BRAND_DESC as BRAND
    FROM `jlr-dl-edw.JLR_DL_EDW_DIM_GPSR_ACCESS_CONFIDENTIAL.VIEW_FACT_SPRINT_METRICS_FLAT`
    WHERE CURRENT_VERSION_FLAG = 'Y' and DATE(CAST(SUBSTR(CALENDAR_PERIOD, 0, 4) AS INT64), CAST(SUBSTR(CALENDAR_PERIOD, -2) AS INT64), 1) >= DATE(CAST(SUBSTR(YEAR_MON_ACT , 0, 4) AS INT64), CAST(SUBSTR(YEAR_MON_ACT, -2) AS INT64), 1)
    )
    
    Select 
    CALENDAR_PERIOD,
    YEAR_MON_ACT,
    LEVEL_3_REPORTING,
    MODEL_FAMILY_DESC as MODEL_FAMILY,
    SUM(RETAIL_FORECAST) as RETAIL_FORECAST,
    EXTRACT(YEAR FROM CALENDAR_PERIOD) AS year,
    EXTRACT(MONTH FROM CALENDAR_PERIOD) AS month,
    BRAND
    FROM table1
    WHERE CALENDAR_PERIOD >= (select min(CALENDAR_PERIOD) from table1) and
    CALENDAR_PERIOD <= (SELECT DATE_ADD((select min(CALENDAR_PERIOD) from table1), INTERVAL {} MONTH))
    GROUP BY CALENDAR_PERIOD, YEAR_MON_ACT, LEVEL_3_REPORTING, MODEL_FAMILY_DESC, BRAND
    ''').format(months)
    )
    results = query_job.result()
    
    # Save queried data as a list of tuples
    data_list = []
    for row in results:
        data_list.append((row.CALENDAR_PERIOD, row.YEAR_MON_ACT, row.LEVEL_3_REPORTING, row.MODEL_FAMILY, row.RETAIL_FORECAST, row.year, row.month, row.BRAND))

    # Convert data list into dataframe
    labels = ['CHD_reporting_date', 'YEAR_MON_ACT', 'LEVEL_3_REPORTING', 'MODEL_FAMILY','RETAIL_FORECAST', 'CHD_Year', 'CHD_Month', 'BRAND']
    sprint_df = pd.DataFrame.from_records(data_list, columns=labels)
    
    # Corrects the model family column
    name_code_dict = func.get_name_code(model_code_name_map_path)
    model_family = sprint_df['MODEL_FAMILY'].unique().tolist()
    sprint_df['MODEL_FAMILY'] = sprint_df['MODEL_FAMILY'].replace(name_code_dict).str.upper()
    
    # Convert dates to date format                                                                        
    sprint_df['CHD_reporting_date'] = pd.to_datetime(sprint_df['CHD_reporting_date'], format = "%Y-%m-%d")
    sprint_df['CHD_reporting_date'] = sprint_df['CHD_reporting_date'] + MonthEnd(0)
    sprint_df['YEAR_MON_ACT'] = pd.to_datetime(sprint_df['YEAR_MON_ACT'], format = "%Y-%m-%d")
    
    # Additional (on top of that in BQ query) aggregation (after cleaning up the MODEL_FAMILY names)
    sprint_df = sprint_df.groupby([col for col in sprint_df.columns.tolist() if col != 'RETAIL_FORECAST'])['RETAIL_FORECAST'].agg('sum').reset_index()                                               
    ### Check: print('Check List:')    
    ### Check: print('Models: {}\n'.format(sprint_df['MODEL_FAMILY'].unique().tolist()))
    ### Check: print('Markets: {}\n'.format(sprint_df['LEVEL_3_REPORTING'].unique().tolist()))
    ### Check: print('First future CHD month-year: {}'.format(min(sprint_df['CHD_reporting_date'])))
    ### Check: print('Last future CHD month-year: {}\n'.format(max(sprint_df['CHD_reporting_date'])))
    
    df = sprint_df.drop(['YEAR_MON_ACT'] ,axis = 1)
    
    if segment_inc_col:
        # Import segment info        
        segment_df = pd.read_csv(input_path.format('segment2.csv'))
        segment_df['MODEL_FAMILY'] = segment_df['MODEL_FAMILY'].str.upper()
        
        # Join the segment info to the data from sprint/BQ
        df = pd.merge(df, segment_df, how = 'left', on =['MODEL_FAMILY'])
        print('Models with NULL segment: {}'.format(df[df['SEGMENT'].isnull()]['MODEL_FAMILY'].unique().tolist()))    
    
    ####################################################################################
    # Create CPDD month/year and assign each with n, n+1, n+2,n+3 handover month data. #
    ####################################################################################
    max_CPDD = max(df['CHD_reporting_date']) - pd.offsets.DateOffset(months=3)
    handover_months = df['CHD_reporting_date'].unique()
    delivery_months = df[df['CHD_reporting_date']<=max_CPDD]['CHD_reporting_date'].unique()

    yy, zz = np.meshgrid(delivery_months.tolist(), handover_months.tolist())
    df_new = {'CPDD_reporting_date': yy.flatten(), 'CHD_reporting_date': zz.flatten()}
    df_new = pd.DataFrame(df_new)
    df_new['CPDD_reporting_date'] = pd.to_datetime(df_new['CPDD_reporting_date'])
    df_new['CHD_reporting_date'] = pd.to_datetime(df_new['CHD_reporting_date'])
    df_new['months_before_handover'] = - (df_new['CPDD_reporting_date'].dt.month + df_new['CPDD_reporting_date'].dt.year*12)\
                                        + (df_new['CHD_reporting_date'].dt.month + df_new['CHD_reporting_date'].dt.year*12)    
    df_new = df_new[(df_new['months_before_handover'] >= 0) & (df_new['months_before_handover'] <= 3)]

    df = pd.merge(df_new, df, how = 'outer', on = 'CHD_reporting_date')   
    df['CPDD_Month'] = df['CPDD_reporting_date'].dt.month
    df['CPDD_Year'] = df['CPDD_reporting_date'].dt.year
    
    ###################################################
    # Modify LEVEL3 & LEVEL5 for country aggregation. #
    ###################################################    
    mkt_mapping = pd.read_csv(aggregate_market_map_path)
    level3_mapping = dict(zip(mkt_mapping['LEVEL_3_REPORTING'],mkt_mapping['TEMP_LEVEL_3_REPORTING']))
    region_mapping = dict(zip(mkt_mapping['LEVEL_3_REPORTING'],mkt_mapping['TEMP_LEVEL_5_REPORTING']))

    # Create a new column for re-labelled LEVEL3 -- for joining with classification results at the final step.
    df['NEW_LEVEL_3_REPORTING'] = df['LEVEL_3_REPORTING']
    df['NEW_LEVEL_3_REPORTING'] = df['NEW_LEVEL_3_REPORTING'].map(level3_mapping)
    df.to_csv(prediction_cases_mkt_unagg_path, index = False)
    
    # Create prediction_cases.csv for prediction/classification.
    df['LEVEL_3_REPORTING'] = df['NEW_LEVEL_3_REPORTING']
    df = df.drop(['NEW_LEVEL_3_REPORTING'], axis = 1)
    df['SEGMENT'] = df['SEGMENT'].fillna('n/a')
    df = df.groupby(['CPDD_reporting_date', 'CHD_reporting_date', 'months_before_handover', 'LEVEL_3_REPORTING',
                     'BRAND', 'MODEL_FAMILY', 'CHD_Month', 'CHD_Year', 'SEGMENT', 'CPDD_Month', 'CPDD_Year']).sum().reset_index()
    df.to_csv(prediction_file_path, index = False)
    return df


df = ClassFeat_Future(num_future_months = 15, segment_inc_col = True)

'''
#####Case when data from BQ has been downloaded as a csv file (outdated)
 
import os
import pandas as pd
from datetime import date

##Location of csv files
os.chdir('C:/Users/hchang/Documents/ACoE_Forecasting/Order Intake/Order Intake/Input')
sprint_df = pd.read_csv('SPRINT_20180202_v1.csv')
segment_df = pd.read_csv('segment2.csv')

model_family = sprint_df['MODEL_FAMILY_DESC'].unique().tolist()
sprint_df['MODEL_FAMILY_DESC'] = sprint_df['MODEL_FAMILY_DESC'].replace({'L460 - Range Rover':'RANGE ROVER',
                                                                        'L461 - Range Rover Sport':'RANGE ROVER SPORT',
                                                                        'L551':'RANGE ROVER EVOQUE',
                                                                        'L552':'DISCOVERY SPORT',
                                                                        'L560':'RANGE ROVER VELAR',
                                                                        'X161':'F-TYPE',
                                                                        'X260':'XF',
                                                                        'X262':'XF',
                                                                        'X391':'XJ',
                                                                        'X540':'E-PACE',
                                                                        'X590':'I-PACE',
                                                                        'X761':'F-PACE',
                                                                        'XF KD':'XF',
                                                                        'DEFENDER KD':'DEFENDER',
                                                                        'FREELANDER KD':'FREELANDER'})
sprint_df['CALENDAR_PERIOD'] = pd.to_datetime(sprint_df['CALENDAR_PERIOD'], format = "%Y-%m-%d")
sprint_df = sprint_df[(sprint_df['CALENDAR_PERIOD'] >= sprint_df['CALENDAR_PERIOD'].min()) & (sprint_df['CALENDAR_PERIOD'] <= sprint_df['CALENDAR_PERIOD'].min() + pd.DateOffset(months=11))]
sprint_df_agg = sprint_df.groupby([col for col in sprint_df.columns.tolist() if col != 'RETAIL_FORECAST'])['RETAIL_FORECAST'].agg('sum').reset_index()                                               
sprint_df_agg['year'] = sprint_df_agg['CALENDAR_PERIOD'].dt.year
sprint_df_agg['month'] = sprint_df_agg['CALENDAR_PERIOD'].dt.month
sprint_df_agg = sprint_df_agg.rename(columns = {'MODEL_FAMILY_DESC': 'MODEL_FAMILY'})

segment_df = pd.read_csv('segment2.csv')
segment_df['MODEL_FAMILY'] = segment_df['MODEL_FAMILY'].str.upper()
df = pd.merge(sprint_df_agg, segment_df, how = 'left', on =['MODEL_FAMILY'])

'''