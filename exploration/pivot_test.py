import pandas as pd
# from pandas import pivot_table
import numpy as np
import utils.bq as ubq
import utils.query_composer as qc

# src_table_id = '509_Shape_Cluster_Test_Cases_Clean'
# df = ubq.bq_table_to_dataframe('jlr-dl-cat', 'earoge', src_table_id, limit=True)
src_table_id = r"C:\Users\earoge\Downloads\results-20190409-102131.csv"
df = pd.read_csv(src_table_id)
print('printing df')
print(df.head())

# Transform
df_piv = qc.PivotQuery(
    df,
    index_col=[
        'StartDate',
        'vfdb_full_vin'
    ],
    pivot_col="vfdb_base_dtc",
    values_col="problem_count",
    prefix="_").write_query('mharms_pivot')
