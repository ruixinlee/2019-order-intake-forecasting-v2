from datetime import datetime as dt

pre_order_curves_columns_mapping = {
    'vista_retail_actuals': 'VISTA_RETAIL_ACTUALS',
    'target_met': 'TARGET_MET',
    'CPDD_reporting_date': 'CPDD_REPORTING_DATE',
    'CHD_reporting_date': 'CHD_REPORTING_DATE',
    'model_family': 'MODEL_FAMILY',
    'CPDD_reporting_date': 'CPDD_REPORTING_DATE',
    'Level_5': 'LEVEL_5_REPORTING',
    'Level_3': 'LEVEL_3_REPORTING',
    'Granularity': 'GRANULARITY',
    '0day_sold_order_level_prec': '_0_DAY_SOLD_ORDER_LEVEL',
    '0day_sold_order_level_cluster': '_0_DAY_SOLD_ORDER_LEVEL_CLUSTER',
    '0day_sold_order_level%_scaled': '_0_DAY_SOLD_ORDER_LEVEL_PERC_SCALED',
    '0day_sold_order_level%_cluster': '_0_DAY_SOLD_ORDER_LEVEL_PERC_CLUSTER',
    '0day_sold_order_level%_band': '_0_DAY_SOLD_ORDER_LEVEL_PERC_BAND',
    'CHD_month': 'CHD_MONTH',
    'CHD_year': 'CHD_YEAR',
    'CPDD_month': 'CPDD_MONTH',
    'CPDD_year': 'CPDD_YEAR',
    'distance': 'DISTANCE_2',
    'is_train': 'IS_TRAIN',
    'level_3_reporting': 'LEVEL_3_REPORTING',
    'level_5_reporting': 'LEVEL_5_REPORTING',
    'months_before_handover': 'MONTHS_BEFORE_HANDOVER',
    'index': 'INDEX',
    'cluster_type': 'CLUSTER_TYPE',
    'confidence_band': 'CONFIDENCE_BAND',
    'weighted': 'WEIGHTED',
    'clusters_type_mean': 'clusters_type_mean',
    'weighted_mean': 'WEIGHTED_MEAN',
    'level_5_mean': 'LEVEL5_MEAN',
    'granularity_mean': 'GRANULARITY_mean',
    'retail_sales_and_forecast': 'RETAIL_ACTUALS',
    'vista_actuals_and_forecast': 'VISTA_RETAIL_ACTUALS',
    'is_train': 'IS_TRAIN',
    'sold_order_level_perc': 'SOLD_ORDER_LEVEL_PERC',
    'year': 'YEAR',
    'month': 'MONTH',
    'day': 'DAY',
    'retail_forecast_cutoff': 'RETAIL_FORECAST_cutoff',
    'classify_cluster_var': 'PRE_ORDER_DAY',
    'classify_cluster_val': 'SOLD_ORDER_LEVEL_PERC',
    'cluster_type': 'clus_pointwise_KMeans'
}
# Create a list of column names that shall be maintained towards the end.

band_attr = [
     'BRAND', 'MODEL_FAMILY', 'SEGMENT',
    'RETAIL_FORECAST', 'CHD_REPORTING_DATE',
    'RETAIL_ACTUALS', 'RETAIL_FORECAST_cutoff',
    'TARGET_MET', 'VISTA_RETAIL_ACTUALS',
    'IS_TRAIN', '_0_DAY_SOLD_ORDER_LEVEL_PERC'
]

level_3_date_truncate = [
    'USA', 'China NSC'
]

class_attr_1 = ['MODEL_FAMILY', 'RETAIL_FORECAST', 'CHD_YEAR', 'CHD_MONTH']

main_config = {
    'level3_ignore': ['Guava', 'Direct Sales'],
    'granularities': [['SEGMENT']],
    # 'granularities': [['SEGMENT', 'MONTHS_BEFORE_HANDOVER']],
    'num_future_months': 15,
    'CHD_reporting_date_cutoff': dt.strptime('2019-09-30', "%Y-%m-%d"),  # TODO: calculate this as end of previous month
    'validation': [True],
    'custom_country': [],
    'cpdd_reporting_start': dt.strptime('2014-05-01', "%Y-%m-%d")
}

pre_order_day_col = [
    "_365", "_364", "_363", "_362", "_361", "_360", "_359", "_358",
    "_357", "_356", "_355", "_354", "_353", "_352", "_351", "_350",
    "_349", "_348", "_347", "_346", "_345", "_344", "_343", "_342",
    "_341", "_340", "_339", "_338", "_337", "_336", "_335", "_334",
    "_333", "_332", "_331", "_330", "_329", "_328", "_327", "_326",
    "_325", "_324", "_323", "_322", "_321", "_320", "_319", "_318",
    "_317", "_316", "_315", "_314", "_313", "_312", "_311", "_310",
    "_309", "_308", "_307", "_306", "_305", "_304", "_303", "_302",
    "_301", "_300", "_299", "_298", "_297", "_296", "_295", "_294",
    "_293", "_292", "_291", "_290", "_289", "_288", "_287", "_286",
    "_285", "_284", "_283", "_282", "_281", "_280", "_279", "_278",
    "_277", "_276", "_275", "_274", "_273", "_272", "_271", "_270",
    "_269", "_268", "_267", "_266", "_265", "_264", "_263", "_262",
    "_261", "_260", "_259", "_258", "_257", "_256", "_255", "_254",
    "_253", "_252", "_251", "_250", "_249", "_248", "_247", "_246",
    "_245", "_244", "_243", "_242", "_241", "_240", "_239", "_238",
    "_237", "_236", "_235", "_234", "_233", "_232", "_231", "_230",
    "_229", "_228", "_227", "_226", "_225", "_224", "_223", "_222",
    "_221", "_220", "_219", "_218", "_217", "_216", "_215", "_214",
    "_213", "_212", "_211", "_210", "_209", "_208", "_207", "_206",
    "_205", "_204", "_203", "_202", "_201", "_200", "_199", "_198",
    "_197", "_196", "_195", "_194", "_193", "_192", "_191", "_190",
    "_189", "_188", "_187", "_186", "_185", "_184", "_183", "_182",
    "_181", "_180", "_179", "_178", "_177", "_176", "_175", "_174",
    "_173", "_172", "_171", "_170", "_169", "_168", "_167", "_166",
    "_165", "_164", "_163", "_162", "_161", "_160", "_159", "_158",
    "_157", "_156", "_155", "_154", "_153", "_152", "_151", "_150",
    "_149", "_148", "_147", "_146", "_145", "_144", "_143", "_142",
    "_141", "_140", "_139", "_138", "_137", "_136", "_135", "_134",
    "_133", "_132", "_131", "_130", "_129", "_128", "_127", "_126",
    "_125", "_124", "_123", "_122", "_121", "_120", "_119", "_118",
    "_117", "_116", "_115", "_114", "_113", "_112", "_111", "_110",
    "_109", "_108", "_107", "_106", "_105", "_104", "_103", "_102",
    "_101", "_100", "_99", "_98", "_97", "_96", "_95", "_94",
    "_93", "_92", "_91", "_90", "_89", "_88", "_87", "_86",
    "_85", "_84", "_83", "_82", "_81", "_80", "_79", "_78",
    "_77", "_76", "_75", "_74", "_73", "_72", "_71", "_70",
    "_69", "_68", "_67", "_66", "_65", "_64", "_63", "_62",
    "_61", "_60", "_59", "_58", "_57", "_56", "_55", "_54",
    "_53", "_52", "_51", "_50", "_49", "_48", "_47", "_46",
    "_45", "_44", "_43", "_42", "_41", "_40", "_39", "_38",
    "_37", "_36", "_35", "_34", "_33", "_32", "_31", "_30",
    "_29", "_28", "_27", "_26", "_25", "_24", "_23", "_22",
    "_21", "_20", "_19", "_18", "_17", "_16", "_15", "_14",
    "_13", "_12", "_11", "_10", "_9", "_8", "_7", "_6",
    "_5", "_4", "_3", "_2", "_1", "_0"
]

tables = {
    "glide_path_table": "400_Glide_Path",
    "prediction_case": "401_Prediction_Case_for_Input",
    "glide_path_shape_cluster": "500_Glide_Path_Shape_Cluster",
    "glide_path_shape_melted_cluster": "500_Glide_Path_Shape_Melted_Cluster",
    # "glide_path_zero_day_cluster": "501_Glide_Path_Zero_Day_Cluster",
    "classified_shape_cluster_info": "502_Classified_Shape_Cluster_Info",
    "classified_shape_cluster_info_test_cases": "503_Classified_Shape_Cluster_Test_Cases",
    # "classified_zero_day_cluster_info": "504_Classified_Zero_Day_Cluster_Info",
    # "classified_zero_day_cluster_info_test_cases": "505_Classified_Zero_Day_Cluster_Test_Cases",
    "prediction_case_unagg": "506_Prediction_Case_Unaggregated",
    "shape_zero_day_combined": "507_Cluster_Shape_Zero_Day_Combined"
}

date_columns = ['CHD_REPORTING_DATE']
