# task.py
import config.schema as schema_field

etl_config = {
    "setup": {
        "extract_raw_tables": {
            "sql_file": "sql/000_Raw_Tables.sql",
            # table_id: k,v pair with k = source and v = destination
            "table_id": {"original_table_id": "new_table_id"}
        },
        "archiving": {
            "sql_file": "sql/000_Archive_Tables.sql",
            # table_id: k,v pair with k = table_id and v = table_schema_field
            "table_id": {
                "400_Glide_Path": schema_field._400_glide_path_schema,
                "401_Prediction_Case_for_Input":
                    schema_field._401_prediction_case_for_input_schema,
                "600_Final": schema_field._600_final_schema
            }
        }
    },

    "sprint": {
        "current": {
            "sql_file": "sql/101_SPRINT_Current_Actuals.sql",
            "table_id": "101_SPRINT_Current_Actuals"
        },

        "historic": {
            "sql_file": "sql/102_SPRINT_Historic_Actuals.sql",
            "table_id": "102_SPRINT_Historic_Actuals"
        },

        "forecast": {
            "sql_file": "sql/103_SPRINT_Forecast.sql",
            "table_id": "103_SPRINT_Forecast"
        },

        "all": {
            "sql_file": "sql/200_SPRINT_Actuals.sql",
            "table_id": "200_SPRINT_Actuals"
        },

        "forecast_filter": {
            "sql_file": "sql/202_SPRINT_Forecast_Filter.sql",
            "table_id": "202_SPRINT_Forecast_Filter"
        },

        "actuals_model_map": {
            "sql_file": "sql/300_SPRINT_Actuals_Model_Map.sql",
            "table_id": "300_SPRINT_Actuals_Model_Map"
        },

        "prediction_case": {
            "sql_file": "sql/301_SPRINT_Prediction_Case.sql",
            "table_id": "301_SPRINT_Prediction_Case"
        },

        "market_aggregate": {
            "sql_file": "sql/303_SPRINT_Market_Aggregate.sql",
            "table_id": "303_SPRINT_Market_Aggregate"
        }
    },

    "vista": {
        "all": {
            "sql_file": "sql/100_VISTA_Data.sql",
            "table_id": "100_VISTA_Data"
        },

        "clean": {
            "sql_file": "sql/201_VISTA_Clean.sql",
            "table_id": "201_VISTA_Clean"
        },

        "segment_map": {
            "sql_file": "sql/302_VISTA_Segment_Map.sql",
            "table_id": "302_VISTA_Segment_Map"
        },

        "end_of_month": {
            "sql_file": "sql/304_VISTA_End_of_Month.sql",
            "table_id": "304_VISTA_End_of_Month"
        },

        "order_aggregate": {
            "sql_file": "sql/305_VISTA_Order_Aggregate.sql",
            "table_id": "305_VISTA_Order_Aggregate"
        },

        "pre_order_days": {
            "sql_file": "sql/306_VISTA_Pre_Order_Days.sql",
            "table_id": "306_VISTA_Pre_Order_Days"
        },

        "mesh_grid": {
            "sql_file": "sql/307_Order_Days_Mesh_Grid.sql",
            "table_id": "307_Order_Days_Mesh_Grid"
        },

        "retail_actual_and_forecast": {
            "sql_file": "sql/308_VISTA_Retail_Actual_And_Forecast.sql",
            "table_id": "308_VISTA_Retail_Actual_And_Forecast"
        },

        "glide_path_unpivot": {
            "sql_file": "sql/309_VISTA_Glide_Path_UNPIVOT.sql",
            "table_id": "309_VISTA_Glide_Path_UNPIVOT"
        },

        "month_before_handover": {
            "sql_file": "sql/310_VISTA_Month_Before_Handover.sql",
            "table_id": "310_VISTA_Month_Before_Handover"
        },

        "retail_and_sold_number": {
            "sql_file": "sql/311_VISTA_Retail_And_Sold_Number.sql",
            "table_id": "311_VISTA_Retail_And_Sold_Number"
        },

        "cumulative_order": {
            "sql_file": "sql/312_VISTA_Cumulative_Order.sql",
            "table_id": "312_VISTA_Cumulative_Order"
        },

        "glide_path_pivot": {
            "src_sql_file": "sql/000_Pivot_Input.sql",
            "piv_sql_file": "sql/313_Glide_Path_Pivot.sql",
            "src_table_id": "312_VISTA_Cumulative_Order",
            "dest_table_id": "313_VISTA_Glide_Path_Pivot",
            "piv_col": "PRE_ORDER_DAYS",
            "piv_idx": ["LEVEL_5_REPORTING",
                        "LEVEL_3_REPORTING",
                        "CPDD_REPORTING_DATE",
                        "BRAND",
                        "SEGMENT",
                        "MODEL_FAMILY",
                        "CHD_REPORTING_DATE",
                        "MONTHS_BEFORE_HANDOVER",
                        "VISTA_RETAIL_ACTUALS"],
            "piv_val": "CUMULATIVE_SOLD_ORDER_NUMBER"
        }
    },

    "final": {
        "glide_path": {
            "sql_file": "sql/400_Glide_Path.sql",
            "table_id": "400_Glide_Path"
        },

        "prediction_case": {
            "sql_file": "sql/401_Prediction_Case_for_Input.sql",
            "table_id": "401_Prediction_Case_for_Input"
        }
    },

    "tear_down": {
        "move_to_access": {
            "table_id": [
                "400_Glide_Path",
                "401_Prediction_Case_for_Input",
                "600_Final"
            ]
        },

        "create_meta_table": {
            "sql_file": "sql/1001_Meta_Table.sql",
            "table_id": "1001_Meta_Table"
        },

        "update_meta_data": {
            'table_sheet_map': {
                '400_Glide_Path': '400_Glide_Path',
                '401_Prediction_Case_for_Input': '401_Prediction_Case_for_Input',
                '600_Final': '600_Final'
            },
            'labels': {
                'data_owner': 'jlr_analytics-commercial_planning'
            },
            "data_dict_filepath": "docs/source/_static/files/order_intake_data_dictionary.xlsx"
        }
    }
}


model_config = {
    "clustering": {
        "file": "models/clustering.py",
        "tables": "500_Order_Intake_Target"
    },

    "etl": {
        "clus_info": {
            "sql_file": "sql/506_Classifed_Cluster_Info.sql",
            "table_id": "506_Classifed_Cluster_Info"
        },

        "zero_day_test_case": {
            "sql_file": "sql/507_Zero_Day_Cluster_Test_Cases_Clean.sql",
            "table_id": "507_Zero_Day_Cluster_Test_Cases_Clean"
        },

        "zero_day_test_case_piv": {
            "sql_file": "sql/508_Zero_Day_Cluster_Test_Cases_Pivot.sql",
            "table_id": "508_Zero_Day_Cluster_Test_Cases_Pivot"
        },

        "shape_test_case": {
            "sql_file": "sql/509_Shape_Cluster_Test_Cases_Clean.sql",
            "table_id": "509_Shape_Cluster_Test_Cases_Clean"
        },

        "shape_test_case_piv": {
            "sql_file": "sql/510_Shape_Cluster_Test_Cases_Pivot.sql",
            "table_id": "510_Shape_Cluster_Test_Cases_Pivot"
        },

        "shape_zero_day": {
            "sql_file": "sql/511_Shape_Zero_Day_Aggregates.sql",
            "table_id": "511_Shape_Zero_Day_Aggregates"
        },

        "final": {
            "sql_file": "sql/600_Final.sql",
            "table_id": "600_Final"
        }
    }
}

etl_sql_test = {
    "calculation_logic_check": {
        "sql_file": "tests/sql/906_Glide_Path_Calculation_Logic.sql",
        "test_table_id": "400_Glide_Path",
        "result_table_id": "906_Glide_Path_Calculation_Logic"
    },

    "test_summary": {
        "expectation_test": {
            "sql_file": "tests/sql/999a_Expectation_Test_Summary.sql",
            "test_tables": [
                '901_Glide_Path_Table_Column_Order_Expectation',
                '902_Glide_Path_Not_Null_Column_Expectation',
                '903_Glide_Path_Date_Format_Expectation',
                '904_Glide_Path_Value_in_Set_Expectation',
                '905_Glide_Path_Column_Regex_Expectations',
                # '906_Glide_Path_Calculation_Logic',
                '907_Glide_Path_Monotonically_Increasing_Columns',
                '908_Prediction_Case_Table_Column_Order_Expectation',
                '909_Prediction_Case_Not_Null_Column_Expectation',
                '910_Prediction_Case_Date_Format_Expectation',
                '911_Prediction_Case_Value_in_Set_Expectation',
                '912_Prediction_Case_Column_Regex_Expectations'
            ],
            "result_table_id": "999a_Expectation_Test_Summary"
        }
    }
}


expectation_tests = {
    "glide_path": {
        "table_id": "400_Glide_Path",
        "expect_columns_to_match_ordered_list": {
            "col_names": [
                "LEVEL_5_REPORTING", "LEVEL_3_REPORTING",
                "BRAND", "SEGMENT", "MODEL_FAMILY", "CHD_REPORTING_DATE",
                "VISTA_RETAIL_ACTUALS", "RETAIL_ACTUALS", "RETAIL_FORECAST",
                "RETAIL_FORECAST_CUTOFF", "TARGET_MET", "_0_DAY_SOLD_ORDER_LEVEL",
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
            ],
            "table_id": "901_Glide_Path_Table_Column_Order_Expectation"
        },

        "expect_column_values_to_not_be_null": {
            "col_names": [
                "LEVEL_5_REPORTING", "LEVEL_3_REPORTING",
                "BRAND", "SEGMENT",
                "MODEL_FAMILY", "CHD_REPORTING_DATE",
                 "VISTA_RETAIL_ACTUALS",
                "RETAIL_ACTUALS", "RETAIL_FORECAST",
                "RETAIL_FORECAST_CUTOFF", "TARGET_MET",
                "_0_DAY_SOLD_ORDER_LEVEL", "_0"
            ],
            "table_id": "902_Glide_Path_Not_Null_Column_Expectation"
        },

        "expect_column_values_to_match_strftime_format": {
            "col_names": [
                'CHD_REPORTING_DATE'
            ],
            "exp_value": "%Y-%m-%d",
            "table_id": "903_Glide_Path_Date_Format_Expectation"
        },

        "expect_column_values_to_be_in_set": {
            "col_names": [
                "LEVEL_5_REPORTING", "LEVEL_3_REPORTING", "BRAND",
                "SEGMENT", "MODEL_FAMILY",
            ],
            "exp_value": {
                "LEVEL_5_REPORTING": [
                    "China",
                    "EU_LowVol", "Europe", "North America",
                    "Overseas", "United Kingdom"
                ],

                "LEVEL_3_REPORTING": [
                    "Japan", "Italy", "Turkey",
                    "Spain", "Korea South", "Taiwan",
                    "Asia Pacific Importers",
                    "Belgium", "Singapore",
                    "Portugal", "Canada",
                    "United Kingdom","South Africa",
                    "China","Brazil","Netherlands",
                    "Switzerland","Russia","Australia",
                    "Austria", "France","Germany","MENA",
                    "LACRO","USA","Rest of European Importers",
                    "CJLR","India",
                    "Colombia","Mexico","Sub Sahara Africa"
                                    ],

                "BRAND": [
                    "JAGUAR", "LAND ROVER"
                ],

                "SEGMENT": [
                    "SUV2", "SUV3", "SUV4", "SUV5", "Sedan3",
                    "Sedan4", "Sedan5", "Sport2", "Sport3"
                ],

                "MODEL_FAMILY": [
                    "DISCOVERY", "DISCOVERY SPORT", "E-PACE", "F-PACE",
                    "F-TYPE", "I-PACE", "RANGE ROVER", "RANGE ROVER EVOQUE",
                    "RANGE ROVER SPORT", "RANGE ROVER VELAR", "XE", "XF", "XJ"
                ]

            },
            "table_id": "904_Glide_Path_Value_in_Set_Expectation"
        },

        "expect_column_values_to_match_regex": {
            "col_names": [
                "LEVEL_5_REPORTING", "LEVEL_3_REPORTING",
                "BRAND", "SEGMENT", "MODEL_FAMILY"
            ],
            "exp_value": "\D+",
            "table_id": "905_Glide_Path_Column_Regex_Expectations"
        },

        "expect_column_pair_values_A_to_be_greater_than_B": {
            "col_names": [
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
            ],
            "table_id": "907_Glide_Path_Monotonically_Increasing_Columns"
        }
    },

    "prediction_case_for_input": {
        "table_id": "401_Prediction_Case_for_Input",
        "expect_columns_to_match_ordered_list": {
            "col_names": [
                "CHD_REPORTING_DATE", "LEVEL_3_REPORTING",
                "BRAND", "MODEL_FAMILY", "CHD_MONTH", "CHD_YEAR",
                "SEGMENT", "RETAIL_FORECAST"
            ],
            "table_id": "908_Prediction_Case_Table_Column_Order_Expectation"
        },

        "expect_column_values_to_not_be_null": {
            "col_names": [
                "CHD_REPORTING_DATE",
                 "LEVEL_3_REPORTING",
                "BRAND", "MODEL_FAMILY", "CHD_MONTH", "CHD_YEAR",
                "SEGMENT", "RETAIL_FORECAST"
            ],
            "table_id": "909_Prediction_Case_Not_Null_Column_Expectation"
        },

        "expect_column_values_to_match_strftime_format": {
            "col_names": [
                'CHD_REPORTING_DATE'
            ],
            "exp_value": "%Y-%m-%d",
            "table_id": "910_Prediction_Case_Date_Format_Expectation"
        },

        "expect_column_values_to_be_in_set": {
            "col_names": [
                "LEVEL_3_REPORTING", "BRAND", "SEGMENT",
                "MODEL_FAMILY"
            ],
            "exp_value": {
                "LEVEL_3_REPORTING": [
                    "Japan", "Italy", "Turkey",
                    "Spain", "Korea South", "Taiwan",
                    "Asia Pacific Importers",
                    "Belgium", "Singapore",
                    "Portugal", "Canada",
                    "United Kingdom", "South Africa",
                    "China", "Brazil", "Netherlands",
                    "Switzerland", "Russia", "Australia",
                    "Austria", "France", "Germany", "MENA",
                    "LACRO", "USA", "Rest of European Importers",
                    "CJLR", "India",
                    "Colombia", "Mexico", "Sub Sahara Africa"
                ],

                "BRAND": [
                    "JAGUAR", "LAND ROVER"
                ],

                "SEGMENT": [
                    "SUV2", "SUV3", "SUV4", "SUV5", "Sedan3",
                    "Sedan4", "Sedan5", "Sport2", "Sport3"
                ],

                "MODEL_FAMILY": [
                    "DISCOVERY", "DISCOVERY SPORT", "E-PACE", "F-PACE",
                    "F-TYPE", "I-PACE", "RANGE ROVER", "RANGE ROVER EVOQUE",
                    "RANGE ROVER SPORT", "RANGE ROVER VELAR", "XE", "XF", "XJ"
                ]
            },
            "table_id": "911_Prediction_Case_Value_in_Set_Expectation"
        },

        "expect_column_values_to_match_regex": {
            "col_names": [
                "LEVEL_3_REPORTING", "BRAND",
                "SEGMENT", "MODEL_FAMILY"
            ],
            "exp_value": "\D+",
            "table_id": "912_Prediction_Case_Column_Regex_Expectations"
        }
    }
}
