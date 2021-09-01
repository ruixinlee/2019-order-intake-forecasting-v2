import logging

import google.cloud.bigquery as bigquery
import great_expectations as ge
import pandas as pd
import pandas.io.json
from great_expectations.dataset import PandasDataset

from config.conf import gcp, logger
from utils.bq import load_df_to_bigquery, bq_query_to_dataframe

# __all__ =['load_data', 'expect_table_columns_to_match_ordered_list',
#           'expect_column_values_to_be_in_set', 'expect_column_values_to_be_null',
#           'expect_column_values_to_be_null', 'expect_column_values_to_not_be_null',
#           'expect_column_values_to_match_strftime_format', 'expect_column_values_to_match_regex',
#           'expect_column_values_to_be_between', 'expect_column_pair_values_a_to_be_greater_than_b',
#           'expect_column_values_to_be_of_type']


class ExpectationTest:
    """
    This class contains an implementation of various expectation
    test functions to be used to check integrity of data sets.
    """

    def __init__(self, project_id: str, table_id: str, dataset_id: str) -> None:
        """
        :param table_id:
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id

    def load_data(self):
        """
        This method enables the loading of data from BigQuery.
        :return:
        """
        logger.info('Attempting to connect via sqlAlchemy...')
        db_conn_str = f'bigquery://{self.project_id}'
        sql_context = ge.get_data_context('SqlAlchemy', db_conn_str)
        table = '{}.{}'.format(self.dataset_id, self.table_id)
        dataset = sql_context.get_dataset(table)
        logger.info("Connection successful")

        return dataset

    def _change_data_context_to_df(self) -> pd.DataFrame:
        """
        Takes in SQL Alchemy data context as input and returns
        data frame data context
        :param:
        :return:
        """
        logger.info("Setting up dataframe context!")
        # read in the dataset
        sql_str = 'SELECT * FROM `{}.{}`'.format(
            self.dataset_id, self.table_id)

        # returns as dataframe object
        df = bq_query_to_dataframe(sql_str)

        return df

    @staticmethod
    def _rename_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        :param df:
        :return:
        """
        old_col_names = df.columns.tolist()
        new_col_names = [name.replace('.', '_') for name in old_col_names]

        df_new = df.rename(index=str, columns=dict(
            zip(old_col_names, new_col_names)))

        return df_new

    def expect_table_columns_to_match_ordered_list(self, dataset, input_columns,
                                                   output_table) -> None:
        """
        Expectation to test if column names are in the right order.
        :param dataset:
        :param input_columns:
        :param output_table:
        :return:
        """
        logger.info("Test: expect_columns_to_match_ordered_list")
        expect_res = dataset.expect_table_columns_to_match_ordered_list(input_columns,
                                                                        result_format="SUMMARY",
                                                                        include_config=True)
        df = pd.io.json.json_normalize(expect_res)
        df = self._rename_column(df)
        load_df_to_bigquery(df, output_table)

        return None

    def expect_column_values_to_be_in_set(self, dataset, input_columns,
                                          expected_values, output_table) -> None:
        """
        Expectation test to check if sets of values in table columns match expected
        set of values.
        :param self:
        :param dataset:
        :param input_columns:
        :param expected_values:
        :param output_table:
        """
        logger.info("Running Test: expect_column_values_to_be_in_set")
        df = pd.DataFrame()

        for i in range(len(input_columns)):
            expect_res = dataset.expect_column_values_to_be_in_set(input_columns[i],
                                                                   expected_values[input_columns[i]],
                                                                   result_format="SUMMARY",
                                                                   include_config=True)
            df_temp = pd.io.json.json_normalize(expect_res)
            df = df.append(df_temp)

        df = self._rename_column(df)
        load_df_to_bigquery(df, output_table)

        return None

    def expect_column_values_to_be_null(self, dataset, input_columns,
                                        output_table, mostly=0.5) -> None:
        """
        :param dataset:
        :param input_columns:
        :param output_table:
        :param mostly:
        :return:
        """
        logger.info("Running Test: expect_column_values_to_be_null")
        df = pd.DataFrame()

        for i in range(len(input_columns)):
            expect_res = dataset.expect_column_values_to_be_null(input_columns[i],
                                                                 mostly=mostly,
                                                                 result_format="SUMMARY",
                                                                 include_config=True)

            df_temp = pd.io.json.json_normalize(expect_res)
            df = df.append(df_temp)

        df = self._rename_column(df)
        load_df_to_bigquery(df, output_table)

        return None

    def expect_column_values_to_not_be_null(self, dataset: str, input_columns: list,
                                            output_table: str, mostly: float = 0.5) -> None:
        """
        :param dataset:
        :param input_columns:
        :param output_table:
        :param mostly:
        :return:
        """
        logger.info("Running Test: expect_column_values_to_not_be_null")
        df = pd.DataFrame()

        for i in range(len(input_columns)):
            expect_res = dataset.expect_column_values_to_not_be_null(input_columns[i],
                                                                     mostly=mostly,
                                                                     result_format="SUMMARY",
                                                                     include_config=True)

            df_temp = pd.io.json.json_normalize(expect_res)
            df = df.append(df_temp)

        df = self._rename_column(df)
        load_df_to_bigquery(df, output_table)

        return None

    def expect_column_values_to_match_strftime_format(self, input_columns: list,
                                                      expected_value: str, output_table: str,
                                                      dataset: pd.DataFrame=None) -> None:
        """
        :param input_columns:
        :param expected_value:
        :param output_table:
        :return:
        """
        logger.info(
            "Running Test: expect_column_values_to_match_strftime_format")

        if dataset is None:
            dataset = self._change_data_context_to_df()
            dataset = PandasDataset(dataset)
        else:
            dataset = PandasDataset(dataset)

        df = pd.DataFrame()

        for i in range(len(input_columns)):
            dataset[input_columns[i]] = dataset[input_columns[i]].astype(str)
            expect_res = dataset.expect_column_values_to_match_strftime_format(input_columns[i],
                                                                               expected_value,
                                                                               result_format="SUMMARY",
                                                                               include_config=True)
            df_temp = pd.io.json.json_normalize(expect_res)
            df = df.append(df_temp)

        df = self._rename_column(df)
        load_df_to_bigquery(df, output_table)

        return None

    def expect_column_values_to_match_regex(self, input_columns: list,
                                            expected_value: str, output_table: str,
                                            dataset: pd.DataFrame=None) -> None:
        """
        :param input_columns:
        :param expected_value:
        :param output_table:
        :return:
        """
        logger.info("Running Test: expect_column_values_to_match_regex")

        if dataset is None:
            dataset = self._change_data_context_to_df()
            dataset = PandasDataset(dataset)
        else:
            dataset = PandasDataset(dataset)

        df = pd.DataFrame()

        for i in range(len(input_columns)):
            expect_res = dataset.expect_column_values_to_match_regex(input_columns[i],
                                                                     expected_value,
                                                                     result_format="SUMMARY",
                                                                     include_config=True)
            df_temp = pd.io.json.json_normalize(expect_res)
            df = df.append(df_temp)

        df = self._rename_column(df)
        load_df_to_bigquery(df, output_table)

        return None

    def expect_column_values_to_be_between(self, dataset, input_columns,
                                           expected_value, output_table, mostly=1) -> None:
        """
        :param dataset:
        :param input_columns:
        :param expected_value:
        :param output_table:
        :param mostly:
        :return:
        """
        logger.info("Running Test: expect_column_values_to_be_between")
        df = pd.DataFrame()

        for i in range(len(input_columns)):
            exp_val = expected_value[i]
            expect_res = dataset.expect_column_values_to_be_between(input_columns[i],
                                                                    min_value=exp_val[0],
                                                                    max_value=exp_val[1],
                                                                    mostly=mostly,
                                                                    result_format="SUMMARY",
                                                                    include_config=True)
            df_temp = pd.io.json.json_normalize(expect_res)
            df = df.append(df_temp)

        df = self._rename_column(df)
        load_df_to_bigquery(df, output_table)

        return None

    def expect_column_pair_values_a_to_be_greater_than_b(self, input_columns,
                                                         output_table, not_strictly=True,
                                                         dataset: pd.DataFrame=None) -> None:
        """
        Expect values in column A to be greater than column B.
        :param input_column: List of columns to compare in the right order. e.g ['b', 'a']. Therefore,
            list columns in increasing order.
        :param output_table: Name of table to output the final result into
        :return:
        """
        logger.info(
            "Running Test: expect_column_pair_values_A_to_be_greater_than_B"
        )

        if dataset is None:
            dataset = self._change_data_context_to_df()
            dataset = PandasDataset(dataset)
        else:
            dataset = PandasDataset(dataset)

        df = pd.DataFrame()
        idx = 1
        while idx < len(input_columns):
            logger.info(f"Comparing {input_columns[idx]} : {input_columns[idx - 1]}")
            expect_res = dataset.expect_column_pair_values_A_to_be_greater_than_B(column_A=input_columns[idx],
                                                                                  column_B=input_columns[idx - 1],
                                                                                  or_equal=not_strictly,
                                                                                  result_format="SUMMARY",
                                                                                  include_config=True)
            df_temp = pd.io.json.json_normalize(expect_res)
            df = df.append(df_temp)
            idx += 1

        df = self._rename_column(df)
        load_df_to_bigquery(df, output_table)

        return None

    def expect_column_values_to_be_of_type(self, input_columns: list,
                                           expected_value: list, output_table: str,
                                           dataset: pd.DataFrame=None) -> None:
        """
        :param test_param:
        :return:
        """
        logger.info("Running Test: expect_column_values_to_be_of_type")

        if dataset is None:
            dataset = self._change_data_context_to_df()
            dataset = PandasDataset(dataset)
        else:
            dataset = PandasDataset(dataset)

        df = pd.DataFrame()

        for i in range(len(input_columns)):
            expect_res = dataset.expect_column_values_to_be_of_type(column=input_columns,
                                                                    type_=expected_value,
                                                                    mostly=1,
                                                                    result_format="SUMMARY",
                                                                    include_config=True)
            df_temp = pd.io.json.json_normalize(expect_res)
            df = df.append(df_temp)

        df = self._rename_column(df)
        load_df_to_bigquery(df, output_table)

        return None
