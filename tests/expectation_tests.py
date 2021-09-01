'''
    expectation_tests.py
'''

from config import tasks
from config.conf import gcp, logger
from utils.expectation_tests_util import ExpectationTest
from utils.bq import bq_table_to_dataframe


def unittest_glidepath_dataset():
    """
    Great Expectation Test for final glide paths table
    :return:
    """

    test_complete = False
    # base test input
    test_input = tasks.expectation_tests['glide_path']

    # Initialise class
    exp_obj = ExpectationTest(project_id=gcp['project'],
                              table_id=test_input['table_id'],
                              dataset_id=gcp['dataset'])

    # load dataset
    dataset = exp_obj.load_data()
    df_context = bq_table_to_dataframe(
        exp_obj.project_id,
        exp_obj.dataset_id,
        exp_obj.table_id
    )

    logger.info(f"Starting Expectation Tests on {test_input['table_id']}")

    # tests
    test_param = test_input['expect_columns_to_match_ordered_list']
    exp_obj.expect_table_columns_to_match_ordered_list(
        dataset=dataset,
        input_columns=test_param['col_names'],
        output_table=test_param['table_id']
    )

    test_param = test_input['expect_column_values_to_not_be_null']
    exp_obj.expect_column_values_to_not_be_null(
        dataset=dataset,
        input_columns=test_param['col_names'],
        output_table=test_param['table_id'],
        mostly=1
    )

    test_param = test_input['expect_column_values_to_match_regex']
    exp_obj.expect_column_values_to_match_regex(
        dataset=df_context,
        input_columns=test_param['col_names'],
        expected_value=test_param['exp_value'],
        output_table=test_param['table_id']
    )

    test_param = test_input['expect_column_values_to_be_in_set']
    exp_obj.expect_column_values_to_be_in_set(
        dataset=dataset,
        input_columns=test_param['col_names'],
        expected_values=test_param['exp_value'],
        output_table=test_param['table_id']
    )

    test_param = test_input['expect_column_values_to_match_strftime_format']
    exp_obj.expect_column_values_to_match_strftime_format(
        dataset=df_context,
        input_columns=test_param['col_names'],
        expected_value=test_param['exp_value'],
        output_table=test_param['table_id']
    )

    test_param = test_input['expect_column_pair_values_A_to_be_greater_than_B']
    exp_obj.expect_column_pair_values_a_to_be_greater_than_b(
        dataset=df_context,
        input_columns=test_param['col_names'],
        output_table=test_param['table_id']
    )

    test_complete = True

    logger.info("Expectation tests Completed!")

    return test_complete


def unittest_prediction_case_dataset():
    """
    Great Expectation Test for final glide paths table
    :return:
    """

    test_complete = False

    # base test input
    test_input = tasks.expectation_tests['prediction_case_for_input']

    # Initialise class
    exp_obj = ExpectationTest(project_id=gcp['project'],
                              table_id=test_input['table_id'],
                              dataset_id=gcp['dataset'])

    # load dataset
    dataset = exp_obj.load_data()
    df_context = bq_table_to_dataframe(
        exp_obj.project_id,
        exp_obj.dataset_id,
        exp_obj.table_id
    )

    logger.info(f"Starting Expectation Tests on {test_input['table_id']}...")

    # tests
    test_param = test_input['expect_columns_to_match_ordered_list']
    exp_obj.expect_table_columns_to_match_ordered_list(
        dataset=dataset,
        input_columns=test_param['col_names'],
        output_table=test_param['table_id']
    )

    test_param = test_input['expect_column_values_to_not_be_null']
    exp_obj.expect_column_values_to_not_be_null(
        dataset=dataset,
        input_columns=test_param['col_names'],
        output_table=test_param['table_id'],
        mostly=1
    )

    test_param = test_input['expect_column_values_to_match_regex']
    exp_obj.expect_column_values_to_match_regex(
        dataset=df_context,
        input_columns=test_param['col_names'],
        expected_value=test_param['exp_value'],
        output_table=test_param['table_id']
    )

    test_param = test_input['expect_column_values_to_be_in_set']
    exp_obj.expect_column_values_to_be_in_set(
        dataset=dataset,
        input_columns=test_param['col_names'],
        expected_values=test_param['exp_value'],
        output_table=test_param['table_id']
    )

    test_param = test_input['expect_column_values_to_match_strftime_format']
    exp_obj.expect_column_values_to_match_strftime_format(
        dataset=df_context,
        input_columns=test_param['col_names'],
        expected_value=test_param['exp_value'],
        output_table=test_param['table_id']
    )

    test_complete = True

    logger.info("Expectation tests Completed!")

    return test_complete


if __name__ == "__main__":
    unittest_glidepath_dataset()

    unittest_prediction_case_dataset()
