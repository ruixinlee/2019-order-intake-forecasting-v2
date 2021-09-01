import os
from pathlib import Path, PureWindowsPath
from google.cloud.bigquery import dbapi
import utils.bq as ubq
from utils.string_formatter import FormatSQLString
from config.conf import gcp, logger


def path_formatter(path_str):
    """
    This function helps format file paths to meet operating system
    requirements. So issues can be avoided in a linux environment.
    :param path_str:
    :return:
    """
    if os.name == 'nt':
        return PureWindowsPath(path_str)
    else:
        return Path(path_str)


def execute_query_task(sql_file,
                       destination_table,
                       dataset_id=gcp['dataset'],
                       run_disposition='WRITE_TRUNCATE',
                       use_legacy=False):
    """
    This function is used to open and execute files containing SQL scripts.
    The query results are stored in given destination table id.
    :param sql_file:
    :param destination_table:
    :param dataset_id:
    :param run_disposition:
    :param use_legacy:
    :return:
    """
    path = path_formatter(sql_file)
    sql_string = FormatSQLString.format_dataset_id(open(path).read(), dataset_id)

    try:
        ubq.bq_query_to_table(destination_table,
                              sql_string,
                              dataset_id=dataset_id,
                              write_disposition=run_disposition,
                              use_legacy=use_legacy)
        success = True
    except dbapi.Error as err:
        logger.exception("The following exception occured: \n" + err)
        logger.info("Something went wrong during query run."
                    "An unknown error was detected in the script.")
        success = False
    return success


def archive_tables(sql_file, source_table, destination_table, run_disposition='WRITE_APPEND'):
    """
    This function is used for archiving tables before overwriting for historical record.
    :param sql_file:
    :param source_table:
    :param destination_table:
    :param run_disposition:
    :return:
    """
    path = path_formatter(sql_file)
    sql_string = open(path).read()
    sql_string.format(source_table)

    try:
        ubq.bq_query_to_table(destination_table, sql_string, write_disposition=run_disposition)
        success = True
    except dbapi.Error as err:
        logger.exception("The following exception occured: \n" + err)
        logger.info("Something went wrong during query run."
                    "An unknown error was detected in the script.")
        success = False
    return success


def move_and_rename_tables(sql_file, source, destination, run_disposition='WRITE_TRUNCATE'):
    path = path_formatter(sql_file)
    sql_string = open(path).read().format(table_id=source)

    try:
        ubq.bq_query_to_table(destination, sql_string, write_disposition=run_disposition)
        success = True
    except dbapi.Error as err:
        logger.exception("The following exception occured: \n" + err)
        logger.info("Something went wrong during query run."
                    "An unknown error was detected in the script.")
        success = False
    return success


def test_summary(sql_file, destination, test_tables, dataset_id=gcp['dataset']):
    """
    :param sql_file:
    :param destination:
    :param test_tables:
    :param dataset_id=gcp['dataset']:
    """
    path = path_formatter(sql_file)
    temp_sql_string = open(path).read()

    lst_sql_str = [
        temp_sql_string.format(table_id=table_id, dataset_id=dataset_id)
        for table_id in test_tables
    ]

    sql_str = '\nunion all\n'.join(lst_sql_str)

    logger.info(f'sql script to execute: {sql_str}')

    try:
        ubq.bq_query_to_table(destination, sql_str)
        run_success = True
    except dbapi.Error as err:
        logger.exception("The following exception occured: \n" + err)
        logger.info("Something went wrong during query run."
                    "An unknown error was detected in the script.")

    return run_success
