# string_formatter.py
"""
    This module is useful for formatting SQL strings read from file.
    See https://docs.python.org/3.6/library/string.html#template-strings
    to see how to use the Template strings method
"""
from string import Template


class FormatSQLString:

    def __init__(self):
        pass

    @staticmethod
    def format_dataset_id(sql_string: str, dataset_id: str) -> str:
        """
        Formats SQL string by replacing its' dataset id
        :param sql_string:
        :param dataset_id:
        :return:
        """
        temp_str = Template(sql_string)
        formatted_str = temp_str.safe_substitute(dataset_id=dataset_id)
        return formatted_str

    @staticmethod
    def format_table_id(sql_string: str, table_id: str) -> str:
        """
        Formats SQL string by replacing its' table id
        :param sql_string:
        :param table_id:
        :return:
        """
        temp_str = Template(sql_string)
        formatted_str = temp_str.safe_substitute(table_id=table_id)
        return formatted_str
