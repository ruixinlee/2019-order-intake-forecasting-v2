# metadata_update.py
"""
    This module is used to update all the data tables
    metadata information. This module only works
    if the suggested data dictionary format presented
    in the analytics data engineering governance is
    used.
"""

import google.cloud.bigquery as bq
import pandas as pd
from google.api_core.exceptions import ClientError

from pathlib import Path
from collections import OrderedDict


class UpdateMetadata:

    def __init__(self, dataset_id: str, metadata_file: str,
                 table_sheet_map: dict, extra_labels: dict=None) -> None:
        """
        class initialisation
        :param dataset_id:
        :param metadata_file:
        :param table_sheet_map: key-value pair where the key point to table name
                                and its corresponding value is the BQ sheet name.
        :param extra_labels: optional- key-value pairs to be used as table label.
        """
        self.dataset_id = dataset_id

        file = Path(metadata_file)

        if file.is_file():
            self.metadata_file = metadata_file
        else:
            raise FileNotFoundError
            print(f'the referenced file: {metadata_file} might not exist.')

        self.table_sheet_map = table_sheet_map
        self.extra_labels = extra_labels
        self.client = bq.Client()

    def _load_metadata_from_file(self, sheet_name: str=None) -> pd.core.frame.DataFrame:
        """
        load excel file into dataframe and parse sheet with metadata detail only
        :param sheet_name: name of sheet corresponding to table name
        :return: df
        """
        if sheet_name:
            try:
                xl = pd.ExcelFile(self.metadata_file)
                df = xl.parse(sheet_name, header=None)
                return df
            except TypeError as err:
                print(f'{err}: Something went wrong during metadata file processing.'
                      'Make sure you specified a sheet name for parsing.')

    def _get_table(self, table_id: str) -> bq.Table:
        """
        Gets big query table to add metadata.
        :return: table
        """
        client = self.client
        dataset_ref = client.dataset(self.dataset_id)
        table_ref = dataset_ref.table(table_id)
        table = client.get_table(table_ref)

        return table

    @staticmethod
    def _get_table_schema(table):
        """
        Retrieves schema from big query table.
        :param table: This is the output of the get_table() method.
        :return:
        """
        schema = table.schema
        return schema

    def _update_table_desc(self, table, table_description):
        """
        Update table description.
        :param table:
        :param table_description:
        :return: None
        """
        if table.description == table_description:
            print('Table description already up to date')
        else:
            try:
                table.description = table_description
                self.client.update_table(table, ['description'])
                assert table.description == table_description
                print(f'Finished updating {table.table_id} description')
            except ClientError as err:
                print(err)

    def _update_table_labels(self, table, dict_table_label):
        """
        Update the table labels.
        :param table:
        :param dict_table_label:
        :return:
        """
        if table.labels == dict_table_label:
            print('table labels already up to date.')
        else:
            try:
                table.labels = dict_table_label
                self.client.update_table(table, ['labels'])
                assert table.labels == dict_table_label
                print(f'Finished updating {table.table_id} label')
            except ClientError as err:
                print(err)

    def _update_table_field_desc(self, table, df_schema):
        """
        Update the table column definitions.
        :param table:
        :param df_schema:
        :return:
        """
        original_schema = self._get_table_schema(table)

        field_name = df_schema[0].tolist()
        type_ = df_schema[1].tolist()
        mode = df_schema[2].tolist()
        description = df_schema[4].tolist()

        assert field_name == [field.name for field in original_schema]

        table_schema = [bq.SchemaField(field_name[i],
                                       type_[i],
                                       mode[i],
                                       description[i]) for i in range(len(field_name))]
        if table.schema == table_schema:
            print('table schema already up to date')
        else:
            try:
                table.schema = table_schema
                self.client.update_table(table, ['schema'])
                assert len(table.schema) == len(original_schema) == len(table_schema)
                print(f'Finished updating {table.table_id} fields definition')
            except ClientError as err:
                print(err)

    @staticmethod
    def _extract_desc(df_meta):
        """
        This extract the table description from dataframe.
        :param df_meta:
        """
        return df_meta.iloc[1, 1]

    @staticmethod
    def _extract_schema(df_meta):
        """
        This extract the schema of a big query table from dataframe.
        :param df_meta:
        """
        return df_meta.iloc[5:]

    def _create_label(self, df_meta: pd.core.frame.DataFrame) -> OrderedDict:
        """
        Create a table label from data from in metadata dataframe and
        addition input
        :param self:
        :param df_meta:
        """
        label = OrderedDict()

        default_key = str(df_meta.iloc[2, 0]).replace(' ', '_').lower().strip(':')
        default_val = str(df_meta.iloc[2, 1]).lower()

        label[default_key] = default_val

        if self.extra_labels:
            label.update(self.extra_labels)

        return label

    def update_table_metadata(self) -> None:
        """
        Runs the metadata table details update
        :param self:
        :return None:
        """

        for table_id, sheet_name in self.table_sheet_map.items():
            print(f'updating metadata for {table_id} using data dictionary sheet - {sheet_name}')

            table = self._get_table(table_id)

            # load the table to df
            df_meta = self._load_metadata_from_file(sheet_name)

            # define inputs
            desc = self._extract_desc(df_meta)
            schema = self._extract_schema(df_meta)
            label = self._create_label(df_meta)

            self._update_table_desc(table, desc)
            self._update_table_labels(table, label)
            self._update_table_field_desc(table, schema)

        print('Metadata update complete.')
