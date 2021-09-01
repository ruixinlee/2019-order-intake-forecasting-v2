import re

import pandas as pd

# from utils.bq import bq_query_to_dataframe
from config import tasks
import os
import sys

sys.path.append(os.getcwd())


class PivotQuery:
    """
    Class to generate a SQL query which creates pivoted tables in BigQuery.
    
    """

    def __init__(self, data, index_col, pivot_col, values_col,
                 project_id=None, dataset_id=None, table_id=None, agg_fun="sum", not_eq_default=0, 
                 add_col_nm_suffix=True, custom_agg_fun=None, prefix=None, suffix=None):
        """
        :paramn self:
        :param data: pandas.core.frame.DataFrame or string
            The input data can either be a pandas dataframe or a string path to the pandas
            data frame. The only requirement of this data is that it must have the column
            on which the pivot it to be done.
        :param index_col: list
            The names of the index columns in the query (the columns on which the group by needs to be performed)
        :param pivot_col: string
            The name of the column on which the pivot needs to be done.
        :param values_col: string
            The name of the column on which aggregation needs to be performed.
        :param agg_fun: string
            The name of the sql aggregation function.
        :param table_id: string
            The name of the table in the query.
        :param not_eq_default: numeric, optional
            The value to take when the case when statement is not satisfied. For example,
            if one is doing a sum aggregation on the value column then the not_eq_default should
            be equal to 0. Because the case statement part of the sql query would look like - 
            
            ... ...
            sum(case when <pivot_col> = <some_pivot_col_value> then values_col else 0)
            ... ...
            Similarly if the aggregation function is min then the not_eq_default should be
            positive infinity.
        :param add_col_nm_suffix: boolean, optional
            If True, then the original values column name will be added as suffix in the new 
            pivoted columns.
        :param custom_agg_fun: string, optional
            Can be used if one wants to give customized aggregation function. The values col name 
            should be replaced with {}. For example, if we want an aggregation function like - 
            sum(coalesce(values_col, 0)) then the custom_agg_fun argument would be - 
            sum(coalesce({}, 0)). 
            If provided this would override the agg_fun argument.
        :param prefix: string, optional
            A fixed string to add as a prefix in the pivoted column names separated by an
            underscore.
        :param suffix: string, optional
            A fixed string to add as a suffix in the pivoted column names separated by an
            underscore.        
        """
        self.query = ""
        self.index_col = list(index_col)
        self.values_col = values_col
        self.pivot_col = pivot_col
        self.not_eq_default = not_eq_default
        self.table_name = self._get_table_name(project_id, dataset_id, table_id)
        self.piv_col_vals = self._get_piv_col_vals(data)
        self.piv_col_names = self._create_piv_col_names(add_col_nm_suffix, prefix, suffix)
        self.function = custom_agg_fun.upper() if custom_agg_fun else agg_fun.upper() + "({})"

    def _get_table_name(self, project_id, dataset_id, table_id):
        """
        Construct query table name in the format <project_id>.<dataset_id>.<table_id>
        :param self:
        :param project_id:
        :param dataset_id:
        :param table_id:
        """
        return "`{}.{}.{}`".format(project_id, dataset_id, table_id)

    def _get_piv_col_vals(self, data):
        """
        Gets all the unique values of the pivot column.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            raise ValueError("Provided data must be a pandas dataframe or a csv file path.")

        if self.pivot_col not in self.data.columns:
            raise ValueError("The provided data must have the column on which pivot is to be done. "
                             "Also make sure that the column name in the data is same as the name "
                             "provided to the pivot_col parameter.")

        return self.data[self.pivot_col].astype(str).unique().tolist()

    def _clean_col_name(self, col_name):
        """
        The pivot column values can have arbitrary strings but in order to 
        convert them to column names some cleaning is required. This method 
        takes a string as input and returns a clean column name.
        Transformation done here
            * replace spaces with underscores
            * replace non alpha numeric characters with underscores
            * replace multiple consecutive underscores with one underscore
            * make all characters upper case
            * remove trailing underscores
        :param self: 
        :param col_name: 
        """
        return re.sub("_+", "_", re.sub("[^0-9a-zA-Z_]+", "", re.sub(" ", "_", col_name))).upper().strip("_")

    def _create_piv_col_names(self, add_col_nm_suffix, prefix, suffix):
        """
        The method created a list of pivot column names of the new pivoted table.
        :param self: 
        :param add_col_nm_suffix: 
        :param prefix: 
        :param suffix: 
        """   
        prefix = prefix + "_" if prefix else ""
        suffix = "_" + suffix if suffix else ""

        if add_col_nm_suffix:
            piv_col_names = ["{0}{1}_{2}{3}".format(prefix,
                                                    self._clean_col_name(piv_col_val),
                                                    self.values_col.upper(),                                
                                                    suffix) 
                            for piv_col_val in self.piv_col_vals]
        else:
            piv_col_names = ["{0}{1}{2}".format(prefix, self._clean_col_name(piv_col_val), suffix) 
                             for piv_col_val in self.piv_col_vals]

        return piv_col_names

    def _add_select_statement(self):
        """
        Adds the select statement part of the query.
        """
        query = "SELECT " + "\n".join([index_col + ", " for index_col in self.index_col]) + "\n"
        return query

    def _add_case_statement(self):
        """
        Adds the case statement part of the query.
        :paramn self:
        # TODO: Do a check for case when value 1 is type string
        """
        case_query = (self.function.format('CASE WHEN {0} = {1} THEN {2} ELSE {3} END') +
                                           " AS {4},\n")

        query = "".join([case_query.format(self.pivot_col,
                                           piv_col_val,
                                           self.values_col,
                                           self.not_eq_default,
                                           piv_col_name,)
                         for piv_col_val, piv_col_name in zip(self.piv_col_vals, 
                                                              self.piv_col_names)])
        query = query[:-2] + "\n"
        return query

    def _add_from_statement(self):
        """
        Adds the from statement part of the query.
        :return:
        """
        query = "FROM {0}\n".format(self.table_name)
        return query

    def _add_group_by_statement(self):
        """
        Adds the group by part of the query.
        :return:
        """
        query = "GROUP BY " + " ".join(["{0},".format(x) for x in self.index_col])
        return query[:-1]

    def generate_query(self):
        """
        Returns the query to create the pivoted table.
        :return:
        """
        self.query = (self._add_select_statement()
                      + self._add_case_statement()
                      + self._add_from_statement()
                      + self._add_group_by_statement())
        return self.query

    def write_query(self, output_file):
        """
        Writes the query to a SQL text file.
        :param output_file:
        :return: 
        """
        with open(output_file, "w") as file:
            file.write(self.generate_query())



if __name__ == "__main__":
    # sql_str = open(task_config['src_sql_file']).read()
    # input_sql_str = FormatSQLString.format_dataset_id(sql_string=sql_str, 
    #                                                       dataset_id=gcp['dataset'])
    # df = ubq.bq_query_to_dataframe(input_sql_str)
    # df = pd.read_csv(r)
    task_config = tasks.etl_config['vista']['glide_path_pivot']
    composer = PivotQuery(data=r'C:\Users\earoge\Downloads\bq-results.csv', 
                          index_col=task_config['piv_idx'],
                          pivot_col=task_config['piv_col'],
                          values_col=task_config['piv_val'],
                          project_id='jlr-dl-cat',
                          dataset_id='earoge',
                          table_id=task_config['src_table_id'],
                          add_col_nm_suffix=False,
                          prefix="_")
        
        # generate query string and write it to file
    composer.write_query("./query.sql")