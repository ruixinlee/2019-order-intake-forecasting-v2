import sys
import os 
sys.path.append(os.getcwd())
import luigi
from config import tasks
import utils.tasks_wrapper as task_wrapper
from tasks import T200_vista
import pandas as pd
import utils.bq as ubq
from utils import query_composer
from config.conf import gcp
from utils.string_formatter import FormatSQLString


class T302VISTASegmentMap(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['segment_map']

    def complete(self):
        return self.success

    def requires(self):
        return [T200_vista.T201VISTAClean()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T304VISTAEndOfMonth(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['end_of_month']

    def complete(self):
        return self.success

    def requires(self):
        return [T302VISTASegmentMap()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T305VISTAOrderAggregate(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['order_aggregate']

    def complete(self):
        return self.success

    def requires(self):
        return [T302VISTASegmentMap()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T306VISTAPreOrderDays(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['pre_order_days']

    def complete(self):
        return self.success

    def requires(self):
        return [T304VISTAEndOfMonth(),
                T305VISTAOrderAggregate()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T307OrderDaysMeshGrid(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['mesh_grid']

    def complete(self):
        return self.success

    def requires(self):
        return [T306VISTAPreOrderDays()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T308VISTARetailActualAndForecast(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['retail_actual_and_forecast']

    def complete(self):
        return self.success

    def requires(self):
        return [T304VISTAEndOfMonth(),
                T305VISTAOrderAggregate()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T309VISTAGlidePathUNPIVOT(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['glide_path_unpivot']

    def complete(self):
        return self.success

    def requires(self):
        return [T306VISTAPreOrderDays(),
                T307OrderDaysMeshGrid()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T310VISTAMonthsBeforeHandover(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['month_before_handover']

    def complete(self):
        return self.success

    def requires(self):
        return [T309VISTAGlidePathUNPIVOT()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T311VISTARetailAndSoldNumber(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['retail_and_sold_number']

    def complete(self):
        return self.success

    def requires(self):
        return [T306VISTAPreOrderDays(),
                T308VISTARetailActualAndForecast(),
                T310VISTAMonthsBeforeHandover()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = True
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T312VISTACumulativeOrder(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['cumulative_order']

    def complete(self):
        return self.success

    def requires(self):
        return [T311VISTARetailAndSoldNumber()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T313VISTAGlidePathPivot(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['glide_path_pivot']

    def complete(self):
        return self.success

    def requires(self):
        return [T312VISTACumulativeOrder()]

    def run(self):

        # with open(self.task_config['piv_sql_file'], 'r') as sql_fp:
        #     sql_str = sql_fp.readlines()

        # if not sql_str:
        #     # Input
        #     sql_str = open(self.task_config['src_sql_file']).read()
        #     input_sql_str = FormatSQLString.format_dataset_id(sql_string=sql_str,
        #                                                       dataset_id=gcp['dataset'])
        #     df = ubq.bq_query_to_dataframe(input_sql_str)

        #     composer = query_composer.PivotQuery(
        #         data=df,
        #         index_col=self.task_config['piv_idx'],
        #         pivot_col=self.task_config['piv_col'],
        #         values_col=self.task_config['piv_val'],
        #         project_id=gcp['project'],
        #         dataset_id=gcp['dataset'],
        #         table_id=self.task_config['src_table_id'],
        #         add_col_nm_suffix=False,
        #         prefix="_"
        #     )

        #     # generate query string and write it to file
        #     composer.write_query(self.task_config['piv_sql_file'])

        # Load output
        query_file = self.task_config['piv_sql_file']
        table_id = self.task_config['dest_table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success
