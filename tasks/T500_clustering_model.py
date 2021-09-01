import luigi
from config import tasks
from tasks import T400_glide_path
from models import clustering
from utils.custom_exceptions import ModelRunError
from config.conf import logger
import utils.tasks_wrapper as task_wrapper


class T500ClusteringModel(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False

    def complete(self):
        return self.success

    def requires(self):
        return [
            T400_glide_path.T400GlidePath(),
            T400_glide_path.T401PredictionCaseForInput()
        ]

    def run(self):
        try:
            clustering.main()
            self.success = True
        except ModelRunError:
            logger.error(
                f"Task {self.task_family} failed. \
                    Something went wrong during model task run."
            )
            self.success = False

        return self.success

    def output(self):
        return self.success


class T506ClassifedClusterInfo(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.model_config['etl']['clus_info']

    def complete(self):
        return self.success

    def requires(self):
        return [
            T500ClusteringModel()
        ]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T507ZeroDayClusterTestCasesClean(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.model_config['etl']['zero_day_test_case']

    def complete(self):
        return self.success

    def requires(self):
        return [
            T500ClusteringModel(),
            T506ClassifedClusterInfo()
        ]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T508ZeroDayClusterTestCasesPivot(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.model_config['etl']['zero_day_test_case_piv']

    def complete(self):
        return self.success

    def requires(self):
        return [T507ZeroDayClusterTestCasesClean()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T509ShapeClusterTestCasesClean(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.model_config['etl']['shape_test_case']

    def complete(self):
        return self.success

    def requires(self):
        return [
            T500ClusteringModel(),
            T508ZeroDayClusterTestCasesPivot()
        ]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T510ShapeClusterTestCasesPivot(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.model_config['etl']['shape_test_case_piv']

    def complete(self):
        return self.success

    def requires(self):
        return [T509ShapeClusterTestCasesClean()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T511ShapeZeroDayAggregates(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.model_config['etl']['shape_zero_day']

    def complete(self):
        return self.success

    def requires(self):
        return [
            T508ZeroDayClusterTestCasesPivot(),
            T510ShapeClusterTestCasesPivot()
        ]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success
