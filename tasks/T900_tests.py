import luigi
from config import tasks
from tasks import T400_glide_path
from tests import expectation_tests
from config.conf import logger
import utils.tasks_wrapper as task_wrapper


class UnitTestGlidePath(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False

    def complete(self):
        return self.success

    def requires(self):
        return [T400_glide_path.T400GlidePath()]

    def run(self):
        try:
            self.success = expectation_tests.unittest_glidepath_dataset()
        except:
            logger.error(f'{self.task_family} failed.')
        return self.success

    def output(self):
        return self.success


class T906GlidePathCalculationLogic(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_sql_test['calculation_logic_check']

    def complete(self):
        return self.success

    def requires(self):
        return [T400_glide_path.T400GlidePath()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['result_table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class UnittestPredictionCase(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False

    def complete(self):
        return self.success

    def requires(self):
        return [T400_glide_path.T401PredictionCaseForInput()]

    def run(self):
        try:
            self.success = expectation_tests.unittest_prediction_case_dataset()
        except:
            logger.error(f'{self.task_family} failed.')
        return self.success

    def output(self):
        return self.success


class T999aExpectationTestSummary(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_sql_test['test_summary']['expectation_test']

    def complete(self):
        return self.success

    def requires(self):
        return [
            UnitTestGlidePath(),
            UnittestPredictionCase(),
            T906GlidePathCalculationLogic()
        ]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['result_table_id']
        test_tables = self.task_config['test_tables']
        self.success = task_wrapper.test_summary(query_file, table_id, test_tables)
        return self.success

    def output(self):
        return self.success
