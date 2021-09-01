import luigi
from config import tasks
from config.conf import gcp
import tasks.T000_setup as T000_setup
import utils.tasks_wrapper as task_wrapper


class T101SPRINTCurrentActuals(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['sprint']['current']

    def complete(self):
        return self.success

    def requires(self):
        return [T000_setup.ArchiveTables()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id,
                                                       dataset_id=gcp['raw_dataset'])
        return self.success

    def output(self):
        return self.success


class T102SPRINTHistoricActuals(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['sprint']['historic']

    def complete(self):
        return self.success

    def requires(self):
        return [T000_setup.ArchiveTables()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id,
                                                       dataset_id=gcp['raw_dataset'])
        return self.success

    def output(self):
        return self.success


class T103SPRINTForecast(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['sprint']['forecast']

    def complete(self):
        return self.success

    def requires(self):
        return [T000_setup.ArchiveTables()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id,
                                                       dataset_id=gcp['raw_dataset'])
        return self.success

    def output(self):
        return self.success
