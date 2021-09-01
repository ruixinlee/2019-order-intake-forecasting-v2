import luigi
from config import tasks
import utils.tasks_wrapper as task_wrapper
from tasks import T300_sprint, T300_vista


class T400GlidePath(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['final']['glide_path']

    def complete(self):
        return self.success

    def requires(self):
        return [T300_sprint.T303SPRINTMarketAggregate(),
                T300_vista.T313VISTAGlidePathPivot()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T401PredictionCaseForInput(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['final']['prediction_case']

    def complete(self):
        return self.success

    def requires(self):
        return [T400GlidePath(),
                T300_sprint.T301SPRINTPredictionCase()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success
