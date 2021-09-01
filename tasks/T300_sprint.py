import luigi
from config import tasks
import utils.tasks_wrapper as task_wrapper
from tasks import T200_sprint


class T300SPRINTActualsModelMap(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['sprint']['actuals_model_map']

    def complete(self):
        return self.success

    def requires(self):
        return [T200_sprint.T200SPRINTActuals()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T301SPRINTPredictionCase(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['sprint']['prediction_case']

    def complete(self):
        return self.success

    def requires(self):
        return [T200_sprint.T202SPRINTForecastFilter()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T303SPRINTMarketAggregate(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['sprint']['market_aggregate']

    def complete(self):
        return self.success

    def requires(self):
        return [T300SPRINTActualsModelMap()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success
