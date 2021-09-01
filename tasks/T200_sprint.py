import luigi
from config import tasks
import utils.tasks_wrapper as task_wrapper
import tasks.T100_sprint as T100_sprint


class T200SPRINTActuals(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['sprint']['all']

    def complete(self):
        return self.success

    def requires(self):
        return [T100_sprint.T101SPRINTCurrentActuals(),
                T100_sprint.T102SPRINTHistoricActuals()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class T202SPRINTForecastFilter(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['sprint']['forecast_filter']

    def complete(self):
        return self.success

    def requires(self):
        return [T100_sprint.T103SPRINTForecast()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


if __name__ == '__main__':
    luigi.build([T200SPRINTActuals()])
