import luigi
from config import tasks
import utils.tasks_wrapper as task_wrapper
import tasks.T100_vista as T100_vista


class T201VISTAClean(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['vista']['clean']

    def complete(self):
        return self.success

    def requires(self):
        return [T100_vista.T100VISTAData()]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


if __name__ == '__main__':
    luigi.build([T201VISTAClean()])
