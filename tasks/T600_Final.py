import luigi
from config import tasks
from tasks import T500_clustering_model, T400_glide_path
from models import clustering
from config.conf import logger
import utils.tasks_wrapper as task_wrapper


class T600Final(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.model_config['etl']['final']

    def complete(self):
        return self.success

    def requires(self):
        return [
            T500_clustering_model.T511ShapeZeroDayAggregates(),
            T400_glide_path.T401PredictionCaseForInput()
        ]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success
