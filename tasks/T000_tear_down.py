import luigi
from config import tasks
from tasks import T500_clustering_model
import utils.bq as ubq
import config.conf as conf
import utils.tasks_wrapper as task_wrapper
from google.api_core.exceptions import NotFound
from tasks import T400_glide_path, \
    T500_clustering_model, T900_tests, T600_Final
import logging
import utils.bq_table_metadata as metadata


class UpdateMetadata(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['tear_down']['update_meta_data']

    def complete(self):
        return self.success

    def requires(self):
        return [
            T600_Final.T600Final()
        ]

    def run(self):
        metadata.UpdateMetadata(
            dataset_id=conf.gcp['dataset'],
            metadata_file=self.task_config['data_dict_filepath'],
            table_sheet_map=self.task_config['table_sheet_map'],
            extra_labels=self.task_config['labels']
        ).update_table_metadata()

        self.success = True

        return self.success

    def output(self):
        return self.success


class MoveFinalTableToAccess(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['tear_down']['move_to_access']
    env = conf.env

    def complete(self):
        return self.success

    def requires(self):
        return [
            T400_glide_path.T400GlidePath(),
            T400_glide_path.T401PredictionCaseForInput(),
            T600_Final.T600Final(),
            UpdateMetadata()
        ]

    def run(self):
        if self.env == "production":
            try:
                for tbl in self.task_config['table_id']:
                    ubq.copy_tables(source_dataset_id=conf.gcp['dataset'],
                                    destination_dataset_id=conf.gcp['access_dataset'],
                                    table_id=tbl)
                self.success = True
            except NotFound:
                logging.exception(f'Task {self.task_family} failed!')
        else:
            logging.info(f'In {self.env} enviroment. No access layer needed')
            self.success = True

        return self.success

    def output(self):
        return self.success


class T1001MetaTable(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['tear_down']['create_meta_table']

    def complete(self):
        return self.success

    def requires(self):
        return [
            T500_clustering_model.T506ClassifedClusterInfo(),
            T600_Final.T600Final(),
            MoveFinalTableToAccess(),
            T900_tests.T999aExpectationTestSummary()
        ]

    def run(self):
        query_file = self.task_config['sql_file']
        table_id = self.task_config['table_id']
        self.success = task_wrapper.execute_query_task(sql_file=query_file,
                                                       destination_table=table_id)
        return self.success

    def output(self):
        return self.success


class Final(luigi.WrapperTask):
    """
    Main purpose of this task is to run
    all pipeline tasks therefore teardown tasks are set as its dependencies.
    """
    param = luigi.Parameter(default=None)

    def requires(self):
        return [
            T1001MetaTable()
        ]
