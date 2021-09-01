import luigi
from config import tasks, conf
import utils.tasks_wrapper as task_wrapper
import utils.bq as ubq
from utils.string_formatter import FormatSQLString


class ExtractLatestRaw(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['setup']['extract_raw_tables']

    def complete(self):
        return self.success

    def requires(self):
        return []

    def run(self):
        query_file = self.task_config['sql_file']
        for src, dest in self.task_config['table_id']:
            self.success = task_wrapper.move_and_rename_tables(query_file, src, dest)
        return self.success

    def output(self):
        return None


class ArchiveTables(luigi.Task):
    param = luigi.Parameter(default=None)
    success = False
    task_config = tasks.etl_config['setup']['archiving']

    def complete(self):
        return self.success

    def requires(self):
        return []

    def run(self):
        table_meta = self.task_config["table_id"]
        query_file = self.task_config['sql_file']
        sql_str = open(query_file).read()
        sql_str = FormatSQLString.format_dataset_id(sql_str, conf.gcp['dataset'])

        for tbl, schema in table_meta.items():
            archive_id = tbl + "_Archive"
            # create partition table if missing
            if not ubq.CheckBQTableExists(archive_id, conf.gcp['dataset']).exists():
                ubq.create_partition_table(
                    table_id=archive_id,
                    schema_field=schema
                )

            # create the partition
            if ubq.CheckBQTableExists(tbl, conf.gcp['dataset']).exists():
                ubq.bq_append_to_table(archive_id, FormatSQLString.format_table_id(sql_str, tbl))
                print("Archived {}!".format(tbl))
            else:
                print('ERR: {} not found! in {}'.format(tbl, conf.gcp['dataset']))

        self.success = True

    def output(self):
        return self.success
