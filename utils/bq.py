import uuid

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from config.conf import gcp
import pprint


def gcs_to_bq_load_auto(gcs_url, table_name, form="CSV", skip_leading=1):
    """
    Loads a csv table from GCS to BigQuery, using autodetect
    :param gcs_url: the gcs url location of the file
    :param table_name: the table_name within the project dataset
    :param form:
    :param skip_leading:
    :return: None
    """
    print("loading {form} from {gcs_url} to {project}.{dataset}.{table_name}".format(gcs_url=gcs_url,
                                                                                     project=gcp['project'],
                                                                                     dataset=gcp['dataset'],
                                                                                     table_name=table_name,
                                                                                     form=form))
    client = bigquery.Client(gcp['project'])
    dataset_ref = client.dataset(gcp['dataset'])
    table_ref = dataset_ref.table(table_name)
    job_name = str(uuid.uuid4())

    job_config = bigquery.LoadJobConfig()
    job_config.create_disposition = 'CREATE_IF_NEEDED'
    if skip_leading:
        job_config.skip_leading_rows = skip_leading
    job_config.source_format = form
    job_config.autodetect = True
    job_config.write_disposition = 'WRITE_TRUNCATE'

    load_job = client.load_table_from_uri(
        gcs_url, table_ref, job_config=job_config,
        job_id_prefix=job_name)

    res = load_job.result()

    print("success - loaded {} rows from {} into {}".format(load_job.output_rows, gcs_url, table_name))

    return res


def bq_query_to_table(table_name, query, dataset_id=gcp['dataset'], write_disposition='WRITE_TRUNCATE', use_legacy=False):
    """
    Runs a BigQuery query and creates/writes to a table in BigQuery with resulting data

    :param write_disposition:
    :param table_name: name of the table in bigquery
    :param query: the SQL query for retrieving data
    :param dataset_id:
    :param use_legacy:
    :return: an instance of query job object

    """
    print("running the following query: {} on table: {}".format(query, table_name))

    client = bigquery.Client()
    query_conf = bigquery.QueryJobConfig()
    table_ref = client.dataset(dataset_id=dataset_id).table(table_name)
    query_conf.create_disposition = 'CREATE_IF_NEEDED'
    query_conf.write_disposition = write_disposition
    query_conf.destination = table_ref
    if use_legacy:
        query_conf.use_legacy_sql = True
    run_query = client.query(query, job_config=query_conf)
    res = run_query.result()

    print('Query results loaded to table {}'.format(table_ref.path))
    return res


def bq_append_to_table(table_name, query):
    """
    Runs a BigQuery query and creates/writes to a table in BigQuery with resulting data

    :param table_name: name of the table in bigquery
    :param query: the SQL query for retrieving data
    :return: an instance of query job object
    """
    client = bigquery.Client(gcp['project'])

    print("running the following query: {} on table: {}".format(query, table_name))

    query_conf = bigquery.QueryJobConfig()
    table_ref = client.dataset(dataset_id=gcp['dataset']).table(table_name)

    query_conf.write_disposition = 'WRITE_APPEND'
    query_conf.destination = table_ref

    run_query = client.query(query, job_config=query_conf)

    res = run_query.result()

    print("success - query complete")

    return res


class CheckBQTableExists:

    def __init__(self, table, dataset):
        self.table = table
        self.dataset = dataset

    def exists(self):
        """
        Check if a provided table exists in the project dataset, if exists returns True, else False.
        :return: Boolean
        """
        client = bigquery.Client()
        dataset_ref = client.dataset(self.dataset)
        table_ref = dataset_ref.table(self.table)
        print("checking if {table} exists within {project}.{dataset}"
              .format(table=self.table, project=gcp['project'], dataset=self.dataset))
        try:
            client.get_table(table_ref)
            print("table found: {}, returning True".format(self.table))
            return True
        except NotFound:
            print("table not found: {}, returning False".format(self.table))
            return False


class CheckTableToAppend:

    def __init__(self, table):
        self.table = table

    def exists(self):
        """
        Check if a provided table exists in the project dataset, if exists returns True, else False.
        :return: Boolean
        """
        client = bigquery.Client(gcp['project'])
        dataset_ref = client.dataset(gcp['dataset'])
        table_ref = dataset_ref.table(self.table)
        print("checking if {table} exists within {project}.{dataset}"
              .format(table=self.table, project=gcp['project'], dataset=gcp['dataset']))
        try:
            client.get_table(table_ref)
            print("table found: {}, returning True".format(self.table))
            return True
        except NotFound:
            print("table not found: {}, returning False".format(self.table))
            return False


def gcs_to_bq_load_schema(gcs_url, table_name, schema, delimiter=","):
    """
    Loads a table from GCS to BigQuery, using a specified schema
    :param gcs_url: the gcs url location of the file
    :param table_name: the table_name within the project dataset
    :param schema: schema of table being loaded into BQ
    :param delimiter: delimiter of csv, defaults to ","
    :return: None

    """
    client = bigquery.Client(gcp['project'])
    dataset_ref = client.dataset(gcp['dataset'])
    print("loading data from gcs at location: {} "
          "to BigQuery, table name: {}, "
          "schema: {}".format(gcs_url, table_name, str(schema)))

    table_ref = dataset_ref.table(table_name)
    job_name = str(uuid.uuid4())

    job_config = bigquery.LoadJobConfig()
    job_config.create_disposition = 'CREATE_IF_NEEDED'
    job_config.skip_leading_rows = 1
    job_config.source_format = 'CSV'
    job_config.field_delimiter = delimiter
    job_config.schema = schema
    job_config.write_disposition = 'WRITE_APPEND'

    load_job = client.load_table_from_uri(
        gcs_url, table_ref, job_config=job_config, job_id_prefix=job_name)

    res = load_job.result()

    print('success - loaded {} rows from {} into {}'.format(load_job.output_rows, gcs_url, table_name))

    return res


def bq_table_to_dataframe(project_id, dataset_id, table_id, limit=False):
    """
    Process string and output as dataframe object.
    :param project_id:
    :param dataset_id:
    :param table_id: table_name string to call a key in dictionary gcp.
    :return: df - dataframe equivalent of query result.
    """
    sql_str = ""

    if limit:
        sql_str = """SELECT *
                       FROM `{project_id}.{dataset_id}.{table_id}`
                      LIMIT 10000""".format(project_id=project_id,
                                            dataset_id=dataset_id,
                                            table_id=table_id)
    else:
        sql_str = """SELECT *
                       FROM `{project_id}.{dataset_id}.{table_id}`""".format(project_id=project_id,
                                                                             dataset_id=dataset_id,
                                                                             table_id=table_id)

    df = bq_query_to_dataframe(sql_str)
    return df


def bq_query_to_dataframe(str_sql):
    """
    Process sql string and output as dataframe object.
    :param str_sql: sql query sting to be processed/read to dataframe.
    :return: df - dataframe equivalent of query result.
    """
    client = bigquery.Client()
    df = client.query(str_sql).to_dataframe()
    return df


def get_table_schema(project_id, dataset_id, table_id):
    """
    :param project_id:
    :param dataset_id:
    :param table_id:
    :return:
    """
    pp = pprint.PrettyPrinter(indent=4)
    client = bigquery.Client(project_id)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    table = client.get_table(table_ref)
    schema = table.schema

    pp.pprint(list(schema))


def list_tables(dataset_id):
    """
    List the tables within a dataset
    :param dataset_id:
    :return table_id:
    """
    client = bigquery.Client(gcp['project'])
    dataset_ref = client.dataset(dataset_id)
    tables = list(client.list_tables(dataset_ref))
    tables_id = [table.table_id for table in tables]

    return tables_id


def load_df_to_bigquery(df, table_name, dataset=gcp['dataset'], if_exists='replace'):
    """
    This function takes a pandas dataframe as input and load
    the data into a big query table.
    :param if_exists:
    :param df:
    :param table_name:
    :param dataset
    :return:
    """
    df.to_gbq(destination_table='{}.{}'.format(dataset, table_name),
              project_id=gcp['project'],
              if_exists=if_exists)


def delete_table(dataset_id, table_id):
    """
    Delete the table from big query dataset
    :param dataset_id:
    :param table_id:
    :return:
    """
    client = bigquery.Client()

    try:
        dataset_ref = client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        client.delete_table(table_ref)

        print('Table {}:{} deleted.'.format(dataset_id, table_id))
    except NotFound as ex:
        print(ex)


def row_count(self, table_ref):
    """[summary]

    :param table_ref: [description]
    :type table_ref: [type]
    :return: [description]
    :rtype: [type]
    """
    # table_ref = client.dataset(dataset_id).table("table_id")
    table = self.client.get_table(table_ref)
    return table.num_rows


def copy_tables(
    source_dataset_id, destination_dataset_id, table_id, project_id=gcp['project']
):
    """
    :param source_dataset_id:
    :param destination_dataset_id:
    :param table_id:
    :param project_id:
    :return:
    """
    client = bigquery.Client()

    job_config = bigquery.CopyJobConfig()
    job_config.write_disposition = "WRITE_TRUNCATE"
    src = client.dataset(source_dataset_id)
    dest = client.dataset(destination_dataset_id)

    if CheckBQTableExists(table_id, destination_dataset_id).exists():
        src_table_ref = src.table(table_id)
        dest_table_ref = dest.table(table_id)
        print(
            f'copying {project_id}.{source_dataset_id}.{table_id} to \
            {project_id}.{destination_dataset_id}.{table_id}'
        )
        job = client.copy_table(
            src_table_ref, dest_table_ref, job_config=job_config
        )

    job.result()
    assert job.state == 'DONE'
    assert row_count(src_table_ref) == row_count(dest_table_ref)

    return


def create_partition_table(table_id, schema_field):
    """
    Creates a partitioned table.
    :param table_id:
    :param schema_field:
    :return:
    """
    client = bigquery.Client()
    dataset_ref = client.dataset(gcp['dataset'])
    table_ref = dataset_ref.table(table_id)

    table = bigquery.Table(table_ref, schema=schema_field)
    table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY,
                                                        expiration_ms=None)

    table = client.create_table(table)

    print('Created table {}, partitioned on column {}'.format(
        table.table_id, table.time_partitioning.field))
