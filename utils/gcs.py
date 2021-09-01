#
# gcs.py
#

import io

from google.api_core.exceptions import NotFound
from google.cloud import storage

from config.conf import gcp

#
# vars
#

storage_client = storage.client.Client(project=gcp['project'])
storage_bucket = storage.bucket.Bucket(storage_client, gcp['bucket'])


#
# funcs
#


def gcs_move_to_archive(file_name, archive_location):
    """
    Move file from one gcs folder to another, within project bucket

    :param file_name: location of gcs file within bucket (within project bucket)
    :param archive_location: archive location of file within bucket (within project bucket)
    """
    print("moving file from location: {}, to location: {}, within {}".format(file_name, archive_location,
                                                                             gcp['bucket']))

    storage_blob = storage.Blob(file_name, storage_bucket)
    storage_bucket.copy_blob(storage_blob, storage_bucket, archive_location)

    return archive_location


def gcs_remove_file(file_name):
    """
    Remove blob from project gcs bucket

    :param file_name: blob file location to be removed (within project bucket)
    :return: file location of file removed
    """
    storage_blob = storage.Blob(file_name, storage_bucket)

    if storage_blob.exists():
        print('removing {file_name} from gcs://{project}'.format(file_name=file_name, project=gcp['project']))
        storage_blob.delete()

    return file_name


def get_file_storage(filename, file_location, file_format):
    """
    retrieves file from Google Cloud Storage based on filename/location

    :param filename: blob file name to be downloaded (within project bucket)
    :param file_location: blob file location to be downloaded (within project bucket)
    :param file_format: blob file format to be downloaded (within project bucket)

    :return: io.Bytes() object of downloaded file
    """
    # create file name string to retrieve
    file_str = '{}{}{}'.format(file_location, filename, file_format)
    print("searching for {}{} in {}".format(filename, file_format, file_str))
    storage_blob = storage.Blob(file_str, storage_bucket)
    # create io Bytes buffer to hold the file in memory
    file_buffer = io.BytesIO()
    try:
        storage_blob.download_to_file(file_buffer)
        # rewind buffer to start for reading
        file_buffer.seek(0)
    except NotFound:
        print("err - reading file from gcp - file not found")
        return False
    return file_buffer


def send_file_to_gcp(filename, file_location, file, content_type="text/csv"):
    """
    store text file in Google Cloud Storage

    :param file: dataframe object which has been modified by the application
    :param filename: name of new file
    :param file_location: file location string (from project bucket, not gs://)
    :param content_type: mime code of new file sent to GCP

    :return bool: True / False depending on success

    """

    file_str = file_location + filename
    # storage blob complete
    storage_blob = storage.Blob(file_str, storage_bucket)
    # attempt to send data file
    try:
        if storage_blob.exists():
            print("err - removing existing file from gcp storage: {}".format(filename))
            storage_blob.delete()
        storage_blob.upload_from_string(file, content_type=content_type)
        return True
    except Exception as e:
        print("err - uploading file to gcp: {}".format(e))
        return False


def send_dataframe_to_gcp(filename, file_location, processed_df):
    """
    store processed dataframe in Google Cloud Storage

    :param processed_df: dataframe object which has been modified by the application
    :param filename: name of new file
    :param file_location: file location string (from project bucket, not gs://)

    :return bool: True / False depending on success
    """

    file_str = file_location + filename
    # storage blob complete
    storage_blob = storage.Blob(file_str, storage_bucket)
    # create in-memory csv string
    csv_str = processed_df.to_csv(index=False)
    # attempt to send data file
    try:
        if storage_blob.exists():
            print("err - removing existing file from gcp storage: {}".format(filename))
            storage_blob.delete()
        storage_blob.upload_from_string(csv_str, content_type='text/csv')
        return True
    except Exception as e:
        print("err - uploading file to gcp: {}".format(e))
        return False


class CheckFileInFolder:
    """
    check file exists in given gcs location (within project bucket)

    :param filename: blob file name to be downloaded (within project bucket)
    :return bool: True if exists, else False

    """

    def __init__(self, filename):
        self.file_name = filename

    def exists(self):
        storage_blob = storage.Blob(self.file_name, storage_bucket)
        print('checking if {file_name} exists in gcs://{project}...'
              .format(file_name=self.file_name, project=gcp['project']))
        if storage_blob.exists():
            print("file: {} exists, returning True".format(self.file_name))
            return True
        print("file: {} does not exist within: {}".format(self.file_name, gcp['project']))
        return False


def create_gcs_location_str(location):
    """
    create string for gcs file locations

    :param location: file folder location
    :return string: gcs file location string
    """

    return "gs://{bucket}/{location}".format(bucket=gcp['bucket'], location=location)


def list_blobs_with_prefix(prefix, delimiter='/'):
    """
    Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        /a/1.txt
        /a/b/2.txt

    If you just specify prefix = '/a', you'll get back:

        /a/1.txt
        /a/b/2.txt

    However, if you specify prefix='/a' and delimiter='/', you'll get back:

        /a/1.txt
    :param prefix:
    :param delimiter:
    :return:
    """
    blobs = storage_bucket.list_blobs(prefix=prefix, delimiter=delimiter)

    lst_blob = []
    for blob in blobs:
        lst_blob.append(blob)

    lst_blob.pop(0)

    return lst_blob
