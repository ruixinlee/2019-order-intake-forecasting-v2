"""
All project configurations are stored here
"""
# conf.py

import logging
from logging.config import fileConfig
import os

# fileConfig('./config/logger.cfg')
logger = logging.getLogger(__name__)

# determine environment to use
# env = DevEnviron().get_environment_config()
get_env = os.getenv('APP_ENV', False)

if get_env:
    env = get_env
else:
    env = 'production'

environments = {
    "development": {
        'gcp': {
            'project': 'jlr-dl-cat',
            'dataset': '2019_Order_Intake_Forecasting_DEV',
            'raw_dataset': '2019_Order_Intake_Forecasting_RAW',
            'dev_dataset': '2019_Order_Intake_Forecasting_DEV',
            'prod_dataset': '2019_Order_Intake_Forecasting_PROD',
            'access_dataset': '2019_Order_Intake_Forecasting_ACCESS'
        },

        'gcs': {
            'project_bucket': '',
            'project_folder': ''
        },

        'luigi': {
            'local_scheduler': False,
            'no_lock': False,
            'host': 'luigi'
        }
    },
    "test": {
        'gcp': {
            'project': 'jlr-dl-cat',
            'dataset': '2019_Order_Intake_Forecasting_TEST',
            'raw_dataset': '2019_Order_Intake_Forecasting_RAW',
            'dev_dataset': '2019_Order_Intake_Forecasting_DEV',
            'prod_dataset': '2019_Order_Intake_Forecasting_PROD',
            'access_dataset': '2019_Order_Intake_Forecasting_ACCESS'
        },

        'gcs': {
            'project_bucket': '',
            'project_folder': ''
        },

        'luigi': {
            'local_scheduler': False,
            'no_lock': False,
            'host': 'localhost'
        }
    },
    "production": {
        'gcp': {
            'project': 'jlr-dl-cat',
            'dataset': '2019_Order_Intake_Forecasting_PROD',
            'raw_dataset': '2019_Order_Intake_Forecasting_RAW',
            'dev_dataset': '2019_Order_Intake_Forecasting_DEV',
            'prod_dataset': '2019_Order_Intake_Forecasting_PROD',
            'access_dataset': '2019_Order_Intake_Forecasting_ACCESS'
        },

        'gcs': {
            'project_bucket': '',
            'project_folder': ''
        },

        'luigi': {
            'local_scheduler': False,
            'no_lock': False,
            'host': 'luigi'
        }
    },
}

gcp = environments[env]['gcp']
gcs = environments[env]['gcs']
task_settings = environments[env]['luigi']
