# entry.py

import os
import warnings

import luigi

import tasks.T000_tear_down as tearDown
from config.conf import logger, task_settings

warnings.simplefilter("ignore", UserWarning)

#
# luigi pipeline management configuration
#

luigi_config = luigi.configuration.get_config()
luigi_config.read("./config/luigi.cfg")

#
# get environment variables
#

RUN_PIPELINE = os.getenv("RUN_ALL", True)

#
# run pipeline
#


def run():
    if RUN_PIPELINE:
        luigi.build(
            [tearDown.Final()],
            local_scheduler=task_settings["local_scheduler"],
            no_lock=task_settings["no_lock"],
            scheduler_host=task_settings["host"]
            # workers=2
            # parallel_scheduling=True
        )


if __name__ == "__main__":
    logger.info("Starting pipeline...")

    run()

    logger.info("Finished Pipeline!")
