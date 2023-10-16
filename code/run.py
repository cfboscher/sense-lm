import argparse

from config.config import Config
from config.config import update_config

from step_1 import step_1
from step_2 import step_2

import os
import json

if __name__ == '__main__':

    # Load initial configuration
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    args = parser.parse_args()

    if args.config_file is not None:
        config = update_config(config, args.config_file)


    # Run Experiments
    if config.steps=="all" or config.steps=="1":
        step_1.run(config)

    if config.steps=="all" or config.steps=="2":
        step_2.run(config)
        pass


