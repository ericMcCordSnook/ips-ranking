import sys
import yaml
from utils import Data_Simulator
import logging
import datetime

def yaml_load(fp):
    with open(fp, "r") as fd:
        config_data = yaml.load(fd)
    return config_data

if __name__ == '__main__':
    config_path = sys.argv[1]
    # Filename is the after the last slash in the path
        # the [:-4] removes the .yml
    config_filename = config_path.split("/")[-1][:-4]
    config_data = yaml_load(config_path)
    log_output = "output/" + config_filename + ".log"
    logging.basicConfig(filename=log_output,
                        level=logging.INFO,
                        format="%(message)s",
                        filemode='w')
    exp_datetime = datetime.datetime.now()
    logging.info("%s\n" % exp_datetime)

    config_data = yaml_load(config_path)
    simulator = Data_Simulator(config_data)
    simulator.simulate()
