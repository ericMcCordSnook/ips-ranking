import sys
import yaml
import time
from experiments import Generic_Experiment
import logging
import datetime

def yaml_load(fp):
    with open(fp, "r") as fd:
        config_data = yaml.load(fd)
    return config_data

def main():
    config_path = sys.argv[1]

    # Filename is the after the last slash in the path
        # the [:-4] removes the .yml
    config_filename = config_path.split("/")[-1][:-4]

    config_data = yaml_load(config_path)
    log_output = "output/random_exp/" + config_filename + ".log"
    logging.basicConfig(filename=log_output,
                        level=logging.INFO,
                        format="%(message)s",
                        filemode='w')
    exp_datetime = datetime.datetime.now()

    logging.info("%s\n" % exp_datetime)
    experiment = Generic_Experiment(config_data)
    start_time = time.time()
    experiment.run()
    end_time = time.time()

    print("\n--- %s seconds ---" % (end_time - start_time))
    logging.info("\n--- %s seconds ---" % (end_time - start_time))

if __name__ == '__main__':
    main()
