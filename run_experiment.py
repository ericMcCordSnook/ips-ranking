import sys
import yaml
from experiments import Generic_Experiment

def yaml_load(fp):
    with open(fp, "r") as fd:
        config_data = yaml.load(fd)
    return config_data

if __name__ == '__main__':
    config_data = yaml_load(sys.argv[1])
    experiment = Generic_Experiment(config_data)
    experiment.run()
