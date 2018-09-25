import yaml
from experiments import Generic_Experiment

def yaml_load(fp):
    with open(fp, "r") as fd:
        config_data = yaml.load(fd)
    return config_data

def yaml_dump(fp, config_data):
    with open(fp, "w") as fd:
        yaml.dump(data, fd)

def create_experiment(fp):
    config_data = yaml_load(fp)
    experiment = Generic_Experiment(config_data)
    return experiment
