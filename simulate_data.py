import sys
import yaml
from utils import Data_Simulator

def yaml_load(fp):
    with open(fp, "r") as fd:
        config_data = yaml.load(fd)
    return config_data

if __name__ == '__main__':
    config_data = yaml_load(sys.argv[1])
    simulator = Data_Simulator(config_data)
    simulator.simulate()
