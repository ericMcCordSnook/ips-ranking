import sys
from configure_experiment import configure_experiment

if __name__ == '__main__':
    experiment = configure_experiment(sys.argv[1])
    experiment.run()
