import sys
from create_experiment import create_experiment

if __name__ == '__main__':
    experiment = create_experiment(sys.argv[1])
    experiment.run()
