from rankobjects import *
from optimizations import *
from heuristics import *
from utils import *
from utils.miscutils import get_data, get_frequency_distribution
from importlib import import_module
import logging

class Generic_Experiment:
    def __init__(self, config_data):
        print(config_data)
        logging.info("Experiment Configuration \n %s \n" % config_data)
        self.config_data = config_data
        self.optimization = None
        self.heuristic = None
        self.weight = None
        self.data = None
        self.configure_experiment()

    def get_class(self, name):
        module, classname = name.rsplit(".", 1)
        try:
            return getattr(import_module(module), classname)
        except AttributeError as e:
            msg = ('%s while trying to import %r from %r'
                   % (e.args[0], classname, module))
            e.args = (msg,) + e.args[1:]
            raise e

    def configure_experiment(self):
        attributes = ["optimization", "heuristic", "weight"]
        for attribute in attributes:
            if attribute in self.config_data:
                attr_obj = self.get_class(self.config_data[attribute])()
                setattr(self, attribute, attr_obj)
        self.data = get_data(self.config_data["data_file"])
        self.optimization.data = self.data
        self.optimization.weight = self.weight
        self.heuristic.num_elements = self.data.shape[1]
        self.heuristic.optimization = self.optimization
        self.heuristic.optimization_params = self.config_data['optimization_params']
        self.heuristic.weight = self.weight
        self.heuristic.set_params(self.config_data['heuristic_params'])

    def output_results(self, format):
        pass

    def run(self):
        results = self.heuristic.run_heuristic({})
        if len(results[0]) > 1:
            path_results = results[0]
            logging.info("Path to ground truth:")
            print("Path to ground truth:")
            for path_result in path_results:
                print(path_result)
                logging.info("%s" % str(path_result))
            precomputed_results = results[1]
            logging.info("\nList of computed values:")
            print("\nList of computed values:")
            for precomputed_result in precomputed_results.items():
                print(precomputed_result)
                logging.info("%s" % str(precomputed_result))
        else:
            single_result = results[0][0]
            ground_truth = single_result[0]
            phi, b, max_log_like = single_result[1]
            logging.info("\nOptimal ground truth was %s with (phi, b, max_log_like) = (%f, %f, %f)" % (ground_truth, phi, b, max_log_like))
            print("\nOptimal ground truth was %s with (phi, b, max_log_like) = (%f, %f, %f)" % (ground_truth, phi, b, max_log_like))
