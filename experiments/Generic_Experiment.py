from rankobjects import *
from optimizations import *
from heuristics import *
from utils import *
from utils.miscutils import get_data
from importlib import import_module

class Generic_Experiment:
    def __init__(self, config_data):
        print("New object created: Generic_Experiment")
        print(config_data)
        self.config_data = config_data
        self.logger = Logger("output/out1.log", )
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
        print("Experiment configuring!")
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
        results = self.heuristic.run_heuristic()
        print("results: \n", results)
        print("Generic_Experiment ran!")
