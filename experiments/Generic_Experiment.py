from rankobjects import *
from algorithms import *
from heuristics import *
from utils import *
from importlib import import_module

class Generic_Experiment:
    def __init__(self, config_data):
        print("New object created: Generic_Experiment")
        print(config_data)
        self.config_data = config_data
        self.logger = Logger("output/out1.log", )
        self.algorithm = None
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
        attributes = ["algorithm", "heuristic", "weight"]
        for attribute in attributes:
            if attribute in self.config_data:
                attr_obj = self.get_class(self.config_data[attribute])()
                setattr(self, attribute, attr_obj)


    # THESE FUNCTIONS MAY NOT BELONG HERE BUT THEY NEED TO BE DONE NONETHELESS

    def create_weight_obj(self):
        pass

    def read_data_file(self):
        pass

    def generate_simulated_data(self):
        pass

    def output_results(self, format):
        pass

    def run(self):
        # do stuff
        print("Generic_Experiment ran!")
