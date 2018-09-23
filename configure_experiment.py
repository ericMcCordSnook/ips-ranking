import yaml
from importlib import import_module

def yaml_load(fp):
    with open(fp, "r") as fd:
        config_data = yaml.load(fd)
    return config_data

def yaml_dump(fp, config_data):
    with open(fp, "w") as fd:
        yaml.dump(data, fd)

def get_class(name):
    module, classname = name.rsplit(".", 1)
    try:
        return getattr(import_module(module), classname)
    except AttributeError as e:
        msg = ('%s while trying to import %r from %r'
               % (e.args[0], classname, module))
        e.args = (msg,) + e.args[1:]
        raise e

def configure_experiment(fp):
    config_data = yaml_load(fp)
    experiment = get_class(config_data['experiment'])()
    #heuristic = get_class(config_data['heuristic'])()
    return experiment
