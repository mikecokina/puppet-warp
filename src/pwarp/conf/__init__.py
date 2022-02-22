import json
import os.path as op

from logging import config


def _load_conf(path):
    with open(path) as f:
        conf_dict = json.loads(f.read())
    config.dictConfig(conf_dict)


def initialize_logger():
    __config__ = op.join(op.dirname(__file__), 'logging_schemas', 'default.json')
    _load_conf(__config__)
