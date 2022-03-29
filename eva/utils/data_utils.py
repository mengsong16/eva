
import os
import numpy as np
import yaml
import math
import collections
import torch

# parse a config yaml file to a list
def parse_config(config):
    """
    Parse yaml config file
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )
    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data

def get_device(config):
    if torch.cuda.is_available():
        return torch.device("cuda:{}".format(int(config.get("gpu_id"))))
    else:
        return torch.device("cpu")
