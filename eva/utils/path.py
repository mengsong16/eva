import yaml
import os
import logging
import numpy as np

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(cur_path[:cur_path.find("/eva")], "eva")
data_path = os.path.join(root_path, "data") 
experiments_path = os.path.join(root_path, "experiments")
config_path = os.path.join(root_path, "configs")