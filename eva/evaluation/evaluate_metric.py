import numpy as np
import torch
import os
from eva.utils.data_utils import parse_config, get_device
from eva.utils.path import *
from eva.envs.common import *
from eva.algorithms.common import *
from eva.utils.maze_utils import MazeVisualizer
from eva.models.metric_model import EncoderMLP

class MetricEvaluator:
    def __init__(self, config_filename="metric.yaml"):
        assert config_filename is not None, "needs config file to initialize trainer"
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        self.env_id = str(self.config.get("env_id"))
        self.env = create_env(self.env_id)

        self.device = get_device(self.config)

        self.seed = int(self.config.get("seed"))
        # seed everything
        seed_other(self.seed)

        # initialize model
        state_dim = int(np.prod(self.env.observation_space.shape))
        self.model = EncoderMLP(input_dim=state_dim, output_dim=int(self.config.get("output_dim")), 
                hidden_dim=int(self.config.get("hidden_dim")), 
                hidden_layer=int(self.config.get("hidden_layer")))
    
    def load_checkpoint(self):
        folder_name = self.config.get("eval_checkpoint_folder")
        folder_path = os.path.join(checkpoints_path, folder_name)
        checkpoint_path = os.path.join(folder_path, f"ckpt_best.pth")
        
        if not os.path.exists(checkpoint_path):
            print("Error: "+checkpoint_path+" does not exists.")
            exit()
        
        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("Loaded checkpoint at: "+str(checkpoint_path))

        # load agent network to the correct device
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        print("Model state loaded.")

    def eval_maze(self):
        self.load_checkpoint()
        mv = MazeVisualizer(self.env)
        self.model.eval()
        mv.embed_state_space(self.model, 
            dim_reduct_type=self.config.get("dim_reduct_type"), 
            color_coding=self.config.get("color_coding"))  
        print("Done")

if __name__ == "__main__": 
    evaluator = MetricEvaluator()
    evaluator.eval_maze()