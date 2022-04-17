import random
import gym
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
import numpy as np
from tqdm.notebook import tqdm
import wandb
import os
import datetime
from eva.algorithms.common import *
from eva.utils.path import *
from eva.utils.data_utils import parse_config, get_device
from eva.replay_buffer.triplet_dataset import Triplet, TripletBuffer, TripletDataset
from eva.models.metric_model import EncoderMLP
from eva.envs.common import *
import pickle

class MetricTrainer:
    def __init__(self, config_filename="metric.yaml"):
        assert config_filename is not None, "needs config file to initialize trainer"
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        self.env_id = str(self.config.get("env_id"))
        self.env = create_env(self.env_id)

        self.seed = int(self.config.get("seed"))

        self.log_to_wandb = self.config.get("log_to_wandb")

        self.device = get_device(self.config)

        # seed everything
        seed_other(self.seed)

        # set experiment name
        self.set_experiment_name()

        # init wandb
        if self.log_to_wandb:
            self.init_wandb()
    
    def set_experiment_name(self):
        self.project_name = 'metric'.lower()

        env_name = self.env_id
        
        loss_type = self.config.get("loss_type")
        self.group_name = (f'{loss_type}-{env_name}').lower()

        # experiment_name - YearMonthDay-HourMiniteSecond
        now = datetime.datetime.now()
        self.experiment_name = now.strftime("%Y%m%d-%H%M%S").lower() 

    def init_wandb(self):
        # initialize this run under project xxx
        wandb.init(
            name=self.experiment_name,
            group=self.group_name,
            project=self.project_name,
            config=self.config,
            dir=os.path.join(root_path)
        )

    def compute_loss(self, batch):
        if self.loss_type == "triplet":
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            # switch input to correct device
            anchors = batch['anchor'].to(self.device)
            positives = batch['positive'].to(self.device)
            negatives = batch['negative'].to(self.device)

            anchor_embeddings = self.model.forward(anchors)
            positive_embeddings = self.model.forward(positives)
            negative_embeddings = self.model.forward(negatives)

            output = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        else:
            print("Error: undefined loss type: %s"%(self.loss_type))
            exit()

        return output

    def get_batch_size(self, data_batch):
        if self.loss_type == "triplet":
            batch_size = data_batch['anchor'].size()[0]
        else:
            print("Error: undefined loss type: %s"%(self.loss_type))
            exit()

        return batch_size

    def save_model(self):
        folder_name = self.group_name + "-" + self.experiment_name
        folder_path = os.path.join(checkpoints_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        checkpoint_path = os.path.join(folder_path, f"ckpt_best.pth")
        # save weights of best model
        torch.save(self.model.state_dict(), checkpoint_path)

        print(f"Best model saved.")


    def train(self):
        state_dim = int(np.prod(self.env.observation_space.shape))
        self.model = EncoderMLP(input_dim=state_dim, output_dim=int(self.config.get("output_dim")), 
                hidden_dim=int(self.config.get("hidden_dim")), 
                hidden_layer=int(self.config.get("hidden_layer")))
        self.loss_type = self.config.get("loss_type")
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=float(self.config.get("learning_rate")))
    
        # load dataset
        assert self.env_id in str(self.config.get("dataset_name")), "Environment name should be consistent with dataset name"
        
        dataset_path = os.path.join(data_path, self.config.get("dataset_name"))
        assert os.path.exists(dataset_path), "Error: %s does not exist"%(dataset_path)
        
        buffer = TripletBuffer()
        with open(dataset_path, 'rb') as file_handler:
            buffer = pickle.load(file_handler)

        # create training dataset and loader
        dataset = TripletDataset(triplets=buffer.get_all_triplets())
    
        train_dataset_loader = TorchDataLoader(dataset, 
                                    batch_size=int(self.config.get("batch_size")), 
                                    shuffle=True, 
                                    num_workers=1)

        # Train model until best loss is achieved
        print('Starting training...')
        # one step per batch
        step = 0
        # per datapoint loss
        best_loss = 1e9
        log_interval = int(self.config.get("log_every_batches"))
        epochs = int(self.config.get("max_train_epochs"))

        num_batches = len(train_dataset_loader)
        num_datapoints = len(dataset)

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_current_epoch = 0
            train_num_datapoints = 0

            # train for one epoch 
            # iterate over the whole dataset once
            # iteration number: number of datapoints / batch size
            for batch_idx, data_batch in enumerate(train_dataset_loader):
                
                self.optimizer.zero_grad()
                # per datapoint loss (reduction=mean)
                loss = self.compute_loss(data_batch)

                loss.backward()

                self.optimizer.step()

                # per batch loss
                cur_batch_size = self.get_batch_size(data_batch)
                train_num_datapoints += cur_batch_size
                train_loss_current_epoch += (loss.item() * cur_batch_size)

                # step counter and log
                if batch_idx % log_interval == 0:
                    print(
                        'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, train_num_datapoints,
                            num_datapoints,
                            100. * (batch_idx+1) / num_batches,
                            loss.item()))
                if self.log_to_wandb:
                    wandb.log({"training loss": loss.item(), "epoch": epoch}, step=step)

                step += 1
                
            # per datapoint loss this epoch
            avg_loss_current_epoch = train_loss_current_epoch / num_datapoints
            print('====> Epoch: {} Average loss: {:.6f}'.format(
                epoch, avg_loss_current_epoch))

            # save best model until now
            if avg_loss_current_epoch < best_loss:
                best_loss = avg_loss_current_epoch
                self.save_model()
            
if __name__ == "__main__": 
    trainer = MetricTrainer()
    trainer.train()