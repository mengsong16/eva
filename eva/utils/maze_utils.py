import os
import time
import torch
import gym
import subprocess as sp

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import csv
import sys
import json
from PIL import Image, ImageDraw 
import math
from gym import spaces
import numpy as np
from eva.utils.path import *

import copy
import seaborn as sns
import pickle

from eva.replay_buffer.trajectory_replay_buffer import TrajectoryBuffer
from eva.algorithms.common import get_one_episode_states
from eva.envs.common import create_env

class StateVisitationDict:
    def __init__(self):
        self._visitation_dict = {}
        self._total = 0

    # add one state
    # state is a float numpy array
    def add(self, state):
        # float array to string key
        state_str = state.tobytes()

        if state_str in self._visitation_dict: 
            self._visitation_dict[state_str] += 1
        else: 
            self._visitation_dict[state_str] = 1

        self._total += 1    

    # normalize to [0,1]
    def get_normalized_visitation_dict(self):
        normalized_visitation_dict = {}
        for state_str, count in self._visitation_dict.items():
            normalized_visitation_dict[state_str] = float(count) / float(self._total)

        return normalized_visitation_dict

class MazeVisualizer:
    def __init__(self, env):
        assert isinstance(env.observation_space, spaces.Box), "observation space needs to be a gym Box"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        
        self.env = env
        self.env_name = env.unwrapped.spec.id   

        # discretized 2d state space
        assert int(np.prod(env.observation_space.shape)) == 2, "observation space needs to be 2-dimensional"
        self.col_n = env.observation_space.high[1] - env.observation_space.low[1] + 1
        self.row_n = env.observation_space.high[0] - env.observation_space.low[0] + 1
        self.low_col = env.observation_space.low[1]
        self.low_row = env.observation_space.low[0]
        
        

    # get embedding for the discretized 2D state space
    # save feature map
    def embed_state_space(self, model, dim_reduct_type="tsne"):
        total_n = self.col_n * self.row_n
        state_map = [None] * total_n
        feature_map = [None] * total_n

        i = 0 
        for row in range(self.row_n):
            for col in range(self.col_n):
                # index to state
                state_coord = np.array([row+self.low_row, col+self.low_col])
                state_feature = model(state_coord).squeeze().numpy()
                
                feature_map[i] = state_feature
                state_map[i] = state_coord

                i += 1
        
        state_map = np.array(state_map)
        feature_map = np.array(feature_map)
        # obstacle_map: [row 1, row 2 ...]
        obstacle_map = self.env.get_obstacle()
        obstacle_map = obstacle_map.reshape((-1, 1))
        
        # save original state map and feature map after embedding
        np.save(os.path.join(figure_path, self.env_name+"-feature-map.npy"), feature_map)
        np.save(os.path.join(figure_path, self.env_name+"-state-map.npy"), state_map) 

        # plot feature map and state map
        if dim_reduct_type == "tsne":
            feature_map_embedded = TSNE(n_components=2).fit_transform(feature_map)
        elif dim_reduct_type == "pca":
            pca = PCA(n_components=2)
            feature_map_embedded = pca.fit_transform(feature_map)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # plot x,y data with c as the color vector, set the line width of the markers to 0
        ax1.scatter(state_map[:,0], state_map[:,1], c=obstacle_map.squeeze(), cmap="Set1", lw=0)
        ax2.scatter(feature_map_embedded[:,0], feature_map_embedded[:,1], c=obstacle_map.squeeze(), cmap="Set1", lw=0)
        ax1.set_title('state map')
        ax2.set_title('feature map')
        #plt.show()
        if dim_reduct_type == "tsne": 
            figure_name = self.env_name + "-feature-map-tsne.png"
        elif dim_reduct_type == "pca":
            figure_name = self.env_name + "-feature-map-pca.png"   
        
        plt.savefig(os.path.join(figure_path, figure_name))  
        print("Embedded the state space. Feature map saved at %s"%(os.path.join(figure_path, figure_name))) 

    # convert visitation dictionary to visitation map
    def visitation_dict2map(self, visitation_dict, h_w_state=True):
        visitation_map = np.zeros((self.row_n, self.col_n), dtype=int) 
        for state_str, count in visitation_dict._visitation_dict.items():
            state = np.frombuffer(state_str, dtype=np.float32)
            
            # discretize state
            state = np.around(state)
            state = state.astype(int) 
            
            # state is in form of [h,w] or [y,x]
            if h_w_state:
                # state to index
                visitation_map[state[0]-self.low_row][state[1]-self.low_col] = count
            # state is in form of [w,h] or [x,y]
            else:
                # state to index
                visitation_map[state[1]-self.low_col][state[0]-self.low_row] = count

        return visitation_map

    def render_visitation_map(self, visitation_map, save_name=None):
        # unvisited area
        mask = np.zeros_like(visitation_map, dtype=bool)
        mask[np.nonzero(visitation_map <= 0)] = True

        # never visited cells are white
        #with sns.axes_style("white#"):
        sns.set_style("ticks")
        
        ax = sns.heatmap(visitation_map, mask=mask, vmin=1, linewidths=.5, linecolor="white", square=True)
        ax.set_xticks(np.arange(visitation_map.shape[1]))
        ax.set_yticks(np.arange(visitation_map.shape[0]))

        for _, spine in ax.spines.items():
            spine.set_visible(True)

        if save_name is None:
            figure_name = self.env_name + '-visitation_map.png'
        else:   
            figure_name = save_name + '-visitation_map.png'
        plt.savefig(os.path.join(figure_path, figure_name))

        print("Visitation map saved at %s"%(os.path.join(figure_path, figure_name))) 

    def get_trajectory_dataset_visitation_map(self, dataset_filename):
        assert self.env_name in dataset_filename, "dataset filename should include the correct env id"

        visitation_dict = StateVisitationDict()

        # load trajectory dataset
        trajectory_buffer = TrajectoryBuffer()
        dataset_path = os.path.join(data_path, dataset_filename)
        assert os.path.exists(dataset_path), "Error: %s does not exist"%(dataset_path)
        
        with open(dataset_path, 'rb') as file_handler:
            trajectory_buffer = pickle.load(file_handler)

        for episode in trajectory_buffer.buffer:
            states = get_one_episode_states(episode) #.tolist()
            
            for state in states:
                visitation_dict.add(state)
        
        visitation_map = self.visitation_dict2map(visitation_dict)
        self.render_visitation_map(visitation_map, save_name = os.path.splitext(dataset_filename)[0])

if __name__ == "__main__":
    env = create_env(env_id="empty-maze-v0")
    mv = MazeVisualizer(env) 
    mv.get_trajectory_dataset_visitation_map(dataset_filename="empty-maze-v0-random.pkl")
