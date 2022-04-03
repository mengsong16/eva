from abc import ABC
from abc import abstractmethod

import numpy as np
import gym
from gym.utils import seeding
from PIL import Image

from gym.envs.registration import EnvSpec

class BaseMazeEnv(gym.Env, ABC):
    # https://github.com/openai/gym/blob/master/gym/core.py

    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second' : 3}
    reward_range = (-float('inf'), float('inf'))
    spec = EnvSpec(id='maze-v0')
    
    def __init__(self):
        self.viewer = None
        #self.seed()
    
    @abstractmethod
    def step(self, action):
        pass
    
    def seed(self, seed):
        np.random.seed(seed)
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def get_image(self):
        pass
    
    # max_width is the width of the square screen
    # img is the image to be shown
    # each time render the whole image
    def render(self, mode='human', max_width=500):
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = max_width / img_width
        img = Image.fromarray(img).resize([int(ratio*img_width), int(ratio*img_height)], resample=Image.NEAREST)
        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                # do not move this to the beginning of the file, otherwise requires screen
                from gym.envs.classic_control.rendering import SimpleImageViewer
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            
            return self.viewer.isopen, img
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
