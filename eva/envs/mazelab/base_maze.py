from abc import ABC
from abc import abstractmethod

from collections import namedtuple
import numpy as np

from .object import Object


class BaseMaze(ABC):
    def __init__(self, m, **kwargs):
        m = np.array(m)
        self.SIZE = m.shape
        # real data structure of maze
        objects = self.make_objects(m)
        assert all([isinstance(obj, Object) for obj in objects])
        # set object as (key, value) pair
        #self.objects = namedtuple('Objects', map(lambda x: x.name, objects), defaults=objects)()
        # before python 3.7, there is not defaults, the following is workaround
        Objects = namedtuple('Objects', map(lambda x: x.name, objects))
        Objects.__new__.__defaults__ = objects   
        self.objects = Objects()
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    # maze size
    @property
    #@abstractmethod
    def size(self):
        #r"""Returns a pair of (height, width). """
        #pass
        return self.SIZE

    
    # convert environment matrix to a list of objects    
    @abstractmethod
    def make_objects(self, m):
        r"""Returns a list of defined objects. """
        pass
    
    # x is an empty array with the same size of maze
    # in the original array, each cell has one object
    # convert the original array to an array of certain property
    def _convert(self, x, name):
        for obj in self.objects:
            pos = np.asarray(obj.positions)
            if pos.size == 0:
                continue
            #print(pos)
            #print(pos.size)
            x[pos[:, 0], pos[:, 1]] = getattr(obj, name, None)
        
        return x

    # only consider background, no agent, goal ...    
    def _convert_background(self, x, name):
        for obj in self.objects:
            pos = np.asarray(obj.positions)
            if pos.size == 0:
                continue
            if obj.name == 'goal' or obj.name == 'agent':
                continue 
            
            x[pos[:, 0], pos[:, 1]] = getattr(obj, name, None)
        
        return x    
    
    def to_name(self):
        x = np.empty(self.size, dtype=object)
        return self._convert(x, 'name')
    
    def to_value(self):
        x = np.empty(self.size, dtype=int)
        return self._convert(x, 'value')
    
    def to_rgb(self):
        x = np.empty((*self.size, 3), dtype=np.uint8)
        return self._convert(x, 'rgb')

    def to_background_rgb(self):
        x = np.empty((*self.size, 3), dtype=np.uint8)
        return self._convert_background(x, 'rgb')    
    
    def to_impassable(self):
        x = np.empty(self.size, dtype=bool)
        return self._convert(x, 'impassable')
    
    def __repr__(self):
        return f'{self.__class__.__name__}{self.size}'
