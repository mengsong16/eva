import heapq
import random
import numpy as np
from collections import OrderedDict

class PrioritizedTrajectoryReplayBuffer(object):
    """Implemented as a priority queue, where the priority value is
    set to be episode's total reward. Note that unlike usual RL buffers,
    we store entire 'trajectories' together, instead of just transitions.
    """
    def __init__(self, size):
        self.size = size
        self.buffer = [] # initialized as a regular list; use heapq functions

    # key: return
    # value: episode (S,A,R,S_)
    def __getitem__(self, key):
        return self.buffer[key]
    
    def __len__(self):
        return len(self.buffer)

    # add one episode in form of [s,a,r,s']
    # Note: s, s' are not augmented, this allows goal changes as HER
    def add_episode(self, S, A, R, S_):
        """ all inputs are numpy arrays; num_rows = timesteps
        S  : states
        A  : actions
        R  : rewards
        S_ : next states
        """
        episode = (S, A, R, S_)
        # no discounting
        episode_return = np.sum(R)
        # heapq is a min-heap

        if S.shape[0] > 1: # ignore episodes that has less than three steps
            # each item in the heap has form (return, (S,A,R,S_))
            item = (episode_return, episode) # -1 for desc ordering
            if len(self.buffer) < self.size:
                heapq.heappush(self.buffer, item) 
            # when buffer is full, always remove the episodes with smallest return    
            else:
                _ = heapq.heappushpop(self.buffer, item) # ignore the popped obj
    
    # get top K episodes [a list]
    def top_episodes(self, K):
        """ Returns K episodes with highest total episode rewards.
        Output: [(state_array, action_array, reward_array, next_state_array), ... ]
        """
        episodes = [x[1] for x in self.buffer[-K:]] # buffer has (-reward, episode)
        return episodes

    def sample_episodes(self, K):
        """ Returns random K episodes.
        Output: [(state_array, action_array, reward_array, next_state_array), ... ]
        """
        if self.__len__() < K:
            print("Error: # episodes to sample < # of episodes in the replay buffer: %d < %d"%(self.__len__(), K))
            exit()
        # no repeat sample
        sampled_items = random.choices(self.buffer, k=K)
        episodes = [x[1] for x in sampled_items] # buffer has (-reward, episode)
        return episodes

    # return a list
    def all_episodes(self):
        episodes = [x[1] for x in self.buffer] 
        return episodes 

class Episode(object):
    def __init__(self, S, A, R, S_):
        self.data = (S, A, R, S_)
        self.len = S.shape[0] + 1

    # episode = (S, A, R, S_)
    # S,A,R,S_ are float numpy arrays
    # return a float numpy array [T, state_dim]
    def get_one_episode_states(self):
        (S, _, _, S_) = self.data
        states = np.append(S, np.expand_dims(S_[-1], axis=0), axis=0)

        return states
    
    def __hash__(self):
        # episode length + the last state
        return hash(str(self.len)+np.array2string(self.data[3][-1]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        
        this_states = self.get_one_episode_states()
        other_states = other.get_one_episode_states()
        return np.array_equal(this_states, other_states)

# exclude repeated trajectories
class TrajectoryBuffer(object):
    """
    Unlimited trajectory buffer
    """
    def __init__(self):
        self.buffer = OrderedDict()
    
    def __len__(self):
        return len(self.buffer)

    # add one episode in form of [s,a,r,s']
    # s, a, r, s' are numpy arrays with the same shape[0] = n
    # episode includes n+1 states
    # s: [n, state_dim], a: [n], r: [n], s': [n, state_dim]
    def add_episode(self, S, A, R, S_):
        """ all inputs are numpy arrays; num_rows = timesteps
        S  : states
        A  : actions
        R  : rewards
        S_ : next states
        """

        episode = Episode(S, A, R, S_)

        if S.shape[0] > 1: # ignore episodes that has less than three steps
            if episode not in self.buffer:
                self.buffer[episode] = None
      
    
    # return a list of episodes
    def get_all_episodes(self):
        return list(self.buffer.keys())

    def print_all_episodes(self):
        episodes = self.get_all_episodes()
        for episode in episodes:
            states = episode.get_one_episode_states()
            print(states) 

    def summary(self):
        self.episode_returns = []
        self.episode_horizons = []
        for episode in self.buffer:
            (S, A, R, S_) = episode.data
            episode_horizon = R.shape[0]
            episode_return = np.sum(R)

            self.episode_returns.append(episode_return)
            self.episode_horizons.append(episode_horizon)

        self.episode_returns = np.array(self.episode_returns, dtype=np.float32)
        self.episode_horizons = np.array(self.episode_horizons, dtype=np.int64)

        print("-------------- Dataset summary --------------")
        print("Number of episodes: %d"%(len(self.buffer)))
        print("------------------------------------------------")
        print("Min return: %f"%(np.min(self.episode_returns, axis=0)))
        print("Mean return: %f"%(np.mean(self.episode_returns, axis=0)))
        print("Max return: %f"%(np.max(self.episode_returns, axis=0)))
        print("------------------------------------------------")
        print("Min horizon: %d"%(np.min(self.episode_horizons, axis=0)))
        print("Mean horizon: %d"%(np.mean(self.episode_horizons, axis=0)))
        print("Max horizon: %d"%(np.max(self.episode_horizons, axis=0)))
        print("------------------------------------------------")

if __name__ == "__main__": 
    buffer = TrajectoryBuffer()
    buffer.add_episode(S=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32), 
                       A=np.array([1, 2], dtype=np.float32), 
                       R=np.array([-1, -1], dtype=np.float32),
                       S_=np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))

    buffer.add_episode(S=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32), 
                       A=np.array([1, 2], dtype=np.float32), 
                       R=np.array([-1, -1], dtype=np.float32),
                       S_=np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))    

    buffer.add_episode(S=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32), 
                       A=np.array([1, 2], dtype=np.float32), 
                       R=np.array([-1, 0], dtype=np.float32),
                       S_=np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))

    buffer.add_episode(S=np.array([[1.0, 1.1], [2.0, 2.0]], dtype=np.float32), 
                       A=np.array([1, 2], dtype=np.float32), 
                       R=np.array([-1, -1], dtype=np.float32),
                       S_=np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))

    buffer.add_episode(S=np.array([[1.0, 1.0]], dtype=np.float32), 
                       A=np.array([1], dtype=np.float32), 
                       R=np.array([-1], dtype=np.float32),
                       S_=np.array([[2.0, 2.0]], dtype=np.float32))  

    buffer.add_episode(S=np.array([[1.0, 2.0], [2.0, 2.0], [4.0, 5.0]], dtype=np.float32), 
                       A=np.array([1, 2, 3], dtype=np.float32), 
                       R=np.array([-1, -1, -1], dtype=np.float32),
                       S_=np.array([[2.0, 2.0], [4.0, 5.0], [3.0, 3.0]], dtype=np.float32))                                                                      

    buffer.print_all_episodes()