import heapq
import random
import numpy as np


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

        if S.shape[0] > 1: # ignore episodes that has only one step
            # each item in the heap has form (return, (S,A,R,S_))
            item = (episode_return, episode) # -1 for desc ordering
            if len(self.buffer) < self.size:
                heapq.heappush(self.buffer, item) 
            # when buffer is full, always remove the episodes with smallest return    
            else:
                _ = heapq.heappushpop(self.buffer, item) # ignore the popped obj
    
    # get top K episodes
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

    def all_episodes(self):
        episodes = [x[1] for x in self.buffer] 
        return episodes 

class TrajectoryBuffer(object):
    """
    Unlimited trajectory buffer
    """
    def __init__(self):
        self.buffer = [] # initialized as a regular list
    
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
        
        if S.shape[0] > 1: # ignore episodes that has only one step
            self.buffer.append(episode)

    def summary(self):
        self.episode_returns = []
        self.episode_horizons = []
        for episode in self.buffer:
            (S, A, R, S_) = episode
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
        print("Min horizon: %f"%(np.min(self.episode_horizons, axis=0)))
        print("Mean horizon: %f"%(np.mean(self.episode_horizons, axis=0)))
        print("Max horizon: %f"%(np.max(self.episode_horizons, axis=0)))
        print("------------------------------------------------")
        