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
        episode_reward = np.sum(R)
        # heapq is a min-heap
        if S.shape[0] > 1: # ignore episodes that has only one step
            # each item in the heap has form (return, (S,A,R,S_))
            item = (episode_reward, episode) # -1 for desc ordering
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
        sampled_items = random.choices(self.buffer, k=K)
        episodes = [x[1] for x in sampled_items] # buffer has (-reward, episode)
        return episodes

    def all_episodes(self):
        return self.buffer    
