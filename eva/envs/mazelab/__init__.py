from eva.envs.mazelab.env import EmptyMazeEnv, UMazeEnv, FourRoomEnv, SimRoomEnv
from gym.envs.registration import register

# register envs to gym 
register(
    id='empty-maze-v0',
    entry_point='eva.envs.mazelab.env:EmptyMazeEnv',
    #max_episode_steps=2000,
)

register(
    id='umaze-v0',
    entry_point='eva.envs.mazelab.env:UMazeEnv',
    #max_episode_steps=2000,
)

register(
    id='four-room-v0',
    entry_point='eva.envs.mazelab.env:FourRoomEnv',
    #max_episode_steps=2000,
)

register(
    id='sim-room-v0',
    entry_point='eva.envs.mazelab.env:SimRoomEnv',
    #max_episode_steps=2000,
)
