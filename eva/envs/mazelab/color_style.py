from dataclasses import dataclass


@dataclass
class DeepMindColor:
    obstacle = (40, 55, 71)
    free = (229, 232, 232)
    agent = (33, 97, 140)
    goal = (148, 49, 38)
    button = (102, 0, 204)
    interruption = (255, 0, 255)
    box = (0, 102, 102)
    lava = (255, 0, 0)
    water = (0, 0, 255)
