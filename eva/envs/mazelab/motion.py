from collections import namedtuple

# coordinate system: [h,w]=[row, col]
#VonNeumannMotion = namedtuple('VonNeumannMotion', 
#                              ['north', 'south', 'west', 'east'], 
#                              defaults=[[-1, 0], [1, 0], [0, -1], [0, 1]])

# before python 3.7, there is not defaults, the following is workaround
VonNeumannMotion = namedtuple('VonNeumannMotion', 
                              ['north', 'south', 'west', 'east'])
VonNeumannMotion.__new__.__defaults__ = ([-1, 0], [1, 0], [0, -1], [0, 1])

#MooreMotion = namedtuple('MooreMotion', 
#                         ['north', 'south', 'west', 'east', 
#                          'northwest', 'northeast', 'southwest', 'southeast'], 
#                         defaults=[[-1, 0], [1, 0], [0, -1], [0, 1], 
#                                   [-1, -1], [-1, 1], [1, -1], [1, 1]])

MooreMotion = namedtuple('MooreMotion', 
                         ['north', 'south', 'west', 'east', 
                          'northwest', 'northeast', 'southwest', 'southeast'])
MooreMotion.__new__.__defaults__ = ([-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1])                        
