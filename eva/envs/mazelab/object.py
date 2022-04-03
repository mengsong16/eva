from dataclasses import dataclass
from dataclasses import field

# value is the number in the cell
@dataclass
class Object:
    r"""Defines an object with some of its properties. 
    
    An object can be an obstacle, free space or food etc. It can also have properties like impassable, positions.
    
    """
    name: str
    value: int
    rgb: tuple
    impassable: bool
    movable: bool
    deadly: bool

    positions: list = field(default_factory=list) # a list of (x,y) locations where this kind of objects exists
