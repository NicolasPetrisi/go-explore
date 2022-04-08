# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from typing import List, Any, Type

class CellRepresentationBase:
    __slots__ = []
    supported_games = ()

    @staticmethod
    def make(env=None) -> Any:
        raise NotImplementedError('Cell representation needs to implement make')

    @staticmethod
    def get_array_length() -> int:
        raise NotImplementedError('Cell representation needs to implement get_array_length')

    @staticmethod
    def get_attributes() -> List[str]:
        raise NotImplementedError('Cell representation needs to implement get_attributes')

    @staticmethod
    def get_attr_max(name) -> int:
        raise NotImplementedError('Cell representation needs to implement get_attr_max')

    def as_array(self) -> np.ndarray:
        raise NotImplementedError('Cell representation needs to implement as_array')



class Generic(CellRepresentationBase):
    """Class to represent Cell used by Fredrik and Nicolas.
       Cells are representated by the x and y position and the done boolean
    """
    __slots__ = ['_x', '_y', '_done', 'tuple']   
    attributes = ('x', 'y', 'done') 
    array_length = 3 #NOTE If you change the number of elements in the tuple, set change this accordingly (lenght of tuple)
    supported_games = ('$generic')

    def __init__(self, atari_env=None):
        self._done = None
        self._x = None
        self._y = None
        self.tuple = None

        if atari_env is not None:
            self._done = atari_env.done
            self._x = atari_env.x
            self._y = atari_env.y
            self.set_tuple()
            

    @staticmethod
    def make(env=None) -> Any:
        """reurns an instance of Generic

        Args:
            env (_type_, optional): enviroment in the the class. Defaults to None.

        Returns:
            Generic: an instance of Generic
        """
        return Generic(env)        

    @property
    def x(self):
        """to be able to reach the x attribute nicly
            NOTE if objcted was not created from CellRepresentationFactory and using it's __call__ method 
            this can be a flot, otherwise it's allways an int

        Returns:
            int: x postion

        """
        return self._x

    @x.setter
    def x(self, value): 
        """setting the x-value

        Args:
            value (any): The value to set the x postion to
        """
        self._x = value
        self.set_tuple()


    @property
    def y(self):
        """to be able to reach the y attribute nicly
            NOTE if objcted was not created from CellRepresentationFactory and using it's __call__ method 
            this can be a flot, otherwise it's allways an int

        Returns:
            int: y postion

        """
        return self._y

    @y.setter
    def y(self, value):
        """setting the x-value

        Args:
            value (any): The value to set the y postion to
        """
        self._y = value
        self.set_tuple()


    def set_tuple(self):
        self.tuple = (self._x, self._y, self._done)


    @staticmethod
    def get_array_length() -> int:
        return Generic.array_length

    @staticmethod
    def get_attributes() -> List[str]:
        return Generic.attributes

    @staticmethod
    def get_attr_max(name) -> int:
        #FN, used in the program to get the one-hot representations, the value is not used for procgen so its just a dummy function now
        return 2 
        
    def as_array(self) -> np.ndarray:
        return np.array(self.tuple)
    
    def __getstate__(self):
        return self.tuple

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, Generic):
            return False
        return self.tuple == other.tuple
    
    def __setstate__(self, d):
        self._x, self._y, self._done = d
        self.tuple = d
        
    def __repr__(self):
        return f'x={self._x} y={self._y} done={self._done}'
    
    def __lt__(self, other):
        if not isinstance(other, Generic):
            return False
        if other._x == self._x:
            return other._y < self._y
        else:
            return other._x < self._x


class CellRepresentationFactory:
    def __init__(self, cell_rep_class: Type[CellRepresentationBase]):
        self.cell_rep_class: Type[CellRepresentationBase] = cell_rep_class
        self.array_length: int = self.cell_rep_class.get_array_length()
        self.grid_resolution = None
        self.grid_res_dict = None
        self.max_values = None
        self.archive = None

    def set_archive(self, archive):
        self.archive = archive

    def __call__(self, env=None):
        """function that is called when a CellRepresentation is called as a function i.e. as cell_representation(self)
           It calles make which calls the init function in the CellRepresentation and then sets atrributes acording to the grid resolution 

        Args:
            env (_type_, optional): enviroment. Defaults to None.

        Returns:
            CellRepresentation: a new cell representation  
        """
        cell_representation = self.cell_rep_class.make(env)

        if env is None:
            return cell_representation
        
        for dimension in self.grid_resolution:
            if dimension.div != 1:
                value = getattr(cell_representation, dimension.attr)
                value = (int(value / dimension.div))
                setattr(cell_representation, dimension.attr, value)
        
        if cell_representation in self.archive.cell_map:
            cell_representation = self.archive.cell_map[cell_representation]
        else:
           self.archive.add_to_cell_map(cell_representation)
        
        return cell_representation

    def set_grid_resolution(self, grid_resolution):
        self.grid_resolution = grid_resolution
        self.grid_res_dict = {}
        self.max_values = []
        for dimension in self.grid_resolution:
            self.grid_res_dict[dimension.attr] = dimension.div
        for attr_name in self.cell_rep_class.get_attributes():
            max_val = self.cell_rep_class.get_attr_max(attr_name)
            if attr_name in self.grid_res_dict:
                max_val, remainder = divmod(max_val, self.grid_res_dict[attr_name])
                if remainder > 0:
                    max_val += 1
            self.max_values.append(max_val)

    def get_max_values(self):
        return self.max_values

    def supported(self, game_name):
        return game_name in self.cell_rep_class.supported_games