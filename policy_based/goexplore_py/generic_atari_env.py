
# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.


# from .basics import *
# from .import_ai import *
# from . import montezuma_env
# from .utils import imdownscale

import numpy as np
import gym
import copy
from typing import Tuple, List
import typing
from atari_reset.atari_reset.wrappers import MyWrapper
import numpy as np
import cv2
import goexplore_py.utils

from goexplore_py.utils import bytes2floatArr, get_goal_pos


def convert_state(state, target_shape, max_pix_value):
    """converts a state to a specified size in grayscale and with a specified max pixel value

    Args:
        state (_type_): observation of the enviroment
        target_shape (tuple): target shape (width, height)
        max_pix_value (int): the maximun pixel value

    Returns:
        _type_: a compressed bytearray with the specified quallities
    """
    if target_shape is None:
        return None
    import cv2
    resized_state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY),
        target_shape,
        interpolation=cv2.INTER_AREA)
    img =  ((resized_state / 255.0) * max_pix_value).astype(np.uint8)
    return cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].flatten().tobytes()

class AtariPosLevel:
    """old code for an atari enviroment, don't think it runs

    Returns:
        _type_: _description_
    """
    __slots__ = ['level', 'score', 'room', 'x', 'y', 'tuple']

    def __init__(self, level=0, score=0, room=0, x=0, y=0):
        self.level = level
        self.score = score
        self.room = room
        self.x = x
        self.y = y

        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self.level, self.score, self.room, self.x, self.y)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, AtariPosLevel):
            return False
        return self.tuple == other.tuple

    def __setstate__(self, d):
        self.level, self.score, self.room, self.x, self.y = d
        self.tuple = d

    def __repr__(self):
        return f'Level={self.level} Room={self.room} Objects={self.score} x={self.x} y={self.y}'

def clip(a, m, M):
    if a < m:
        return m
    if a > M:
        return M
    return a


class MyAtari(MyWrapper):
    TARGET_SHAPE = None
    MAX_PIX_VALUE = None
    screen_width = 64
    screen_height = 64
    x_repeat = 1
    
    @staticmethod
    def get_attr_max(name):
        if name == 'x':
            return MyAtari.screen_width
        elif name == 'y':
            return MyAtari.screen_height 
        else:
            return MyAtari.attr_max[name]

    def __init__(self, env, name, target_shape = (25,25), max_pix_value = 16 , x_repeat=1, end_on_death=False, cell_representation=None, level_seed=0):
        super(MyAtari, self).__init__(env)
        self.name = name
        self.env.reset()
        self.state = []
        self.x_repeat = x_repeat
        self.rooms = {}
        self.unprocessed_state = None
        self.end_on_death = end_on_death
        self.prev_lives = 0
       
        self.pos = None
        self.cell_representation = cell_representation
        self.done = 0

        self.image = None
        self.target_shape = target_shape
        self.max_pix_value = max_pix_value

        self.x = 0
        self.y = 0
        self.org_seed = level_seed
        self.level_seed = level_seed

    def __getattr__(self, e):
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        """reseting an enviroment to the start state

        Returns:
            np.ndarray: observation of the start frame (64,64,3) in procgen
        """
        #unprocessed, reward, done, lol = self.env.reset()
        unprocessed = self.env.reset()
        self.done = 0
        self.level_seed = self.org_seed
        self.unprocessed_state = unprocessed
        self.pos_from_unprocessed_state(self.get_face_pixels(self.get_full_res_image()))
        self.pos = self.cell_representation(self)


        goal = get_goal_pos(self.get_full_res_image())
        oldx = self.x
        oldy = self.y
        self.x = goal[0]
        self.y = goal[1]
        self.done = 1
        self.goal_cell = self.cell_representation(self)
        self.done = 0
        self.x = oldx
        self.y = oldy
        if self.x == 6 and self.y == 18 and self.done ==1:
            print("wrong from reset")
        return unprocessed

    def get_full_res_image(self):
        """A higher resolution image of the frame

        Returns:
            _type_: Full image for video/image, resolution (512,512,3) in procgen
        """
        return self.env.render(mode="rgb_array")

    def get_restore(self):
        """This method does not run, maybe an relic from robustified version?

        Returns:
            _type_: _description_
        """
        return (
            self.unwrapped.clone_state(),
            copy.copy(self.state),
            self.env._elapsed_steps
        )

    def restore(self, data):
        """This method does not run, maybe an relic from robustified version?

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        (
            full_state,
            state,
            elapsed_steps
        ) = data
        self.state = copy.copy(state)
        self.env.reset()
        self.env._elapsed_steps = elapsed_steps
        self.env.unwrapped.restore_state(full_state)
        return copy.copy(self.state)

    def step(self, action) -> typing.Tuple[np.ndarray, float, bool, dict]:
        """Perform the action on the enviroment

        Args:
            action (int): number describing the action to be taken

        Returns:
            self.unprocessed_state (np.ndarray): the observation after taking the action\n
            reward (float): reward for taking the action\n
            done (bool): if the enviroment is done after taking the action\n
            lol (dict): level inormation after taking the action
        """
        self.unprocessed_state, reward, done, lol = self.env.step(action)
        self.done = done
        self.level_seed = lol['level_seed']
        prev_level = lol['prev_level_seed']

        # FN assuming that the spisodes terminate when reward is found and the position of the agent is at the goal when it happens
        # This is because procgen end the episodes before the agent actaully enters the goal space
        if reward > 0 or done == 1:
            self.pos = self.goal_cell
            return self.unprocessed_state, reward, done, lol
        #if self.score > 0:
        #    self.score = reward

        self.pos_from_unprocessed_state(self.get_face_pixels(self.get_full_res_image()))
        self.pos = self.cell_representation(self)
        if self.x == 6 and self.y == 18 and self.done ==1:
            print("wrong from step")
        return self.unprocessed_state, reward, done, lol

    def get_pos(self):
        """Get the current pos, a GenericCellRepresentation

        Returns:
            CellRepresentation: Cell represenation of the current state, should be a GenericCellRepresentation
        """
        return self.pos
    
    def pos_from_unprocessed_state(self, face_pixels):
        """sets the x and y position of agent, aquired from an observation

        Args:
            face_pixels (_type_): pixels specific for only the agent

        Returns:
            _type_: old pos if no pixels where specified, otherwise no return
        """
        face_pixels = [(y, x) for y, x in face_pixels] #  * self.x_repeat
        if len(face_pixels) == 0:
            assert self.pos is not None, 'No face pixel and no previous pos'
            return self.pos  # Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        self.x = x
        self.y = y

    #TODO make generic or at least not this bad
    def get_face_pixels(self, unprocessed_state):
        """get location of pixels unique for the agent

        Args:
            unprocessed_state (_type_): observation of the enviroment

        Returns:
            set: a set of y and x postion of pixels unique for the agent 
        """
        COLOR = (187,203,204)
        indices = np.where(np.all(unprocessed_state == COLOR, axis=-1))
        indexes = zip(indices[0], indices[1])
        return set(indexes)

    def render_with_known(self, known_positions, resolution, show=True, filename=None, combine_val=max,
                          get_val=lambda x: x.score, minmax=None):
        """Not used, perhaps a relic from robustified

        Args:
            known_positions (_type_): _description_
            resolution (_type_): _description_
            show (bool, optional): _description_. Defaults to True.
            filename (_type_, optional): _description_. Defaults to None.
            combine_val (_type_, optional): _description_. Defaults to max.
            get_val (_type_, optional): _description_. Defaults to lambdax:x.score.
            minmax (_type_, optional): _description_. Defaults to None.
        """
        pass