
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

from goexplore_py.utils import bytes2floatArr


def convert_state(state, target_shape, max_pix_value):
    if target_shape is None:
        return None
    import cv2
    resized_state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY),
        target_shape,
        interpolation=cv2.INTER_AREA)
    img =  ((resized_state / 255.0) * max_pix_value).astype(np.uint8)
    return cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].flatten().tobytes()

class AtariPosLevel:
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

    #TODO does this affect anything?
    # def __getstate__(self):
    #     return self.__dict__

    # def __setstate__(self, ob):
    #     self.__dict__ = ob

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
    def __init__(self, env, name, target_shape = (25,25), max_pix_value = 16 , x_repeat=2, end_on_death=False, cell_representation=None, seed_lvl=0):
        super(MyAtari, self).__init__(env)
        self.name = name
        #self.unwrapped.seed() #TODO seed is ignored in procgen XD
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
        self.seed_lvl = seed_lvl

    def __getattr__(self, e):
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        unprocessed = self.env.reset()
        self.unprocessed_state = unprocessed
        self.pos_from_unprocessed_state(self.get_face_pixels(unprocessed))
        self.state = [convert_state(self.unprocessed_state, self.target_shape, self.max_pix_value)]
        self.image = bytes2floatArr(convert_state(self.unprocessed_state, self.target_shape, self.max_pix_value)) #currently unused, only x,y and done are used
        self.pos = self.cell_representation(self)

        return unprocessed

    #The full image in order to get a better picture/video
    def get_full_res_image(self):
        return self.env.render(mode="rgb_array")

    def get_restore(self):
        return (
            self.unwrapped.clone_state(),
            copy.copy(self.state),
            self.env._elapsed_steps
        )

    def restore(self, data):
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
        self.unprocessed_state, reward, done, lol = self.env.step(action)
        print("trolololol: " + str(lol))
        #self.seed_lvl = 
        self.pos_from_unprocessed_state(self.get_face_pixels(self.unprocessed_state))
        self.state.append(convert_state(self.unprocessed_state, self.target_shape, self.max_pix_value))
        self.state.pop(0)

        self.image = bytes2floatArr(convert_state(self.unprocessed_state, self.target_shape, self.max_pix_value))

        #FN, gives a GenericCellRepresentation with the values in this enviroment
        self.pos = self.cell_representation(self)

        return self.unprocessed_state, reward, done, lol

    #FN, Returns the current pos, a GenericCellrepreentation
    def get_pos(self):
        return self.pos
    
    #FN, get our current position. This is taken from the picture where face_pixels are the pixels that match our agent
    def pos_from_unprocessed_state(self, face_pixels):
        face_pixels = [(y, x) for y, x in face_pixels] #  * self.x_repeat
        if len(face_pixels) == 0:
            assert self.pos is not None, 'No face pixel and no previous pos'
            return self.pos  # Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        self.x = x
        self.y = y

    #FN, Get the mouse position from the frame by looking for it's RGB values.
    #TODO make generic or at least not this bad
    def get_face_pixels(self, unprocessed_state):
        COLOR = (187,203,204)
        indices = np.where(np.all(unprocessed_state == COLOR, axis=-1))
        indexes = zip(indices[0], indices[1])
        return set(indexes)

    #FN, don't know what this does yet
    def render_with_known(self, known_positions, resolution, show=True, filename=None, combine_val=max,
                          get_val=lambda x: x.score, minmax=None):
        pass