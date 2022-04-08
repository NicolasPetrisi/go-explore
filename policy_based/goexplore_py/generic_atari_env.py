# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gym
import copy
import typing
from atari_reset.atari_reset.wrappers import MyWrapper
import numpy as np

from goexplore_py.utils import get_goal_pos, AGENT_COLOR


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

    def __init__(self, env,  name, distribution_mode="hard", cell_representation=None, level_seed=0, use_sequential_levels=False,  num_levels = 1, restrict_themes = True, pos_seed = 0):
        self.name = name
        
        self.state = [] #TODO unused?
        self.unprocessed_state = None
        self.rooms = {}
        self.pos = None
        self.cell_representation = cell_representation
        self.done = 0

        self.x = 0
        self.y = 0
        self.org_seed = level_seed # FN, This is if sequentiall levels are used to check if we are ack on the startring level
        self.level_seed = level_seed

        # FN, parameters need when running procgen
        self.distribution_mode = distribution_mode
        self.use_sequential_levels = use_sequential_levels
        self.num_levels = num_levels
        self.restrict_themes = restrict_themes
        self.pos_seed = pos_seed

        self.env = self.make_env()
        super(MyAtari, self).__init__(self.env)
        self.env.reset()

    def __getattr__(self, e):
        return getattr(self.env, e)

    def make_env(self):
        return gym.make(self.name, distribution_mode=self.distribution_mode, render_mode="rgb_array" , start_level=self.org_seed, use_sequential_levels=self.use_sequential_levels, num_levels = self.num_levels, restrict_themes = self.restrict_themes, pos_seed = self.pos_seed)
    
    def reset(self) -> np.ndarray:
        """reseting an enviroment to the start state

        Returns:
            np.ndarray: observation of the start frame (64,64,3) in procgen
        """
        self.env = self.make_env()
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
        self.level_seed = lol['level_seed']

        # FN, Assuming that the episodes terminate when reward is found and the position of the agent is at the goal when it happens
        # This is because procgen end the episodes before the agent actaully enters the goal space
        if reward > 0:
            self.done = done

        self.pos_from_unprocessed_state(self.get_face_pixels(self.get_full_res_image()))
        self.pos = self.cell_representation(self)

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
        face_pixels = [(y, x) for y, x in face_pixels]
        if len(face_pixels) == 0:
            assert self.pos is not None, 'No face pixel and no previous pos'
            return self.pos  # FN, Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        self.x = x
        self.y = y

    def get_face_pixels(self, unprocessed_state):
        """get location of pixels unique for the agent

        Args:
            unprocessed_state (_type_): observation of the enviroment

        Returns:
            set: a set of y and x postion of pixels unique for the agent 
        """
        indices = np.where(np.all(unprocessed_state == AGENT_COLOR, axis=-1))
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