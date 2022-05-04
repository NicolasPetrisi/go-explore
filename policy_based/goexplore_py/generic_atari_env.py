# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

import code
from tkinter.tix import CELL
from turtle import listen, pos
from xmlrpc.client import Boolean
from goexplore_py.cell_representations import CellRepresentationBase
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

        # FN, this varibale is for when we want to be able to reproduce a random sequence of starting positions in procgen, should start at least 2
        self.reset_counter = 2  
        self.env = self.make_env()
        super(MyAtari, self).__init__(self.env)
        self.env.reset()
        self.potential_cells = None

    def __getattr__(self, e):
        return getattr(self.env, e)

    def make_env(self):
        if self.pos_seed < -1:
            chosen_pos = -self.reset_counter
        else:
            chosen_pos = self.pos_seed
        return gym.make(self.name, distribution_mode=self.distribution_mode, render_mode="rgb_array" , start_level=self.org_seed, use_sequential_levels=self.use_sequential_levels, num_levels = self.num_levels, restrict_themes = self.restrict_themes, pos_seed = chosen_pos)
    
    def reset(self) -> np.ndarray:
        """reseting an enviroment to the start state

        Returns:
            np.ndarray: observation of the start frame (64,64,3) in procgen
        """
        self.reset_counter += 1
        self.env = self.make_env()
        unprocessed = self.env.reset()
        self.done = 0
        self.level_seed = self.org_seed
        self.unprocessed_state = unprocessed
        self.pos_from_unprocessed_state(self.get_face_pixels(self.get_full_res_image()))
        self.pos = self.cell_representation(env=self)


        goal = get_goal_pos(self.get_full_res_image())
        oldx = self.x
        oldy = self.y
        self.x = goal[0]
        self.y = goal[1]
        self.done = 1
        self.goal_cell = self.cell_representation(env=self)
        self.done = 0
        self.x = oldx
        self.y = oldy

        if self.potential_cells is None:
            self.potential_cells = self.get_reachable_cells()
        return self.get_full_res_image()

    def get_full_res_image(self):
        """A higher resolution image of the frame

        Returns:
            _type_: Full image for video/image, resolution (512,512,3) in procgen
        """
        return self.env.render(mode="rgb_array")



    def get_reachable_cells(self) -> typing.List[CellRepresentationBase]:
        """ function that returns every cell that's not a way in the procgen game maze. 
            To determine if a cell is a wall the rgb-value is checked for the x,y cordinates
            fo the cell and see if they match the rgb-value of walls. Note that this returns a list 
            of single cells, i.e. hampu cells are not regarded in this funcion

        Returns:
            List[CellRepresentationBase]: a list of all cells that are not walls
        """
        potential_cells = list()
        for x in range(25):
            for y in range(25):
                cell = self.cell_representation(None)
                cell.x = x
                cell.y = y
                cell.done = 0
                potential_cells.append(cell)

        def is_wall(start_x,end_x,start_y,end_y, full_image) -> bool:

            if full_image is None:
                print("Imgae is none")
                return True
            if len(full_image) == 0:
                print("empty image")
                return True
            r_g_b = list()
            for i in range(3):
                mean = np.mean(full_image[start_y:end_y , start_x:end_x, i])
                r_g_b.append(mean)

            # FN, since the walls have a pattern and the number of pixels of a cells may differ with one pixel in x and y
            # a mean in used instead of a hard-coded value to be more robust
            min_r_wall = 185
            max_r_wall = 205
            min_g_wall = 135
            max_g_wall = 155
            min_b_wall = 85
            max_b_wall = 105

            return r_g_b[0] > min_r_wall and r_g_b[0] < max_r_wall\
                and r_g_b[1] > min_g_wall and r_g_b[1] < max_g_wall\
                and r_g_b[2] > min_b_wall and r_g_b[2] < max_b_wall\


        saved_cells = list(potential_cells)
        full_res_image = self.get_full_res_image()
        for p_cell in potential_cells:
            start_x = int(p_cell.x * 20.48)
            end_x = int(start_x + 20.48)
            start_y = int(p_cell.y * 20.48)
            end_y = int(start_y + 20.48)

            if is_wall(start_x,end_x,start_y,end_y, full_res_image):
                saved_cells.remove(p_cell)
            
            # FN, assuming that the program is done in the goal cell we set the _done varaible in the cell accordingly
            elif p_cell.x == self.goal_cell.x and p_cell.y == self.goal_cell.y:
                tmp_cell = p_cell
                saved_cells.remove(p_cell)
                tmp_cell.done = 1
                saved_cells.append(tmp_cell)


        for cell in saved_cells:
            print("this cell is not a wall: ",  cell)
        return saved_cells


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

        old_pos = self.pos

        self.pos_from_unprocessed_state(self.get_face_pixels(self.get_full_res_image()))

        self.pos = self.cell_representation(env=self, came_from=old_pos)


        return self.get_full_res_image(), reward, done, lol

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