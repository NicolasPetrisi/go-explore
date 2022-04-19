# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
import sys
from collections import deque, defaultdict
from typing import Any, Dict, Set, List, Optional, Tuple
from goexplore_py.cell_representations import CellRepresentationBase
import goexplore_py.data_classes as data_classes
from goexplore_py.trajectory_manager import CellTrajectoryManager
from goexplore_py.utils import get_neighbours
import horovod.tensorflow as hvd
import goexplore_py.globals as global_const

from goexplore_py.cell_mapping import CellMapping

class StochasticArchive:
    def __init__(self, optimize_score, cell_selector, cell_trajectory_manager, max_failed, reset_on_update):
        # Archive parameters
        self.optimize_score: bool = optimize_score
        self.max_failed: int = max_failed
        self.failed_threshold: float = 0.9
        self.reset_on_update: bool = reset_on_update

        self.max_cell_size = 10

        # Core data
        # This information is required to reset the state of the archive
        self.archive: Dict[Any, data_classes.CellInfoStochastic] = dict()
        self.cell_trajectory_manager: CellTrajectoryManager = cell_trajectory_manager
        self.cell_id_to_key_dict: Dict[int, Any] = {-1: None}
        self.cells_reached_dict: Dict[Any, deque] = dict()
        
        # Micro to macro cell mapping
        self.cell_map: CellMapping = CellMapping()

        # Convenience data
        # This information is necessary to make the archive run properly, but it can be calculated from the core data
        self.cell_key_to_id_dict: Dict[Any, int] = {None: -1}
        self.local_cell_counter: int = 0
        self.cell_selector = cell_selector
        self.max_score: float = -float('inf')
        self.frames: int = 0
        self.frame_skip: int = 1

        # Temporary data
        # This information is reset every iteration, so it does not need to be restored
        self.updated_cells: Set[Any] = set()
        self.updated_info = defaultdict(data_classes.CellInfoStochastic)
        self.new_cells: Dict[Any, int] = dict()

    def add_to_cell_map(self, cell_key):
        self.cell_map.add_cell(cell_key)

    def get_state(self, hampu_cells_save=False):
        """Get the current state of the archive.

        Args:
            hampu_cells_save (bool, optional): If the save should merge the cells together as Hampu Cells before saving the archive state. Defaults to False.

        Returns:
            Dict[string, Any]: The current state of the archive as a dictionary.
        """
        cell_key_set = set(self.cell_id_to_key_dict.values())
        for key in self.archive:
            assert key in self.cell_key_to_id_dict, 'key:' + str(key) + ' has no recorded id!'
            assert key in cell_key_set, 'key:' + str(key) + ' has no inverse id!'        


        save_trajectories = None

        # FN, NOTE: The resulting cells we end up with when merging together individual cells are refered to as "Hampu Cells".
        if hampu_cells_save:
            for key in self.archive.keys():
                self.cell_map.add_cell(key)

            # FN, Find which cell has what neighbour to correctly merge them together as Hampu Cells in the later steps.
            # A cell neighbour is defined from the trajectories. If a cell comes after another in a trajectory, they are
            # assumed to be neighbours.
            cell_neighbours: Dict[CellRepresentationBase, Set[CellRepresentationBase]]
            cell_neighbours = get_neighbours(self.cell_trajectory_manager.cell_trajectories, self.cell_id_to_key_dict)
            # cell_neighbours: Dict[CellRepresentationBase, Set[CellRepresentationBase]] = dict()
            # for traj in self.cell_trajectory_manager.cell_trajectories.values():
            #     prev_cell_key = None

            #     for cell_id in traj.cell_ids:
            #         cell_key = self.cell_id_to_key_dict[cell_id]

            #         if cell_key not in cell_neighbours:
            #             cell_neighbours[cell_key] = set()

            #         if prev_cell_key is None:
            #             prev_cell_key = cell_key
            #             continue

            #         # FN, Define the two cells to be each other's neighbours.
            #         cell_neighbours[cell_key].add(prev_cell_key)
            #         cell_neighbours[prev_cell_key].add(cell_key)

            #         prev_cell_key = cell_key

            # FN, Create so called "Hampu Cells". Merging together neighbouring cells with each other to create larger cells.
            mapped_cells: set[CellRepresentationBase] = set()
            for cell_key in cell_neighbours.keys():

                #FN, If this is already mapped to a cell this run, don't map others to it, it may cause chain mapping.
                if cell_key in mapped_cells or self.archive[self.cell_map[cell_key]].score > 0:
                    continue

                # FN, Sort the neighbouring cells according to their cell size, making Hampu Cells prioritizing merging with smaller neighbours before the larger.
                neighbours = list(cell_neighbours[cell_key])
                neighbours.sort(key=self.cell_map.get_cell_size)

                for neighbour_cell_key in neighbours:

                    if self.cell_map.get_cell_size(cell_key) + self.cell_map.get_cell_size(neighbour_cell_key) <= self.max_cell_size and \
                        not self.archive[self.cell_map[neighbour_cell_key]].score > 0:
                        
                        #FN, Only add cells that have not been mapped to another cell yet.
                        if not neighbour_cell_key in mapped_cells:
                            self.cell_map[neighbour_cell_key] = self.cell_map[cell_key]
                            mapped_cells.add(neighbour_cell_key)
                            mapped_cells.add(cell_key)

            # FN, Update the cell_id mapping according to the mapping already done in cell_map.
            cell_id_map = dict()
            items = list(self.cell_id_to_key_dict.items())
            for cell_id, cell_key in items:
                if cell_id != -1:
                    mapped_cell_id = self.cell_key_to_id_dict[self.cell_map[cell_key]]
                    cell_id_map[cell_id] = mapped_cell_id

                    if cell_id != mapped_cell_id:
                        self.cell_id_to_key_dict.pop(cell_id)
            
            #FN, Extract the cell info of all cells that have survived the creation of the Hampu Cells (those that were not removed).
            cell_infos = list()
            for key in self.cell_map.values():
                if self.archive[key] not in cell_infos:
                    cell_infos.append(self.archive[key])

            #FN, Modify the remaining objects which we save as the archive.
            save_trajectories = self.cell_trajectory_manager.update_trajectories(cell_id_map, cell_infos)
            keys = list(self.archive.keys())
            for key in keys:
                if key != self.cell_map[key]:
                    info = self.archive.pop(key)
                    self.cells_reached_dict.pop(key, None)
                    self.archive[self.cell_map[key]].add(info)



        #FN, The final state of the archive to return.
        state = {'archive': self.archive,
                 'trajectory_manager_state': self.cell_trajectory_manager.get_state(save_trajectories),
                 'cell_id_to_key_dict': self.cell_id_to_key_dict,
                 'cells_reached_dict': self.cells_reached_dict,
                 'cell_mapping': self.cell_map.get_mapping(),
                 }
        return state

    def set_state(self, state):
        # Directly set attributes
        self.archive = state['archive']
        self.cell_trajectory_manager.set_state(state['trajectory_manager_state'])
        self.cell_id_to_key_dict = state['cell_id_to_key_dict']
        self.cells_reached_dict = state['cells_reached_dict']
        self.cell_map = CellMapping()
        self.cell_map.load_cell_mapping(state['cell_mapping'])

        # Derived attributes
        self.local_cell_counter = 0
        my_min_id = hvd.rank() * (sys.maxsize // hvd.size())
        my_max_id = (hvd.rank() + 1) * (sys.maxsize // hvd.size())
        for cell_id, cell_key in self.cell_id_to_key_dict.items():
            if my_min_id <= cell_id < my_max_id:
                local_cell_id = cell_id - hvd.rank() * (sys.maxsize // hvd.size())
                if local_cell_id > self.local_cell_counter:
                    self.local_cell_counter = local_cell_id
            if cell_id not in self.cell_key_to_id_dict:
                self.cell_key_to_id_dict[cell_key] = cell_id
            elif my_min_id <= cell_id < my_max_id:
                self.cell_key_to_id_dict[cell_key] = cell_id
            if cell_key is not None:
                self.cell_selector.cell_update(cell_key)
                if self.archive[cell_key].score > self.max_score:
                    self.max_score = self.archive[cell_key].score

                # This ensures that all cell data is copied to child processes
                self.updated_cells.add(cell_key)
                self.updated_info[cell_key] = self.archive[cell_key]
        self.local_cell_counter += 1

        for key in self.cell_key_to_id_dict:
            if key is not None:
                assert key in self.archive, 'key:' + str(key) + ' not in archive!'

        for key in self.archive:
            assert key in self.cell_key_to_id_dict, 'key:' + str(key) + ' has no recorded id!'

    def get_name(self):
        raise NotImplementedError('get_name needs to be implemented in archive!')

    def _get_cell(self, elem):
        return elem.cells[self.get_name()]

    def get_cell_from_env(self, env):
        raise NotImplementedError('get_cell_from_env needs to be implemented in archive!')

    def get_cells(self, env):
        cells = {self.get_name(): self.get_cell_from_env(env)}
        return cells

    def get_archive(self, name):
        assert name == self.get_name()
        return self

    def clear_cache(self, _active=True):
        pass

    def get_new_cell_id(self):
        result = self.local_cell_counter + hvd.rank() * (sys.maxsize // hvd.size())
        self.local_cell_counter += 1
        return result

    # Synchronization methods:
    def sync_info(self, info_received):
        for info in info_received:
            cell_trajectory_info, archive_info = info
            self.cell_trajectory_manager.sync_traj_start(cell_trajectory_info)
            self.sync_cells(archive_info)
            self.cell_trajectory_manager.sync_traj_end()

        for cell_key in self.archive:
            self.archive[cell_key].nb_sub_goal_failed = max(min(self.archive[cell_key].nb_sub_goal_failed,
                                                                self.max_failed), 0)
            if self.archive[cell_key].nb_sub_goal_failed > self.max_failed * self.failed_threshold:
                self.archive[cell_key].nb_failures_above_thresh += 1
            else:
                self.archive[cell_key].nb_failures_above_thresh = 0

            if self.archive[cell_key].should_reset:
                self.archive[cell_key].nb_chosen = 0
                self.archive[cell_key].nb_reached = 0
                self.archive[cell_key].nb_actions_taken_in_cell = 0
                self.archive[cell_key].nb_seen = 0
                self.archive[cell_key].nb_reset += 1

    def get_info_to_sync(self) -> Tuple[Dict[int, Any], Dict[Any, data_classes.CellInfoStochastic],
                                        Dict[Any, Any], Dict[Any, int], Dict[CellRepresentationBase, CellRepresentationBase]]:
        updated_cell_id_to_key_dict = {}
        updated_cell_info = {}
        for cell in self.updated_cells:
            updated_cell_id_to_key_dict[self.cell_key_to_id_dict[cell]] = cell
            updated_cell_info[cell] = self.archive[cell]
        info_to_sync = (updated_cell_id_to_key_dict, updated_cell_info, self.updated_info, self.new_cells, self.cell_map)
        return info_to_sync

    def sync_cells(self, info_to_sync: Tuple[Dict[int, Any], Dict[Any, data_classes.CellInfoStochastic], Any, Any, Dict[CellRepresentationBase, CellRepresentationBase]]):
        updated_cell_id_to_key_dict, updated_cell_info, updated_info, new_cells, cell_map = info_to_sync
        self.cell_map = cell_map
        self.cell_id_to_key_dict.update(updated_cell_id_to_key_dict)
        for cell_id, cell_key in updated_cell_id_to_key_dict.items():
            if cell_key not in self.cell_key_to_id_dict:
                self.cell_key_to_id_dict[cell_key] = cell_id
                self.cell_id_to_key_dict[cell_id] = cell_key

        for cell_key, cell_info in updated_cell_info.items():
            if cell_key in new_cells:
                if cell_key in self.archive:
                    current_frame = self.archive[cell_key].frame
                    if cell_info.frame < current_frame:
                        old_traj_id = self.archive[cell_key].first_cell_traj_id
                        old_exp_strat = self.archive[cell_key].ret_discovered
                        self.archive[cell_key].frame = cell_info.frame
                        self.archive[cell_key].first_cell_traj_id = cell_info.first_cell_traj_id
                        self.archive[cell_key].ret_discovered = cell_info.ret_discovered
                        if old_exp_strat == global_const.EXP_STRAT_NONE:
                            self.cell_trajectory_manager.cell_trajectories[old_traj_id].ret_new_cells -= 1
                        else:
                            self.cell_trajectory_manager.cell_trajectories[old_traj_id].exp_new_cells -= 1
                    elif current_frame < cell_info.frame:
                        traj_id = cell_info.first_cell_traj_id
                        if new_cells[cell_key] == global_const.EXP_STRAT_NONE:
                            self.cell_trajectory_manager.cell_trajectories[traj_id].ret_new_cells -= 1
                        else:
                            self.cell_trajectory_manager.cell_trajectories[traj_id].exp_new_cells -= 1
                    else:
                        raise RuntimeError('Frames should never be equal!')

            if self.should_accept_cell(cell_key, cell_info.score, cell_info.trajectory_len, cell_info.cell_traj_id):
                self.update_cell(cell_key, cell_info)
                self.cell_selector.cell_update(cell_key)

        for cell_key, cell_info in updated_info.items():
            self.update_cell_info(cell_key, cell_info)

    def update_cell_info(self, cell_key, cell_info):
        self.archive[cell_key].nb_chosen += cell_info.nb_chosen
        self.archive[cell_key].nb_reached += cell_info.nb_reached
        self.archive[cell_key].nb_actions_taken_in_cell += cell_info.nb_actions_taken_in_cell
        self.archive[cell_key].nb_sub_goal_failed += cell_info.nb_sub_goal_failed
        self.archive[cell_key].nb_failures_above_thresh += cell_info.nb_failures_above_thresh
        self.archive[cell_key].nb_seen += cell_info.nb_seen
        self.archive[cell_key].should_reset = self.archive[cell_key].should_reset or cell_info.should_reset

    def update_cell(self, cell_key, cell_info):
        if cell_key not in self.archive:
            self.add_cell(cell_key)
            self.archive[cell_key].ret_discovered = cell_info.ret_discovered
            self.archive[cell_key].frame = cell_info.frame
            self.archive[cell_key].first_cell_traj_id = cell_info.first_cell_traj_id
            self.archive[cell_key].traj_disc = cell_info.traj_disc
            self.archive[cell_key].total_traj_length = cell_info.total_traj_length
        elif self.archive[cell_key].cell_traj_id != -1:
            self.cell_trajectory_manager.decrement_reference(self.archive[cell_key].cell_traj_id)
        self.archive[cell_key].trajectory_len = cell_info.trajectory_len
        self.archive[cell_key].score = cell_info.score
        self.archive[cell_key].cell_traj_id = cell_info.cell_traj_id
        self.archive[cell_key].cell_traj_end = cell_info.cell_traj_end
        self.archive[cell_key].should_reset = cell_info.should_reset

        if cell_info.cell_traj_id != -1:
            self.cell_trajectory_manager.increment_reference(cell_info.cell_traj_id)
        if cell_info.score > self.max_score:
            self.max_score = cell_info.score

    def update(self, mb_data, return_goals_chosen, return_goals_reached, sub_goals, inc_ents):
        self.update_archive(mb_data)
        self.update_goal_info(return_goals_chosen, return_goals_reached, sub_goals, inc_ents)

    def clear_info_to_sync(self):
        self.updated_cells = set()
        self.updated_info = defaultdict(data_classes.CellInfoStochastic)
        self.new_cells = dict()
        self.cell_trajectory_manager.clear_info_to_sync()

    def update_archive(self, mb_data):
        current_trajectory_id = None
        current_cell_key = None
        current_cell_id = None
        self.clear_info_to_sync()
        self.cell_trajectory_manager.start_update()

        for element in mb_data:
            assert len(element) == len(mb_data[0]), 'All trajectory information should have the same length!'

        # Because observations and actions are one element longer than, for example, rewards, this effectively truncates
        # the last action and the last reward from the trajectory.
        for (cell_key, game_reward, trajectory_id, done, ob, goal, action, reward, sil, exp_strat, traj_index,
             traj_len) in zip(*mb_data):
            if sil:
                continue

            if trajectory_id != current_trajectory_id:  # <- This is an optimization to reduce the number of lookups
                current_trajectory_id = trajectory_id
                self.cell_trajectory_manager.switch_trajectory(trajectory_id)
                current_cell_id = self.cell_trajectory_manager.get_current_cell_id()
                current_cell_key = self.cell_id_to_key_dict[current_cell_id]
            if cell_key != current_cell_key:  # <- This works if the current trajectory is represented as cells
                # We have to give IDs to newly discovered cells, even if we do not want to add them to our archive,
                # otherwise we can not properly keep track of our trajectories.
                # This also means the cell needs to be synchronized and, with the code as written, it needs to be added
                # to the archive. In other words, freezing the archive is currently not supported.
                if cell_key not in self.cell_key_to_id_dict:
                    new_id = self.get_new_cell_id()
                    self.cell_key_to_id_dict[cell_key] = new_id
                    self.cell_id_to_key_dict[new_id] = cell_key
                current_cell_id = self.cell_key_to_id_dict[cell_key]
                # We have visited this cell, so its weight (or that of any of its neighbors) may have changed
                self.cell_selector.cell_update(cell_key)
            current_cell_key = cell_key

            new_cell = current_cell_key not in self.archive
            self.cell_trajectory_manager.update_trajectory(current_cell_id, game_reward, ob, goal, action, reward,
                                                           current_cell_key, exp_strat, new_cell)
            length = self.cell_trajectory_manager.current_length()
            score = self.cell_trajectory_manager.current_score()

            traj_id = self.cell_trajectory_manager.cell_trajectory_id
            traj_length = self.cell_trajectory_manager.get_current_trajectory_length()
            if self.should_accept_cell(current_cell_key, score, length, traj_id):
                if current_cell_key in self.archive:
                    should_reset = self.reset_on_update
                    ret_discovered = self.archive[current_cell_key].ret_discovered
                else:
                    ret_discovered = exp_strat
                    should_reset = False
                cell_info = data_classes.CellInfoStochastic(score,
                                                            length,
                                                            traj_id,
                                                            traj_length,
                                                            ret_discovered,
                                                            self.frames,
                                                            traj_id,
                                                            traj_index,
                                                            traj_len,
                                                            should_reset)
                self.update_cell(current_cell_key, cell_info)
                self.updated_cells.add(current_cell_key)
                if new_cell:
                    self.new_cells[current_cell_key] = ret_discovered

            cell_info = self.archive[current_cell_key]
            u_cell_info = self.updated_info[current_cell_key]
            cell_info.nb_actions_taken_in_cell += 1
            u_cell_info.nb_actions_taken_in_cell += 1
            u_cell_info.should_reset = u_cell_info.should_reset or cell_info.should_reset
            if not self.cell_trajectory_manager.already_seen(current_cell_key):
                cell_info.nb_seen += 1
                u_cell_info.nb_seen += 1
                cell_info.nb_sub_goal_failed -= 1
                u_cell_info.nb_sub_goal_failed -= 1
                self.cell_trajectory_manager.seen(current_cell_key)

            if done:
                # Reduces the reference count of the trajectory, allowing it to be deleted
                self.cell_trajectory_manager.end_trajectory(self.frames)
            self.frames += self.frame_skip

        self.update_neighbours()
        
        self.cell_trajectory_manager.finish_update()


    def update_neighbours(self):
        print("Updating...")
        cleared_cells: set() = set()
        for traj in self.cell_trajectory_manager.cell_trajectories.values():            
            prev_cell_id = None
            for cell_id in traj.cell_ids:
                cell_key = self.cell_id_to_key_dict[cell_id]
                cell: data_classes.CellInfoStochastic = self.archive[cell_key]

                if cell_key not in cleared_cells:
                    cell.neighbours.clear()
                    cleared_cells.add(cell_key)

                if prev_cell_id is not None:
                    prev_cell_key = self.cell_id_to_key_dict[prev_cell_id]
                    prev_cell: data_classes.CellInfoStochastic = self.archive[prev_cell_key]

                    # NOTE FN, this assumes that it is possible to go both ways between cells. Both from A -> B and B -> A.
                    cell.neighbours.add(prev_cell_key)
                    prev_cell.neighbours.add(cell_key)

                prev_cell_id = cell_id

        print("Update complete...")


    def otf_trajectory(self, from_cell: CellRepresentationBase, to_cell: CellRepresentationBase, max_depth: int):
        """On The Fly trajectory creates trajectories on the fly using the neighbours of cells.

        Args:
            from_cell (CellRepresentationBase): The origin cell.
            to_cell (CellRepresentationBase): The destination cell.
            max_depth (int): For how far the search is allowed to go.

        Returns:
            list (CellRepresentationBase): The path between the two given cells, None if no path could be found or to_cell is None.
        """
        
        if to_cell is None or from_cell == to_cell:
            return None
        
        queue = list()
        visited = set()

        queue.append([from_cell])
        visited.add(from_cell)

        while queue:
            current_traj = queue.pop(0)

            current_cell = current_traj[-1]

            print("cur traj:", current_traj)
            print("current_CELLASD:", current_cell)
            
            for neighbour in self.archive[current_cell].neighbours:
                if neighbour not in visited:
                    tmp_list = list(current_traj)
                    tmp_list.append(neighbour)

                    if neighbour == to_cell:
                        return tmp_list

                    print("I should append now??")
                    if len(tmp_list) + 1 < max_depth:
                        print("APPENDING")
                        queue.append(tmp_list)

                    visited.add(neighbour)
            
        # FN, We will reach here only if we weren't able to find the goal cell.
        print("NO PATH FOUND")
        return None


    def update_goal_info(self, return_goals_chosen, return_goals_reached, sub_goals, inc_ents):
        for goal_key, reached, sub_goal_key, inc_ent in zip(return_goals_chosen, return_goals_reached, sub_goals,
                                                            inc_ents):
            # - update chosen and reached information
            cell_info = self.archive[goal_key]
            u_cell_info = self.updated_info[goal_key]
            cell_info.nb_chosen += 1
            u_cell_info.nb_chosen += 1
            if reached:
                cell_info.nb_reached += 1
                u_cell_info.nb_reached += 1
            else:
                sub_goal_info = self.archive[sub_goal_key]
                u_sub_goal_info = self.updated_info[sub_goal_key]
                if not inc_ent:
                    sub_goal_info.nb_sub_goal_failed += 1
                    u_sub_goal_info.nb_sub_goal_failed += 1

            if goal_key not in self.cells_reached_dict:
                self.cells_reached_dict[goal_key] = deque(maxlen=100)
            self.cells_reached_dict[goal_key].append(reached)
            self.cell_selector.cell_update(goal_key)

    def should_accept_cell(self, potential_cell_key, cur_score, full_traj_len, current_traj_id):
        if potential_cell_key not in self.archive:
            return True
        potential_cell = self.archive[potential_cell_key]
        if current_traj_id != -1:
            c_tie_breaker = self.cell_trajectory_manager.cell_trajectories[current_traj_id].tie_breaker
        else:
            c_tie_breaker = 1
        if potential_cell.cell_traj_id != -1:
            p_tie_breaker = self.cell_trajectory_manager.cell_trajectories[potential_cell.cell_traj_id].tie_breaker
        else:
            p_tie_breaker = 1
        if self.optimize_score:
            if cur_score > potential_cell.score:
                return True
            elif cur_score < potential_cell.score:
                return False
            elif full_traj_len < potential_cell.trajectory_len:
                return True
            elif full_traj_len > potential_cell.trajectory_len:
                return False
            # Ensure that all workers and processes accept the same trajectory when all other cases are equal
            elif c_tie_breaker < p_tie_breaker:
                return True
            elif c_tie_breaker > p_tie_breaker:
                return False
            elif current_traj_id < potential_cell.cell_traj_id:
                return True
            else:
                return False
        else:
            if full_traj_len < potential_cell.trajectory_len:
                return True
            elif full_traj_len > potential_cell.trajectory_len:
                return False
            # Ensure that all workers and processes accept the same trajectory when all other cases are equal
            elif c_tie_breaker < p_tie_breaker:
                return True
            elif c_tie_breaker > p_tie_breaker:
                return False
            elif current_traj_id < potential_cell.cell_traj_id:
                return True
            else:
                return False

    def get_new_cell_info(self):
        return data_classes.CellInfoStochastic()

    def add_cell(self, cell_key: Any):
        """
        Force-adds a cell to the archive.

        The new cell will get an ID (i.e. index) if it did not already have one and it will overwrite previous cell
        information if the cell was already in the archive. Because the new cell will get an ID, this function should
        not be called in any of the worker processes, as it will desynchronize the cell IDs for that process.

        @param cell_key: The cell-key of the cell to add.
        @return: None
        """
        cell = self.get_new_cell_info()
        if cell_key not in self.cell_key_to_id_dict:
            cell_id = self.get_new_cell_id()
            self.cell_id_to_key_dict[cell_id] = cell_key
            self.cell_key_to_id_dict[cell_key] = cell_id
        self.archive[cell_key] = cell


class DomainKnowledgeArchive(StochasticArchive):
    def __init__(self, optimize_score, selector, cell_trajectory_manager, grid_info, max_failed, reset_on_update):
        super(DomainKnowledgeArchive, self).__init__(optimize_score, selector, cell_trajectory_manager, max_failed,
                                                     reset_on_update)
        self._domain_knowledge_cell_cache = None
        self.grid_info = grid_info

    def clear_cache(self, _active=True):
        self._domain_knowledge_cell_cache = None
        super(DomainKnowledgeArchive, self).clear_cache()

    def get_name(self):
        return 'domain_knowledge_cell'

    def get_cell_from_env(self, env):
        if self._domain_knowledge_cell_cache is None:
            self._domain_knowledge_cell_cache = env.recursive_call_method('get_pos')
        return self._domain_knowledge_cell_cache


class FirstRoomOnlyArchive(DomainKnowledgeArchive):
    def should_accept_cell(self, potential_cell_key, potential_cell, cur_score, full_traj_len):
        if potential_cell_key.room != 1:
            return False
        return super(FirstRoomOnlyArchive, self).should_accept_cell(potential_cell_key,
                                                                    potential_cell,
                                                                    cur_score,
                                                                    full_traj_len)


class ArchiveCollection:
    def __init__(self):
        self.active_archive: Optional[StochasticArchive] = None
        self.archive_dict: Dict[str, StochasticArchive] = {}

    def add_archive(self, archive: StochasticArchive, name: Optional[str] = None):
        if len(self.archive_dict) == 0:
            self.active_archive = archive
        if name is None:
            name = archive.get_name()
        assert name not in self.archive_dict, 'Archive with name ' + name + ' already in collection'
        self.archive_dict[name] = archive

    def get_archive(self, name):
        return self.archive_dict[name]

    def get_cells(self, env):
        cells = {}
        for archive in self.archive_dict.values():
            cells[archive.get_name()] = archive.get_cell_from_env(env)
        return cells

    def clear_cache(self):
        for archive in self.archive_dict.values():
            active = archive == self.active_archive
            archive.clear_cache(active)
