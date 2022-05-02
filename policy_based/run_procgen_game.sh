#!/bin/sh

# The settings below are for testing the code locally
# For the full experiment settings, change each setting to each "full experiment" value.

Game=$1

if [ $2 != '-' ];
then
    ModelLoad="--load_path $2"
else
    ModelLoad=""
fi

if [ $3 != '-' ];
then
    ArchLoad="--expl_state $3"
else
    ArchLoad=""
fi

# Max Hours
if [ $4 != '-' ];
then
    MaxTime="--mh $4"
else
    MaxTime=""
fi

# Max Computational Steps
if [ $5 != '-' ];
then
    MaxSteps="--mcs $5"
else
    MaxSteps=""
fi

# Level Seed
if [ $6 != '-' ];
then
    LevelSeed="--level_seed $6"
else
    LevelSeed=""
fi

# Position Seed
if [ $7 != '-' ];
then
    PosSeed="--pos_seed $7"
else
    PosSeed=""
fi

# Networking Test
if [ $8 = 'True' ];
then
    TestNetwork="--freeze_network --test_mode"
else
    TestNetwork=""
fi

if [ $9 = 'True' ];
then
    video_all_ep="--video_all_ep"
else
    video_all_ep=""
fi


# Full experiment: 16
NB_MPI_WORKERS=${10}
# Full experiment: 16
NB_ENVS_PER_WORKER=1

if [ ${11} != '-' ];
then
    Folder="--folder ${11}"
else
    Folder=""
fi

if [ ${12} != '-' ];
then
    TrajectoryFile="--trajectory_file ${12}"
else
    TrajectoryFile=""
fi

if [ ${13} != '-' ];
then
    Continue="--continue"
else
    Continue=""
fi
# Full experiment: different for each run
#SEED=0

# Full experiment: 200000000
CHECKPOINT=50000


#ModelLoad="--load_path /home/fredrik/temp/0600_ce76f8b4a8734e8bbb8fc957f5144d38/000001430098_model.joblib"
# The game is run with both sticky actions and noops. Also, for Montezuma's Revenge, the episode ends on death.
GAME_OPTIONS="--game generic_${Game}"

# Both trajectory reward (goal_reward_factor) are 1, except for reaching the final cell, for which the reward is 3.
# Extrinsic (game) rewards are clipped to [-2, 2]. Because most Atari games have large rewards, this usually means that extrinsic rewards are twice that of the trajectory rewards.
REWARD_OPTIONS="--game_reward_factor 1 --goal_reward_factor 3 --clip_game_reward 1 --rew_clip_range=-2,2 --final_goal_reward 3"

# Cell selection is relative to: 1 / (1 + 0.5*number_of_actions_taken_in_cell). 
CELL_SELECTION_OPTIONS="--selector_weights=max_score_cell,nb_actions_taken_in_cell,1,1,0.5 --base_weight 0"
#CELL_SELECTION_OPTIONS="--selector_weights=attr,nb_actions_taken_in_cell,1,1,0.5 --base_weight 0"
#CELL_SELECTION_OPTIONS="--selector_weights=target_cell,x,20,y,18,done,0 --base_weight 0"

# When the agent takes too long to reach the next cell, its intropy increases according to (inc_ent_fac*steps)^ent_inc_power.
# When exploring, this entropy increase starts when it takes more than expl_inc_ent_thresh (50) actions to reach the goal.
# When returning, entropy increase starts relative to the time it originally took to reach the target cell.
ENTROPY_INC_OPTIONS="--entropy_strategy dynamic_increase --inc_ent_fac 0.01 --ent_inc_power 2 --ret_inc_ent_fac 1 --expl_inc_ent_thresh 50 --expl_ent_reset=on_goal_reached --legacy_entropy 0"

# The cell representation for Montezuma's Revenge is a domain knowledge representation including level, room, number of keys, and the x, y coordinate of the agent.
# The x, y coordinate is discretized into bins of 36 by 18 pixels (note that the pixel of the x axis are doubled, so this is 18 by 18 on the orignal frame)
#CELL_REPRESENTATION_OPTIONS="--cell_representation level_room_keys_x_y --resolution=36,18"

CELL_REPRESENTATION_OPTIONS="--cell_representation generic"

# When following a trajectory, the agent is allowed to reach the goal cell, or any of the subsequent soft_traj_win_size (10) - 1 cells.
# While returning, the episode is terminated if it takes more than max_actions_to_goal (1000) to reach the current goal
# While exploring, the episode is terminated if it takes more than max_actions_to_new_cell (1000) to discover a new cell
# When the the final cell is reached, there is a random_exp_prob (0.5) chance that we explore by taking random actions, rather than by sampling from the policy.
EPISODE_OPTIONS="--trajectory_tracker sparse_soft --soft_traj_win_size 10 --random_exp_prob 0.33 --delay 0"

CHECKPOINT_OPTIONS="--checkpoint_compute ${CHECKPOINT} --clear_checkpoints all --final_goal_or_sub_goal sub_goal"
TRAINING_OPTIONS="--goal_rep raw --no_exploration_gradients --sil=none"
MISC_OPTIONS="--explorer hampu --early_stopping --log_files __main__ ${video_all_ep} ${ModelLoad} ${ArchLoad} ${MaxTime} ${MaxSteps} ${LevelSeed} ${PosSeed} ${TestNetwork} ${Folder} ${TrajectoryFile} ${Continue}"
mpirun -n ${NB_MPI_WORKERS} python3 goexplore_start.py --base_path ~/temp --nb_envs ${NB_ENVS_PER_WORKER} ${REWARD_OPTIONS} ${CELL_SELECTION_OPTIONS} ${ENTROPY_INC_OPTIONS} ${CHECKPOINT_OPTIONS} ${CELL_REPRESENTATION_OPTIONS} ${EPISODE_OPTIONS} ${GAME_OPTIONS} ${TRAINING_OPTIONS} ${MISC_OPTIONS}
