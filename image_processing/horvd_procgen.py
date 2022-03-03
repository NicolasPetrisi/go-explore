import gym
from gym.wrappers import Monitor
import numpy as np
import cv2
import tensorflow as tf
import horovod.tensorflow as hvd

#horovod_and mpi4
def init_hvd(nb_envs, nb_cpu, seed_offset):
    hvd.init()
    master_seed = hvd.rank() * (nb_envs + 1) + seed_offset
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=nb_cpu, # threads
                            inter_op_parallelism_threads=nb_cpu) # independent tasks
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    return master_seed

def to_np(array):
    arr = cv2.imencode('.png', array, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].flatten().tobytes()
    print(arr)
    print(cv2.imdecode(np.frombuffer(arr, np.uint8), 0))
    return cv2.imdecode(np.frombuffer(array, np.uint8), 0)

def make_env(a,b):
    env = gym.make("procgen:procgen-maze-v0", render_mode="rgb_array")
    print(env.unwrapped.env.env.get_combos())
    env = Monitor(env, './video', force = True)
    return env

for episode in range(5):
    nb_cpu = 2
    nb_envs = 2
    m_seed = init_hvd(nb_cpu, nb_envs, 0)
    a = hvd.local_rank()
    envs = [make_env(i + nb_envs * hvd.rank(), a) for i in range(nb_envs)]
    env = envs[0]
    observation = env.reset()
    print(observation.shape)
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info  = env.step(action)
        score += reward
    print( " score  " + str(score))
    #print(to_np(observation))
env.close()