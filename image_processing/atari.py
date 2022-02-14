import gym
from gym.wrappers import Monitor
import numpy as np
import cv2

def to_np(array):
    arr = cv2.imencode('.png', array, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].flatten().tobytes()
    print(arr)
    print(cv2.imdecode(np.frombuffer(arr, np.uint8), 0))
    return cv2.imdecode(np.frombuffer(array, np.uint8), 0)

#env = gym.make('AlienNoFrameskip-v4')
#env = Monitor(env, './video', force = True)


env = gym.make("procgen:procgen-maze-v0", render_mode="rgb_array")    
env = Monitor(env, './video', force = True, video_callable=lambda episode_id: True)



for episode in range(5):
    observation = env.reset()
    print(observation.shape)
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info  = env.step(action)
        score += reward
    print( " score  " + str(score))
    print(to_np(observation))
env.close()