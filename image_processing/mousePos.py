import gym
from gym.wrappers import Monitor
import numpy as np
import cv2
import imageio
from PIL import Image

env = gym.make("procgen:procgen-maze-v0", render_mode="rgb_array", start_level=1344269901, num_levels = 1)    
env = Monitor(env, './video', force = True, video_callable=lambda episode_id: True)


def getMousePos(image):
    COLOR = (187,203,204)
    indices = np.where(np.all(image == COLOR, axis=-1))
    indexes = zip(indices[0], indices[1])
    setter = set(indexes)
    print(setter)

def pos_from_unprocessed_state(self, face_pixels):
    """sets the x and y position of agent, aquired from an observation

    Args:
        face_pixels (_type_): pixels specific for only the agent

    Returns:
        _type_: old pos if no pixels where specified, otherwise no return
    """
    face_pixels = [(y, x) for y, x in face_pixels] #  * self.x_repeat
    if len(face_pixels) == 0:
        print("no face pixels!!")
        return(-1,-1)
    y, x = np.mean(face_pixels, axis=0)
    return (x, y)

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

for episode in range(2):
    observation = env.reset()
    print(observation.shape)
    done = False
    score = 0
    i = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info  = env.step(action)
        facePixels(observation)
        score += reward
        imageThing(observation, f'{i}:env.step', 64)
        imageThing(env.render(mode="rgb_array"), f'{i}:env.render', 512)
        i+=1
    print( " score  " + str(score))
env.close()