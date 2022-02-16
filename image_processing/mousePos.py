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

def facePixels(image):
    print(set(zip(*np.where(image[:, :, 2] == 204))))

def imageThing(image,name,size):
    f_out = np.zeros((size, size, 3), dtype=np.uint8) #TODO org (210, 160,3)
    f_out[:, :, 0:3] = np.cast[np.uint8](image)[:, :, :]
        #f_out = f_out.repeat(2, axis=1)
    filename =  f'{name}.png'
    im = Image.fromarray(f_out)
    im.save(filename)

for episode in range(1):
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