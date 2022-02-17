# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import random
import numpy as np
import os
import glob
import hashlib
from contextlib import contextmanager
import cv2


class TimedPickle:
    def __init__(self, data, name, enabled=True):
        self.data = data
        self.name = name
        self.enabled = enabled

    def __getstate__(self):
        return time.time(), self.data, self.name, self.enabled

    def __setstate__(self, s):
        tstart, self.data, self.name, self.enabled = s
        if self.enabled:
            print(f'pickle time for {self.name} = {time.time() - tstart} seconds')


@contextmanager
def use_seed(seed):
    # Save all the states
    python_state = random.getstate()
    np_state = np.random.get_state()

    # Seed all the rngs (note: adding different values to the seeds
    # in case the same underlying RNG is used by all and in case
    # that could be a problem. Probably not necessary)
    random.seed(seed + 2)
    np.random.seed(seed + 3)

    # Yield control!
    yield

    # Reset the rng states
    random.setstate(python_state)
    np.random.set_state(np_state)


def get_code_hash():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    all_code = ''
    for f in sorted(glob.glob(cur_dir + '**/*.py', recursive=True)):
        # We assume all whitespace is irrelevant, as well as comments
        with open(f) as fh:
            for line in fh:
                line = line.partition('#')[0]
                line = line.rstrip()

                all_code += ''.join(line.split())

    code_hash = hashlib.sha256(all_code.encode('utf8')).hexdigest()

    return code_hash


def clip(value, low, high):
    return max(min(value, high), low)


def bytes2floatArr(array):
    """gives a np.array from a byte array of an image

    Args:
        array (byteArray): byte array representation of the image

    Returns:
        _type_: np.array represenatation of the image
    """
    return cv2.imdecode(np.frombuffer(array, np.uint8), 0).astype(np.float32)

def floatArr2bytes(array):
    """gives a byte array from a np.array representation of an image

    Args:
        array (np.array): np.array representation of the image

    Returns:
        byteArray: byte array representation of the image
    """
    return cv2.imencode('.png', array, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].flatten().tobytes()

import matplotlib.pyplot as plt
import sys


def make_sub_list(input_list,seperator):
    """Help method to make plots. transform a list of strings sperated by commas to \n
       a list of list of the string, splitted at the commas

    Args:
        input_list (list): list of strings, taken from log.txt
        seperator (string/char): sperator sign

    Returns:
        list: list of list of strings
    """
    final = []
    for line in input_list:
        tmp = line.split(seperator)
        almost_final = []
        for word in tmp:
            almost_final.append(word.strip())
        final.append(almost_final)
    return final

def get_values(filepath, x_name, y_name):
    """extract the x and y values from log.txt file to be used when plotting the graph

    Args:
        filepath (string): filepth to the log file
        x_name (string): name of the attribue in the log file to be used as x-value
        y_name (_type_): name of the attribue in the log file to be used as y-value

    Returns:
        x_values (list): x-values to be used in the graph
        y_values (list): y-values to be used in the graph
    """
    with open(filepath) as f:
        lines = f.readlines()
        first = make_sub_list(lines, ',')

        x_index = first[0].index(x_name)
        y_index = first[0].index(y_name)
        x_values = []
        y_values = []
        for line in first[1:]:
            if line[y_index] != 'nan':
                y = float(line[y_index])
                if y > -0.1:
                    x_values.append(int(line[x_index]))
                    y_values.append(float(line[y_index]))
        return x_values,y_values

def plot_values(x_values, y_values, name="plot.png"):
    """plots the x and y values to a graph and aves it

    Args:
        x_values (list): x-values in the graph
        y_values (list): y-values in the graph
        name (str, optional): name of the saved graph. Defaults to "plot.png".
    """
    plt.clf()
    plt.plot(x_values, y_values)
    plt.savefig(name)

def make_plot(filename, x_name, y_name):
    """makes a plot with x_name as x values and y_name as y values.\n
       The plot will be saved in temp/run_numberandhash/plots/

    Args:
        filename (string): filepath to the log file
        x_name (string): name of the atribute to be used as x value
        y_name (string): name of the atribute to be used as y value
    """
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    x, y = get_values(filename, x_name , y_name)
    plot_values(x,y,f'./plots/{y_name}_of_{x_name}.png')



