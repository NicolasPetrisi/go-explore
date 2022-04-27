from importlib_metadata import DistributionFinder
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

word_dict = dict()

word_dict['it'] = 'Iteration'
word_dict['ret_suc'] = 'Return Sucess'
word_dict['ret_cum_suc'] = 'Cumulative Return Success'
word_dict['exp_suc'] = 'Exploration Success'
word_dict['opt_len'] = 'Optimal Length'
word_dict['dist_from_opt'] = 'Distance from Optimal Length'
word_dict['len_mean'] = 'Length Mean'
word_dict['frames'] = 'Frames'
word_dict['rew_mean'] = 'Reward Mean'
word_dict['score'] = 'Score'
word_dict['ep'] = 'Episode'
word_dict['cells'] = 'Cells'
word_dict['arch_suc'] = 'Archive Success'
word_dict['policy_loss'] = 'Policy Loss'
word_dict['value_loss'] = 'Value Loss'
word_dict['l2_loss'] = 'L2 Loss'
word_dict['policy_entropy'] = 'Policy Entropy'
word_dict['cells_found_ret'] = 'Cells Found when Returning'
word_dict['cells_found_rand'] = 'Cells Found when Random Exploring'
word_dict['cells_found_policy'] = 'Cells Found when Policy Exploring'



def make_sub_list(input_list,seperator):
    final = []
    for line in input_list:
        tmp = line.split(seperator)
        almost_final = []
        for word in tmp:
            almost_final.append(word.strip())
        final.append(almost_final)
    return final

def get_values(filepath, x_name, y_name):
    with open(filepath) as f:
        lines = f.readlines()
        first = make_sub_list(lines, ',')
        x_index = first[0].index(x_name)
        y_index = first[0].index(y_name)
        x_values = []
        y_values = []
        for line in first[1:]:
            y = float(line[y_index])
            if y > -0.1:
                x_values.append(int(line[x_index]))
                y_values.append(float(line[y_index]))
        return x_values,y_values

def plot_values(word_dict, x_values, y_values, x_label, y_label, title, name="plot.png"):
    plt.clf()
    plt.plot(x_values, y_values)
    if y_label in word_dict:
        label_name = word_dict[y_label]
    else:
        label_name = y_label
        print("Y-label name was not in the word dictionary! Using log file label name instead")
    plt.ylabel(label_name)

    if x_label in word_dict:
        label_name = word_dict[x_label]
    else:
        label_name = x_label
        print("X-label name was not in the word dictionary! Using log file label name instead")
    plt.xlabel(label_name)
    
    # NOTE: If you want a fixed y-axis. Use this line!
    #plt.ylim([0,1])
    
    plt.title(title)
    plt.savefig(name)

def main(word_dict):

    if not os.path.isdir('plots'):
        os.mkdir('plots')


    files = []

    for i in range(1, len(sys.argv)):
        files.append(sys.argv[i])

    x_axis_label = "frames"
    y_axis_label = "ret_suc"
    x_values = []
    y_values = []
    min_len = sys.maxsize
    for file in files:
        x, y = get_values(file, x_axis_label, y_axis_label)
        x_values.append(x)
        y_values.append(y)
        min_len = np.min([min_len, len(x)])

    for i in range(len(x_values)):
        x_values[i] = x_values[i][:min_len]
        y_values[i] = y_values[i][:min_len]

    for i in range(len(x_values)):
        plot_values(word_dict, x_values[i], y_values[i], x_axis_label, y_axis_label, files[i].split('.')[0], "plots/" + files[i].split('.')[0] + "_" + y_axis_label + "_of_" + x_axis_label + ".png")


main(word_dict)