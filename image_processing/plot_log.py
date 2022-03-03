import matplotlib.pyplot as plt
import sys
import os

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

def plot_values(x_values, y_values, x_label, y_label, seed, name="plot.png"):
    plt.clf()
    plt.plot(x_values, y_values)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title("Level Seed = " + str(seed))
    plt.savefig(name)

def main():

    if not os.path.isdir('plots'):
        os.mkdir('plots')

    seed = 0
    x_axis_label = "frames"
    y_axis_label = "cells"
    x, y = get_values(str(sys.argv[1]), x_axis_label, y_axis_label)
    plot_values(x, y, x_axis_label, y_axis_label, seed, "plots/" + y_axis_label + "_of_" + x_axis_label + ".png")

    seed = 0
    x_axis_label = "frames"
    y_axis_label = "ret_suc"
    x, y = get_values(str(sys.argv[1]), x_axis_label, y_axis_label)
    plot_values(x, y, x_axis_label, y_axis_label, seed, "plots/" + y_axis_label + "_of_" + x_axis_label + ".png")

main()