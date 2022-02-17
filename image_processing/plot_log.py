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

def plot_values(x_values, y_values, name="plot.png"):
    plt.clf()
    plt.plot(x_values, y_values)
    plt.savefig(name)

def main():
    x, y = get_values(str(sys.argv[1]),"frames","score" )
    plot_values(x,y,"score_of_frames.png")
    x, y = get_values(str(sys.argv[1]),"frames","ret_suc" )
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    plot_values(x,y, 'plots/ret_suc_of_frames.png' )

main()