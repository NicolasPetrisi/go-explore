import os

# Set any unwished arguments (except game_name) to "-" if they are not desired.
game_name = "maze"
load_path = "-"
hours_per_level = "0.1"
levels = 1


print(os.listdir('/home/nicolas/temp/')[-1])


for i in range(levels):
    os.system("sh generic_atari_game.sh " + game_name + " " + load_path + " " + hours_per_level)