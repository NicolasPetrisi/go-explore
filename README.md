# Go-Explore

This is the code used in ["First return, then explore" Adapted and Evaluated for Dynamic Environments](link_to_paper). Code given in the [First return, then Explore](https://arxiv.org/abs/2004.12919) paper can be found at: https://github.com/uber-research/go-explore 

The code for Go-Explore with a policy-based exploration phase is located in the `policy_based` subdirectory. The modied procgen code can be found in the `procgen` subdirectory.

## Requirements

To be able to run the code conda is required. The only tested conda version is Miniconda3 so no guarante that any other conda works. A guide to install Miniconda3 can be found at (https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)


Ubuntu 22.04, 20.04 or WSL2 20.04, 18.04 is required for the code to work. Using Ubuntu 18.04 causes the program to deadlock when exiting because of multiprocessing in python.
The program deadlocks when trying to exit the program because of suspected orphan threads left alive in procgen. 

Ubuntu 22.04, 20.04 and WSL2 20.04, 18.04 all appear to clean up the orphans which enables the program to exit as intended, however Ubuntu 18.04 instead deadlocks in this scenario.
Any other versions than those listed have not been tested.


To try out the code do the following:

```
git clone https://github.com/NicolasPetrisi/go-explore
cd go-explore
chmod a+x init.sh
./init.sh
conda activate go-explore
cd policy-based
python program_runner.py
```
The results will be in the ~/temp folder. In this folder all runs will be saved where plots, videos, model, archive and a log file for the run can be found.

To change the settings of the run modify the program_runner.py file perferably. Some settings can only be changed in the run_procgen_game.sh file but do this only if you are certain on what your doing.

##Notes

If a run is either interupted using CTRL-C or through a crash there is a high risk for zombie processes to stay alive. Use 'ps -a' and check for any python processes still running. Given that there shouldn't be any for any other reason, kill these zombies using 'killall python3'.
