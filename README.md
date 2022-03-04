# Go-Explore

This is the code for [First return then explore](https://arxiv.org/abs/2004.12919), the new Go-explore paper. Code for the [original paper](https://arxiv.org/abs/1901.10995) can be found in this repository under the tag "v1.0" or the release "Go-Explore v1". 

The code for Go-Explore with a deterministic exploration phase followed by a robustification phase is located in the `robustified` subdirectory. The code for Go-Explore with a policy-based exploration phase is located in the `policy_based` subdirectory. The `policy_based` directory is the only one used in the master thesis found at(future link to master thesis).

## Requirements

To be able to run the code conda is required. The only tested conda version is Miniconda3 so no guarante that any other conda works. A guide to install Miniconda3 can be found at (https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

To try out the code do the following:

```
git clone https://github.com/NicolasPetrisi/go-explore
cd go-explore
chmod a+x init.sh
./init.sh
conda activate go-explore
cd policy-based
sh generic_atari_env maze - 0.1 123 # The arguments are: game to play, model to load, time to train and level-seed
```
The results will be in the ~/temp folder. Here plots about the run, videos of some episodes, model, archive and a log file will be written