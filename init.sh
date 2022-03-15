#!/bin/bash
#Scrip to download all the requiremnets for policy-basde go explore into a conda enviroment called go-explore
#This bash scrip require that miniconda is installed
#to run do:
#chmod a+x init.sh
#./init.sh
sudo apt-get update
#
#Packages in order to run procgens enviroment, need at least fore ubuntu 20.04 on wsl2 
sudo apt-get install build-essential -y
sudo apt install mesa-common-dev libglu1-mesa-dev -y
#Package to get mpi4py to run on ubuntu 20.04 on wsl2
sudo apt install libopenmpi-dev -y
#To be able to reach conda
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
#create and activate the procgen enviroment, needed to be able to run modified procgen enviroments
cd procgen
conda env update --name go-explore --file environment.yml
conda activate go-explore
pip install -e .
python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='coinrun')" #This shuld print building....done!
cd ..
#installing the requirements for policy-based go-explore, these need to be installed in order
pip install tensorflow==1.15.2
pip install mpi4py==3.0.1
pip install gym==0.15.4
pip install horovod==0.23.0
pip install baselines@git+https://github.com/openai/baselines@ea25b9e8b234e6ee1bca43083f8f3cf974143998
pip install Pillow==9.0.1
pip install imageio==2.16.1
pip install matplotlib==3.5.1
pip install loky==3.1.0
pip install joblib==1.1.0
pip install dataclasses==0.6
pip install opencv-python==4.5.5.62
pip install cloudpickle==1.2.2
#
#To test it, do:
#conda activate go-explore
#cd policy-based
#sh generic_atari_env maze - 0.1 10000 123