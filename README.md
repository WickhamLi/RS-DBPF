# Regime switching differentiable bootstrap particle filter

This repository provides the code to enable reproducibility of the numerical experiments presented in the paper **[Differentiable Bootstrap Particle Filters for Regime-Switching Models](https://arxiv.org/abs/2302.10319)**. 


## Structure

- [NNs.py](https://github.com/WickhamLi/RS-DBPF/blob/master/NNs.py): neural network structures. 
- [data_generation.py](https://github.com/WickhamLi/RS-DBPF/blob/master/data_generation.py): generate synthetic datasets for the numerical experiment.  
- [main_rsdbpf.py](https://github.com/WickhamLi/RS-DBPF/blob/master/main_rsdbpf.py): train, validate, or test the model.
- [classes.py](https://github.com/WickhamLi/RS-DBPF/blob/master/classes.py): implementation of different particle filtering algorithms evaluated in the paper (MM-PF, DBPF, RS-DBPF, and RS-PF).
- [utils.py](https://github.com/WickhamLi/RS-DBPF/blob/master/utils.py): useful functions.
- [Figures.ipynb](https://github.com/WickhamLi/RS-DBPF/blob/master/Figures.ipynb): data analysis. 
- [requirements.txt](https://github.com/WickhamLi/RS-DBPF/blob/master/requirements.txt): required environment construction. 


## Prerequisites

### Install python Packages 

To install the required python packages, run the command:

```
pip install -r requirements.txt
```

### Create datasets

Run the file [data_generation.py](https://github.com/WickhamLi/RS-DBPF/blob/master/data_generation.py): 

```
python data_generation.py
``` 

to create the synthetic dataset for training, validation and testing sets.

The essential parameter is: 
- ```-dy/--dynamics``` model switching dynamics, available options: **Mark|Poly**(i.e., Markovian|Polya urn), can be appended for both two.

The generated dataset will be stored in the folder ```./datasets/```.  Some optional parameters are listed as follows:

- ```-de/--device``` choose which device to use, available options: **cpu|cuda|mps**, default: **cpu**.
- ```-nt/--timestep``` number of time steps, default: **50**.
- ```-tr/--trajectorynum``` number of created trajectories, default: **2000**.
- ```-mu/--muu``` mean of the disturbance u in the dynamica model, default: **0.**.
- ```-su/--sigmau``` standard deviation of the disturbance u in the dynamica model, default: **$0.1^{0.5}$**.
- ```-mv/--muv``` mean of the disturbance v in the measurement model, default: **0.**.
- ```-sv/--sigmav``` standard deviation of the disturbance v in the measurement model, default: **$0.1^{0.5}$**.
- ```--ts/--testsize``` percentage of test datasets over all created trajectories, default: **0.25**.


## Arguments

#### Basics 
    
- ```-de/--device``` choose which device to use, available options: **cpu|cuda|mps**, default: **cpu**.
- ```-nt/--timestep``` number of time steps, default: **50**.
- ```-nptr/--trainparticlenum``` number of particles for training, default: **200**.
- ```-npte/--testparticlenum``` number of particles for testing, default: **2000**.
- ```-mu/--muu``` mean of the disturbance u in the dynamica model, default: **0.**.
- ```-su/--sigmau``` standard deviation of the disturbance u in the dynamica model, default: **$0.1^{0.5}$**.
- ```-mv/--muv``` mean of the disturbance v in the measurement model, default: **0.**.
- ```-sv/--sigmav``` standard deviation of the disturbance v in the measurement model, default: **$0.1^{0.5}$**.
- ```-re/--resample``` resampling method, available options: **mul|sys**(i.e., multinomial|systematic), default: **mul**.
- ```-tr/--train``` whether train differentiable particle filter, default: **true**.
- ```-te/--test``` whether test model performance, default: **true**.
- ```-ep/--epochnum``` epoch number for training, default: **60**.
- ```-lr/--learningrate``` learning rate for training, default: **$5*10^{-2}$**.
- ```-ttr/--trainingtrajectorynum``` number of training trajectories, default: **1000**.
- ```-tb/--trainingbatchsize``` training batch size, default: **100**.
- ```-vb/--valbatchsize``` validation batch size, default: **500**.
- ```-teb/--testbatchsize``` testing batch size, default: **500**.

#### RS-DBPF

- ```-rsdpf/--regimeswitchingdpf``` whether operate regime switching differentiable particle filter, default: **true**.
- ```-dy/--dynamics``` model switching dynamics, available options: **Mark|Poly**(i.e., Markovian|Polya urn), can be appended for both two.
- ```-nn/--neuralnetwork``` whether use neural network, default: **true**.
- ```-pr/--proposal``` proposal type of model index sampling, available options: **Boot|Uni|Deter**(i.e., Bootstrap|Uniform|Deterministic), can be appended for all three.

#### DBPF

- ```-sidpf/--singlemodeldpf``` whether operate single model differentiable particle filter, default: **true**.
- ```-nn/--neuralnetwork``` whether use neural network, default: **true**.

#### RS-PF

- ```-rspf/--regimeswitchingpf``` whether test regime switching particle filter, default: **true**.
- ```-dy/--dynamics``` model switching dynamics, available options: **Mark|Poly**(i.e., Markovian|Polya urn), can be appended for both two.
- ```-pr/--proposal``` proposal type of model index sampling, available options: **Boot|Uni|Deter**(i.e., Bootstrap|Uniform|Deterministic), can be appended for all three.

#### MM-PF

- ```-mmpf/--multimodelpf``` whether test multi-model particle filter, default: **true**.
- ```-ga/--gamma``` hyperparameter gamma for multi-model particle filter, can be appended for all needed gamma parameters.


## Citation

If you find this code is useful for your research, please cite our paper: 
```
@article{li2023differentiable, 
  title={Differentiable bootstrap particle filters for regime-switching models},
  author={Li, Wenhan and Chen, Xiongjie and Wang, Wenwu and Elvira, V{\'\i}ctor and Li, Yunpeng},
  journal={arXiv preprint arXiv:2302.10319},
  year={2023}
}
```
