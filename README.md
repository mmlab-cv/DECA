[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deca-deep-viewpoint-equivariant-human-pose/pose-estimation-on-itop-top-view)](https://paperswithcode.com/sota/pose-estimation-on-itop-top-view?p=deca-deep-viewpoint-equivariant-human-pose)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deca-deep-viewpoint-equivariant-human-pose/pose-estimation-on-itop-front-view)](https://paperswithcode.com/sota/pose-estimation-on-itop-front-view?p=deca-deep-viewpoint-equivariant-human-pose)
[![arXiv](https://img.shields.io/badge/arXiv-2108.08557-00ff00.svg)](https://arxiv.org/abs/2108.08557)

# DECA
Official code for the ICCV 2021 paper "DECA: Deep viewpoint-Equivariant human pose estimation using Capsule Autoencoders".
All the code is written using Pytorch Lightning. Please use [Pipenv](https://pipenv.pypa.io/en/latest/) to configure the virtual environment required to run the code.

![Teaser Image](/img/teaser.png)

## How to run
Use the following command to configure the virtual environment:
```
pipenv install
```
To configure all the network parameters, including the dataset paths and hyperparameters, please edit the file:
```
config/config_TV.cfg
```
or add each parameter as a runtime flag while executing the main.py file as follows:
```
python main.py --flagfile config/config_TV.cfg
```
As an example, to run the network in training mode with a dataset stored in <datasetpath>, you can run the following command:
```
python main.py --flagfile config/config_TV.cfg --mode train --dataset_dir <datasetpath>
```
