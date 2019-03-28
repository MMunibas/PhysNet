# PhysNet

Tensorflow implementation of PhysNet (see https://arxiv.org/abs/1902.08408) for details

## Requirements

To run this software, you need:

- python3 (tested with version 3.6.3)
- TensorFlow (tested with version 1.10.1)



## How to use

Edit the config.txt file to specify hyperparameters, dataset location, training/validation set size etc.
(see "train.py" for a list of all options)

Then, simply run

```
python3 train.py 
```

in a terminal to start training. 

The included "config.txt" assumes that the dataset "sn2_reactions.npz" is present. It can be downloaded from: https://zenodo.org/record/2605341. In order to use a different dataset, it needs to be formatted in the same way as this example ("sn2_reactions.npz"). Please refer to the README file of the dataset (available from https://zenodo.org/record/2605341) for details.


## How to cite

If you find this software useful, please cite:

```
Unke, O. T. and Meuwly, M. "PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges" arxiv:1902.08408 (2019).
```


