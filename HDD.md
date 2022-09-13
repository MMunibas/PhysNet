# Introduction 
This repository implements the PhysNet architecture in Tensorflow. Utility functions for training and inference are also provided. 

# Code structure 

    - neural_network 
        - grimme_d3 
        - layers 
        activation_fn.py
        NeuralNetwork.py
    - training 
        AMSGrad.py
        DataContainer.py
        DataProvider.py
        DataQueue.py
        Trainer.py
    NNCalculator.py
    train.py 

Implementation of the model and utility functions for training are separated into two directories - ```neural_network``` and ```training``` respectively.   

## Model Implementation - neural_network
### grimme_d3 
This contains the implementation of Grimme's D3 method in Tensorflow.  
Grimme, Stefan, et al. "A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu." The Journal of chemical physics 132.15 (2010): 154104.

### layers 
- NeuronLayer.py - This contains the base class for a neural layer which specifies the number of inputs and outputs, and the activation function that the layer will use. Every subsequent layer inherits from this class.  
- InteractionLayer.py and InteractionBlock.py - InteractionLayer implements the interaction module, which refines the interaction between an atom and surrounding atoms in the molecule. InteractionBlock obtains the final representation obtained after passing output of InteractionLayer through residual layers 
- OutputBlock.py - Implements the output block, which passes the output of the interaction block through several residual layers and a dense layer for the final output of a module 
- RBFLayer.py - Implements radial basis function which is used to model the distances between an atom and its surrounding atoms 
- ResidualLayer.py - Implements layer with skip connections used throughout the model for refining outputs 

```NeuralNetwork.py``` implements the overall architecture, putting together the various layers as well as the atom embeddings. It also uses DFT-D3 dispersion correction

## Training utilities
This contains various utility functions useful for training the model.   
- AMSGrad.py - Implementation of AMSGrad, a gradient descent algorithm that is an improvement on Adam  
- DataContainer.py - Data structure for storing the data  
- DataProvider.py - Data structure for reading the dataset and creating test-train-dev splits, as well as computing various metrics related to it 
- DataQueue.py - Data structure for feeding data to model, which places the data on threads for execution  
- Trainer.py - Helper data structure for easy training of model using a high level interface  

train.py is used to train the model, and NNCalculator.py is used for running inference. It requires the model checkpoint, and data wrapped using the Atoms datastructure from the ASE library.   
