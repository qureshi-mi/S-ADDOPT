# S-ADDOPT
The code demonstrates the performance of SADDOPT with other comparable algorithms over directed graphs including GP, SGP, ADDOPT and SADDOPT presented in the paper: "https://arxiv.org/abs/2005.07785". We include the experiments for classification using logistic regression and neural networks. 

## Dependencies and Setup
All code runs on Python 3.6.7. We use some code from: "https://github.com/Hvass-Labs/TensorFlow-Tutorials" to download the CIFAR-10 dataset.

## Running Experiments
There are three main scripts:
1) LogisticRegression/ExponentialNetwork.py
2) LogisticRegression/GeometricNetwork.py
3) NeuralNetwork/NeuralNetwork.py

The rest of the files contain the classes and methods used to implement these files. The user needs to run "LogisticRegression/ExponentialNetwork.py" to simulate the results for classification of MNIST and CIFAR-10 datasets using logistic regression over an exponential graph of 16 nodes. Run "LogisticRegression/GeometricNetwork.py" to simulate the classification of MNIST and CIFAR-10 datasets using logistic regression over a 500 node geometric graph. Simulate "NeuralNetwork/NeuralNetwork.py" to get the results for classification using neural networks distributed over a geometric graph of 500 nodes.

## Generalization
This work can be used to characterize any dataset. The user only need to load custom dataset as commented in the code.

## Note
If you use this code for your research, please cite the paper.
